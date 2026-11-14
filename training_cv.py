"""DeepDTA splits: five train/validation folds, one fixed test, mean of test metrics.

Uses the original fold indices, not a new random KFold. Hyperparameters are supplied
by the caller; this entry point does not perform DeepDTA's hyperparameter search.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from training import (ROOT, MODEL_TYPE, TrainingConfig, build_parser as training_parser,
                      configure_model, train, predicting)
from utils import (FourSourceSplit, IndexBatches, load_data, seed_everything,
                   unwrap_model, cpu_snapshot, save_checkpoint, save_test_predictions,
                   dataset_signature, model_signature, capture_rng, restore_rng,
                   find_resume_checkpoint)

PROTOCOL = 'deepdta_fixed_test_5fold_validation_v1'


@dataclass
class CVConfig(TrainingConfig):
    raw_data_dir: Path | None = None
    fold: int | None = None
    check_folds: bool = False

    def __post_init__(self):
        default_output = self.output_dir is None
        super().__post_init__()
        if default_output:
            self.output_dir = self.output_dir.with_name(self.output_dir.name + '_deepdta_cv')
        self.raw_data_dir = Path(self.raw_data_dir) if self.raw_data_dir else ROOT / 'data'
        if self.fold is not None and self.fold not in range(5):
            raise ValueError('fold must be 0, 1, 2, 3 or 4')
        if self.resume_path is not None and self.fold is None:
            raise ValueError('--resume-path requires --fold; otherwise use automatic per-fold resume')


class PairSubset:
    """Only slice pair indices/labels; reuse the existing entity bank on every GPU."""
    def __init__(self, split, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        if (indices.ndim != 1 or not len(indices) or indices.min() < 0
                or indices.max() >= len(split) or len(indices.unique()) != len(indices)):
            raise ValueError('Invalid or empty pair subset')
        self.dataset = split.dataset
        self.ligand_ids, self.protein_ids = split.ligand_ids, split.protein_ids
        self.pairs = {key: value.index_select(0, indices) if isinstance(value, torch.Tensor)
                      else [value[i] for i in indices.tolist()]
                      for key, value in split.pairs.items()}
        self.labels = self.pairs['y'].float().reshape(-1)

    __len__ = FourSourceSplit.__len__
    pair_ids = FourSourceSplit.pair_ids


def match_original_rows(split, original_rows):
    """Return original-order -> prepared-row mapping, checking identities AND labels."""
    if len(split) != len(original_rows):
        raise ValueError(f'{split.dataset}: prepared split has {len(split)} rows; '
                         f'DeepDTA expects {len(original_rows)}')
    def matches(index, row):
        return (split.pair_ids(index) == (row['ligand_id'], row['protein_id'])
                and np.isclose(float(split.labels[index]), row['affinity'], rtol=0, atol=2e-6))

    # row_id is the position in the ORIGINAL FLATTENED SPLIT, not an index into Y.
    # Validate every identity/label before trusting it. This disambiguates KIBA
    # records whose distinct raw ligand IDs become the same canonical SMILES.
    try:
        positions = [int(value) for value in split.pairs['row_id']]
    except (TypeError, ValueError):
        positions = []
    if (sorted(positions) == list(range(len(split)))
            and all(matches(i, original_rows[pos]) for i, pos in enumerate(positions))):
        return np.argsort(positions).astype(np.int64)
    lookup = {}
    for i in range(len(split)):
        key = split.pair_ids(i)
        lookup.setdefault(key, []).append(i)
    result = []
    seen = set()
    for row in original_rows:
        key = (row['ligand_id'], row['protein_id'])
        candidates = [i for i in lookup.get(key, []) if i not in seen and matches(i, row)]
        if not candidates:
            raise ValueError(f'Pair missing/duplicated or affinity differs from DeepDTA: {key}')
        if len(candidates) > 1:
            raise ValueError('Ambiguous duplicate DeepDTA records: preserve original flattened-split '
                             'row_id when reordering prepared data')
        i = candidates[0]
        seen.add(i)
        result.append(i)
    return np.asarray(result, dtype=np.int64)


def original_folds(train_data, test_data, raw_data_dir):
    from create_data import raw_fold_rows

    folder = Path(raw_data_dir) / train_data.dataset
    groups = json.loads((folder / 'folds/train_fold_setting1.txt').read_text())
    test_ids = json.loads((folder / 'folds/test_fold_setting1.txt').read_text())
    if len(groups) != 5 or any(not isinstance(g, list) or not g for g in groups):
        raise ValueError('Expected exactly five non-empty DeepDTA training folds')
    flat = [i for group in groups for i in group]
    all_ids = flat + test_ids
    if (any(type(i) is not int or i < 0 for i in all_ids)
            or len(set(all_ids)) != len(all_ids)):
        raise ValueError('DeepDTA fold indices overlap, repeat, or are invalid')
    sequences = json.loads((folder / 'proteins.txt').read_text())
    # raw_fold_rows only checks membership in graph_source; graph data is already in v4.
    expected_train = raw_fold_rows(train_data.dataset, raw_data_dir, 'train', sequences, sequences)
    expected_test = raw_fold_rows(train_data.dataset, raw_data_dir, 'test', sequences, sequences)
    train_order = match_original_rows(train_data, expected_train)
    match_original_rows(test_data, expected_test)
    # Preserve original record-level splits. Canonicalized ligand identities can
    # coincide across raw KIBA records; changing their folds would change the benchmark.
    boundaries = np.cumsum([len(g) for g in groups])[:-1]
    return list(np.split(train_order, boundaries))


def regression_metrics(labels, predictions):
    """MSE/RMSE/Pearson and exact CI in O(n log n), excluding equal-label pairs."""
    y, p = np.asarray(labels, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    if (y.ndim != 1 or p.shape != y.shape or not len(y)
            or not np.isfinite(y).all() or not np.isfinite(p).all()):
        raise ValueError('Metrics require matching, non-empty, finite prediction/label vectors')
    mse = float(np.mean((y - p) ** 2))
    ranks = np.searchsorted(np.unique(p), p) + 1
    tree = np.zeros(int(ranks.max()) + 1, dtype=np.int64)

    def count(rank):
        total = 0
        while rank:
            total += int(tree[rank])
            rank -= rank & -rank
        return total

    order = np.argsort(y, kind='stable')
    concordant, comparable, previous, start = 0.0, 0, 0, 0
    while start < len(order):
        end = start + 1
        while end < len(order) and y[order[end]] == y[order[start]]:
            end += 1
        for index in order[start:end]:
            rank = int(ranks[index])
            below = count(rank - 1)
            concordant += below + 0.5 * (count(rank) - below)
            comparable += previous
        for index in order[start:end]:
            rank = int(ranks[index])
            while rank < len(tree):
                tree[rank] += 1
                rank += rank & -rank
        previous += end - start
        start = end
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.ptp(y) > 0 and np.ptp(p) > 0 else None
    return {'mse': mse, 'rmse': mse ** 0.5,
            'ci': concordant / comparable if comparable else None, 'pearson': pearson}


def write_json(path, value):
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def run_fold(config, fold_id, train_data, valid_data, test_data, store, experiment_id):
    output = config.output_dir / f'fold_{fold_id}'
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    train_batches = IndexBatches(train_data, store, config.batch_size, config.shuffle, config.seed)
    valid_batches = IndexBatches(valid_data, store, config.test_batch_size)
    test_batches = IndexBatches(test_data, store, config.test_batch_size)
    model = configure_model(config.model_config, store)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    identity = {'protocol': PROTOCOL, 'fold': fold_id, 'experiment_id': experiment_id,
                'model_config': dict(config.model_config), 'model_signature': model_signature(model),
                'dataset_signature': dataset_signature((train_data, valid_data, test_data))}
    best, history, start_epoch = None, [], 1
    resume_path, resume = (find_resume_checkpoint([output, config.input_dir], identity, config.resume_path)
                           if config.auto_resume or config.resume_path else (None, None))
    if resume is not None:
        unwrap_model(model).load_state_dict(resume['model_state_dict'])
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        best = cpu_snapshot(resume['best_checkpoint'])
        history, start_epoch = list(resume['history']), resume['epoch'] + 1
        restore_rng(resume['rng_state'], train_batches, store.devices)
        print(f'Fold {fold_id}: resume {resume_path}, next epoch {start_epoch}')
        del resume
    print(f'Fold {fold_id}: {len(train_data):,} train / {len(valid_data):,} validation / '
          f'{len(test_data):,} fixed test; checkpoint selection = validation MSE')
    with (output / 'history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_mse', 'validation_mse', 'seconds'])
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        for epoch in range(start_epoch, config.epochs + 1):
            started = time.perf_counter()
            train_mse = train(model, train_batches, optimizer, epoch, config.log_interval)
            labels, predictions = predicting(model, valid_batches)
            if not np.isfinite(predictions).all():
                raise FloatingPointError('Validation prediction contains NaN/Inf')
            validation_mse = float(np.mean((labels.astype(np.float64) - predictions.astype(np.float64)) ** 2))
            if best is None or validation_mse < best['validation_mse']:
                best = {'model_type': MODEL_TYPE, **identity,
                        'model_state_dict': cpu_snapshot(unwrap_model(model).state_dict()),
                        'epoch': epoch, 'validation_mse': validation_mse,
                        'selection_split': 'validation', 'dataset': config.dataset}
                save_checkpoint(best, output / 'best_model.pt')
            record = {'epoch': epoch, 'train_mse': train_mse, 'validation_mse': validation_mse,
                      'seconds': time.perf_counter() - started}
            history.append(record)
            writer.writerow(record)
            handle.flush()
            if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
                save_checkpoint({
                    'checkpoint_kind': 'training_resume_v1', **identity, 'epoch': epoch,
                    'model_state_dict': cpu_snapshot(unwrap_model(model).state_dict()),
                    'optimizer_state_dict': cpu_snapshot(optimizer.state_dict()),
                    'best_checkpoint': best, 'best_predictions': None,
                    'history': history, 'rng_state': capture_rng(train_batches, store.devices),
                }, output / f'training_checkpoint_epoch_{epoch:04d}.pt')
            print(f'Fold {fold_id}, epoch {epoch}: train={train_mse:.6f}, '
                  f'validation={validation_mse:.6f}, best epoch={best["epoch"]}')
    if best is None:
        raise RuntimeError('No trained checkpoint available')
    # Evaluate the held-out test only after checkpoint selection is finished.
    unwrap_model(model).load_state_dict(best['model_state_dict'])
    save_checkpoint(best, output / 'best_model.pt')
    labels, predictions = predicting(model, test_batches)
    metrics = regression_metrics(labels, predictions)
    save_test_predictions(output / 'test_predictions.csv', test_data, labels, predictions)
    result = {'protocol': PROTOCOL, 'experiment_id': experiment_id, 'fold': fold_id,
              'requested_epochs': config.epochs, 'completed_epoch': history[-1]['epoch'],
              'best_epoch': best['epoch'], 'validation_mse': best['validation_mse'],
              'n_train': len(train_data), 'n_validation': len(valid_data), 'n_test': len(test_data),
              'test_metrics': metrics}
    write_json(output / 'metrics.json', result)
    print(f'Fold {fold_id} test metrics: {metrics}')
    return result


def summarize(output_dir, experiment_id, epochs):
    records = []
    for fold in range(5):
        path = Path(output_dir) / f'fold_{fold}/metrics.json'
        if path.is_file():
            record = json.loads(path.read_text())
            if (record.get('experiment_id') == experiment_id and record.get('fold') == fold
                    and record.get('requested_epochs') == epochs):
                records.append(record)
    metrics = {}
    for key in ('mse', 'rmse', 'ci', 'pearson'):
        values = [r['test_metrics'][key] for r in records]
        defined = values and all(v is not None for v in values)
        metrics[key] = {'mean': float(np.mean(values)) if defined else None,
                        'std': float(np.std(values, ddof=1)) if defined and len(values) > 1 else None}
    summary = {'protocol': PROTOCOL, 'experiment_id': experiment_id, 'requested_epochs': epochs,
               'complete': len(records) == 5, 'completed_folds': [r['fold'] for r in records],
               'aggregation': 'mean of per-model test metrics, not ensemble predictions',
               'std_ddof': 1, 'test_metrics': metrics, 'folds': records}
    write_json(Path(output_dir) / 'cv_summary.json', summary)
    with (Path(output_dir) / 'cv_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['fold', 'best_epoch', 'validation_mse',
                                                    'mse', 'rmse', 'ci', 'pearson'])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in ('fold', 'best_epoch', 'validation_mse')} | r['test_metrics'])
        for stat in ('mean', 'std'):
            writer.writerow({'fold': stat, **{key: value[stat] for key, value in metrics.items()}})
    return summary


def run_cross_validation(config: CVConfig):
    seed_everything(config.seed)
    if config.check_folds:
        train_data = FourSourceSplit(config.data_dir / 'train.pt', 'train', config.dataset)
        test_data = FourSourceSplit(config.data_dir / 'test.pt', 'test', config.dataset)
        groups = original_folds(train_data, test_data, config.raw_data_dir)
        result = {'dataset': config.dataset, 'validation_sizes': [len(g) for g in groups],
                  'training_sizes': [len(train_data) - len(g) for g in groups], 'test_size': len(test_data)}
        print(json.dumps(result, indent=2))
        return result
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    train_data, test_data, store, unused_train, unused_test = load_data(
        config.data_dir, config.dataset, device, config.batch_size, config.test_batch_size,
        config.max_gpus, config.shuffle, config.seed)
    del unused_train, unused_test
    groups = original_folds(train_data, test_data, config.raw_data_dir)
    probe_model = configure_model(config.model_config, store)
    architecture = model_signature(probe_model)
    del probe_model
    experiment = {'protocol': PROTOCOL, 'dataset': config.dataset,
                  'dataset_signature': dataset_signature((train_data, test_data)),
                  'fold_signature': hashlib.sha256(b''.join(g.tobytes() for g in groups)).hexdigest(),
                  'model_config': dict(config.model_config), 'model_signature': architecture,
                  'optimizer': 'Adam', 'lr': config.lr,
                  'batch_size': config.batch_size, 'test_batch_size': config.test_batch_size,
                  'shuffle': config.shuffle, 'seed': config.seed}
    experiment_id = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = config.output_dir / 'experiment.json'
    if manifest.exists() and json.loads(manifest.read_text()) != experiment:
        raise ValueError('Output belongs to a different CV experiment; choose a new --output-dir')
    write_json(manifest, experiment)
    with (config.output_dir / 'fold_assignments.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['prepared_train_index', 'row_id', 'validation_fold', 'ligand_id', 'protein_id'])
        for fold, indices in enumerate(groups):
            for i in indices:
                writer.writerow([int(i), train_data.pairs['row_id'][i], fold, *train_data.pair_ids(i)])
    for fold in range(5) if config.fold is None else [config.fold]:
        training_indices = np.concatenate([g for i, g in enumerate(groups) if i != fold])
        run_fold(config, fold, PairSubset(train_data, training_indices),
                 PairSubset(train_data, groups[fold]), test_data, store, experiment_id)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        summary = summarize(config.output_dir, experiment_id, config.epochs)
        print(f'Completed {len(summary["completed_folds"])}/5 folds. Summary: '
              f'{config.output_dir / "cv_summary.json"}')
    return summary


def build_parser():
    parser = training_parser()
    parser.description = __doc__
    parser.add_argument('--raw-data-dir', type=Path, help='Parent of davis/ and kiba/ containing original folds, Y, ligands_can.txt, proteins.txt')
    parser.add_argument('--fold', type=int, choices=range(5), help='Run just one fold (0..4); default: all five')
    parser.add_argument('--check-folds', action='store_true', help='Validate prepared splits against original DeepDTA pairs/labels without training')
    return parser


def main(argv=None):
    return run_cross_validation(CVConfig(**vars(build_parser().parse_args(argv))))


if __name__ == '__main__':
    main()
