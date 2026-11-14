"""Five-fold prediction averaging, exported into both standalone notebooks."""
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from utils import (FourSourceSplit, build_cpu_tables, DeviceFourSourceStore,
                   execution_devices, IndexBatches, dataset_signature, unwrap_model)
from training import MODEL_TYPE, configure_model, predicting
from training_cv import regression_metrics, write_json


AGGREGATION = 'arithmetic_mean_of_5_fold_predictions_then_metrics_v1'


def ensemble_metrics(labels, predictions_by_fold):
    if set(predictions_by_fold) != set(range(5)):
        raise ValueError('Ensemble requires exactly 5 folds with distinct IDs: 0, 1, 2, 3, 4')
    labels = np.asarray(labels, dtype=np.float64)
    predictions = [np.asarray(predictions_by_fold[i], dtype=np.float64) for i in range(5)]
    # Validate every model before averaging: never silently drop NaN/missing folds.
    individual = {str(i): regression_metrics(labels, p) for i, p in enumerate(predictions)}
    matrix = np.stack(predictions, axis=1)
    mean_prediction = matrix.mean(axis=1)
    return matrix, mean_prediction, regression_metrics(labels, mean_prediction), individual


def export_ensemble(output_dir, rows, predictions_by_fold, metadata):
    """Rows and every prediction vector must refer to the same ordered test samples."""
    labels = np.asarray([float(row['affinity']) for row in rows], dtype=np.float64)
    matrix, mean_prediction, metrics, individual = ensemble_metrics(labels, predictions_by_fold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / 'ensemble_predictions.csv'
    temporary = prediction_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_id', 'ligand_id', 'protein_id', 'y_true',
                         *[f'y_pred_fold_{i}' for i in range(5)], 'y_pred_mean'])
        for index, row in enumerate(rows):
            writer.writerow([row['row_id'], row['ligand_id'], row['protein_id'],
                             labels[index], *matrix[index], mean_prediction[index]])
    temporary.replace(prediction_path)
    summary = {**metadata, 'aggregation': AGGREGATION, 'complete': True,
               'completed_folds': list(range(5)), 'n_test': len(labels),
               'test_metrics': metrics, 'per_model_test_metrics': individual,
               'prediction_file': prediction_path.name}
    write_json(output_dir / 'ensemble_metrics.json', summary)
    with (output_dir / 'ensemble_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', *metrics])
        writer.writeheader()
        for fold, values in individual.items():
            writer.writerow({'model': f'fold_{fold}', **values})
        writer.writerow({'model': 'ensemble_mean_prediction', **metrics})
    return summary


def read_prediction_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'row_id', 'ligand_id', 'protein_id', 'affinity', 'predicted_affinity'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing ID/affinity/predicted_affinity columns')
        rows = list(reader)
    keys = [(r['row_id'], r['ligand_id'], r['protein_id']) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError(f'{path}: empty test set or duplicate (row_id, ligand_id, protein_id) keys')
    return rows, keys


def summarize(output_dir, experiment_id, epochs):
    """Replacement for the original notebook's metric-mean summary; retain CV training."""
    output_dir = Path(output_dir)
    records = []
    for fold in range(5):
        path = output_dir / f'fold_{fold}/metrics.json'
        if path.is_file():
            record = json.loads(path.read_text())
            if (record.get('experiment_id') == experiment_id and record.get('fold') == fold
                    and record.get('requested_epochs') == epochs
                    and record.get('completed_epoch', 0) >= epochs):
                records.append(record)
    summary = {'experiment_id': experiment_id, 'requested_epochs': epochs,
               'aggregation': AGGREGATION, 'complete': False,
               'completed_folds': [r['fold'] for r in records], 'test_metrics': None,
               'folds': records}
    # Invalidate an old complete export before attempting a new aggregation.
    for filename in ('ensemble_predictions.csv', 'ensemble_metrics.json', 'ensemble_metrics.csv'):
        (output_dir / filename).unlink(missing_ok=True)
    write_json(output_dir / 'cv_summary.json', summary)
    if len(records) < 5:
        print(f'Ensemble pending: all 5 folds are required, but only {len(records)} are available. No partial average.')
        return summary
    reference_rows, reference_keys = read_prediction_rows(output_dir / 'fold_0/test_predictions.csv')
    reference_labels = np.asarray([float(r['affinity']) for r in reference_rows])
    predictions = {}
    for record in records:
        fold = record['fold']
        rows, keys = read_prediction_rows(output_dir / f'fold_{fold}/test_predictions.csv')
        if set(keys) != set(reference_keys):
            raise ValueError(f'Fold {fold}: test sample IDs differ from fold 0')
        by_key = dict(zip(keys, rows))
        ordered = [by_key[key] for key in reference_keys]
        labels = np.asarray([float(row['affinity']) for row in ordered])
        if not np.array_equal(labels, reference_labels):
            raise ValueError(f'Fold {fold}: test labels differ from fold 0')
        predictions[fold] = np.asarray([float(row['predicted_affinity']) for row in ordered])
        calculated = regression_metrics(labels, predictions[fold])
        for key, value in calculated.items():
            saved = record['test_metrics'].get(key)
            if ((value is None) != (saved is None)
                    or value is not None and not np.isclose(value, saved, rtol=1e-10, atol=1e-12)):
                raise ValueError(f'Fold {fold}: CSV does not match metrics.json ({key})')
        if record.get('n_test') != len(ordered):
            raise ValueError(f'Fold {fold}: test sample count does not match metrics.json')
    summary = export_ensemble(output_dir, reference_rows, predictions, summary)
    write_json(output_dir / 'cv_summary.json', summary)
    return summary


def load_fold_checkpoint(path):
    """Accept best_model.pt or extract the validation-selected best from a resume file."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True, mmap=True)
    if isinstance(payload, dict) and payload.get('checkpoint_kind') == 'training_resume_v1':
        payload = payload.get('best_checkpoint')
    required = {'fold', 'dataset', 'experiment_id', 'model_config', 'model_state_dict',
                'selection_split', 'validation_mse', 'epoch', 'model_type'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f'{path}: expected a CV checkpoint with metadata (best_model.pt or resume), '
                         'not a bare state_dict or single-run checkpoint')
    if type(payload['fold']) is not int or payload['fold'] not in range(5):
        raise ValueError(f'{path}: fold must be in 0..4')
    if (payload['selection_split'] != 'validation'
            or not np.isfinite(payload['validation_mse']) or payload['epoch'] < 1):
        raise ValueError(f'{path}: invalid validation-selected checkpoint')
    if payload['model_type'] != MODEL_TYPE:
        raise ValueError(f'{path}: model architecture is not supported by this notebook')
    return payload


def validate_checkpoint_set(checkpoint_paths, dataset):
    paths = [Path(path).expanduser().resolve() for path in checkpoint_paths]
    if len(paths) != 5 or len(set(paths)) != 5:
        raise ValueError('CHECKPOINT_PATHS must contain exactly 5 files with distinct paths')
    by_fold, common = {}, None
    for path in paths:
        payload = load_fold_checkpoint(path)
        fold = payload['fold']
        if payload['dataset'] != dataset:
            raise ValueError(f'{path}: checkpoint dataset differs from {dataset}')
        if fold in by_fold:
            raise ValueError(f'Two checkpoints have the same fold {fold}; all folds 0..4 are required')
        identity = {key: payload.get(key) for key in
                    ('experiment_id', 'model_config', 'model_type', 'model_signature', 'protocol')}
        if not identity['experiment_id'] or not isinstance(identity['model_config'], dict):
            raise ValueError(f'{path}: missing experiment identity or model config')
        if common is not None and identity != common:
            raise ValueError('All 5 checkpoints must have the same experiment, architecture, and model config')
        common = identity
        by_fold[fold] = {'path': str(path), 'fold': fold, 'best_epoch': payload['epoch'],
                         'validation_mse': payload['validation_mse']}
        del payload
    return by_fold, common


def run_ensemble_inference(checkpoint_paths, test_path, output_dir, dataset='davis',
                           batch_size=512, device='cuda:0', max_gpus=1):
    """Test-only bank, one model at a time, same test row order, no training or fold JSON."""
    if dataset not in ('davis', 'kiba'):
        raise ValueError('DATASET must be davis or kiba')
    by_fold, common = validate_checkpoint_set(checkpoint_paths, dataset)
    config = common['model_config']
    test_data = FourSourceSplit(Path(test_path), 'test', dataset,
                               protein_embedding_dim=config.get('protein_embedding_dim', 640))
    signature = dataset_signature((test_data,))
    rows = [{'row_id': str(test_data.pairs['row_id'][i]),
             'ligand_id': test_data.pair_ids(i)[0], 'protein_id': test_data.pair_ids(i)[1],
             'affinity': float(test_data.labels[i])} for i in range(len(test_data))]
    tables, ligand_map, protein_map = build_cpu_tables((test_data,))
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    devices = execution_devices(device, use_data_parallel=max_gpus > 1, max_gpus=max_gpus)
    store = DeviceFourSourceStore(tables, devices, ligand_map, protein_map)
    test_data.data = None
    del tables
    batches = IndexBatches(test_data, store, batch_size, shuffle=False)
    expected_labels = test_data.labels.numpy()
    predictions = {}
    for fold in range(5):
        item = by_fold[fold]
        payload = load_fold_checkpoint(item['path'])
        model = configure_model(config, store)
        unwrap_model(model).load_state_dict(payload['model_state_dict'], strict=True)
        del payload
        labels, predictions[fold] = predicting(model, batches)
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f'Fold {fold}: test order or labels changed during inference')
        print(f'Fold {fold}, best epoch {item["best_epoch"]}: '
              f'{regression_metrics(labels, predictions[fold])}')
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return export_ensemble(output_dir, rows, predictions, {
        'dataset': dataset, 'experiment_id': common['experiment_id'],
        'model_config': config, 'test_path': str(Path(test_path).resolve()),
        'test_signature': signature, 'checkpoints': [by_fold[i] for i in range(5)],
    })
