#!/usr/bin/env python3
"""Export standalone CV notebook and external original-fold JSON files.

Only reads raw identities/labels/folds at export time. Does not prepare embeddings,
instantiate a model, or train. Kaggle needs the v4 .pt files and one fold JSON.
"""
import ast
import json
from pathlib import Path
import sys
import textwrap

from sync_notebook import code, markdown, without_nodes

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebook/four-source-cv.ipynb'
FOLD_DIR = ROOT / 'data/cv_folds'


def export_fold_files():
    sys.path.insert(0, str(ROOT))
    from create_data import raw_fold_rows

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    for name in ('davis', 'kiba'):
        folder = ROOT / 'data' / name
        groups = json.loads((folder / 'folds/train_fold_setting1.txt').read_text())
        test_ids = json.loads((folder / 'folds/test_fold_setting1.txt').read_text())
        ids = [i for group in groups for i in group] + test_ids
        if (len(groups) != 5 or any(not g for g in groups)
                or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError(f'{name}: invalid original folds')
        sequences = json.loads((folder / 'proteins.txt').read_text())
        splits = {s: raw_fold_rows(name, ROOT / 'data', s, sequences, sequences)
                  for s in ('train', 'test')}
        lids = sorted({r['ligand_id'] for rows in splits.values() for r in rows})
        pids = sorted({r['protein_id'] for rows in splits.values() for r in rows})
        lm, pm = {v: i for i, v in enumerate(lids)}, {v: i for i, v in enumerate(pids)}
        payload = {'format_version': 1, 'dataset': name,
                   'ligand_ids': lids, 'protein_ids': pids,
                   'fold_sizes': [len(g) for g in groups],
                   **{s: [[lm[r['ligand_id']], pm[r['protein_id']], r['affinity']]
                          for r in rows] for s, rows in splits.items()}}
        path = FOLD_DIR / f'{name}_folds.json'
        path.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
        print(f'Exported {path}')


FOLD_LOADER = '''def original_folds(train_data, test_data, raw_data_dir=None):
    """Restore original interaction folds; verify pairs AND labels in both .pt files."""
    if test_data.dataset != train_data.dataset:
        raise ValueError('Train/test must belong to the same dataset')
    name = train_data.dataset
    if FOLDS_PATH is not None:
        path = Path(FOLDS_PATH)
    else:
        filename = f'{name}_folds.json'
        path = Path(DATA_DIR) / filename
        if not path.is_file():
            if Path('/kaggle/input').is_dir():
                candidates = sorted(Path('/kaggle/input').rglob(filename))
            else:
                candidates = [REPO_ROOT / 'data/cv_folds' / filename]
                candidates = [p for p in candidates if p.is_file()]
            if len(candidates) != 1:
                raise ValueError(f'Found {len(candidates)} files named {filename}; set FOLDS_PATH explicitly.')
            path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f'Fold file not found: {path}; update FOLDS_PATH.')
    payload = json.loads(path.read_text())
    if payload.get('format_version') != 1 or payload.get('dataset') != name:
        raise ValueError(f'{path}: incorrect dataset or fold file format')
    def expected_rows(split):
        return [{'ligand_id': payload['ligand_ids'][li],
                 'protein_id': payload['protein_ids'][pi], 'affinity': y}
                for li, pi, y in payload[split]]
    train_order = match_original_rows(train_data, expected_rows('train'))
    match_original_rows(test_data, expected_rows('test'))
    sizes = payload['fold_sizes']
    if (len(sizes) != 5 or any(type(n) is not int or n < 1 for n in sizes)
            or sum(sizes) != len(train_order)):
        raise ValueError('Invalid fold sizes')
    print('Fold file:', path)
    return list(np.split(train_order, np.cumsum(sizes)[:-1]))
'''


CONFIG = '''from pathlib import Path

DATASET = 'davis'  # 'davis' or 'kiba'
DATA_DIR = None  # Path('/kaggle/input/.../davis') containing both train.pt and test.pt.
FOLDS_PATH = None  # Discover davis_folds.json/kiba_folds.json, or set an explicit Path.
REPO_ROOT = next((p for base in (Path.cwd(), *Path.cwd().parents)
                  for p in (base, base / 'DATN_DTA')
                  if (p / 'training.py').is_file() and (p / 'models').is_dir()), Path.cwd())
if DATASET not in ('davis', 'kiba'):
    raise ValueError('DATASET must be davis or kiba')
if DATA_DIR is None:
    if Path('/kaggle/input').is_dir():
        candidates = sorted({p.parent for p in Path('/kaggle/input').rglob('train.pt')
                             if (p.parent / 'test.pt').is_file()})
        matching = [p for p in candidates if DATASET in p.as_posix().lower()]
        candidates = matching or candidates
        if len(candidates) != 1:
            raise ValueError(f'Found {len(candidates)} datasets; set DATA_DIR explicitly: {candidates}')
        DATA_DIR = candidates[0]
    else:
        DATA_DIR = REPO_ROOT / 'data/four_source_embeddings' / DATASET
DATA_DIR = Path(DATA_DIR)
OUTPUT_DIR = (Path('/kaggle/working') if Path('/kaggle/working').is_dir()
              else REPO_ROOT / 'outputs') / f'{DATASET}_four_source_deepdta_cv'
INPUT_DIR = Path('/kaggle/input')
FOLD = None  # None: run all five folds sequentially; 0..4: run one fold.
CHECK_FOLDS_ONLY = False  # True: validate data/folds on CPU without training.
TRAIN_BATCH_SIZE = 512  # Global batch, split across GPUs when MAX_GPUS > 1.
TEST_BATCH_SIZE = 512
MAX_GPUS = 2
CUDA_DEVICE = 'cuda:0'
LR = 0.0005  # Fixed throughout training.
NUM_EPOCHS = 1000  # Per fold, including epochs completed before resuming.
CHECKPOINT_EVERY = 100
AUTO_RESUME = True
RESUME_PATH = None  # Full checkpoint path; requires FOLD=0..4 when specified.
SEED = 42
SHUFFLE_TRAIN = False
LOG_INTERVAL = 20
MODEL_CONFIG = {'num_features_xd': 78, 'num_features_xt': 41, 'n_output': 1,
                'output_dim': 128, 'dropout': 0.2, 'protein_embedding_dim': 640}
print('Dataset:', DATASET, '| data:', DATA_DIR, '| output:', OUTPUT_DIR)
'''


def main():
    config = CONFIG
    if TARGET.exists():
        previous = json.loads(TARGET.read_text())
        config = next((''.join(c['source']) for c in previous['cells']
                       if c.get('id') == 'config'), config)
    if not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FOLDS_PATH'
               for t in n.targets) for n in ast.parse(config).body):
        config += "\nFOLDS_PATH = None  # Path('/kaggle/input/.../davis_folds.json'); None: auto-discover.\n"
    training = without_nodes((ROOT / 'training.py').read_text(), lambda n: (
        isinstance(n, ast.ImportFrom) and n.module == 'utils') or (
        isinstance(n, ast.FunctionDef) and n.name in ('run_training', 'build_parser', 'main')) or (
        isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"))
    training = training.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = REPO_ROOT')
    training = training.replace('    from models.gat_gcn import GATnGCN_Model\n', '')
    cv = without_nodes((ROOT / 'training_cv.py').read_text(), lambda n: (
        isinstance(n, ast.ImportFrom) and n.module in ('training', 'utils')) or (
        isinstance(n, ast.FunctionDef) and n.name in ('original_folds', 'build_parser', 'main')) or (
        isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"))
    export_fold_files()
    cells = [
        markdown('''# Four sources: five-fold validation with DeepDTA splits and a fixed test set

Add **train.pt, test.pt in v4 format and davis_folds.json or kiba_folds.json** as inputs,
select DATASET, then Run All. FOLDS_PATH=None discovers the fold file; an explicit Path is also supported.
This standalone notebook supports DAVIS/KIBA without the repo, raw data, or repeated preprocessing.
Original folds are loaded from a separate JSON file. Pair IDs and labels are checked
before use; mismatched data never triggers an automatic random split.

Each run trains on four folds, selects the best validation MSE on the remaining fold, and evaluates
on the same test.pt. Each fold starts with a fresh model. Reports contain the mean and sample
standard deviation of five models' metrics, without prediction ensembling. The shared four-source bank
is indexed in forward. The model is exported directly from models/gat_gcn.py.

This uses DeepDTA splits and metric aggregation with your fixed hyperparameters;
it does not reproduce the paper's validation-CI hyperparameter search.
[DeepDTA, sections 3.3-3.4](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129291/).
Regenerate from the repo: `python notebook/sync_cv_notebook.py`.
''', 'overview'),
        code(config, 'config'),
        code('''import importlib.util
import subprocess
import sys

if not CHECK_FOLDS_ONLY and importlib.util.find_spec('torch_geometric') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch-geometric'])
''', 'dependencies'),
        code((ROOT / 'utils.py').read_text(), 'data-and-checkpoints'),
        code('if not CHECK_FOLDS_ONLY:\n' + textwrap.indent(
            (ROOT / 'models/gat_gcn.py').read_text(), '    '), 'four-source-model'),
        code(training, 'training-library'),
        code(cv + '\n' + FOLD_LOADER, 'cross-validation-library'),
        markdown('''## Running and resuming

`FOLD=None` runs folds 0-4 sequentially, each for NUM_EPOCHS epochs at a fixed learning rate.
`CHECK_FOLDS_ONLY=True` validates splits on CPU without creating a model.
To run a single fold, set FOLD=0..4; results are provisional until all five folds finish.

Full checkpoints are saved every 100 epochs and at the final epoch, under fold_0/.../fold_4.
In a later session, add saved outputs as input and retain the configuration with FOLD=None to find
compatible checkpoints per fold. Completed folds are loaded and evaluated without further training;
partial folds resume. `best_model.pt` supports inference only and lacks the state required to resume.
Save Kaggle outputs before switching sessions; checkpoints from the old single-run workflow
are incompatible with CV. To rebuild the full summary across sessions, use FOLD=None with
checkpoints from completed folds so their results are regenerated in the current output directory.
''', 'run-notes'),
        code('''config = CVConfig(
    dataset=DATASET, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, input_dir=INPUT_DIR,
    fold=FOLD, check_folds=CHECK_FOLDS_ONLY,
    device=CUDA_DEVICE, max_gpus=MAX_GPUS,
    batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
    lr=LR, epochs=NUM_EPOCHS, checkpoint_every=CHECKPOINT_EVERY,
    auto_resume=AUTO_RESUME, resume_path=RESUME_PATH,
    seed=SEED, shuffle=SHUFFLE_TRAIN, log_interval=LOG_INTERVAL,
    model_config=MODEL_CONFIG)
result = run_cross_validation(config)
''', 'run'),
        code('''if CHECK_FOLDS_ONLY:
    print('Validated folds:', result)
else:
    print('Completed:', len(result['completed_folds']), '/ 5 folds')
    print('All five folds complete' if result['complete'] else 'PROVISIONAL results: fewer than five folds')
    for metric, values in result['test_metrics'].items():
        print(f"{metric}: mean={values['mean']}, std={values['std']}")
    print('Per-fold metrics and mean/std:', OUTPUT_DIR / 'cv_results.csv')
''', 'report'),
    ]
    nb = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.0'}},
        'nbformat': 4, 'nbformat_minor': 5}
    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    print(f'Exported {TARGET} ({TARGET.stat().st_size / 1024:.0f} KiB)')


if __name__ == '__main__':
    main()
