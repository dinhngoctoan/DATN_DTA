#!/usr/bin/env python3
"""Export two standalone notebooks from the original CV notebook and ensemble.py.

Does not run training, load datasets, or modify the original notebook.
Regeneration replaces the two generated notebooks, including edited config cells.
"""
import ast
import copy
import json
from pathlib import Path
import textwrap

from sync_notebook import code, markdown, without_nodes

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'notebook'


def selected_source(source, functions, assignments=()):
    """Keep imports and selected definitions, including function decorators."""
    lines = source.splitlines(keepends=True)
    result = []
    for node in ast.parse(source).body:
        keep = isinstance(node, (ast.Import, ast.ImportFrom))
        keep |= isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions
        keep |= isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in assignments for target in node.targets)
        if keep:
            start = min([node.lineno] + [n.lineno for n in getattr(node, 'decorator_list', [])])
            result.append(''.join(lines[start - 1:node.end_lineno]) + '\n\n')
    return ''.join(result)


def write_notebook(name, cells, metadata):
    for cell in cells:
        source = ''.join(cell['source'])
        source = source.replace('DeepDTA', 'reference split').replace('deepdta_', '').replace('_deepdta', '')
        cell['source'] = source.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
            cell['execution_count'], cell['outputs'] = None, []
    path = HERE / name
    path.write_text(json.dumps({'cells': cells, 'metadata': metadata,
                               'nbformat': 4, 'nbformat_minor': 5}, ensure_ascii=False, indent=1) + '\n')
    print(f'Exported {path.name}: {len(cells)} cells, {path.stat().st_size / 1024:.0f} KiB')


def main():
    original = json.loads((HERE / 'four-source-cv.ipynb').read_text())
    cells = {c['id']: ''.join(c['source']) for c in original['cells']}
    runtime = without_nodes((ROOT / 'ensemble.py').read_text(), lambda n:
                            isinstance(n, ast.ImportFrom)
                            and n.module in ('utils', 'training', 'training_cv'))
    aggregation = selected_source(runtime, {'ensemble_metrics', 'export_ensemble',
                                            'read_prediction_rows', 'summarize'}, {'AGGREGATION'})
    training_cells = copy.deepcopy(original['cells'])
    replacements = {
        'overview': markdown('''# Four sources: five-fold CV and prediction ensembling

Four-source model with fixed splits and checkpoint selection by validation MSE.
Each fold uses four partitions for training and one for validation.
**Test aggregation follows DrugForm-DTA**: all five models predict the same test samples;
average predictions per sample, then compute MSE, RMSE, CI (concordance index), and Pearson.
No additional 90/10 validation split or UniQSAR checkpoint-weight averaging is applied.

Inputs: `train.pt`, `test.pt` in v4 format and `davis_folds.json`/`kiba_folds.json`.
This standalone notebook needs no repository on Kaggle. Ensemble metrics are withheld
until all folds are available; the mean of per-fold metrics is not the primary result.

Outputs: `ensemble_predictions.csv` contains true labels, per-fold predictions, and their mean;
`ensemble_metrics.json`/`.csv` contains ensemble and individual-model metrics for comparison;
`cv_summary.json` records whether all five folds are complete. Undefined metrics are stored as null.
Regenerate both notebooks: `python notebook/sync_ensemble_notebooks.py`.
''', 'overview'),
        'config': code(cells['config'].replace("f'{DATASET}_four_source_deepdta_cv'",
                                               "f'{DATASET}_four_source_cv_ensemble'"), 'config'),
        'cross-validation-library': code(without_nodes(cells['cross-validation-library'],
            lambda n: isinstance(n, ast.FunctionDef) and n.name == 'summarize').replace(
                'one fixed test, mean of test metrics.', 'one fixed test; ensemble summary in next cell.'),
            'cross-validation-library'),
        'run-notes': markdown('''## Running and resuming

`FOLD=None`: run all five folds sequentially. `FOLD=0..4`: train one fold; aggregate only when
the current output directory contains five compatible results completed through NUM_EPOCHS.
`CHECK_FOLDS_ONLY=True`: validate splits on CPU only.

Resuming restores optimizer/RNG/history and requires the same protocol and experiment.
To combine folds across sessions,
add previous outputs as input and run FOLD=None: completed folds are loaded and reevaluated;
incomplete folds resume training. `best_model.pt` supports inference, not resuming.

Aggregate only predictions from each fold's best validation-selected checkpoint.
CSV rows are matched by (row_id, ligand_id, protein_id) and labels rather than relying
on row order. Different experiments and unfinished training runs cannot be mixed.
To evaluate five existing models only, use `four-source-ensemble-predict.ipynb`.
''', 'run-notes'),
        'report': code('''if CHECK_FOLDS_ONLY:
    print('Validated folds:', result)
else:
    print('Completed:', len(result['completed_folds']), '/ 5 folds')
    if result['complete']:
        print('Metrics from the mean prediction of five models:')
        for metric, value in result['test_metrics'].items():
            print(f'{metric}: {value}')
        print('Prediction:', OUTPUT_DIR / 'ensemble_predictions.csv')
        print('Ensemble and per-fold metrics:', OUTPUT_DIR / 'ensemble_metrics.csv')
    else:
        print('Ensemble pending: fewer than five folds. Continue the remaining folds.')
''', 'report'),
    }
    training_cells = [replacements.get(c['id'], c) for c in training_cells]
    position = next(i for i, c in enumerate(training_cells) if c['id'] == 'run-notes')
    training_cells.insert(position, code(aggregation, 'ensemble-library'))
    write_notebook('four-source-cv-ensemble.ipynb', training_cells, original['metadata'])

    data_source = selected_source(cells['data-and-checkpoints'], {
        'FourSourceSplit', 'pack_segments', 'pack_graphs', 'same_graph', 'build_cpu_tables',
        'execution_devices', 'take_segments', 'DeviceFourSourceStore', 'IndexBatches',
        'unwrap_model', 'dataset_signature'})
    model_source = textwrap.dedent('\n'.join(cells['four-source-model'].splitlines()[1:]))
    inference_source = selected_source(cells['training-library'], {'configure_model', 'predicting'},
                                       {'MODEL_TYPE'})
    metric_source = selected_source(cells['cross-validation-library'], {'regression_metrics', 'write_json'})
    inference_runtime = selected_source(runtime, {'ensemble_metrics', 'export_ensemble',
        'load_fold_checkpoint', 'validate_checkpoint_set', 'run_ensemble_inference'}, {'AGGREGATION'})
    predict_cells = [
        markdown('''# Four sources: load five models and average their test predictions

Standalone inference notebook requiring only **five CV checkpoints and test.pt in v4 format**.
No train.pt, raw data, or fold files are needed. Accepts `best_model.pt` from the original
CV or ensemble notebook, as well as resume checkpoints from which the validation-selected
`best_checkpoint` is extracted instead of the final epoch weights.

Set TEST_PATH and CHECKPOINT_PATHS below, then Run All. The five files must cover
folds 0..4 with the same dataset, experiment, and architecture. Bare state_dict files and models
with different architectures are rejected. Model and bank code is embedded in the library cells.
Models are loaded one at a time and share the test bank to reduce memory usage.

Output: `ensemble_predictions.csv` (ID, y_true, y_pred_fold_0..4, y_pred_mean),
`ensemble_metrics.json`/`.csv` (MSE, RMSE, CI, Pearson). **CI means concordance index**,
not a confidence interval. Undefined metrics, such as CI when all labels are equal,
are stored as null. Test labels are required for metrics; keep the original scale in test.pt.
''', 'overview'),
        code('''from pathlib import Path

DATASET = 'davis'  # 'davis' or 'kiba'
REPO_ROOT = next((p for base in (Path.cwd(), *Path.cwd().parents)
                  for p in (base, base / 'DATN_DTA')
                  if (p / 'training.py').is_file() and (p / 'models').is_dir()), Path.cwd())

# On Kaggle, replace these with the corresponding Add Input paths.
TEST_PATH = REPO_ROOT / 'data/four_source_embeddings' / DATASET / 'test.pt'
MODEL_DIR = REPO_ROOT / 'outputs' / f'{DATASET}_four_source_cv_ensemble'
CHECKPOINT_PATHS = [MODEL_DIR / f'fold_{i}' / 'best_model.pt' for i in range(5)]
# Alternatively, specify five Paths from different directories; file order does not matter.

OUTPUT_DIR = (Path('/kaggle/working') if Path('/kaggle/working').is_dir()
              else REPO_ROOT / 'outputs') / f'{DATASET}_ensemble_predict'
TEST_BATCH_SIZE = 512  # Reduce if GPU memory is insufficient.
CUDA_DEVICE = 'cuda:0'  # Fall back to CPU when CUDA is unavailable.
MAX_GPUS = 1

if len(CHECKPOINT_PATHS) != 5:
    raise ValueError('Exactly five checkpoints are required')
for path in [TEST_PATH, *CHECKPOINT_PATHS]:
    if not Path(path).is_file():
        raise FileNotFoundError(f'File not found: {path}; update TEST_PATH/CHECKPOINT_PATHS.')
''', 'config'),
        code('''import importlib.util
import subprocess
import sys

if importlib.util.find_spec('torch_geometric') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch-geometric'])
''', 'dependencies'),
        code(data_source, 'test-data-library'),
        code(model_source, 'four-source-model'),
        code(inference_source, 'inference-library'),
        code(metric_source + '\n' + inference_runtime, 'ensemble-library'),
        code('''result = run_ensemble_inference(
    checkpoint_paths=CHECKPOINT_PATHS, test_path=TEST_PATH, output_dir=OUTPUT_DIR,
    dataset=DATASET, batch_size=TEST_BATCH_SIZE, device=CUDA_DEVICE, max_gpus=MAX_GPUS)

print('Metrics from the mean prediction of five models:')
for name, value in result['test_metrics'].items():
    print(f'{name}: {value}')
print('Prediction:', OUTPUT_DIR / 'ensemble_predictions.csv')
print('Metric:', OUTPUT_DIR / 'ensemble_metrics.json')

import pandas as pd
display(pd.read_csv(OUTPUT_DIR / 'ensemble_predictions.csv').head(10))
''', 'run-and-report'),
    ]
    for cell in predict_cells:
        if cell['id'] in ('test-data-library', 'four-source-model', 'inference-library', 'ensemble-library'):
            cell['metadata'] = {'jupyter': {'source_hidden': True}, 'collapsed': True}
    write_notebook('four-source-ensemble-predict.ipynb', predict_cells, original['metadata'])


if __name__ == '__main__':
    main()
