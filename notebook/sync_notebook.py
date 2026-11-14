#!/usr/bin/env python3
"""Export repository modules into the standalone Kaggle notebook (without executing it)."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebook/davis-attention-original copy.ipynb'


def without_nodes(source, predicate):
    lines = source.splitlines(keepends=True)
    for node in reversed(ast.parse(source).body):
        if predicate(node):
            del lines[node.lineno - 1:node.end_lineno]
    return ''.join(lines)


def markdown(text, cell_id):
    return {'cell_type': 'markdown', 'id': cell_id, 'metadata': {},
            'source': text.splitlines(keepends=True)}


def code(text, cell_id):
    compile(text, cell_id, 'exec')
    return {'cell_type': 'code', 'id': cell_id, 'metadata': {},
            'execution_count': None, 'outputs': [], 'source': text.splitlines(keepends=True)}


def main():
    nb = json.loads(TARGET.read_text())
    old_config = next((''.join(c['source']) for c in nb['cells']
                       if c['cell_type'] == 'code' and 'DATA_DIR = None' in ''.join(c['source'])), None)
    # Keep the editable configuration on repeat exports. Model/data/training cells come from .py.
    config = old_config if old_config and 'REPO_ROOT =' in old_config else '''from pathlib import Path

DATASET = 'davis'  # 'davis' or 'kiba'
DATA_DIR = None  # Set Path('/kaggle/input/...') if multiple datasets are available.
REPO_ROOT = next((p for base in (Path.cwd(), *Path.cwd().parents)
                  for p in (base, base / 'DATN_DTA')
                  if (p / 'training.py').is_file() and (p / 'models').is_dir()), Path.cwd())
if DATA_DIR is None:
    if Path('/kaggle/input').is_dir():
        candidates = sorted({p.parent for p in Path('/kaggle/input').rglob('train.pt')
                             if (p.parent / 'test.pt').is_file()})
        if len(candidates) != 1:
            raise ValueError(f'Found {len(candidates)} datasets; set DATA_DIR explicitly.')
        DATA_DIR = candidates[0]
    else:
        DATA_DIR = REPO_ROOT / 'data/four_source_embeddings' / DATASET
OUTPUT_DIR = (Path('/kaggle/working') if Path('/kaggle/working').is_dir()
              else REPO_ROOT / 'outputs') / f'{DATASET}_drug_gat8x64_gat4x64_dual_embedding_maxmean'
TRAIN_BATCH_SIZE = 512  # Global batch, split across GPUs.
TEST_BATCH_SIZE = 512
MAX_GPUS = 2  # Set to 1 to use a single GPU.
CUDA_DEVICE = 'cuda:0'
LR = 0.0005
NUM_EPOCHS = 1000  # Total epochs, including those completed before resuming.
CHECKPOINT_EVERY = 100
AUTO_RESUME = True
RESUME_PATH = None  # Full checkpoint Path; None scans output and /kaggle/input.
SEED = 42
SHUFFLE_TRAIN = False
LOG_INTERVAL = 20
'''
    if 'MODEL_CONFIG =' not in config:
        config += "\nMODEL_CONFIG = {'num_features_xd': 78, 'num_features_xt': 41, 'n_output': 1,\n                'output_dim': 128, 'dropout': 0.2, 'protein_embedding_dim': 640}\n"
    training = (ROOT / 'training.py').read_text()
    training = without_nodes(training, lambda n: (
        isinstance(n, ast.ImportFrom) and n.module == 'utils') or (
        isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"))
    training = training.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = REPO_ROOT')
    training = training.replace('    from models.gat_gcn import GATnGCN_Model\n', '')
    nb['cells'] = [
        markdown('''# DTA: drug/protein graph + drug/protein embedding

Model: drug GAT 8x64 -> GAT 4x64; protein GAT 8x64 -> GCN 256.
Four branches with max+mean pooling -> concatenation -> MLP 256->512->128->1. MSE loss, fixed LR.

Standalone notebook: add `train.pt` / `test.pt` in v4 format as inputs, select DATASET, then Run All.
Full checkpoints are saved every 100 epochs and at the final epoch; compatible checkpoints are found automatically.
`best_model.pt` is for inference only, not a full resume checkpoint.

Library cells are exported from `utils.py`, `models/gat_gcn.py`, and `training.py` using
`python notebook/sync_notebook.py`. Edit the repository model and regenerate to keep both versions aligned.
''', 'overview'),
        code(config, 'config'),
        code('''import importlib.util
import subprocess
import sys

if importlib.util.find_spec('torch_geometric') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch-geometric'])
''', 'dependencies'),
        code((ROOT / 'utils.py').read_text(), 'data-and-checkpoints'),
        code((ROOT / 'models/gat_gcn.py').read_text(), 'four-source-model'),
        code(training, 'training-library'),
        code('''config = TrainingConfig(
    dataset=DATASET, data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
    device=CUDA_DEVICE, max_gpus=MAX_GPUS,
    batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
    lr=LR, epochs=NUM_EPOCHS, checkpoint_every=CHECKPOINT_EVERY,
    auto_resume=AUTO_RESUME, resume_path=RESUME_PATH,
    seed=SEED, shuffle=SHUFFLE_TRAIN, log_interval=LOG_INTERVAL,
    model_config=MODEL_CONFIG)
result = run_training(config)
print('Best epoch:', result['best_epoch'], '| test MSE:', result['best_mse'])
''', 'run'),
        code('''# run_training restores the best checkpoint before returning the model.
model = result['model']
pair_indices, _ = next(iter(result['test_batches']))
with torch.no_grad():
    example_predictions = model(pair_indices)
print('Example predictions:', example_predictions[:5].cpu().flatten())
del pair_indices, example_predictions
''', 'inference-example'),
        markdown('''Outputs: `best_model.pt`, `best_test_predictions.csv`, `history.csv`, and
`training_checkpoint_epoch_XXXX.pt`. Add a full checkpoint as input in the next session to resume.
Checkpoints retain current weights and optimizer state, a separate best model, history, and RNG state.
Changes to architecture, split, LR, or batch size prevent automatic resume from an old checkpoint.
Moving code from a notebook to a module does not itself break checkpoint compatibility for the same architecture.
Test MSE selects the best checkpoint in this legacy workflow, so the test set is not an independent evaluation set.
Save `/kaggle/working` files as Kaggle outputs to reuse them in another session.
''', 'outputs'),
    ]
    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    print(f'Exported {TARGET}')


if __name__ == '__main__':
    main()
