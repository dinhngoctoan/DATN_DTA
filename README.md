# DATN_DTA: four-source drug-target affinity prediction

The current workflow combines **drug graphs, drug embeddings, protein graphs, and
protein embeddings**. The default model in `models/gat_gcn.py` matches the current
`davis-attention-original copy.ipynb` notebook, including two drug GAT layers.

## Repository layout

```text
DATN_DTA/
├── create_data.py                       # Prepare and validate v4 data
├── training.py                          # Single-run training and checkpoint resume
├── training_cv.py                       # Five validation folds and a fixed test set
├── ensemble.py                          # Average five models' test predictions
├── utils.py                             # Splits, device banks, batching, checkpoints
├── models/
│   ├── gat_gcn.py                       # Default four-source GATnGCN_Model
│   ├── gin.py                           # GIN_Model: ligand GAT/GCN, protein GIN
│   └── legacy/                          # Historical ECFP/CNN ablations
├── data/
│   ├── davis/, kiba/                    # Raw graphs, sequences, folds, affinities
│   ├── cv_folds/                        # Fold JSON files for standalone notebooks
│   └── four_source_embeddings/
│       ├── davis/{train,test}.pt
│       └── kiba/{train,test}.pt
├── notebook/
│   ├── davis-attention-original copy.ipynb # Current standalone single-run notebook
│   ├── four-source-cv.ipynb             # CV with mean/std of per-model metrics
│   ├── four-source-cv-ensemble.ipynb    # CV followed by prediction ensembling
│   ├── four-source-ensemble-predict.ipynb # Inference from five existing checkpoints
│   ├── sync_notebook.py                 # Export single-run notebook
│   ├── sync_cv_notebook.py              # Export CV notebook and fold JSON files
│   └── sync_ensemble_notebooks.py       # Export both ensemble notebooks
└── tests/                              # CPU infrastructure and notebook checks
```

`models/legacy`, `davis-attention-original.ipynb`, `davis-attention.ipynb`,
`davis_attention_gpu_bank.py`, and `architecture.png` are historical experiments
or documentation. The old original notebook depends on an earlier `utils` API;
use the `original copy` notebook for the current single-run workflow.
Workspace notebooks outside `DATN_DTA` remain separate from this repository.

## Default model

| Source | Encoder | Pooling and projection |
| --- | --- | --- |
| Drug graph, 78 node features | GAT 8 heads x 64 -> ReLU -> GAT 4 heads x 64 -> ReLU; both use `concat=True` | max+mean -> 512 -> Linear 256 -> ReLU/Dropout -> Linear 128 |
| Protein graph, 41 node features | GAT 8 heads x 64 -> ReLU -> GCN 512->256 -> ReLU | max+mean -> 512 -> Linear 256 -> ReLU/Dropout -> Linear 128 |
| Chemformer drug embeddings `[L,512]` | Fixed precomputed embeddings | max+mean -> 1024 -> Linear 128 -> BatchNorm -> ReLU/Dropout |
| ESM protein embeddings `[L,640]` | Fixed precomputed embeddings | max+mean -> 1280 -> Linear 128 -> BatchNorm -> ReLU/Dropout |

Each side concatenates its graph and embedding vectors into 256 features, then
applies Linear 128 + ReLU. Drug and protein vectors are concatenated and passed
to an MLP **256->512->128->1**. There is no cross-attention, MoE, auxiliary loss,
or sequence CNN. BatchNorm uses running statistics for singleton device batches.
Protein `edge_attr` is retained when present in the source, but the current
GAT/GCN uses only `x` and `edge_index`.

`models/gin.py` exports the alternative `GIN_Model`: a ligand GAT/GCN encoder,
a five-layer protein GIN encoder, and the two embedding branches. Its ligand
embedding branch uses max pooling; its protein embedding branch uses max+mean.
The training CLI continues to select `models.gat_gcn.GATnGCN_Model` by default.

## Environment

Use Python 3.10+, a PyTorch build matching your CUDA environment, and PyTorch
Geometric for training. Data preparation also requires RDKit and NumPy. Install
the appropriate PyTorch build first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Run the following commands from `DATN_DTA`. Default data paths are resolved
relative to the Python modules, so scripts can also be invoked from elsewhere.

## Data preparation

**Existing valid v4 files can be used directly.** Validate both datasets:

```bash
python create_data.py --datasets davis kiba --check-only
```

Prepare missing datasets; existing valid files are checked and preserved:

```bash
python create_data.py --datasets davis kiba
```

Default inputs:

- `data/{dataset}_train.csv` and `data/{dataset}_test.csv`, preserving labels and order.
- `data/{dataset}/proteins.txt` and `protein_to_graph.pkl`.
- `../ligand_embeddings.pt`: dictionary from `lig_<sha256(SMILES)>` to `[L,512]` tensors.
- `../protein_embeddings.pt`: dictionary from `prot_<sha256(sequence)>` to `[L,640]` tensors.

When CSV files are absent, preparation reads `ligands_can.txt`, `Y`, and the
original folds. Only raw DAVIS affinities are converted from nM to pKd; CSV labels
and KIBA affinities are not converted again. Override inputs with `--data-dir`,
`--ligand-embeddings`, `--protein-embeddings`, and `--output-dir`. Invalid existing
outputs cause an error rather than being overwritten. Chemformer and ESM are
neither downloaded nor rerun.

The default output contains only `train.pt` and `test.pt`, each a self-contained
v4 dictionary with all four sources and pair indices. No manifest is required.
Use `--write-pairs-csv` for optional readable pair tables. Original token lengths
and storage dtypes are preserved without padding, truncation, or preprocessing
pooling. See [README_dual_embeddings.md](README_dual_embeddings.md).

## Single-run training

```bash
python training.py --dataset davis --max-gpus 2
python training.py --dataset kiba --max-gpus 1

python training.py --dataset kiba \
  --data-dir /path/to/kiba \
  --output-dir /path/to/run \
  --batch-size 512 --test-batch-size 512 --epochs 1000 --lr 0.0005
```

Defaults: Adam, **fixed LR 0.0005**, **1000 total epochs**, global batch 512,
shuffle=False, seed=42, and MSE loss. `--max-gpus 2` splits the global batch across
at most two GPUs; `--max-gpus 1` uses one. CPU is used when CUDA is unavailable.
Named flags replace old numeric arguments such as `training.py 0 2 0`.
Use `python training.py --help` for all options. Importing modules does not load
datasets or start training.

Train/test entities are merged into tensor banks and uploaded once per GPU.
DataParallel splits pair indices; each model looks up its graph/token batch
inside `forward`. Banks use concatenated tensors and offsets without padding.
Each pair gets its own graph in the batch. Protein IDs with equal sequences
share embeddings but retain separate graphs. Banks are excluded from
`state_dict` and are not rebroadcast on every forward pass.

**This legacy workflow selects checkpoints using test MSE each epoch.** The test
set therefore participates in model selection and is not an independent final
evaluation set. Use the CV workflow below for validation-based selection.

## Five-fold validation with DeepDTA splits

`training_cv.py` uses the five existing folds in
`data/{dataset}/folds/train_fold_setting1.txt`. Each model trains on four folds,
selects its checkpoint by **MSE on the remaining validation fold**, then evaluates
on the **same fixed test set** from `test_fold_setting1.txt`.

No additional 10% validation split is created. The standard CV report contains
**the mean of five per-model test metrics** and sample standard deviation
(`ddof=1`), not averaged predictions. Hyperparameters are supplied by the user;
the script does not implement the paper's hyperparameter search.
Reference: [DeepDTA, sections 3.3-3.4](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129291/).

```bash
# Run all five folds sequentially.
python training_cv.py --dataset davis --max-gpus 2
python training_cv.py --dataset kiba --max-gpus 2

# Validate v4 pairs and labels against original folds without creating a model.
python training_cv.py --dataset davis --check-folds

# Run individual folds across separate sessions.
python training_cv.py --dataset davis --fold 0 --epochs 1000
python training_cv.py --dataset davis --fold 1 --epochs 1000
```

Keep the same configuration and output directory when running folds separately.
After each fold, compatible results are aggregated; `cv_summary.json` sets
`complete=true` only when all five are available. Choose LR, epoch budget, and
architecture using validation rather than test metrics.

| Dataset | Training samples per run | Validation samples per run | Fixed test samples |
| --- | ---: | ---: | ---: |
| DAVIS | 20,036 or 20,037 | 5,010 or 5,009 | 5,010 |
| KIBA | 78,836 | 19,709 | 19,709 |

`--data-dir` points to v4 `train.pt/test.pt`. `--raw-data-dir` defaults to `data/`;
on another machine, supply a directory containing `{dataset}/Y`,
`ligands_can.txt`, `proteins.txt`, and both files under `folds/`.

```bash
python training_cv.py --dataset davis \
  --data-dir /kaggle/input/my-embeddings/davis \
  --raw-data-dir /kaggle/input/my-raw-data/data \
  --output-dir /kaggle/working/davis_deepdta_cv \
  --epochs 1000 --batch-size 512 --max-gpus 2
```

Ligand IDs, protein IDs, and affinities are checked against the original data.
Reordered v4 rows are supported; missing samples, changed splits, or changed
labels raise errors instead of silently generating new folds. KIBA can contain
distinct raw ligand IDs that canonicalize to the same SMILES. Original records
and folds are preserved. When both pair IDs and labels are duplicated, retain
the original `row_id` to disambiguate reordered records. `row_id` is validated
against pair identity and label and denotes a position in the flattened original
split, not a direct index into the affinity matrix. These are record-level
splits and do not guarantee disjoint drugs or proteins after canonicalization.

Default CV outputs:

```text
outputs/{dataset}_drug_gat8x64_gat4x64_dual_embedding_maxmean_deepdta_cv/
├── experiment.json             # Configuration and model/data/fold signatures
├── fold_assignments.csv        # Validation fold assigned to each training row
├── fold_0/ ... fold_4/
│   ├── best_model.pt            # Selected by validation MSE
│   ├── history.csv              # Per-epoch training and validation MSE
│   ├── test_predictions.csv     # Predictions on the shared test set
│   ├── metrics.json             # Test MSE, RMSE, CI, and Pearson
│   └── training_checkpoint_epoch_XXXX.pt
├── cv_results.csv              # Per-fold metrics, mean, and standard deviation
└── cv_summary.json             # complete=false until all five folds are available
```

Automatic per-fold resume restores model, optimizer, RNG, and best validation
checkpoint. Checkpoints from the single-run workflow, another fold, or another
configuration are rejected. To select a file explicitly, use
`--fold 2 --resume-path /path/to/checkpoint.pt`. Increase `--epochs` to extend the
total training budget; summaries do not mix results with different epoch targets.
Use a new output directory when changing experiment settings.

## Ensemble predictions from five models

`ensemble.py` averages the five predictions **for each test sample**, then
computes MSE, RMSE, CI, and Pearson against the true labels. This differs from
the mean of per-model metrics reported by `training_cv.py`.

Train with an explicit output directory:

```bash
python training_cv.py --dataset davis --output-dir outputs/davis_cv
```

Once all five checkpoints exist, run inference (`ensemble.py` has no direct CLI):

```bash
python - <<'PY'
from pathlib import Path
from ensemble import run_ensemble_inference

model_dir = Path('outputs/davis_cv')
result = run_ensemble_inference(
    checkpoint_paths=[model_dir / f'fold_{i}' / 'best_model.pt' for i in range(5)],
    test_path=Path('data/four_source_embeddings/davis/test.pt'),
    output_dir=Path('outputs/davis_ensemble'),
    dataset='davis',
    device='cuda:0',
    max_gpus=1,
)
print(result['test_metrics'])
PY
```

All five checkpoints must cover folds 0..4 with the same experiment, dataset,
and architecture. Resume files are accepted by extracting their best
validation-selected checkpoint. Bare state dictionaries and single-run
checkpoints are not accepted. Models load sequentially and share a test bank.
Only the five checkpoints and v4 `test.pt` are needed for this inference path.

Outputs: `ensemble_predictions.csv` contains sample IDs, `y_true`,
`y_pred_fold_0` through `y_pred_fold_4`, and `y_pred_mean`.
`ensemble_metrics.json` and `.csv` contain ensemble and individual-model metrics.
CI means concordance index. Undefined metrics are stored as null.

## Checkpoints and resume

Single-run outputs default to
`outputs/{dataset}_drug_gat8x64_gat4x64_dual_embedding_maxmean`, or under
`/kaggle/working` on Kaggle. `best_model.pt` and `best_test_predictions.csv`
update when test MSE improves. `history.csv` records per-epoch metrics.
Full `training_checkpoint_epoch_XXXX.pt` files are saved every 100 epochs and
at the final epoch, including current weights, Adam state, a separate best
checkpoint, history, and Python/NumPy/CPU/CUDA/shuffle RNG states.

Automatic resume scans the output directory and `/kaggle/input`, validates
compatibility, and uses the compatible checkpoint with the highest epoch.
Without one, training starts from scratch. To specify a checkpoint:

```bash
python training.py --dataset davis --resume-path /path/to/training_checkpoint_epoch_0300.pt
```

Epoch 300 resumes at 301 and still stops at the configured total (default 1000).
The earlier best model is preserved separately. `best_model.pt` alone lacks the
state needed to resume. Changed model/split/batch/LR settings are rejected;
changing GPU count is allowed but may prevent exact random replay. Equivalent
notebook/module architectures remain compatible; old drug GAT->GCN weights do
not match the default two-GAT drug encoder. Use `--no-auto-resume` and a new
output directory for an independent run. Save/download checkpoints or persist
Kaggle outputs before ending a session.

## Standalone Kaggle notebooks

- `davis-attention-original copy.ipynb`: single-run workflow; add both v4 split files.
- `four-source-cv.ipynb`: CV with mean/std of per-model metrics.
- `four-source-cv-ensemble.ipynb`: CV followed by mean-prediction ensemble metrics.
- `four-source-ensemble-predict.ipynb`: evaluate five existing checkpoints on `test.pt`.

For CV notebooks, add `train.pt`, `test.pt`, and the matching
`data/cv_folds/davis_folds.json` or `kiba_folds.json`. Set DATASET and Run All.
FOLDS_PATH=None discovers the fold file; set explicit DATA_DIR/FOLDS_PATH if
multiple inputs are present. The repository and raw data are not required at
notebook runtime. `CHECK_FOLDS_ONLY=True` validates folds on CPU without a model.
`FOLD=None` runs all folds; `FOLD=0..4` runs one. Each fold uses a fixed LR and
1000 epochs by default, with selection based on validation MSE.

For resume across sessions, add saved outputs as inputs, keep the configuration,
and use FOLD=None. Completed folds are loaded and evaluated without retraining;
incomplete folds resume. Keep all completed-fold checkpoints to regenerate the
full summary in the current output directory. Ensemble reports require all five
folds completed through the requested epoch budget and align predictions by
sample IDs and labels.

Edit repository `.py` files and regenerate notebooks in this order:

```bash
python notebook/sync_notebook.py
python notebook/sync_cv_notebook.py
python notebook/sync_ensemble_notebooks.py
```

Exporters preserve editable configuration cells where supported, replace library
cells from source, and clear generated notebook outputs. Move notebook-only model
edits into the repository before exporting. CV export reads raw identities and
folds to write separate JSON files; it does not train or modify embedding files.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover small-data preparation, pair ordering, graph offsets, shared
embeddings, full token lengths, checkpoint/RNG handling, notebook synchronization,
fold membership, overlapping raw records, tied CI values, validation-based
checkpoint selection, resume, and both metric and prediction aggregation.
Training-loop tests use tiny CPU substitutes rather than training a DTA model.
