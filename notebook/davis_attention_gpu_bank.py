# Python version of davis-attention.ipynb for copying cells into Kaggle.
# Each # %% block corresponds to a notebook cell; update paths in the config cell.


# %% [markdown] intro
# # DAVIS: GPU embedding banks -> cross-attention -> affinity
#
# Inputs: `ligand_embeddings.pt`, `protein_embeddings.pt`, and **output** CSV files
# `davis_train.csv` and `davis_test.csv` containing `ligand_id`, `protein_id`, and `affinity`.
# Only PyTorch and NumPy are needed on Kaggle; enable T4 x2 to use two GPUs.
#
# **Flow:** load original embeddings -> pad each unique entity once -> upload banks and masks
# to each GPU once -> fetch pair indices per step -> index embeddings on that GPU
# -> learned MLP reduces 512/640 features to 256 **inside the model** -> bidirectional cross-attention -> masked max
# pooling → concat → MLP → affinity.
#
# No DataLoader, workers, collation, padding, or CPU embedding copies occur in the training loop.
# Embeddings remain fixed; Chemformer/ESM do not need to run again. The CPU manages only indices
# and small length metadata used to identify removable padding per batch.
#
# **Model selection:** use test MSE each epoch. Test data participates in model selection,
# so this is not an independent test evaluation. Keep the original DAVIS pKd labels.

# %% config
from pathlib import Path
import csv
import math
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# Update this path for the embedding dataset added as input on Kaggle.
EMBEDDING_DIR = Path('/kaggle/input/YOUR_EMBEDDING_DATASET/dta_embeddings')
LIGAND_DICT_PATH = EMBEDDING_DIR / 'ligand_embeddings.pt'
PROTEIN_DICT_PATH = EMBEDDING_DIR / 'protein_embeddings.pt'
TRAIN_CSV = EMBEDDING_DIR / 'davis_train.csv'
TEST_CSV = EMBEDDING_DIR / 'davis_test.csv'
OUTPUT_DIR = Path('/kaggle/working/davis_attention_mlp256_outputs')

SEED = 42
# GLOBAL batch size, split across GPUs by DataParallel.
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
USE_DATA_PARALLEL = True
LR = 0.0005
NUM_EPOCHS = 1000
LOG_INTERVAL = 20
SNAPSHOT_EPOCHS = (500, 700, 1000)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_CONFIG = {'embed_dim': 256, 'num_heads': 8, 'dropout': 0.2}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
print('Device:', DEVICE, '| PyTorch:', torch.__version__)
if DEVICE.type == 'cuda':
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        print(f'GPU {gpu_id}: {props.name} | {props.total_memory / 1024**3:.2f} GiB')

# %% [markdown] preprocess-notes
# ## 1. Prepare GPU embedding banks once
#
# Keep only IDs used by DAVIS. Sort IDs to build `ligand_id_to_index` and
# `protein_id_to_index`. Pad each entity once; mask entries of `True` indicate padding.
# Preserve original float32 embeddings: **512 ligand features / 640 protein features**, without pooling.
#
# For the current DAVIS data: ligand bank `[68, 80, 512]`, protein bank `[379, 2549, 640]`,
# about **2.314 GiB per GPU**, excluding masks and batch/training memory. Each GPU holds
# its own copy. A single GPU or CPU also works. Release temporary CPU dictionaries/banks after upload.
#
# Train/test samples retain only `(lig_idx, prot_idx, y)`. Indices and labels reside on the primary GPU.
# Shuffling retains all samples, including a smaller final batch. Actual lengths are stored
# as CPU metadata to remove **excess padding** before batch indexing, without dropping real tokens.
#
# GPU banks live in a separate data object, **not model buffers or parameters**:
# DataParallel replicates only the small model and does not broadcast embedding banks per forward pass;
# checkpoints also exclude these banks. Rerunning bank/model cells while retaining old banks
# can temporarily increase VRAM usage; restart the session before switching from an old notebook.

# %% data-definitions
def read_pairs(path):
    rows = []
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'ligand_id', 'protein_id', 'affinity'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: expected an OUTPUT CSV with columns {sorted(required)}')
        for position, row in enumerate(reader):
            where = f'{path}, row {position + 2}'
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{where}: invalid CSV row')
            lid, pid = row['ligand_id'].strip(), row['protein_id'].strip()
            try:
                y = float(row['affinity'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{where}: invalid affinity') from exc
            if not lid or not pid or not math.isfinite(y) or abs(y) > np.finfo(np.float32).max:
                raise ValueError(f'{where}: empty ID or affinity is non-finite in float32')
            rows.append({'row_id': row.get('row_id', str(position)),
                         'ligand_id': lid, 'protein_id': pid, 'affinity': y})
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    return rows


def load_embedding_bank(path, required_ids, width):
    raw = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: expected an ID -> tensor dictionary')
    missing = required_ids.difference(raw)
    if missing:
        raise ValueError(f'{path}: missing {len(missing)} IDs, e.g. {sorted(missing)[:3]}')
    bank = {}
    for key in sorted(required_ids):
        value = raw[key]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != width
                or not value.is_floating_point()):
            raise ValueError(f'{path}: embedding {key} must be a float tensor with shape [L > 0, {width}]')
        value = value.detach().to(dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all().item():
            raise ValueError(f'{path}: embedding {key} contains NaN/Inf')
        bank[key] = value
    # Raw data contains DAVIS and KIBA; only the bank remains after this function returns.
    return bank



def execution_devices(device, use_data_parallel=True):
    device = torch.device(device)
    if device.type != 'cuda':
        return [device]
    primary = device.index if device.index is not None else torch.cuda.current_device()
    ids = [primary]
    if use_data_parallel:
        ids += [i for i in range(torch.cuda.device_count()) if i != primary]
    return [torch.device('cuda', i) for i in ids]


def pack_embedding_bank(bank):
    ids = sorted(bank)
    lengths = np.array([bank[key].shape[0] for key in ids], dtype=np.int64)
    # Call pad_sequence only during preprocessing, never during training.
    padded = pad_sequence([bank[key] for key in ids], batch_first=True)
    mask = torch.arange(padded.shape[1])[None, :] >= torch.from_numpy(lengths)[:, None]
    return padded, mask, lengths, {key: i for i, key in enumerate(ids)}


class DeviceEmbeddingStore:
    """Immutable data shared by reference across DataParallel replicas."""
    def __init__(self, cpu_tables, devices):
        self.tables = {}
        self.bytes_per_device = sum(t.numel() * t.element_size() for t in cpu_tables.values())
        for device in devices:
            device = torch.device(device)
            if device.type == 'cuda':
                free, _ = torch.cuda.mem_get_info(device)
                if self.bytes_per_device > free:
                    raise RuntimeError(
                        f'{device}: bank requires {self.bytes_per_device / 1024**3:.3f} GiB, '
                        f'but only {free / 1024**3:.3f} GiB VRAM is free. Restart the session to release old data.')
            self.tables[str(device)] = {
                key: value.detach().to(device) for key, value in cpu_tables.items()
            }
            print(f'Embedding bank and masks on {device}: {self.bytes_per_device / 1024**3:.3f} GiB')

    def on(self, device):
        key = str(device)
        if key not in self.tables:
            raise ValueError(f'Embedding bank is not prepared on {key}')
        return self.tables[key]


class IndexBatches:
    """Iterate pair indices without loading, copying, or padding embeddings on CPU."""
    def __init__(self, rows, ligand_id_to_index, protein_id_to_index,
                 ligand_lengths, protein_lengths, batch_size, device,
                 shuffle=False, seed=42):
        if batch_size < 1 or not rows:
            raise ValueError('Batch size must be >= 1 and the pair table must be non-empty')
        self.batch_size = batch_size
        self.size = len(rows)
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        pairs = np.array([(ligand_id_to_index[r['ligand_id']],
                           protein_id_to_index[r['protein_id']]) for r in rows], dtype=np.int64)
        self.lengths = np.column_stack((ligand_lengths[pairs[:, 0]],
                                        protein_lengths[pairs[:, 1]]))
        self.pairs = torch.from_numpy(pairs).to(self.device)
        self.labels = torch.tensor([r['affinity'] for r in rows], dtype=torch.float32,
                                   device=self.device).reshape(-1, 1)

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

    def reset_seed(self, seed):
        self.generator.manual_seed(seed)

    def __iter__(self):
        order = (torch.randperm(self.size, generator=self.generator)
                 if self.shuffle else torch.arange(self.size))
        order_cpu = order.numpy()
        order_device = order.to(self.device)  # Transfer the small index array once per epoch.
        for start in range(0, self.size, self.batch_size):
            stop = min(start + self.batch_size, self.size)
            indices = order_device[start:stop]
            max_ligand, max_protein = self.lengths[order_cpu[start:stop]].max(axis=0)
            yield {
                'pair_indices': self.pairs.index_select(0, indices),
                'y': self.labels.index_select(0, indices),
                'max_ligand': int(max_ligand), 'max_protein': int(max_protein),
            }

# %% preprocess-run
for input_path in (TRAIN_CSV, TEST_CSV, LIGAND_DICT_PATH, PROTEIN_DICT_PATH):
    if not input_path.is_file():
        raise FileNotFoundError(f'File not found: {input_path}. Update the configuration cell.')

train_rows = read_pairs(TRAIN_CSV)
test_rows = read_pairs(TEST_CSV)
all_rows = train_rows + test_rows

ligand_bank = load_embedding_bank(
    LIGAND_DICT_PATH, {row['ligand_id'] for row in all_rows}, 512)
ligand_table, ligand_mask, ligand_lengths, ligand_id_to_index = pack_embedding_bank(ligand_bank)
del ligand_bank
protein_bank = load_embedding_bank(
    PROTEIN_DICT_PATH, {row['protein_id'] for row in all_rows}, 640)
protein_table, protein_mask, protein_lengths, protein_id_to_index = pack_embedding_bank(protein_bank)
del protein_bank, all_rows

runtime_devices = execution_devices(DEVICE, USE_DATA_PARALLEL)
# Canonicalize cuda:N so the bank and model use the same device key.
DEVICE = runtime_devices[0]
embedding_store = DeviceEmbeddingStore({
    'ligand': ligand_table, 'protein': protein_table,
    'ligand_mask': ligand_mask, 'protein_mask': protein_mask,
}, runtime_devices)
del ligand_table, protein_table, ligand_mask, protein_mask

train_batches = IndexBatches(
    train_rows, ligand_id_to_index, protein_id_to_index, ligand_lengths, protein_lengths,
    TRAIN_BATCH_SIZE, DEVICE, shuffle=True, seed=SEED)
test_batches = IndexBatches(
    test_rows, ligand_id_to_index, protein_id_to_index, ligand_lengths, protein_lengths,
    TEST_BATCH_SIZE, DEVICE, shuffle=False, seed=SEED)

print(f'Preprocess xong: {len(train_rows):,} train / {len(test_rows):,} test')
print(f'Unique: {len(ligand_id_to_index):,} ligands / {len(protein_id_to_index):,} proteins')
print('Maximum lengths:', int(ligand_lengths.max()), '/', int(protein_lengths.max()))
print('Original 512/640 embeddings are ready on:', [str(d) for d in runtime_devices])

# %% [markdown] model-notes
# ## 2. Model
#
# ```text
# Ligand  [B, Ld, 512] → MLP(512 → 256 → 256) ── D [B, Ld, 256]
# Protein [B, Lp, 640] → MLP(640 → 256 → 256) ── P [B, Lp, 256]
#
# D_new = CrossAttention(query=D, key=P, value=P)
# P_new = CrossAttention(query=P, key=D, value=D)
#
# masked_max(D_new, Ld axis) -> [B, 256]
# masked_max(P_new, Lp axis) -> [B, 256]
# concat → [B, 512] → MLP(512 → 512 → 256 → 1)
# ```
#
# Two independent input MLPs use `Linear -> ReLU -> Dropout(0.2) -> Linear`
# along the feature dimension of each token/residue, preserving both sequence lengths.
# These MLPs train jointly with attention and the prediction head inside forward.
# Original GPU embeddings stay fixed. Both eight-head attention modules use the original D/P
# inputs. Masks exclude padded keys/values and padded queries during max pooling.
#
# Reuse the existing 512/640 embedding dictionaries. Train the 256-feature MLP architecture anew;
# old average-pooling checkpoints are incompatible. New checkpoints store
# `model_config`; the default output `davis_attention_mlp256_outputs` preserves previous results.
# `IndexedEmbeddingDTA` retains per-GPU bank lookup without changing the data flow.

# %% model
class EmbeddingCrossAttentionDTA(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.2):
        super().__init__()
        # Separate MLPs for each embedding type, applied independently per token/residue.
        self.ligand_projection = nn.Sequential(
            nn.Linear(512, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(640, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ligand_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.protein_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    @staticmethod
    def masked_max(sequence, padding_mask):
        return sequence.masked_fill(padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)

    def forward(self, ligand, protein, ligand_mask, protein_mask):
        # Reduce feature dimensions with MLPs while preserving token/residue counts.
        ligand = self.ligand_projection(ligand)    # [B, Ld, 512] -> [B, Ld, 256]
        protein = self.protein_projection(protein) # [B, Lp, 640] -> [B, Lp, 256]
        ligand_new, _ = self.ligand_attention(
            ligand, protein, protein, key_padding_mask=protein_mask, need_weights=False)
        protein_new, _ = self.protein_attention(
            protein, ligand, ligand, key_padding_mask=ligand_mask, need_weights=False)
        ligand_vector = self.masked_max(ligand_new, ligand_mask)
        protein_vector = self.masked_max(protein_new, protein_mask)
        return self.mlp(torch.cat((ligand_vector, protein_vector), dim=-1))


class IndexedEmbeddingDTA(nn.Module):
    """Accept indices; each GPU retrieves embeddings from its own fixed bank."""
    def __init__(self, core, embedding_store):
        super().__init__()
        self.core = core
        # Plain Python object: neither register_buffer nor nn.Module.
        # DataParallel replicas share the store without moving or rebroadcasting banks.
        self.embedding_store = embedding_store

    def forward(self, pair_indices, max_ligand, max_protein):
        tables = self.embedding_store.on(pair_indices.device)
        ligand_indices, protein_indices = pair_indices[:, 0], pair_indices[:, 1]
        # Slice a view before index_select to avoid copying all global padding into the batch.
        ligand = tables['ligand'][:, :max_ligand].index_select(0, ligand_indices)
        protein = tables['protein'][:, :max_protein].index_select(0, protein_indices)
        ligand_mask = tables['ligand_mask'][:, :max_ligand].index_select(0, ligand_indices)
        protein_mask = tables['protein_mask'][:, :max_protein].index_select(0, protein_indices)
        return self.core(ligand, protein, ligand_mask, protein_mask)

# %% [markdown] training-notes
# ## 3. Training and MSE
#
# Adam, learning rate `0.0005`, MSE; report train/test MSE and select the checkpoint
# with the lowest test MSE. Batch size is **global** across GPUs, default 8. Preserve actual token lengths.
#
# `IndexBatches` creates an index order once per epoch. Each step retrieves pair indices and labels on the
# primary GPU; DataParallel splits only indices across GPUs. Each replica indexes the embedding bank
# already on its device. MSE uses all predictions gathered onto the primary GPU,
# so uneven batches are weighted correctly by sample count.
#
# The train/eval loop has no DataLoader, workers, collation, or padding. The CPU only selects
# batch length metadata; GPU indexing still allocates batch tensors. Removing excess padding keeps
# attention from always using the dataset-wide maximum length.
#
# The original bank uses about 2.314 GiB/GPU; activations, gradients, and batches require extra VRAM.
# Reduce batch size on OOM. Initialize banks once; GPU 0 also holds the optimizer and gathered outputs.

# %% training-definitions
def regression_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if labels.shape != predictions.shape or not labels.size:
        raise ValueError('Labels and predictions must have the same non-empty shape')
    if not (np.isfinite(labels).all() and np.isfinite(predictions).all()):
        raise ValueError('Labels or predictions contain NaN/Inf')
    mse = float(np.mean((labels - predictions) ** 2))
    return {'mse': mse}



def unwrap_model(model):
    model = model.module if isinstance(model, nn.DataParallel) else model
    return model.core if isinstance(model, IndexedEmbeddingDTA) else model


def configure_model(core, device, embedding_store, use_data_parallel=True):
    devices = execution_devices(device, use_data_parallel)
    for target in devices:
        embedding_store.on(target)  # Fail early if a required GPU bank is missing.
    model = IndexedEmbeddingDTA(core, embedding_store).to(devices[0])
    if len(devices) > 1:
        gpu_ids = [d.index for d in devices]
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0], dim=0)
    return model


def predict_batch(model, batch):
    return model(batch['pair_indices'], batch['max_ligand'], batch['max_protein'])


def train_epoch(model, batches, optimizer, epoch, log_interval=20):
    model.train()
    squared_error, count = 0.0, 0
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        predictions = predict_batch(model, batch)
        loss = nn.functional.mse_loss(predictions, batch['y'])
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Loss contains NaN/Inf at epoch {epoch}, batch {step}')
        loss.backward()
        optimizer.step()
        batch_size = batch['y'].shape[0]
        squared_error += loss.item() * batch_size
        count += batch_size
        if log_interval and (step == 1 or step % log_interval == 0):
            print(f'Epoch {epoch} | batch {step}/{len(batches)} | MSE {loss.item():.6f}')
    return squared_error / count


@torch.inference_mode()
def predicting(model, batches):
    model.eval()
    labels, predictions = [], []
    for batch in batches:
        predictions.append(predict_batch(model, batch).cpu().numpy().reshape(-1))
        labels.append(batch['y'].cpu().numpy().reshape(-1))
    return np.concatenate(labels), np.concatenate(predictions)


def save_checkpoint(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_trained_model(path, device, embedding_store, use_data_parallel=True):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = EmbeddingCrossAttentionDTA(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return configure_model(model, device, embedding_store, use_data_parallel).eval(), checkpoint


def write_predictions(path, rows, labels, predictions):
    if not len(rows) == len(labels) == len(predictions):
        raise ValueError('Metadata row and prediction counts do not match')
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'row_id', 'ligand_id', 'protein_id', 'affinity', 'predicted_affinity'])
        writer.writeheader()
        for row, label, prediction in zip(rows, labels, predictions):
            writer.writerow({**row, 'affinity': float(label),
                             'predicted_affinity': float(prediction)})


def run_training(model, device, train_batches, test_batches, test_rows,
                 output_dir, model_config, num_epochs, lr, log_interval,
                 snapshot_epochs, seed):
    if num_epochs < 1:
        raise ValueError('NUM_EPOCHS must be >= 1')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse = float('inf')
    best_checkpoint = None
    history = []
    fields = ['epoch', 'train_mse', 'mse', 'seconds', 'is_best']
    # Rerunning this cell starts a new run and overwrites outputs with the same names.
    with (output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, num_epochs + 1):
            started = time.perf_counter()
            train_mse = train_epoch(model, train_batches, optimizer, epoch, log_interval)
            labels, predictions = predicting(model, test_batches)
            metrics = regression_metrics(labels, predictions)
            improved = metrics['mse'] < best_mse
            if improved:
                best_mse = metrics['mse']
                best_checkpoint = {
                    'format_version': 1, 'epoch': epoch,
                    'input_representation': 'raw_embeddings_512_640',
                    'model_config': dict(model_config),
                    'model_state_dict': {key: value.detach().cpu().clone()
                                         for key, value in unwrap_model(model).state_dict().items()},
                    'metrics': dict(metrics), 'selection_split': 'test',
                    'train_config': {'seed': seed, 'lr': lr, 'num_epochs': num_epochs,
                                     'batch_size': train_batches.batch_size,
                                     'test_batch_size': test_batches.batch_size,
                                     'parallel_gpu_ids': list(model.device_ids)
                                         if isinstance(model, nn.DataParallel) else []},
                }
                save_checkpoint(best_checkpoint, output_dir / 'best_model.pt')
                write_predictions(output_dir / 'best_test_predictions.csv',
                                  test_rows, labels, predictions)
                with (output_dir / 'best_metrics.csv').open('w', newline='', encoding='utf-8') as best_handle:
                    best_writer = csv.DictWriter(best_handle, fieldnames=['epoch', *metrics])
                    best_writer.writeheader()
                    best_writer.writerow({'epoch': epoch, **metrics})
            record = {'epoch': epoch, 'train_mse': train_mse, **metrics,
                      'seconds': time.perf_counter() - started, 'is_best': improved}
            writer.writerow(record)
            handle.flush()
            history.append(record)
            print(f'Epoch {epoch}: train MSE={train_mse:.6f}, test MSE={metrics["mse"]:.6f} '
                  f'| best epoch={best_checkpoint["epoch"]}')
            if epoch in snapshot_epochs:
                save_checkpoint(best_checkpoint, output_dir / f'best_until_epoch_{epoch}.pt')
    unwrap_model(model).load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    return history, best_checkpoint

# %% training-run
# Rerunning this cell trains from scratch. Change OUTPUT_DIR to retain previous results.
seed_everything(SEED)
train_batches.reset_seed(SEED)
model = configure_model(
    EmbeddingCrossAttentionDTA(**MODEL_CONFIG), DEVICE, embedding_store, USE_DATA_PARALLEL)
if isinstance(model, nn.DataParallel):
    print(f'DataParallel: GPUs {model.device_ids} | index each GPU\'s local bank')
else:
    print(f'Indexing embeddings on {DEVICE}')
print(f'Global batch sizes: train={TRAIN_BATCH_SIZE}, test={TEST_BATCH_SIZE}')
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters()):,}')
history, best_checkpoint = run_training(
    model=model, device=DEVICE, train_batches=train_batches, test_batches=test_batches,
    test_rows=test_rows, output_dir=OUTPUT_DIR, model_config=MODEL_CONFIG,
    num_epochs=NUM_EPOCHS, lr=LR, log_interval=LOG_INTERVAL,
    snapshot_epochs=SNAPSHOT_EPOCHS, seed=SEED)
print('Best epoch:', best_checkpoint['epoch'])
print('Best test MSE:', best_checkpoint['metrics']['mse'])
print('Output:', OUTPUT_DIR)

# %% [markdown] outputs-notes
# ## 4. Outputs and inference
#
# Trong `OUTPUT_DIR`:
# - `best_model.pt`: core weights, configuration, epoch, and best test MSE.
# - `history.csv`: train/test MSE per epoch (`mse` denotes test MSE).
# - `best_metrics.csv`: best epoch and test MSE.
# - `best_test_predictions.csv`: pair IDs, labels, and predictions in test CSV order.
# - `best_until_epoch_500.pt`, `...700.pt`, `...1000.pt`: best checkpoints through each milestone.
#
# Checkpoints **exclude GPU embedding banks** and have no `module.`/`core.` prefixes;
# the core loads on CPU or one/two GPUs. Use only checkpoints with the matching architecture:
# the new input MLPs; old average-pooling checkpoints are incompatible. For a new inference session, rebuild banks from
# the two original dictionaries, then call `load_trained_model(..., embedding_store=embedding_store)`.
# Old index order is irrelevant: IDs are remapped using the current data tables.
#
# For new-pair inference, add the required embeddings to the bank first, then create unshuffled
# `IndexBatches` from the corresponding IDs; placeholder labels of `0.0` are accepted. The example below
# reuses existing GPU banks without uploading embeddings again when loading the checkpoint.

# %% inference-example
inference_model, saved_checkpoint = load_trained_model(
    OUTPUT_DIR / 'best_model.pt', DEVICE, embedding_store,
    use_data_parallel=USE_DATA_PARALLEL)
example_batch = next(iter(test_batches))
with torch.inference_mode():
    example_predictions = predict_batch(inference_model, example_batch).cpu().flatten()
print('Checkpoint epoch:', saved_checkpoint['epoch'])
for row, prediction in zip(test_rows[:len(example_predictions)], example_predictions):
    print(row['ligand_id'], row['protein_id'], 'prediction =', round(prediction.item(), 4))
del inference_model, example_batch
