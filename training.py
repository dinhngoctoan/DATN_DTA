"""Train the four-source DTA model with fixed LR and resumable checkpoints."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (load_data, seed_everything, unwrap_model, save_checkpoint,
                   save_test_predictions, cpu_snapshot, model_signature,
                   dataset_signature, capture_rng, restore_rng, find_resume_checkpoint)

ROOT = Path(__file__).resolve().parent
MODEL_TYPE = 'GATnGCN_DrugGAT8x64_GAT4x64_ProteinGATGCN_DualEmbeddingMaxMean_SmallMLP'
DEFAULT_MODEL_CONFIG = {'num_features_xd': 78, 'num_features_xt': 41, 'n_output': 1,
                        'output_dim': 128, 'dropout': 0.2, 'protein_embedding_dim': 640}


@dataclass
class TrainingConfig:
    dataset: str = 'davis'
    data_dir: Path | None = None
    output_dir: Path | None = None
    device: str = 'cuda:0'
    max_gpus: int = 2
    batch_size: int = 512
    test_batch_size: int = 512
    lr: float = 0.0005
    epochs: int = 1000
    checkpoint_every: int = 100
    auto_resume: bool = True
    resume_path: Path | None = None
    input_dir: Path = Path('/kaggle/input')
    shuffle: bool = False
    seed: int = 42
    log_interval: int = 20
    model_config: dict = field(default_factory=lambda: dict(DEFAULT_MODEL_CONFIG))

    def __post_init__(self):
        if self.dataset not in ('davis', 'kiba'):
            raise ValueError('dataset must be davis or kiba')
        if min(self.batch_size, self.test_batch_size, self.max_gpus, self.epochs,
               self.checkpoint_every) < 1 or self.lr <= 0 or not np.isfinite(self.lr):
            raise ValueError('Batch sizes, epochs, max_gpus, checkpoint_every and LR must be positive')
        if self.log_interval < 0:
            raise ValueError('log_interval must be >= 0')
        self.data_dir = Path(self.data_dir) if self.data_dir else ROOT / 'data/four_source_embeddings' / self.dataset
        base = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else ROOT / 'outputs'
        self.output_dir = Path(self.output_dir) if self.output_dir else base / f'{self.dataset}_drug_gat8x64_gat4x64_dual_embedding_maxmean'
        self.input_dir = Path(self.input_dir)
        self.resume_path = Path(self.resume_path) if self.resume_path else None


def configure_model(model_config, store):
    from models.gat_gcn import GATnGCN_Model

    for target in store.devices:
        store.on(target)
    core = GATnGCN_Model(data_store=store, **model_config).to(store.device)
    if len(store.devices) > 1:
        gpu_ids = [target.index for target in store.devices]
        return nn.DataParallel(core, device_ids=gpu_ids, output_device=gpu_ids[0], dim=0)
    return core

def train(model, batches, optimizer, epoch, log_interval=20):
    model.train()
    total_loss, count = 0.0, 0
    for batch_idx, (pair_indices, labels) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        output = model(pair_indices)
        # Optimize MSE over the full batch gathered from all GPUs.
        loss = F.mse_loss(output, labels)
        value = loss.item()
        if not np.isfinite(value):
            raise FloatingPointError(f'Loss contains NaN/Inf at epoch {epoch}, batch {batch_idx + 1}')
        loss.backward()
        optimizer.step()
        size = pair_indices.size(0)
        total_loss += value * size
        count += size
        if log_interval and (batch_idx == 0 or (batch_idx + 1) % log_interval == 0):
            print(f'Epoch {epoch} [{count}/{batches.size}] MSE: {value:.6f}')
    return total_loss / count

@torch.no_grad()
def predicting(model, batches):
    model.eval()
    labels, predictions = [], []
    for pair_indices, batch_labels in batches:
        output = model(pair_indices)
        predictions.append(output.cpu().reshape(-1))
        labels.append(batch_labels.cpu().reshape(-1))
    return torch.cat(labels).numpy(), torch.cat(predictions).numpy()

def run_training(config: TrainingConfig):
    """Run up to the total epoch target; importing this module never starts a run."""
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    train_data, test_data, data_store, train_batches, test_batches = load_data(
        config.data_dir, config.dataset, device, config.batch_size, config.test_batch_size,
        config.max_gpus, config.shuffle, config.seed)
    # Match notebook initialization after bank creation.
    seed_everything(config.seed)
    train_batches.reset_seed(config.seed)
    model = configure_model(config.model_config, data_store)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_identity = {
        'model_config': dict(config.model_config),
        'model_signature': model_signature(model),
        'dataset_signature': dataset_signature((train_data, test_data)),
        'train_config': {'optimizer': 'Adam', 'lr': config.lr, 'batch_size': config.batch_size,
                         'test_batch_size': config.test_batch_size, 'shuffle': config.shuffle,
                         'seed': config.seed, 'loss': 'mse'},
    }
    best_mse, best_epoch, best_checkpoint = float('inf'), -1, None
    best_predictions, history, start_epoch = None, [], 1
    resume_path, resume = (find_resume_checkpoint(
        [config.output_dir, config.input_dir], run_identity, config.resume_path)
        if config.auto_resume or config.resume_path is not None else (None, None))
    if resume is not None:
        unwrap_model(model).load_state_dict(resume['model_state_dict'], strict=True)
        # PyTorch load_state_dict places optimizer state on each parameter's device.
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        best_checkpoint = cpu_snapshot(resume['best_checkpoint'])
        best_mse, best_epoch = best_checkpoint['test_mse'], best_checkpoint['epoch']
        best_predictions = cpu_snapshot(resume['best_predictions'])
        history = list(resume['history'])
        start_epoch = resume['epoch'] + 1
        # Restore best-model artifacts even if this run produces no further improvement.
        save_checkpoint(best_checkpoint, config.output_dir / 'best_model.pt')
        save_test_predictions(config.output_dir / 'best_test_predictions.csv', test_data,
                              test_data.labels.cpu().numpy(), best_predictions.numpy())
        restore_rng(resume['rng_state'], train_batches, data_store.devices)
        print(f'Resuming from {resume_path}: completed epoch {start_epoch - 1}; '
              f'continuing at {start_epoch}, best MSE={best_mse:.6f} at epoch {best_epoch}')
        del resume
    else:
        print('No compatible resume checkpoint or config.auto_resume=False: training from scratch.')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('DataParallel:', isinstance(model, nn.DataParallel), '| devices:',
          [str(target) for target in data_store.devices])
    if start_epoch > config.epochs:
        print(f'Checkpoint reached epoch {start_epoch - 1}; target is {config.epochs}, no further training.')

    # Preserve the legacy checkpoint-selection rule based on test MSE.
    # Test data participates in model selection and is not an independent evaluation set.
    with (config.output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_mse', 'test_mse', 'seconds'])
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        for epoch in range(start_epoch, config.epochs + 1):
            started = time.perf_counter()
            train_mse = train(model, train_batches, optimizer, epoch, config.log_interval)
            labels, predictions = predicting(model, test_batches)
            if not np.isfinite(predictions).all():
                raise FloatingPointError('Test predictions contain NaN/Inf')
            test_mse = float(np.mean((labels.astype(np.float64) - predictions.astype(np.float64)) ** 2))
            if test_mse < best_mse:
                best_mse, best_epoch = test_mse, epoch
                best_checkpoint = {
                    'model_type': MODEL_TYPE,
                    'model_config': dict(config.model_config),
                    'model_signature': run_identity['model_signature'],
                    'model_state_dict': cpu_snapshot(unwrap_model(model).state_dict()),
                    'epoch': epoch, 'test_mse': test_mse, 'selection_split': 'test',
                    'dataset': config.dataset, 'dataset_format_version': 4, 'loss_type': 'mse',
                    'fusion_head': 'dense_mlp_256_512_128',
                    'pair_fusion': 'concat_ligand_protein',
                    'ligand_graph_encoder': 'gat_8x64_gat_4x64_maxmean_fc256_out128',
                    'protein_graph_encoder': 'gat_8x64_gcn512to256_maxmean_fc256_out128',
                    'data_loading': 'per_device_bank_indexed_in_forward',
                    'attention_fusion': 'direct_graph_embedding_concat',
                    'ligand_embedding_pooling': 'concat_max_mean_over_tokens_in_model',
                    'protein_embedding_pooling': 'concat_max_mean_over_tokens_in_model',
                }
                best_predictions = torch.from_numpy(predictions.copy()).cpu()
                save_checkpoint(best_checkpoint, config.output_dir / 'best_model.pt')
                save_test_predictions(config.output_dir / 'best_test_predictions.csv', test_data, labels, predictions)
            record = {'epoch': epoch, 'train_mse': train_mse, 'test_mse': test_mse,
                      'seconds': time.perf_counter() - started}
            history.append(record)
            writer.writerow(record)
            handle.flush()
            # Save the CURRENT model and optimizer state, with a separate best checkpoint.
            if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
                payload = {
                    'checkpoint_kind': 'training_resume_v1', **run_identity,
                    'epoch': epoch,
                    'model_state_dict': cpu_snapshot(unwrap_model(model).state_dict()),
                    'optimizer_state_dict': cpu_snapshot(optimizer.state_dict()),
                    'best_checkpoint': best_checkpoint,
                    'best_predictions': best_predictions,
                    'history': list(history),
                    'rng_state': capture_rng(train_batches, data_store.devices),
                }
                path = config.output_dir / f'training_checkpoint_epoch_{epoch:04d}.pt'
                save_checkpoint(payload, path)
                del payload
                print('Saved resume checkpoint:', path)
            print(f'Epoch {epoch}: train MSE={train_mse:.6f}, test MSE={test_mse:.6f} '
                  f'| best={best_mse:.6f} at epoch {best_epoch}')

    if best_checkpoint is not None:
        # The resume checkpoint retains the final state; inference uses the best model.
        unwrap_model(model).load_state_dict(best_checkpoint['model_state_dict'])
        model.eval()
        print('Best model:', config.output_dir / 'best_model.pt')

    return {'model': model, 'best_epoch': best_epoch, 'best_mse': best_mse,
            'history': history, 'output_dir': config.output_dir, 'test_batches': test_batches}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['davis', 'kiba'], default='davis')
    parser.add_argument('--data-dir', type=Path, help='Directory containing format-v4 train.pt/test.pt')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-gpus', type=int, default=2, help='1 disables DataParallel')
    parser.add_argument('--batch-size', type=int, default=512, help='GLOBAL batch across GPUs')
    parser.add_argument('--test-batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005, help='Fixed LR, no scheduler')
    parser.add_argument('--epochs', type=int, default=1000, help='TOTAL epochs, including resumed history')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--auto-resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume-path', type=Path)
    parser.add_argument('--input-dir', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=20)
    return parser


def main(argv=None):
    config = TrainingConfig(**vars(build_parser().parse_args(argv)))
    return run_training(config)


if __name__ == '__main__':
    main()
