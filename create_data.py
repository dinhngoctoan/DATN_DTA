#!/usr/bin/env python3
"""Prepare four-source format-v4 splits from graph sources and precomputed embeddings.

Entry point: python create_data.py --datasets davis kiba
Output: {output_dir}/{dataset}/{train,test}.pt, each a self-contained dictionary.
No padding, token pooling, sequence CNN, ECFP, model downloads or embedding inference.
Existing complete datasets are validated and reused; never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import tempfile

import numpy as np
import torch
from rdkit import Chem, rdBase

from utils import FourSourceSplit

FORMAT_VERSION = 4


def protein_embedding_id(sequence):
    return 'prot_' + hashlib.sha256(sequence.encode('utf-8')).hexdigest()

ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
                'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

def ligand_id(smiles):
    return 'lig_' + hashlib.sha256(smiles.encode('utf-8')).hexdigest()

def one_hot(value, choices, unknown=False):
    if value not in choices:
        if not unknown:
            raise ValueError(f'Value {value} is not in {choices}')
        value = choices[-1]
    return [value == item for item in choices]

def ligand_graph(smiles):
    """Match the repository's smile_to_graph: 78 features per atom and the same edge order."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    features = []
    # Preserve the original implicit-valence definition; suppress only deprecation warnings.
    with rdBase.BlockLogs():
        for atom in mol.GetAtoms():
            feature = np.array(
                one_hot(atom.GetSymbol(), ATOM_SYMBOLS, unknown=True)
                + one_hot(atom.GetDegree(), list(range(11)))
                + one_hot(atom.GetTotalNumHs(), list(range(11)), unknown=True)
                + one_hot(atom.GetImplicitValence(), list(range(11)), unknown=True)
                + [atom.GetIsAromatic()])
            features.append(feature / feature.sum())
    # Preserve node and edge order as in nx.Graph(edges).to_directed().
    adjacency = {}
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency.setdefault(u, {})[v] = None
        adjacency.setdefault(v, {})[u] = None
    edges = [(u, v) for u, neighbors in adjacency.items() for v in neighbors]
    edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return {'x': torch.tensor(np.asarray(features), dtype=torch.float32),
            'edge_index': edge_index}

def protein_graph(value, protein_id):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f'{protein_id}: expected graph (features, edges[, edge_features])')
    x = torch.as_tensor(np.asarray(value[0]), dtype=torch.float32).clone()
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 41 or not torch.isfinite(x).all():
        raise ValueError(f'{protein_id}: node features must be finite floats with shape [N>0,41]')
    raw_edges = np.asarray(value[1])
    if raw_edges.size == 0:
        raw_edges = np.empty((0, 2), dtype=np.int64)
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError(f'{protein_id}: protein_to_graph edges must have shape [E,2]')
    if not np.issubdtype(raw_edges.dtype, np.number) or not np.isfinite(raw_edges).all():
        raise ValueError(f'{protein_id}: edges contain non-finite values')
    if not np.equal(raw_edges, np.floor(raw_edges)).all():
        raise ValueError(f'{protein_id}: edge indices must be integers')
    edges = torch.tensor(raw_edges, dtype=torch.long).t().contiguous()
    if edges.numel() and (edges.min() < 0 or edges.max() >= len(x)):
        raise ValueError(f'{protein_id}: node index outside the graph')
    graph = {'x': x, 'edge_index': edges}
    if len(value) > 2 and value[2] is not None:
        attrs = torch.as_tensor(np.asarray(value[2]), dtype=torch.float32).clone()
        if attrs.ndim < 1 or attrs.shape[0] != edges.shape[1] or not torch.isfinite(attrs).all():
            raise ValueError(f'{protein_id}: edge_attr has the wrong edge count or contains NaN/Inf')
        graph['edge_attr'] = attrs  # Retained for future use; the current GCN does not use edge attributes.
    return graph

def read_pairs(path, sequences, graph_source):
    rows = []
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = {'compound_iso_smiles', 'target_sequence', 'affinity'}
        if not required.issubset(columns):
            raise ValueError(f'{path}: required columns: {sorted(required)}')
        for i, row in enumerate(reader):
            where = f'{path}: row {i + 2}'
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{where}: CSV row has missing or extra fields')
            smiles = row['compound_iso_smiles'].strip()
            pid = row['target_sequence'].strip()  # Original graph lookup ID, not the sequence hash.
            try:
                y = float(row['affinity'])
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{where}: invalid affinity') from exc
            if not smiles or not math.isfinite(y) or abs(y) > np.finfo(np.float32).max:
                raise ValueError(f'{where}: empty SMILES or non-finite affinity')
            if pid not in sequences or pid not in graph_source:
                raise ValueError(f'{where}: missing sequence or graph for protein {pid!r}')
            lid = ligand_id(smiles)
            if 'ligand_id' in row and row['ligand_id'] != lid:
                raise ValueError(f'{where}: ligand_id does not match SMILES')
            if 'protein_id' in row:
                seq_id = 'prot_' + hashlib.sha256(sequences[pid].encode('utf-8')).hexdigest()
                if row['protein_id'] != seq_id:
                    raise ValueError(f'{where}: embedding protein_id does not match the sequence')
            rows.append({'row_id': row.get('row_id', str(i)), 'ligand_id': lid,
                         'protein_id': pid, 'smiles': smiles, 'affinity': y,
                         'affinity_text': row['affinity']})
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    return rows

def collect_ligand_embeddings(ids, embeddings):
    matrices, lengths = [], []
    for lid in ids:
        if lid not in embeddings:
            raise ValueError(f'Missing embedding for {lid}; use the embedding notebook output dictionary')
        value = embeddings[lid]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 512
                or not value.is_floating_point()):
            raise ValueError(f'{lid}: embedding must be an unpadded float tensor with shape [L>0,512]')
        value = value.detach().to(device='cpu')
        if not torch.isfinite(value).all():
            raise ValueError(f'{lid}: embedding contains NaN/Inf')
        # Clone to save only this entity's storage, preserving its dtype and values.
        matrices.append(value.clone())
        lengths.append(len(value))
    return matrices, torch.tensor(lengths, dtype=torch.long)

def pair_tensors(rows, ligand_map, protein_map):
    return {
        'ligand_index': torch.tensor([ligand_map[r['ligand_id']] for r in rows], dtype=torch.long),
        'protein_index': torch.tensor([protein_map[r['protein_id']] for r in rows], dtype=torch.long),
        'y': torch.tensor([r['affinity'] for r in rows], dtype=torch.float32),
        'row_id': [r['row_id'] for r in rows],
    }

def write_pair_csv(path, rows, ligand_map, protein_map):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'row_id', 'ligand_id', 'protein_id', 'ligand_index', 'protein_index',
            'compound_iso_smiles', 'affinity'])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'row_id': r['row_id'], 'ligand_id': r['ligand_id'], 'protein_id': r['protein_id'],
                'ligand_index': ligand_map[r['ligand_id']], 'protein_index': protein_map[r['protein_id']],
                'compound_iso_smiles': r['smiles'], 'affinity': r['affinity_text'],
            })

def raw_fold_rows(name, data_dir, split, sequences, graphs):
    """Fallback when CSV is absent: retain original DeepDTA fold/order and affinity scale."""
    folder = Path(data_dir) / name
    ligands = json.loads((folder / 'ligands_can.txt').read_text())
    with (folder / 'Y').open('rb') as handle:
        affinity = np.asarray(pickle.load(handle, encoding='latin1'), dtype=np.float64)
    if name == 'davis':
        observed = ~np.isnan(affinity)
        if np.any(affinity[observed] <= 0):
            raise ValueError('DAVIS raw affinity must be positive before conversion to pKd')
        affinity = -np.log10(affinity / 1e9)
    rows, cols = np.where(~np.isnan(affinity))
    fold_name = 'train_fold_setting1.txt' if split == 'train' else 'test_fold_setting1.txt'
    fold = json.loads((folder / 'folds' / fold_name).read_text())
    if split == 'train':
        fold = [index for group in fold for index in group]
    smiles_list = []
    for smiles in ligands.values():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid raw SMILES: {smiles}')
        smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    protein_ids = list(sequences)
    result = []
    for row_id, index in enumerate(fold):
        if not isinstance(index, int) or index < 0 or index >= len(rows):
            raise ValueError(f'{name}/{split}: fold index out of bounds')
        smiles, pid = smiles_list[rows[index]], protein_ids[cols[index]]
        y = float(affinity[rows[index], cols[index]])
        if pid not in graphs or not math.isfinite(y):
            raise ValueError(f'{name}/{split}: missing graph or invalid affinity for {pid}')
        result.append({'row_id': str(row_id), 'ligand_id': ligand_id(smiles),
                       'protein_id': pid, 'smiles': smiles, 'affinity': y,
                       'affinity_text': str(y)})
    if not result:
        raise ValueError(f'{name}/{split}: empty fold')
    return result


def build_split(name, split, rows, sequences, graph_source, drug_embeddings,
                protein_embeddings, cache):
    """Entity dictionaries/lists are deduplicated; pairs keep the original row order."""
    smiles_by_id = {row['ligand_id']: row['smiles'] for row in rows}
    lids, pids = sorted(smiles_by_id), sorted({row['protein_id'] for row in rows})
    ligand_map = {key: i for i, key in enumerate(lids)}
    protein_map = {key: i for i, key in enumerate(pids)}
    for lid in lids:
        if lid not in cache['ligand']:
            embedding, _ = collect_ligand_embeddings([lid], drug_embeddings)
            cache['ligand'][lid] = (ligand_graph(smiles_by_id[lid]), embedding[0])
    embedding_ids = []
    for pid in pids:
        sequence = sequences[pid]
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f'{pid}: empty protein sequence')
        eid = protein_embedding_id(sequence)
        if eid not in cache['embedding']:
            if eid not in protein_embeddings:
                raise ValueError(f'{pid}: missing protein embedding {eid}')
            value = protein_embeddings[eid]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != (len(sequence), 640) or not value.is_floating_point()
                    or not torch.isfinite(value).all()):
                raise ValueError(f'{pid}: expected finite [{len(sequence)},640], no padding/BOS/EOS')
            cache['embedding'][eid] = value.detach().cpu().clone()
        if pid not in cache['protein']:
            cache['protein'][pid] = protein_graph(graph_source[pid], pid)
        embedding_ids.append(eid)
    lengths = torch.tensor([len(sequences[pid]) for pid in pids], dtype=torch.long)
    return {
        'format_version': FORMAT_VERSION, 'dataset': name, 'split': split,
        'ligand_ids': lids, 'protein_ids': pids,
        'ligand_id_to_index': ligand_map, 'protein_id_to_index': protein_map,
        'ligand_smiles': [smiles_by_id[lid] for lid in lids],
        'ligand_graphs': [cache['ligand'][lid][0] for lid in lids],
        'ligand_embeddings': [cache['ligand'][lid][1] for lid in lids],
        'ligand_embedding_token_lengths': torch.tensor(
            [len(cache['ligand'][lid][1]) for lid in lids], dtype=torch.long),
        'protein_graphs': [cache['protein'][pid] for pid in pids],
        'protein_embedding_ids': embedding_ids,
        'protein_embeddings': [cache['embedding'][eid] for eid in embedding_ids],
        'protein_sequence_lengths': lengths,
        'protein_embedding_token_lengths': lengths.clone(),
        'pairs': pair_tensors(rows, ligand_map, protein_map),
        'config': {
            'ligand_feature_dim': 78, 'protein_feature_dim': 41,
            'ligand_embedding_dim': 512, 'protein_embedding_dim': 640,
            'ligand_embedding_pooling': 'none', 'protein_embedding_pooling': 'none',
            'ligand_embedding_layout': 'list_of_unpadded_token_matrices',
            'protein_embedding_layout': 'list_of_unpadded_token_matrices',
            'ligand_embedding_dtype': 'preserved_from_source',
            'protein_embedding_dtype': 'preserved_from_source',
            'ligand_embedding_special_tokens': 'preserved_from_source',
            'protein_embedding_special_tokens': 'excluded_in_source',
            'protein_embedding_truncation': 'none',
            'protein_key': 'original_dataset_protein_id',
            'protein_embedding_key': 'prot_<sha256(original_sequence)>',
            'ligand_graph_smiles': 'unchanged_csv_smiles_including_salts',
        },
    }


def validate_existing(target, name):
    """Validate each split independently without allocating a GPU bank."""
    from utils import pack_graphs
    for split in ('train', 'test'):
        data = FourSourceSplit(target / f'{split}.pt', split, name)
        # Validate graph shapes/indices too; temporary packed tensors are released per source.
        for key, width in [('ligand_graphs', 78), ('protein_graphs', 41)]:
            pack_graphs(data.data[key], width)
        print(f'{name}/{split}: {len(data):,} pairs, format v4 OK', flush=True)
        del data


def prepare_dataset(name, data_dir, drug_embeddings, protein_embeddings, output_dir,
                    write_pairs=False):
    target = Path(output_dir) / name
    if target.exists():
        validate_existing(target, name)
        print(f'Reusing {target}; no files overwritten.', flush=True)
        return target
    data_dir = Path(data_dir)
    sequences = json.loads((data_dir / name / 'proteins.txt').read_text())
    with (data_dir / name / 'protein_to_graph.pkl').open('rb') as handle:
        graph_source = pickle.load(handle)  # Existing trusted graph source in the repo.
    cache = {'ligand': {}, 'protein': {}, 'embedding': {}}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{name}-v4-', dir=target.parent) as tmp:
        staged = Path(tmp) / name
        staged.mkdir()
        for split in ('train', 'test'):
            csv_path = data_dir / f'{name}_{split}.csv'
            rows = (read_pairs(csv_path, sequences, graph_source) if csv_path.is_file()
                    else raw_fold_rows(name, data_dir, split, sequences, graph_source))
            bundle = build_split(name, split, rows, sequences, graph_source,
                                 drug_embeddings, protein_embeddings, cache)
            torch.save(bundle, staged / f'{split}.pt')
            if write_pairs:
                write_pair_csv(staged / f'{split}_pairs.csv', rows,
                               bundle['ligand_id_to_index'], bundle['protein_id_to_index'])
            print(f'{name}/{split}: wrote {len(rows):,} pairs', flush=True)
        # Validate before publishing so an interrupted preparation cannot appear complete.
        validate_existing(staged, name)
        os.rename(staged, target)
    return target


class FourSourceDataset(torch.utils.data.Dataset):
    """Read a format-v4 split; samples contain references, not padded/copied tensors."""
    def __init__(self, path, split=None):
        if split is None:
            split = Path(path).stem
        self.data = FourSourceSplit(path, split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        b, pairs = self.data.data, self.data.pairs
        li, pi = int(pairs['ligand_index'][index]), int(pairs['protein_index'][index])
        return {'ligand_graph': b['ligand_graphs'][li],
                'ligand_embedding': b['ligand_embeddings'][li],
                'protein_graph': b['protein_graphs'][pi],
                'protein_embedding': b['protein_embeddings'][pi],
                'y': pairs['y'][index], 'row_id': pairs['row_id'][index],
                'ligand_id': b['ligand_ids'][li], 'protein_id': b['protein_ids'][pi]}


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['davis', 'kiba'], default=['davis'])
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--ligand-embeddings', type=Path, default=root.parent / 'ligand_embeddings.pt')
    parser.add_argument('--protein-embeddings', type=Path, default=root.parent / 'protein_embeddings.pt')
    parser.add_argument('--output-dir', type=Path, default=root / 'data/four_source_embeddings')
    parser.add_argument('--check-only', action='store_true', help='Validate existing splits; do not prepare data')
    parser.add_argument('--write-pairs-csv', action='store_true', help='Optional human-readable pair tables')
    args = parser.parse_args(argv)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    pending = []
    for name in dict.fromkeys(args.datasets):
        target = args.output_dir / name
        if args.check_only or target.exists():
            validate_existing(target, name)
        else:
            pending.append(name)
    if not pending:
        print('All requested format-v4 datasets are ready; no preprocessing needed.')
        return
    banks = []
    for path in (args.ligand_embeddings, args.protein_embeddings):
        if not path.is_file():
            raise FileNotFoundError(f'Missing precomputed embedding dictionary: {path}')
        bank = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if not isinstance(bank, dict):
            raise ValueError(f'{path}: expected ID -> token matrix dictionary')
        banks.append(bank)
    for name in pending:
        prepare_dataset(name, args.data_dir, *banks, args.output_dir, args.write_pairs_csv)


if __name__ == '__main__':
    main()
