"""CPU infrastructure tests only: never execute the DTA model/forward/backward."""
import ast
import copy
import csv
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from create_data import (build_split, ligand_id, prepare_dataset,
                                     raw_fold_rows, FourSourceDataset)
from create_data import protein_embedding_id
from training import TrainingConfig, build_parser
from utils import (FourSourceSplit, build_cpu_tables, DeviceFourSourceStore,
                   IndexBatches, dataset_signature, save_checkpoint,
                   find_resume_checkpoint, capture_rng, restore_rng, cpu_snapshot,
                   execution_devices)


def fixture(name='davis', split='train'):
    sequences = {'p0': 'ACD', 'p1': 'ACD'}  # Equal sequence, distinct graph identities.
    graph_source = {
        'p0': (np.full((3, 41), 1, dtype=np.float32), [[0, 1], [1, 0]]),
        'p1': (np.full((2, 41), 2, dtype=np.float32), []),
    }
    smiles = ['CC', '[Na+]']
    drug = {ligand_id(s): torch.full((i + 2, 512), float(i + 1), dtype=torch.float16)
            for i, s in enumerate(smiles)}
    protein = {protein_embedding_id('ACD'): torch.arange(3 * 640, dtype=torch.float32).view(3, 640)}
    rows = [{'row_id': str(i), 'ligand_id': ligand_id(s), 'protein_id': p,
             'smiles': s, 'affinity': y, 'affinity_text': str(y)}
            for i, (s, p, y) in enumerate([('CC', 'p1', 7.), ('[Na+]', 'p0', 8.), ('CC', 'p1', 9.)])]
    bundle = build_split(name, split, rows, sequences, graph_source, drug, protein,
                         {'ligand': {}, 'protein': {}, 'embedding': {}})
    return bundle, sequences, graph_source, drug, protein, rows


class FourSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def write_split(self, name='davis', split='train'):
        bundle = fixture(name, split)[0]
        path = self.root / f'{name}_{split}.pt'
        torch.save(bundle, path)
        return FourSourceSplit(path, split, name)

    def test_lookup_offsets_pair_order_and_shared_embedding(self):
        train, test = self.write_split(), self.write_split(split='test')
        tables, lm, pm = build_cpu_tables((train, test))
        self.assertEqual(tables['protein_embedding_ptr'].numel(), 2)  # One sequence bank.
        self.assertEqual(tables['protein_node_ptr'].numel(), 3)  # Two distinct graphs.
        store = DeviceFourSourceStore(tables, ['cpu'], lm, pm)
        batches = IndexBatches(train, store, batch_size=2)
        for pairs, labels in batches:
            drug, prot = store.lookup(pairs)
            for j, (li, pi) in enumerate(pairs.tolist()):
                lid, pid = sorted(lm)[li], sorted(pm)[pi]
                old_li, old_pi = train.ligand_ids.index(lid), train.protein_ids.index(pid)
                for obj, key, idx in [(drug, 'ligand', old_li), (prot, 'protein', old_pi)]:
                    graph = train.data[key + '_graphs'][idx]
                    selected = obj.batch == j
                    self.assertTrue(torch.equal(obj.x[selected], graph['x']))
                    offset = int((obj.batch < j).sum())
                    if obj.edge_index.numel():
                        selected_edges = obj.batch[obj.edge_index[0]] == j
                        actual = obj.edge_index[:, selected_edges] - offset
                    else:
                        actual = obj.edge_index
                    self.assertTrue(torch.equal(actual, graph['edge_index']))
                self.assertTrue(torch.equal(drug.drug_embedding[drug.drug_embedding_batch == j],
                                            train.data['ligand_embeddings'][old_li]))
                self.assertTrue(torch.equal(prot.protein_embedding[prot.protein_embedding_batch == j],
                                            train.data['protein_embeddings'][old_pi]))
        all_labels = torch.cat([labels for _, labels in batches]).flatten()
        self.assertTrue(torch.equal(all_labels, torch.tensor([7., 8., 9.])))
        self.assertEqual(len(batches), 2)  # Keep final singleton batch.
        self.assertTrue(torch.equal(next(iter(batches))[0], next(iter(batches))[0]))

    def test_kiba_and_wrong_split_or_mixed_dataset(self):
        kiba = self.write_split('kiba')
        self.assertEqual(kiba.dataset, 'kiba')
        with self.assertRaises(ValueError):
            FourSourceSplit(self.root / 'kiba_train.pt', 'test', 'kiba')
        with self.assertRaises(ValueError):
            FourSourceSplit(self.root / 'kiba_train.pt', 'train', 'davis')
        with self.assertRaises(ValueError):
            build_cpu_tables((kiba, self.write_split('davis')))

    def test_invalid_pairs_embeddings_and_graphs_fail(self):
        b = fixture()[0]
        path = self.root / 'bad.pt'
        for mutate in [lambda x: x['pairs']['ligand_index'].fill_(999),
                       lambda x: x['pairs']['y'].fill_(float('nan')),
                       lambda x: x['ligand_embeddings'].__setitem__(0, torch.zeros(0, 512))]:
            bad = copy.deepcopy(b)
            mutate(bad)
            torch.save(bad, path)
            with self.assertRaises(ValueError):
                FourSourceSplit(path, 'train')
        bad = copy.deepcopy(b)
        bad['protein_graphs'][0]['edge_index'] = torch.tensor([[0], [999]])
        torch.save(bad, path)
        with self.assertRaises(ValueError):
            build_cpu_tables((FourSourceSplit(path, 'train'),))

    def test_conflicting_entity_data_rejected(self):
        a, b = self.write_split(), self.write_split(split='test')
        b.data['ligand_embeddings'][0] = b.data['ligand_embeddings'][0] + 1
        with self.assertRaisesRegex(ValueError, 'Ligand'):
            build_cpu_tables((a, b))

    def test_preparation_no_manifest_idempotence_and_exact_labels(self):
        _, seq, graphs, drug, protein, rows = fixture()
        raw = self.root / 'raw'
        (raw / 'davis').mkdir(parents=True)
        (raw / 'davis/proteins.txt').write_text(json.dumps(seq))
        with (raw / 'davis/protein_to_graph.pkl').open('wb') as f:
            pickle.dump(graphs, f)
        for split in ('train', 'test'):
            with (raw / f'davis_{split}.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['compound_iso_smiles', 'target_sequence', 'affinity'])
                for r in rows:
                    writer.writerow([r['smiles'], r['protein_id'], r['affinity']])
        target = prepare_dataset('davis', raw, drug, protein, self.root / 'out')
        self.assertEqual({p.name for p in target.iterdir()}, {'train.pt', 'test.pt'})
        before = (target / 'train.pt').read_bytes()
        prepare_dataset('davis', raw, {}, {}, self.root / 'out')
        self.assertEqual(before, (target / 'train.pt').read_bytes())
        sample = FourSourceDataset(target / 'train.pt')[0]
        self.assertEqual(sample['protein_id'], 'p1')
        self.assertEqual(sample['y'].item(), 7.)
        self.assertEqual(sample['ligand_embedding'].dtype, torch.float32)  # Runtime dtype.
        saved = torch.load(target / 'train.pt', weights_only=True)
        self.assertEqual(saved['ligand_embeddings'][0].dtype, torch.float16)  # Storage dtype preserved.
        self.assertNotIn('protein_tokens', saved)
        self.assertEqual(saved['protein_embeddings'][0].shape, (3, 640))

    def test_long_protein_tokens_are_not_truncated(self):
        _, sequences, graphs, drug, _, rows = fixture()
        sequences = {pid: 'A' * 1007 for pid in sequences}
        proteins = {protein_embedding_id('A' * 1007): torch.ones(1007, 640, dtype=torch.float16)}
        b = build_split('kiba', 'train', rows, sequences, graphs, drug, proteins,
                        {'ligand': {}, 'protein': {}, 'embedding': {}})
        self.assertEqual(b['protein_embeddings'][0].shape, (1007, 640))
        self.assertEqual(b['protein_embeddings'][0].dtype, torch.float16)
        self.assertIs(b['protein_embeddings'][0], b['protein_embeddings'][1])

    def test_raw_fold_fallback_converts_davis_once_preserves_order(self):
        folder = self.root / 'davis'
        (folder / 'folds').mkdir(parents=True)
        (folder / 'ligands_can.txt').write_text(json.dumps({'d0': 'CC', 'd1': 'CO'}))
        (folder / 'folds/train_fold_setting1.txt').write_text('[[2, 0], [1]]')
        (folder / 'folds/test_fold_setting1.txt').write_text('[3]')
        with (folder / 'Y').open('wb') as f:
            pickle.dump(np.array([[10., 100.], [1000., 10000.]]), f)
        rows = raw_fold_rows('davis', self.root, 'train', {'p0': 'AC', 'p1': 'DE'}, {'p0': (), 'p1': ()})
        self.assertEqual([r['affinity'] for r in rows], [6., 8., 7.])
        self.assertEqual([r['protein_id'] for r in rows], ['p0', 'p0', 'p1'])

    def test_checkpoint_selects_latest_compatible_and_restores_rng(self):
        split = self.write_split()
        tables, lm, pm = build_cpu_tables((split,))
        store = DeviceFourSourceStore(tables, ['cpu'], lm, pm)
        batches = IndexBatches(split, store, 2, shuffle=True)
        state = capture_rng(batches, store.devices)
        expected = {'model_signature': 'same-model', 'dataset_signature': dataset_signature((split,))}
        def payload(epoch):
            return {'checkpoint_kind': 'training_resume_v1', **expected, 'epoch': epoch,
                    'model_state_dict': {'w': torch.tensor([epoch])},
                    'optimizer_state_dict': {'state': {}, 'param_groups': []},
                    'best_checkpoint': {'model_state_dict': {'w': torch.tensor([97])},
                                        'epoch': 97, 'test_mse': 0.15},
                    'best_predictions': torch.tensor([7., 8., 9.]),
                    'history': [{'epoch': i} for i in range(1, epoch + 1)], 'rng_state': state}
        save_checkpoint(payload(100), self.root / 'checkpoint_100.pt')
        save_checkpoint(payload(200), self.root / 'checkpoint_200.pt')
        wrong = payload(900)
        wrong['model_signature'] = 'wrong-model'
        save_checkpoint(wrong, self.root / 'checkpoint_900.pt')
        save_checkpoint({'model_state_dict': {}}, self.root / 'best_model.pt')
        path, saved = find_resume_checkpoint([self.root], expected)
        self.assertEqual(path.name, 'checkpoint_200.pt')
        self.assertEqual(saved['epoch'] + 1, 201)
        self.assertEqual(saved['best_checkpoint']['epoch'], 97)
        self.assertEqual(saved['model_state_dict']['w'].item(), 200)
        self.assertEqual(saved['best_checkpoint']['model_state_dict']['w'].item(), 97)
        a = (random.random(), np.random.rand(), torch.rand(2), [x.clone() for x, _ in batches])
        restore_rng(saved['rng_state'], batches, store.devices)
        b = (random.random(), np.random.rand(), torch.rand(2), [x.clone() for x, _ in batches])
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a[3], b[3])))
        with self.assertRaises(ValueError):
            find_resume_checkpoint([], expected, self.root / 'checkpoint_900.pt')
        original = torch.tensor([1.])
        snapshot = cpu_snapshot({'x': original})
        original.fill_(2)
        self.assertEqual(snapshot['x'].item(), 1.)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_cli_and_notebook_model_match_without_execution(self):
        config = TrainingConfig(**vars(build_parser().parse_args(['--dataset', 'kiba', '--max-gpus', '1'])))
        self.assertEqual(config.lr, 0.0005)
        self.assertEqual(config.epochs, 1000)
        self.assertEqual(config.data_dir, ROOT / 'data/four_source_embeddings/kiba')
        self.assertEqual(execution_devices('cpu'), [torch.device('cpu')])
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=0)
        nb = json.loads((ROOT / 'notebook/davis-attention-original copy.ipynb').read_text())
        source = (ROOT / 'models/gat_gcn.py').read_text()
        embedded = next(''.join(c['source']) for c in nb['cells'] if c.get('id') == 'four-source-model')
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(embedded)))
        self.assertIn('self.conv2 = GATConv(512, 64, heads=4, concat=True)', source)
        self.assertIn('self.conv_prot2 = GCNConv(512, 256)', source)
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), c.get('id', 'cell'), 'exec')


if __name__ == '__main__':
    unittest.main()
