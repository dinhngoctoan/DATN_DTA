"""Exercise exported notebooks on tiny CPU data; no full DTA training or PyG needed."""
import ast
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def notebook_module(filename, identifiers, name):
    notebook = json.loads((ROOT / 'notebook' / filename).read_text())
    module = ModuleType(name)
    sys.modules[name] = module
    module.REPO_ROOT = ROOT
    for cell in notebook['cells']:
        if cell['id'] in identifiers:
            exec(compile(''.join(cell['source']), cell['id'], 'exec'), module.__dict__)
    return notebook, module


def fixture(part):
    lids = ['a_train_only', 'b', 'c'] if part == 'train' else ['c', 'b']
    pids = ['p0', 'p1']
    n = 10 if part == 'train' else 4
    return {
        'format_version': 4, 'dataset': 'davis', 'split': part,
        'ligand_ids': lids, 'protein_ids': pids, 'protein_embedding_ids': ['e0', 'e1'],
        'ligand_graphs': [{'x': torch.full((1, 78), float(ord(lid[0]) - 96)),
                           'edge_index': torch.empty(2, 0, dtype=torch.long)} for lid in lids],
        'protein_graphs': [{'x': torch.full((1, 41), float(i + 1)),
                            'edge_index': torch.empty(2, 0, dtype=torch.long)} for i in range(2)],
        'ligand_embeddings': [torch.zeros(1, 512) for _ in lids],
        'protein_embeddings': [torch.zeros(1, 640) for _ in pids],
        'pairs': {'ligand_index': torch.arange(n) % len(lids),
                  'protein_index': torch.arange(n) % len(pids),
                  'row_id': [str(i) for i in range(n)], 'y': torch.arange(n).float() / 10 + 5},
    }


class TinyBankModel(torch.nn.Module):
    def __init__(self, store):
        super().__init__()
        self.store = store
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, indices):
        drug, protein = self.store.lookup(indices)
        # One-node fixture graphs. Use features, not local IDs: the test-only bank
        # deliberately has different entity indices from the merged train/test bank.
        return self.linear(torch.stack((drug.x[:, 0], protein.x[:, 0]), dim=1))


class EnsembleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.train_nb, cls.cv = notebook_module('four-source-cv-ensemble.ipynb',
            {'data-and-checkpoints', 'training-library', 'cross-validation-library', 'ensemble-library'},
            '_ensemble_cv_notebook')
        cls.predict_nb, cls.infer = notebook_module('four-source-ensemble-predict.ipynb',
            {'test-data-library', 'inference-library', 'ensemble-library'}, '_ensemble_predict_notebook')

    @classmethod
    def tearDownClass(cls):
        for module in (cls.cv, cls.infer):
            sys.modules.pop(module.__name__, None)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_standalone_cells_compile_and_model_is_unchanged(self):
        original = json.loads((ROOT / 'notebook/four-source-cv.ipynb').read_text())
        original_model = next(c for c in original['cells'] if c['id'] == 'four-source-model')
        wrapped = ast.parse(''.join(original_model['source'])).body[0]
        original_ast = ast.dump(ast.Module(body=wrapped.body, type_ignores=[]))
        for notebook in (self.train_nb, self.predict_nb):
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                compile(source, cell['id'], 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, ('utils', 'training', 'training_cv',
                                                       'ensemble', 'models.gat_gcn'))
            model_cell = next(c for c in notebook['cells'] if c['id'] == 'four-source-model')
            model_ast = ast.parse(''.join(model_cell['source']))
            if isinstance(model_ast.body[0], ast.If):
                model_ast = ast.Module(body=model_ast.body[0].body, type_ignores=[])
            self.assertEqual(ast.dump(model_ast), original_ast)
        run_ids = [c['id'] for c in self.train_nb['cells']]
        self.assertLess(run_ids.index('ensemble-library'), run_ids.index('run'))

    def test_averages_predictions_before_metrics_and_handles_ci_ties(self):
        y = np.array([1., 1., 2., 3.])
        predictions = {i: y + i - 2 for i in range(5)}
        for module in (self.cv, self.infer):
            _, mean, metrics, individual = module.ensemble_metrics(y, predictions)
            np.testing.assert_array_equal(mean, y)
            self.assertEqual(metrics['mse'], 0)
            self.assertEqual(metrics['ci'], 1)
            self.assertEqual(np.mean([m['mse'] for m in individual.values()]), 2)
            with self.assertRaisesRegex(ValueError, '5 fold'):
                module.ensemble_metrics(y, {i: y for i in range(4)})
            bad = {**predictions, 4: np.array([np.nan] * 4)}
            with self.assertRaises(ValueError):
                module.ensemble_metrics(y, bad)
        y, p = np.array([2, 1, 1, 3]), np.array([0, 0, 2, 1])
        scores = [float(p[i] > p[j]) + .5 * (p[i] == p[j])
                  for i in range(4) for j in range(4) if y[i] > y[j]]
        self.assertAlmostEqual(self.infer.regression_metrics(y, p)['ci'], np.mean(scores))
        self.assertIsNone(self.infer.regression_metrics([1, 1], [2, 3])['ci'])

    def prepare_csv_results(self):
        y = np.array([1., 2., 3.])
        for fold in range(5):
            folder = self.root / f'fold_{fold}'
            folder.mkdir()
            prediction = y + fold - 2
            rows = [{'row_id': str(i), 'ligand_id': 'same_drug', 'protein_id': 'same_protein',
                     'affinity': y[i], 'predicted_affinity': prediction[i]} for i in range(3)]
            # Same pair may occur several times: row_id disambiguates it.
            if fold % 2:
                rows.reverse()
            with (folder / 'test_predictions.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.cv.write_json(folder / 'metrics.json', {
                'fold': fold, 'experiment_id': 'test', 'requested_epochs': 2, 'completed_epoch': 2,
                'n_test': 3, 'test_metrics': self.cv.regression_metrics(y, prediction)})

    def test_cv_summary_aligns_rows_and_refuses_partial_stale_mismatched_results(self):
        self.prepare_csv_results()
        result = self.cv.summarize(self.root, 'test', 2)
        self.assertEqual(result['test_metrics']['mse'], 0.)
        with (self.root / 'ensemble_predictions.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([float(r['y_pred_mean']) for r in rows], [1., 2., 3.])
        self.assertEqual([float(r['y_pred_fold_1']) for r in rows], [0., 1., 2.])
        path = self.root / 'fold_4/metrics.json'
        saved = json.loads(path.read_text())
        self.cv.write_json(path, {**saved, 'completed_epoch': 1})
        partial = self.cv.summarize(self.root, 'test', 2)
        self.assertFalse(partial['complete'])
        self.assertIsNone(partial['test_metrics'])
        self.assertFalse((self.root / 'ensemble_predictions.csv').exists())
        self.cv.write_json(path, saved)
        csv_path = self.root / 'fold_4/test_predictions.csv'
        csv_path.write_text(csv_path.read_text().replace('same_drug', 'wrong_drug'))
        with self.assertRaisesRegex(ValueError, 'ID'):
            self.cv.summarize(self.root, 'test', 2)

    def test_five_fold_train_export_and_test_only_inference_agree(self):
        data = self.root / 'data'
        data.mkdir()
        for part in ('train', 'test'):
            torch.save(fixture(part), data / f'{part}.pt')
        config = self.cv.CVConfig(data_dir=data, output_dir=self.root / 'cv',
            input_dir=self.root / 'missing', device='cpu', max_gpus=1, epochs=2,
            batch_size=2, test_batch_size=3, checkpoint_every=1, log_interval=0, auto_resume=False)
        make_model = lambda _, store: TinyBankModel(store)
        with patch.object(self.cv, 'configure_model', side_effect=make_model), \
             patch.object(self.cv, 'original_folds', return_value=list(np.split(np.arange(10), 5))):
            trained = self.cv.run_cross_validation(config)
        self.assertTrue(trained['complete'])
        paths = [config.output_dir / f'fold_{i}/best_model.pt' for i in range(5)]
        # A resume file has deliberately wrong LAST weights but a valid BEST snapshot.
        resume = torch.load(config.output_dir / 'fold_2/training_checkpoint_epoch_0002.pt', weights_only=True)
        for value in resume['model_state_dict'].values():
            value.fill_(999)
        torch.save(resume, self.root / 'resume.pt')
        paths[2] = self.root / 'resume.pt'
        test_only = self.root / 'test_only'
        test_only.mkdir()
        torch.save(fixture('test'), test_only / 'test.pt')
        with patch.object(self.infer, 'configure_model', side_effect=make_model):
            inferred = self.infer.run_ensemble_inference(paths[::-1], test_only / 'test.pt',
                self.root / 'inference', dataset='davis', device='cpu', batch_size=3)
        self.assertEqual(trained['test_metrics'], inferred['test_metrics'])
        self.assertEqual((config.output_dir / 'ensemble_predictions.csv').read_text(),
                         (self.root / 'inference/ensemble_predictions.csv').read_text())
        with self.assertRaisesRegex(ValueError, '5 file'):
            self.infer.validate_checkpoint_set(paths[:4] + [paths[0]], 'davis')
        with self.assertRaisesRegex(ValueError, 'dataset'):
            self.infer.validate_checkpoint_set(paths, 'kiba')
        copied = self.root / 'duplicate_fold.pt'
        payload = torch.load(paths[0], weights_only=True)
        torch.save(payload, copied)
        with self.assertRaisesRegex(ValueError, 'same fold'):
            self.infer.validate_checkpoint_set(paths[:4] + [copied], 'davis')
        payload['fold'], payload['experiment_id'] = 4, 'other_experiment'
        torch.save(payload, copied)
        with self.assertRaisesRegex(ValueError, 'same experiment'):
            self.infer.validate_checkpoint_set(paths[:4] + [copied], 'davis')


if __name__ == '__main__':
    unittest.main()
