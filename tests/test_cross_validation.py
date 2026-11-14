"""Fold identity/leakage, CI ties, checkpoint selection, and resume on tiny CPU data."""
import csv
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import training_cv as cv
from create_data import raw_fold_rows
from utils import FourSourceSplit, dataset_signature, resume_problem


def split_from_rows(rows):
    split = FourSourceSplit.__new__(FourSourceSplit)
    split.dataset = 'davis'
    split.ligand_ids = sorted({r['ligand_id'] for r in rows})
    split.protein_ids = sorted({r['protein_id'] for r in rows})
    split.pairs = {
        'ligand_index': torch.tensor([split.ligand_ids.index(r['ligand_id']) for r in rows]),
        'protein_index': torch.tensor([split.protein_ids.index(r['protein_id']) for r in rows]),
        'y': torch.tensor([r['affinity'] for r in rows], dtype=torch.float32),
        'row_id': [str(i) for i in range(len(rows))],
    }
    split.labels = split.pairs['y']
    return split


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        folder = self.root / 'davis'
        (folder / 'folds').mkdir(parents=True)
        self.groups = [[8, 0], [3, 9], [1, 7], [2, 6], [5, 4]]
        (folder / 'folds/train_fold_setting1.txt').write_text(json.dumps(self.groups))
        (folder / 'folds/test_fold_setting1.txt').write_text('[10, 11]')
        (folder / 'ligands_can.txt').write_text(json.dumps({'d0': 'CC', 'd1': 'CO', 'd2': 'CN'}))
        seq = {'p0': 'AC', 'p1': 'AD', 'p2': 'AE', 'p3': 'AF'}
        (folder / 'proteins.txt').write_text(json.dumps(seq))
        with (folder / 'Y').open('wb') as handle:
            pickle.dump(10. ** np.arange(1., 13.).reshape(3, 4), handle)
        self.rows = raw_fold_rows('davis', self.root, 'train', seq, seq)
        self.test_rows = raw_fold_rows('davis', self.root, 'test', seq, seq)
        self.train, self.test = split_from_rows(self.rows), split_from_rows(self.test_rows)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_fold_membership_survives_prepared_row_reordering(self):
        prepared = split_from_rows(self.rows[::-1])
        groups = cv.original_folds(prepared, self.test, self.root)
        self.assertEqual([len(g) for g in groups], [2] * 5)
        self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(10)))
        self.assertEqual(groups[0].tolist(), [9, 8])
        for fold, validation in enumerate(groups):
            training = np.concatenate([g for i, g in enumerate(groups) if i != fold])
            self.assertFalse(set(training) & set(validation))
            self.assertEqual(len(training), 8)
        subset = cv.PairSubset(prepared, groups[0])
        self.assertEqual(subset.pair_ids(0), self.train.pair_ids(0))
        self.assertTrue(torch.equal(subset.labels, self.train.labels[:2]))
        self.assertNotEqual(dataset_signature((subset,)), dataset_signature((cv.PairSubset(prepared, groups[1]),)))

    def test_rejects_changed_test_labels_pairs_and_overlapping_folds(self):
        self.test.labels[0] += 0.1
        with self.assertRaisesRegex(ValueError, 'affinity'):
            cv.original_folds(self.train, self.test, self.root)
        self.test = split_from_rows(self.test_rows[::-1])  # Reordering is allowed.
        cv.original_folds(self.train, self.test, self.root)
        self.test.pairs['protein_index'][0] = self.test.pairs['protein_index'][1]
        with self.assertRaisesRegex(ValueError, 'missing/duplicated'):
            cv.original_folds(self.train, self.test, self.root)
        self.groups[0][0] = 10  # Also present in test.
        (self.root / 'davis/folds/train_fold_setting1.txt').write_text(json.dumps(self.groups))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            cv.original_folds(self.train, self.test, self.root)

    def test_metrics_match_brute_force_with_ties(self):
        rng = np.random.default_rng(7)
        y, pred = rng.integers(0, 5, 100), rng.integers(0, 5, 100)
        scores = [float(pred[i] > pred[j]) + 0.5 * (pred[i] == pred[j])
                  for i in range(len(y)) for j in range(len(y)) if y[i] > y[j]]
        actual = cv.regression_metrics(y, pred)
        self.assertAlmostEqual(actual['ci'], float(np.mean(scores)))
        self.assertAlmostEqual(actual['mse'], float(np.mean((y - pred) ** 2)))
        self.assertEqual(cv.regression_metrics([1, 2, 3], [1, 2, 3])['ci'], 1.)
        self.assertEqual(cv.regression_metrics([1, 2, 3], [3, 2, 1])['ci'], 0.)
        self.assertEqual(cv.regression_metrics([1, 2, 3], [2, 2, 2])['ci'], 0.5)
        self.assertIsNone(cv.regression_metrics([1, 1], [2, 3])['ci'])

    def test_duplicate_canonical_records_require_verified_original_row_ids(self):
        rows = [self.rows[0], self.rows[1], dict(self.rows[0])]
        prepared = split_from_rows(rows)
        self.assertEqual(cv.match_original_rows(prepared, rows).tolist(), [0, 1, 2])
        reordered = cv.PairSubset(prepared, [2, 0, 1])
        self.assertEqual(cv.match_original_rows(reordered, rows).tolist(), [1, 2, 0])
        reordered.pairs['row_id'] = ['changed'] * 3
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            cv.match_original_rows(reordered, rows)

    def test_validation_selects_checkpoint_test_only_after_training_and_resume(self):
        config = cv.CVConfig(output_dir=self.root / 'out', input_dir=self.root / 'empty',
                             raw_data_dir=self.root, epochs=2, checkpoint_every=1, max_gpus=1)
        train = cv.PairSubset(self.train, np.arange(2, 10))
        valid = cv.PairSubset(self.train, [0, 1])
        lids = sorted(set(self.train.ligand_ids + self.test.ligand_ids))
        pids = sorted(set(self.train.protein_ids + self.test.protein_ids))
        store = SimpleNamespace(device=torch.device('cpu'), devices=[torch.device('cpu')],
                                ligand_id_to_index={v: i for i, v in enumerate(lids)},
                                protein_id_to_index={v: i for i, v in enumerate(pids)})
        calls = []

        def tiny_train(model, batches, optimizer, epoch, log_interval):
            with torch.no_grad():
                model.weight.fill_(epoch)
            return 1.

        def fake_predict(model, batches):
            epoch = round(model.weight.item())
            labels = batches.labels.flatten().numpy()
            is_test = np.array_equal(labels, self.test.labels.numpy())
            calls.append(('test' if is_test else 'validation', epoch))
            # Validation prefers epoch 1; test would prefer epoch 2.
            error = (3 - epoch) if is_test else epoch
            return labels, labels + error

        with patch.object(cv, 'configure_model', side_effect=lambda *_: torch.nn.Linear(1, 1, bias=False)), \
             patch.object(cv, 'train', side_effect=tiny_train), \
             patch.object(cv, 'predicting', side_effect=fake_predict):
            result = cv.run_fold(config, 0, train, valid, self.test, store, 'experiment')
            self.assertEqual(calls, [('validation', 1), ('validation', 2), ('test', 1)])
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['test_metrics']['mse'], 4.)
            checkpoint = config.output_dir / 'fold_0/training_checkpoint_epoch_0002.pt'
            saved = torch.load(checkpoint, weights_only=True)
            self.assertIsNone(resume_problem(saved, {'fold': 0}))
            self.assertIsNotNone(resume_problem(saved, {'fold': 1}))
            self.assertEqual(saved['model_state_dict']['weight'].item(), 2.)
            config.epochs = 3
            calls.clear()
            result = cv.run_fold(config, 0, train, valid, self.test, store, 'experiment')
            self.assertEqual(calls, [('validation', 3), ('test', 1)])
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['completed_epoch'], 3)
            with (config.output_dir / 'fold_0/history.csv').open() as handle:
                self.assertEqual([int(r['epoch']) for r in csv.DictReader(handle)], [1, 2, 3])

    def test_summary_averages_metrics_and_excludes_other_runs(self):
        output = self.root / 'out'
        for fold in range(5):
            path = output / f'fold_{fold}'
            path.mkdir(parents=True)
            cv.write_json(path / 'metrics.json', {
                'experiment_id': 'a', 'fold': fold, 'requested_epochs': 2,
                'best_epoch': 1, 'validation_mse': 0.5,
                'test_metrics': {'mse': float(fold), 'rmse': float(fold) ** 0.5,
                                 'ci': 0.8, 'pearson': 0.9}})
        result = cv.summarize(output, 'a', 2)
        self.assertTrue(result['complete'])
        self.assertEqual(result['test_metrics']['mse']['mean'], 2.)
        self.assertAlmostEqual(result['test_metrics']['mse']['std'], np.std(np.arange(5), ddof=1))
        self.assertFalse(cv.summarize(output, 'b', 2)['complete'])
        self.assertEqual(cv.summarize(output, 'a', 3)['completed_folds'], [])

    def test_all_five_folds_share_fixed_test_and_start_fresh_models(self):
        config = cv.CVConfig(output_dir=self.root / 'all_folds', raw_data_dir=self.root,
                             input_dir=self.root / 'empty', epochs=1, max_gpus=1)
        lids = sorted(set(self.train.ligand_ids + self.test.ligand_ids))
        pids = sorted(set(self.train.protein_ids + self.test.protein_ids))
        store = SimpleNamespace(device=torch.device('cpu'), devices=[torch.device('cpu')],
                                ligand_id_to_index={v: i for i, v in enumerate(lids)},
                                protein_id_to_index={v: i for i, v in enumerate(pids)})
        training_labels, validation_labels, test_labels = [], [], []

        def make_model(*_):
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.zero_()
            return model

        def tiny_train(model, batches, *_):
            self.assertEqual(model.weight.item(), 0.)
            with torch.no_grad():
                model.weight.fill_(1.)
            training_labels.append(set(batches.labels.flatten().tolist()))
            return 1.

        def fake_predict(model, batches):
            labels = batches.labels.flatten().numpy()
            if np.array_equal(labels, self.test.labels.numpy()):
                test_labels.append(labels.copy())
            else:
                validation_labels.append(set(labels.tolist()))
            return labels, labels + 0.5

        with patch.object(cv, 'load_data', return_value=(self.train, self.test, store, None, None)), \
             patch.object(cv, 'configure_model', side_effect=make_model), \
             patch.object(cv, 'train', side_effect=tiny_train), \
             patch.object(cv, 'predicting', side_effect=fake_predict):
            result = cv.run_cross_validation(config)
        self.assertTrue(result['complete'])
        self.assertEqual(len(test_labels), 5)
        self.assertEqual(len(training_labels), 5)
        self.assertEqual(sum(map(len, validation_labels)), 10)
        self.assertEqual(set.union(*validation_labels), set(self.train.labels.tolist()))
        for train, valid in zip(training_labels, validation_labels):
            self.assertFalse(train & valid)
            self.assertEqual(len(train), 8)
        self.assertEqual(result['test_metrics']['mse']['mean'], 0.25)
        self.assertEqual(result['test_metrics']['mse']['std'], 0.)


if __name__ == '__main__':
    unittest.main()
