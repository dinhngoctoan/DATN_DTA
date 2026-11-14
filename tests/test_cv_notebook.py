"""Validate standalone export and original folds; never instantiate/run a DTA model."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import training_cv


class CVNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((ROOT / 'notebook/four-source-cv.ipynb').read_text())
        cls.cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        cls.code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        cls.module = ModuleType('_cv_notebook_test')
        sys.modules[cls.module.__name__] = cls.module
        cls.module.REPO_ROOT = ROOT
        cls.module.FOLDS_PATH = None
        cls.module.DATA_DIR = ROOT / 'data/four_source_embeddings/davis'
        # No dependency installer, model cell, training call, or GPU bank allocation.
        for name in ('data-and-checkpoints', 'training-library', 'cross-validation-library'):
            exec(compile(cls.cells[name], name, 'exec'), cls.module.__dict__)

    @classmethod
    def tearDownClass(cls):
        del sys.modules[cls.module.__name__]

    def test_cells_compile_and_have_no_repo_runtime_imports(self):
        for cell in self.code_cells:
            source = ''.join(cell['source'])
            compile(source, cell['id'], 'exec')
            self.assertIsNone(cell['execution_count'])
            self.assertEqual(cell['outputs'], [])
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, ('training', 'training_cv', 'utils',
                                                   'create_data', 'models.gat_gcn'))
                if isinstance(node, ast.Name):
                    self.assertNotEqual(node.id, '__file__')

    def test_model_and_cv_training_match_repository(self):
        wrapped = ast.parse(self.cells['four-source-model']).body[0]
        exported = ast.Module(body=wrapped.body, type_ignores=[])
        self.assertEqual(ast.dump(exported), ast.dump(ast.parse((ROOT / 'models/gat_gcn.py').read_text())))
        library = ast.parse(self.cells['cross-validation-library'])
        functions = {n.name: ast.dump(n) for n in library.body if isinstance(n, ast.FunctionDef)}
        for node in ast.parse((ROOT / 'training_cv.py').read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name not in ('original_folds', 'build_parser', 'main'):
                self.assertEqual(functions[node.name], ast.dump(node), node.name)

    def test_fold_file_errors(self):
        from types import SimpleNamespace
        split = SimpleNamespace(dataset='davis')
        with tempfile.TemporaryDirectory() as folder:
            self.module.FOLDS_PATH = Path(folder) / 'missing.json'
            try:
                with self.assertRaises(FileNotFoundError):
                    self.module.original_folds(split, split)
                self.module.FOLDS_PATH.write_text('{"format_version":1,"dataset":"kiba"}')
                with self.assertRaisesRegex(ValueError, 'dataset'):
                    self.module.original_folds(split, split)
            finally:
                self.module.FOLDS_PATH = None

    def test_external_folds_match_raw_folds_for_existing_prepared_data(self):
        for name in ('davis', 'kiba'):
            folder = ROOT / 'data/four_source_embeddings' / name
            if not all((folder / f'{split}.pt').is_file() for split in ('train', 'test')):
                self.skipTest('Local v4 data not present')
            splits = []
            for part in ('train', 'test'):
                data = torch.load(folder / f'{part}.pt', map_location='cpu', weights_only=True, mmap=True)
                # Access only pair metadata: embedding/graph tensors stay memory-mapped.
                split = self.module.FourSourceSplit.__new__(self.module.FourSourceSplit)
                split.dataset = data['dataset']
                split.ligand_ids, split.protein_ids = data['ligand_ids'], data['protein_ids']
                split.pairs = data['pairs']
                split.labels = split.pairs['y'].float().reshape(-1)
                splits.append(split)
                del data
            train, test = splits
            self.module.DATA_DIR = folder
            self.module.FOLDS_PATH = None
            expected = training_cv.original_folds(train, test, ROOT / 'data')
            actual = self.module.original_folds(train, test)
            self.module.FOLDS_PATH = ROOT / 'data/cv_folds' / f'{name}_folds.json'
            for a, b in zip(self.module.original_folds(train, test), actual):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(len(actual), 5)
            np.testing.assert_array_equal(np.sort(np.concatenate(actual)), np.arange(len(train)))
            # Reordered .pt pairs retain original row_id, including duplicate KIBA pairs.
            order = torch.arange(len(train) - 1, -1, -1)
            reordered = self.module.PairSubset(train, order)
            for a, b in zip(self.module.original_folds(reordered, test), actual):
                np.testing.assert_array_equal(a, len(train) - 1 - b)
            bad_test = self.module.PairSubset(test, torch.arange(len(test)))
            bad_test.labels[0] += 1
            with self.assertRaises(ValueError):
                self.module.original_folds(train, bad_test)
            print(f'{name}: verified original folds {[len(g) for g in actual]}, fixed test={len(test)}')
        self.module.FOLDS_PATH = None


if __name__ == '__main__':
    unittest.main()
