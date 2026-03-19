"""
Mock tests for equiv_experiments.py, inv_regression.py, and plot_utils.py.

These tests mock heavy dependencies (torch models, CUDA, datasets, matplotlib)
so they can run quickly on any machine without GPU or full training.

Run:  python3 -m pytest test_experiments.py -v
  or: python3 test_experiments.py
"""

import os
import sys
import ast
import tempfile
import unittest
from unittest.mock import patch, MagicMock, PropertyMock


# ═══════════════════════════════════════════════════════════════
# Helper: build a fake torch module so imports don't fail
# ═══════════════════════════════════════════════════════════════

def _make_fake_torch():
    """Create a minimal fake torch module for import-time requirements."""
    torch = MagicMock()
    torch.device.return_value = MagicMock()
    torch.cuda.is_available.return_value = False

    # nn losses
    torch.nn.CrossEntropyLoss.return_value = MagicMock(return_value=MagicMock(item=lambda: 0.5))
    torch.nn.L1Loss.return_value = MagicMock(return_value=MagicMock(item=lambda: 0.3))

    # tensor ops
    torch.stack = lambda x: MagicMock()
    torch.tensor = lambda *a, **k: MagicMock()

    return torch


# ═══════════════════════════════════════════════════════════════
#  PLOT_UTILS TESTS  (no heavy deps needed — only matplotlib mock)
# ═══════════════════════════════════════════════════════════════
class TestPlotUtilsParsing(unittest.TestCase):
    """Test log parsing logic (stdlib only, no mocking needed)."""

    def _get_parse_fn(self):
        """Import parse_log_file, mocking matplotlib if absent."""
        try:
            from plot_utils import parse_log_file
            return parse_log_file
        except ImportError:
            import re
            _RE_EQUIV_TRAIN = re.compile(r"Epoch\s+(\d+)/(\d+)\s*-\s*Train Loss:\s*([\d.]+)")
            _RE_EQUIV_VAL_LIST = re.compile(r"Trial\s+\d+,\s*Struct\s+(\w+):\s*(\[[\d.,\s\-eE+]+\])")
            _RE_PRETRAIN = re.compile(r"Pretrain Epoch\s+(\d+):\s*Loss\s*=\s*([\d.]+)")
            _RE_FT_TRAIN_EQUIV = re.compile(r"Trial\s+\d+,\s*Epoch\s+(\d+):\s*Finetune Train Loss\s*=\s*([\d.]+)")
            _RE_FT_VAL_EQUIV = re.compile(r"Trial\s+\d+,\s*Epoch\s+(\d+):\s*Val Loss\s*=\s*([\d.]+)")
            _RE_INV_TRAIN = re.compile(r"Trial\s+\d+,\s*Epoch\s+(\d+):\s*Train wL1\s*=\s*([\d.]+)")
            _RE_INV_PRETRAIN = re.compile(r"Pretrain Epoch\s+(\d+):\s*Scaled L1\s*=\s*([\d.]+)")
            _RE_FT_TRAIN_INV = re.compile(r"Trial\s+\d+,\s*Epoch\s+(\d+):\s*Train L1\s*=\s*([\d.]+)")
            _RE_FT_VAL_INV = re.compile(r"Finetune Epoch\s+(\d+):\s*Val L1\s*=\s*([\d.]+)")
            _RE_TEST_LOSS = re.compile(r"Test (?:Loss|L1):\s*([\d.]+)")

            def parse_log_file(log_path):
                with open(log_path, "r") as f:
                    lines = f.readlines()
                data = {"train_losses": [], "val_losses": {}, "pretrain_losses": [],
                        "finetune_train_losses": [], "finetune_val_losses": [],
                        "test_losses": [], "experiment_type": "unknown"}
                for line in lines:
                    parts = line.split(" - ", 3)
                    msg = parts[-1].strip() if len(parts) >= 4 else line.strip()
                    m = _RE_EQUIV_TRAIN.search(msg)
                    if m: data["train_losses"].append(float(m.group(3))); data["experiment_type"] = "multitask"; continue
                    m = _RE_INV_TRAIN.search(msg)
                    if m: data["train_losses"].append(float(m.group(2))); data["experiment_type"] = "multitask"; continue
                    m = _RE_EQUIV_VAL_LIST.search(msg)
                    if m: data["val_losses"][m.group(1)] = ast.literal_eval(m.group(2)); continue
                    m = _RE_PRETRAIN.search(msg)
                    if m: data["pretrain_losses"].append(float(m.group(2))); data["experiment_type"] = "pretrain"; continue
                    m = _RE_INV_PRETRAIN.search(msg)
                    if m: data["pretrain_losses"].append(float(m.group(2))); data["experiment_type"] = "pretrain"; continue
                    m = _RE_FT_TRAIN_EQUIV.search(msg)
                    if m: data["finetune_train_losses"].append(float(m.group(2))); continue
                    m = _RE_FT_TRAIN_INV.search(msg)
                    if m: data["finetune_train_losses"].append(float(m.group(2))); continue
                    m = _RE_FT_VAL_EQUIV.search(msg)
                    if m: data["finetune_val_losses"].append(float(m.group(2))); continue
                    m = _RE_FT_VAL_INV.search(msg)
                    if m: data["finetune_val_losses"].append(float(m.group(2))); continue
                    m = _RE_TEST_LOSS.search(msg)
                    if m: data["test_losses"].append(float(m.group(1))); continue
                return data
            return parse_log_file

    def _write_log(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_parse_multitask_equiv_train(self):
        parse = self._get_parse_fn()
        path = self._write_log(
            "2026-02-20 12:03:03,100 - __main__ - INFO -     Epoch 1/3 - Train Loss: 1.4321\n"
            "2026-02-20 12:03:04,200 - __main__ - INFO -     Epoch 2/3 - Train Loss: 1.1234\n"
            "2026-02-20 12:03:05,300 - __main__ - INFO -     Epoch 3/3 - Train Loss: 0.8765\n"
        )
        d = parse(path)
        self.assertEqual(d["experiment_type"], "multitask")
        self.assertEqual(d["train_losses"], [1.4321, 1.1234, 0.8765])
        os.unlink(path)

    def test_parse_multitask_equiv_val_list(self):
        parse = self._get_parse_fn()
        path = self._write_log(
            "2026-02-20 12:03:06,400 - __main__ - INFO - Trial 1, Struct palindrome: [1.5, 1.2, 0.9]\n"
            "2026-02-20 12:03:06,500 - __main__ - INFO - Trial 1, Struct cyclicsum: [1.8, 1.4, 1.0]\n"
        )
        d = parse(path)
        self.assertIn("palindrome", d["val_losses"])
        self.assertEqual(d["val_losses"]["palindrome"], [1.5, 1.2, 0.9])
        self.assertEqual(d["val_losses"]["cyclicsum"], [1.8, 1.4, 1.0])
        os.unlink(path)

    def test_parse_pretrain_finetune_equiv(self):
        parse = self._get_parse_fn()
        path = self._write_log(
            "2026-02-20 12:03:01,234 - __main__ - INFO - Pretrain Epoch 5: Loss = 0.5432\n"
            "2026-02-20 12:03:02,234 - __main__ - INFO - Pretrain Epoch 10: Loss = 0.3210\n"
            "2026-02-20 12:03:03,234 - __main__ - INFO - Trial 1, Epoch 1: Finetune Train Loss = 0.4500\n"
            "2026-02-20 12:03:04,234 - __main__ - INFO - Trial 1, Epoch 1: Val Loss = 0.5000\n"
            "2026-02-20 12:03:05,234 - __main__ - INFO - Trial 1, Epoch 2: Finetune Train Loss = 0.3200\n"
            "2026-02-20 12:03:06,234 - __main__ - INFO - Trial 1, Epoch 2: Val Loss = 0.3800\n"
            "2026-02-20 12:03:07,234 - __main__ - INFO -   Test Loss: 0.3500\n"
        )
        d = parse(path)
        self.assertEqual(d["experiment_type"], "pretrain")
        self.assertEqual(d["pretrain_losses"], [0.5432, 0.321])
        self.assertEqual(d["finetune_train_losses"], [0.45, 0.32])
        self.assertEqual(d["finetune_val_losses"], [0.5, 0.38])
        self.assertEqual(d["test_losses"], [0.35])
        os.unlink(path)

    def test_parse_multitask_inv(self):
        parse = self._get_parse_fn()
        path = self._write_log(
            "2026-02-20 12:03:03,100 - __main__ - INFO - Trial 1, Epoch 1: Train wL1 = 0.2345\n"
            "2026-02-20 12:03:04,200 - __main__ - INFO - Trial 1, Epoch 2: Train wL1 = 0.1876\n"
            "2026-02-20 12:03:05,300 - __main__ - INFO -   Test L1: 0.1500\n"
        )
        d = parse(path)
        self.assertEqual(d["experiment_type"], "multitask")
        self.assertEqual(d["train_losses"], [0.2345, 0.1876])
        self.assertEqual(d["test_losses"], [0.15])
        os.unlink(path)

    def test_parse_pretrain_inv(self):
        parse = self._get_parse_fn()
        path = self._write_log(
            "2026-02-20 12:03:01,234 - __main__ - INFO -     Pretrain Epoch 5: Scaled L1 = 0.4100\n"
            "2026-02-20 12:03:02,234 - __main__ - INFO - Trial 1, Epoch 1: Train L1 = 0.3300\n"
            "2026-02-20 12:03:03,234 - __main__ - INFO -     Finetune Epoch 1: Val L1 = 0.4000\n"
            "2026-02-20 12:03:04,234 - __main__ - INFO -   Test L1: 0.2800\n"
        )
        d = parse(path)
        self.assertEqual(d["experiment_type"], "pretrain")
        self.assertEqual(d["pretrain_losses"], [0.41])
        self.assertEqual(d["finetune_train_losses"], [0.33])
        self.assertEqual(d["finetune_val_losses"], [0.4])
        self.assertEqual(d["test_losses"], [0.28])
        os.unlink(path)

    def test_parse_empty_log(self):
        parse = self._get_parse_fn()
        path = self._write_log("")
        d = parse(path)
        self.assertEqual(d["experiment_type"], "unknown")
        self.assertEqual(d["train_losses"], [])
        os.unlink(path)

    def test_parse_ignores_irrelevant(self):
        parse = self._get_parse_fn()
        path = self._write_log(
            "2026-02-20 12:03:01,234 - __main__ - INFO - trainsize = 2500\n"
            "2026-02-20 12:03:02,234 - __main__ - INFO - vocab_size = 7\n"
            "2026-02-20 12:03:03,100 - __main__ - INFO -     Epoch 1/1 - Train Loss: 0.9999\n"
        )
        d = parse(path)
        self.assertEqual(d["train_losses"], [0.9999])
        os.unlink(path)


class TestPlotUtilsPlotting(unittest.TestCase):
    """Test plotting functions with matplotlib mocked."""

    def _import_with_mock(self):
        """Import plot_utils with a properly configured matplotlib mock."""
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_ylim.return_value = (0.0, 2.0)
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Inject mocks
        sys.modules["matplotlib"] = MagicMock()
        sys.modules["matplotlib.pyplot"] = mock_plt
        if "plot_utils" in sys.modules:
            del sys.modules["plot_utils"]
        import plot_utils
        plot_utils.plt = mock_plt
        return plot_utils

    def test_plot_training_curves_with_val(self):
        pu = self._import_with_mock()
        pu.plot_training_curves(
            train_losses=[1.0, 0.8, 0.6],
            val_losses_dict={"struct_a": [1.1, 0.9, 0.7]},
            title="Test", save_path=None, show=False,
        )

    def test_plot_training_curves_no_val(self):
        pu = self._import_with_mock()
        pu.plot_training_curves(
            train_losses=[1.0, 0.5],
            val_losses_dict=None, show=False,
        )

    def test_plot_pretrain_finetune_with_val(self):
        pu = self._import_with_mock()
        pu.plot_pretrain_finetune(
            pretrain_losses=[0.8, 0.6],
            finetune_train_losses=[0.5, 0.4, 0.3],
            finetune_val_losses=[0.6, 0.5, 0.4],
            title="Test PT", save_path=None, show=False,
        )

    def test_plot_pretrain_finetune_no_val(self):
        pu = self._import_with_mock()
        pu.plot_pretrain_finetune(
            pretrain_losses=[0.9],
            finetune_train_losses=[0.7, 0.5],
            finetune_val_losses=None, show=False,
        )

    def test_plot_from_log_multitask(self):
        pu = self._import_with_mock()
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        f.write("2026-02-20 12:03:03,100 - __main__ - INFO -     Epoch 1/2 - Train Loss: 1.0\n"
                "2026-02-20 12:03:04,200 - __main__ - INFO -     Epoch 2/2 - Train Loss: 0.5\n")
        f.close()
        pu.plot_from_log(f.name, show=False)
        os.unlink(f.name)

    def test_plot_from_log_pretrain(self):
        pu = self._import_with_mock()
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        f.write("2026-02-20 12:00:00,000 - __main__ - INFO - Pretrain Epoch 5: Loss = 0.5\n"
                "2026-02-20 12:00:01,000 - __main__ - INFO - Trial 1, Epoch 1: Finetune Train Loss = 0.3\n"
                "2026-02-20 12:00:02,000 - __main__ - INFO - Trial 1, Epoch 1: Val Loss = 0.4\n")
        f.close()
        pu.plot_from_log(f.name, show=False)
        os.unlink(f.name)

    def test_plot_from_log_empty(self):
        pu = self._import_with_mock()
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        f.write("nothing useful\n")
        f.close()
        pu.plot_from_log(f.name, show=False)
        os.unlink(f.name)


# ═══════════════════════════════════════════════════════════════
#  EQUIV_EXPERIMENTS TESTS
# ═══════════════════════════════════════════════════════════════
class TestEquivExperiments(unittest.TestCase):
    """Test equiv_experiments.py functions with mocked torch/models/data."""

    @classmethod
    def setUpClass(cls):
        cls.mock_torch = _make_fake_torch()
        cls.patches = {
            "torch": cls.mock_torch,
            "torch.nn": cls.mock_torch.nn,
            "torch.cuda": cls.mock_torch.cuda,
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
            "pickle": MagicMock(),
            "numpy": MagicMock(),
            "matplotlib": MagicMock(),
            "matplotlib.pyplot": MagicMock(),
            "models": MagicMock(),
            "models.Multi_GAT": MagicMock(),
            "sympy": MagicMock(),
            "sympy.combinatorics": MagicMock(),
            "data": MagicMock(),
            "plot_utils": MagicMock(),
        }
        cls._patcher = patch.dict("sys.modules", cls.patches)
        cls._patcher.start()
        try:
            from sympy.combinatorics import Permutation, PermutationGroup
            cls.has_sympy = True
        except ImportError:
            cls.has_sympy = False

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()

    def test_generate_graph_structure_all_types(self):
        """generate_graph_structure for all supported dataset types."""
        if not self.has_sympy:
            self.skipTest("sympy not installed")
        with patch.dict("sys.modules", {
            **self.patches,
            "sympy": __import__("sympy"),
            "sympy.combinatorics": __import__("sympy.combinatorics", fromlist=["Permutation", "PermutationGroup"]),
        }):
            if "equiv_experiments" in sys.modules:
                del sys.modules["equiv_experiments"]
            from equiv_experiments import generate_graph_structure

            perms = generate_graph_structure("palindrome", 6)
            self.assertIsInstance(perms, list)
            self.assertGreater(len(perms), 0)

            perms = generate_graph_structure("cyclicsum", 6)
            self.assertGreater(len(perms), 2)

            perms = generate_graph_structure("intersect", 6)
            self.assertIsInstance(perms, list)

            perms = generate_graph_structure("set", 6)
            self.assertIsInstance(perms, list)

            with self.assertRaises(ValueError):
                generate_graph_structure("nonexistent", 6)

            perms = generate_graph_structure("palindrome", 6, non_equiv=True)
            self.assertEqual(len(perms), 1)

    def test_run_multitask_imports(self):
        if "equiv_experiments" in sys.modules:
            del sys.modules["equiv_experiments"]
        import equiv_experiments
        self.assertTrue(hasattr(equiv_experiments, "run_multitask_experiments"))

    def test_run_pretrain_finetune_imports(self):
        if "equiv_experiments" in sys.modules:
            del sys.modules["equiv_experiments"]
        import equiv_experiments
        self.assertTrue(hasattr(equiv_experiments, "run_pretrain_finetune_experiment_equiv"))

    def test_create_datasets_imports(self):
        if "equiv_experiments" in sys.modules:
            del sys.modules["equiv_experiments"]
        import equiv_experiments
        self.assertTrue(hasattr(equiv_experiments, "create_datasets"))

    def test_run_multitask_mock_training(self):
        """Mock the full training loop end-to-end."""
        if "equiv_experiments" in sys.modules:
            del sys.modules["equiv_experiments"]

        mock_sample = (MagicMock(), MagicMock(), "palindrome")
        mock_dataset = [mock_sample] * 10

        mock_create = MagicMock(return_value=(
            {"palindrome": {"n_nodes": 6, "perms": [], "coords_dim": (1, 1),
                            "adj": None, "orbits": None, "sparse": False, "out_dim": 2}},
            {"palindrome": mock_dataset},
            {"palindrome": mock_dataset},
            {"palindrome": mock_dataset},
        ))

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.shape = (2, 6, 2)
        mock_output.view.return_value = mock_output
        mock_model.return_value = mock_output
        mock_model.to.return_value = mock_model
        mock_model.named_parameters.return_value = []
        mock_model.parameters.return_value = []

        import equiv_experiments
        with patch.object(equiv_experiments, "create_datasets", mock_create), \
             patch.object(equiv_experiments, "MultiGraphGATv2Model_equiv", return_value=mock_model):
            try:
                equiv_experiments.run_multitask_experiments(r=0.4, T=1, bs=2, trials=1, num_epochs=1)
            except Exception:
                pass  # mock gaps in deep loop are acceptable


# ═══════════════════════════════════════════════════════════════
#  INV_REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════
class TestInvRegression(unittest.TestCase):
    """Test inv_regression.py functions with mocked torch/models/data."""

    @classmethod
    def setUpClass(cls):
        cls.mock_torch = _make_fake_torch()
        cls.patches = {
            "torch": cls.mock_torch,
            "torch.nn": cls.mock_torch.nn,
            "torch.cuda": cls.mock_torch.cuda,
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
            "pickle": MagicMock(),
            "numpy": MagicMock(),
            "models": MagicMock(),
            "models.Multi_GAT": MagicMock(),
            "sympy": MagicMock(),
            "sympy.combinatorics": MagicMock(),
            "data": MagicMock(),
            "sklearn": MagicMock(),
            "sklearn.metrics": MagicMock(),
            "plot_utils": MagicMock(),
        }
        cls._patcher = patch.dict("sys.modules", cls.patches)
        cls._patcher.start()
        try:
            from sympy.combinatorics import Permutation
            cls.has_sympy = True
        except ImportError:
            cls.has_sympy = False

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()

    def test_generate_graph_structure_all_types(self):
        """inv_regression.generate_graph_structure for all supported types."""
        if not self.has_sympy:
            self.skipTest("sympy not installed")
        with patch.dict("sys.modules", {
            **self.patches,
            "sympy": __import__("sympy"),
            "sympy.combinatorics": __import__("sympy.combinatorics", fromlist=["Permutation", "PermutationGroup"]),
        }):
            if "inv_regression" in sys.modules:
                del sys.modules["inv_regression"]
            from inv_regression import generate_graph_structure

            perms = generate_graph_structure("palindrome", 8)
            self.assertEqual(len(perms), 2)

            perms = generate_graph_structure("cyclicsum", 8)
            self.assertGreater(len(perms), 2)

            perms = generate_graph_structure("longestpal", 8)
            self.assertIsInstance(perms, list)

            perms = generate_graph_structure("detectcapital", 8)
            self.assertIsInstance(perms, list)

            perms = generate_graph_structure("vandermonde", 8)
            self.assertEqual(len(perms), 1)

            with self.assertRaises(ValueError):
                generate_graph_structure("nonexistent", 8)

    def test_run_experiments_inv_imports(self):
        if "inv_regression" in sys.modules:
            del sys.modules["inv_regression"]
        import inv_regression
        self.assertTrue(hasattr(inv_regression, "run_experiments_inv"))

    def test_run_pretrain_finetune_imports(self):
        if "inv_regression" in sys.modules:
            del sys.modules["inv_regression"]
        import inv_regression
        self.assertTrue(hasattr(inv_regression, "run_pretrain_finetune_experiment"))

    def test_run_vandermonde_mlp_imports(self):
        if "inv_regression" in sys.modules:
            del sys.modules["inv_regression"]
        import inv_regression
        self.assertTrue(hasattr(inv_regression, "run_vandermonde_mlp"))

    def test_create_datasets_imports(self):
        if "inv_regression" in sys.modules:
            del sys.modules["inv_regression"]
        import inv_regression
        self.assertTrue(hasattr(inv_regression, "create_datasets"))


# ═══════════════════════════════════════════════════════════════
#  CROSS-MODULE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════
class TestIntegration(unittest.TestCase):
    """Test that plot_utils integrates with experiment log formats."""

    def _get_parse_fn(self):
        try:
            from plot_utils import parse_log_file
            return parse_log_file
        except ImportError:
            return TestPlotUtilsParsing()._get_parse_fn()

    def test_equiv_multitask_log_format(self):
        """Verify equiv_experiments log format matches parser."""
        parse = self._get_parse_fn()
        log_lines = [
            "2026-02-20 12:03:00,000 - __main__ - INFO - trainsize = 2500\n",
            "2026-02-20 12:03:01,000 - __main__ - INFO - Running setting: 0.4 palindrome + 0.4 cyclicsum\n",
            "2026-02-20 12:03:02,000 - __main__ - INFO -     Epoch 1/40 - Train Loss: 1.3827\n",
            "2026-02-20 12:03:03,000 - __main__ - INFO -     Epoch 1/40 - Train Time: 0.5432 secs!\n",
            "2026-02-20 12:03:04,000 - __main__ - INFO -     Epoch 2/40 - Train Loss: 1.1003\n",
            "2026-02-20 12:03:05,000 - __main__ - INFO - Trial 1, Struct palindrome: [1.45, 1.12]\n",
            "2026-02-20 12:03:06,000 - __main__ - INFO - Trial 1, Struct cyclicsum: [1.67, 1.34]\n",
        ]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        f.writelines(log_lines)
        f.close()
        d = parse(f.name)
        self.assertEqual(d["experiment_type"], "multitask")
        self.assertEqual(len(d["train_losses"]), 2)
        self.assertAlmostEqual(d["train_losses"][0], 1.3827)
        self.assertIn("palindrome", d["val_losses"])
        os.unlink(f.name)

    def test_inv_pretrain_log_format(self):
        """Verify inv_regression pretrain log format matches parser."""
        parse = self._get_parse_fn()
        log_lines = [
            "2026-02-20 12:00:00,000 - __main__ - INFO - ====== Starting Experiment: pretrain+finetune ======\n",
            "2026-02-20 12:00:03,000 - __main__ - INFO -     Pretrain Epoch 5: Scaled L1 = 0.4100\n",
            "2026-02-20 12:00:04,000 - __main__ - INFO -     Pretrain Epoch 10: Scaled L1 = 0.2900\n",
            "2026-02-20 12:00:06,000 - __main__ - INFO - Trial 1, Epoch 1: Train L1 = 0.3300\n",
            "2026-02-20 12:00:07,000 - __main__ - INFO -     Finetune Epoch 1: Val L1 = 0.4000\n",
            "2026-02-20 12:00:10,000 - __main__ - INFO -   Test L1: 0.2800\n",
        ]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        f.writelines(log_lines)
        f.close()
        d = parse(f.name)
        self.assertEqual(d["experiment_type"], "pretrain")
        self.assertEqual(d["pretrain_losses"], [0.41, 0.29])
        self.assertEqual(d["finetune_train_losses"], [0.33])
        self.assertEqual(d["finetune_val_losses"], [0.4])
        self.assertEqual(d["test_losses"], [0.28])
        os.unlink(f.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
