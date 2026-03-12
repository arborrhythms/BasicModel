"""BasicModel test suite — exercises all core modules.

Covers:
  - Model.py   layer tests (LinearLayer, Reversible*, PiLayer, SigmaLayer, etc.)
  - SPNN.py    classical neural network
  - SigmaPi.py product-sum network
  - SymPercept.py  bidirectional linear learning
  - Ergodic.py ergodic model construction
  - BasicModel.py  full model creation, forward/reverse pass, weight persistence
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

# Ensure bin/ is importable
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


# ---------------------------------------------------------------------------
# Model.py — Layer tests
# ---------------------------------------------------------------------------
class TestLinearLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import LinearLayer
        layer = LinearLayer(nInput=4, nOutput=3)
        x = torch.randn(2, 4)
        y = layer(x)
        self.assertEqual(y.shape, (2, 3))

    def test_with_identity_weight(self):
        from Model import LinearLayer
        W = torch.eye(4)
        layer = LinearLayer(nInput=4, nOutput=4, W=W)
        x = torch.randn(1, 4)
        y = layer(x)
        self.assertEqual(y.shape, (1, 4))


class TestReversibleRotationLayer(unittest.TestCase):
    def test_forward_reverse_identity(self):
        from Model import ReversibleRotationLayer
        dim = 4
        layer = ReversibleRotationLayer(dim)
        x = torch.randn(2, dim)
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4),
                        f"Rotation reverse error: {(x - x_rec).abs().max():.6f}")


class TestReversibleDiagonalLayer(unittest.TestCase):
    def test_forward_reverse_identity(self):
        from Model import ReversibleDiagonalLayer
        layer = ReversibleDiagonalLayer(nInput=4, nOutput=4)
        x = torch.randn(2, 4)
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4),
                        f"Diagonal reverse error: {(x - x_rec).abs().max():.6f}")


class TestReversibleLinearLayer(unittest.TestCase):
    def test_forward_reverse_square(self):
        from Model import ReversibleLinearLayer
        layer = ReversibleLinearLayer(nInput=4, nOutput=4)
        x = torch.randn(2, 4)
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-3),
                        f"Linear reverse error: {(x - x_rec).abs().max():.6f}")


class TestSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import SigmaLayer
        layer = SigmaLayer(nInput=8, nOutput=4)
        x = torch.randn(2, 8)
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))


class TestPiLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import PiLayer
        layer = PiLayer(nInput=6, nOutput=3)
        x = torch.randn(2, 6)
        y = layer(x)
        self.assertEqual(y.shape, (2, 3))


class TestReversibleSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import ReversibleSigmaLayer
        layer = ReversibleSigmaLayer(nInput=4, nOutput=4)
        x = torch.randn(2, 4) * 0.3
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))

    def test_reverse_shape(self):
        from Model import ReversibleSigmaLayer
        layer = ReversibleSigmaLayer(nInput=4, nOutput=4)
        y = torch.randn(2, 4) * 0.3
        x = layer.reverse(y)
        self.assertEqual(x.shape, (2, 4))


class TestAttentionLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import AttentionLayer
        layer = AttentionLayer(nInput=8, nOutput=4)
        x = torch.randn(2, 8)
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))


class TestNormLayer(unittest.TestCase):
    def test_forward_runs(self):
        from Model import NormLayer
        layer = NormLayer(4, 4)
        x = torch.randn(3, 4)
        y = layer(x)
        # NormLayer may append extra features (mean, std)
        self.assertEqual(y.shape[0], 3)
        self.assertGreaterEqual(y.shape[1], 4)


class TestMemory(unittest.TestCase):
    def test_mem_update(self):
        from Model import Mem
        Mem.test()  # Runs the built-in test


# ---------------------------------------------------------------------------
# SPNN.py — Classical neural network
# ---------------------------------------------------------------------------
class TestSPNN(unittest.TestCase):
    def test_xor_creation(self):
        from SPNN import SPNN
        net = SPNN("sigmoid", False)
        self.assertIsNotNone(net)
        self.assertTrue(hasattr(net, 'W1'))
        self.assertTrue(hasattr(net, 'W2'))

    def test_xor_training(self):
        from SPNN import SPNN
        net = SPNN("tanh", False)
        net.loadXOR()
        # Run a few epochs — just check it doesn't crash
        for _ in range(3):
            net.run()


# ---------------------------------------------------------------------------
# SigmaPi.py — Product-sum network
# ---------------------------------------------------------------------------
class TestSigmaPi(unittest.TestCase):
    def test_logical_function_net_creation(self):
        from SigmaPi import LogicalFunctionNet
        net = LogicalFunctionNet(nInput=2, nHidden=4, nOutput=1)
        self.assertIsNotNone(net)
        self.assertTrue(hasattr(net, 'hidden'))
        self.assertTrue(hasattr(net, 'output'))


# ---------------------------------------------------------------------------
# SymPercept.py — Bidirectional linear learning
# ---------------------------------------------------------------------------
class TestSymPercept(unittest.TestCase):
    def test_bidirectional_creation(self):
        from SymPercept import BidirectionalLinearNumpy
        model = BidirectionalLinearNumpy()
        self.assertIsNotNone(model)
        self.assertEqual(model.dim, 2)

    def test_forward_reverse(self):
        from SymPercept import BidirectionalLinearNumpy
        model = BidirectionalLinearNumpy()
        x = np.random.randn(1, 2)
        y = model.forward(x)
        x_rec = model.inverse(y)
        err = np.abs(x - x_rec).max()
        self.assertLess(err, 1e-4,
                        f"SymPercept reverse error: {err:.6f}")


# ---------------------------------------------------------------------------
# DerivedModel construction (now in BasicModel.py)
# ---------------------------------------------------------------------------
class TestDerivedModel(unittest.TestCase):
    def test_derived_model_creation(self):
        from BasicModel import DerivedModel
        model = DerivedModel()
        self.assertIsNotNone(model)

    def test_derived_model_traditional(self):
        """DerivedModel with ergodic=False, certainty=False produces valid output."""
        from BasicModel import DerivedModel
        model = DerivedModel()
        model.ergodic   = False
        model.certainty = False
        model.quantized = False
        model.create(nInput=28*28, nConcepts=20, nOutput=10)
        x = torch.randn(2, 28*28, 1)  # batch of 2, flattened MNIST, dim=1
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_derived_model_ergodic(self):
        """DerivedModel with ergodic=True uses SigmaLayer path."""
        from BasicModel import DerivedModel
        model = DerivedModel()
        model.ergodic   = True
        model.certainty = True
        model.quantized = False
        model.create(nInput=28*28, nConcepts=20, nOutput=10)
        x = torch.randn(2, 28*28, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)


# ---------------------------------------------------------------------------
# BasicModel.py — Full model
# ---------------------------------------------------------------------------
class TestBasicModelCreation(unittest.TestCase):
    def test_encodings(self):
        from BasicModel import PositionalEncoding, TemporalEncoding
        PositionalEncoding.test()
        TemporalEncoding.test()

    def test_config_loading(self):
        from BasicModel import BasicModel
        cfg = BasicModel.load_config()
        # model.xml should exist and parse
        self.assertIsInstance(cfg, dict)
        if cfg:
            self.assertIn("architecture", cfg)


class TestWeightPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        from Model import LinearLayer
        layer = LinearLayer(nInput=4, nOutput=3)
        x = torch.randn(1, 4)
        y_before = layer(x).detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(layer.state_dict(), path)
            # Create a fresh layer and load
            layer2 = LinearLayer(nInput=4, nOutput=3)
            layer2.load_state_dict(torch.load(path, weights_only=True))
            y_after = layer2(x).detach()
            self.assertTrue(torch.allclose(y_before, y_after, atol=1e-6),
                            "Weight save/load should preserve outputs")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
