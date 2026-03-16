"""BasicModel test suite — exercises all core modules.

Covers:
  - Model.py   layer tests (LinearLayer, Invertible*, PiLayer, SigmaLayer, etc.)
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

# Prevent OMP fork-safety crash on macOS when multiple libs load OpenMP
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

# Ensure bin/ is importable
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from BasicModel import TheDevice


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


class TestInvertibleRotationLayer(unittest.TestCase):
    def test_forward_reverse_identity(self):
        from Model import InvertibleRotationLayer
        dim = 4
        layer = InvertibleRotationLayer(dim)
        x = torch.randn(2, dim)
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4),
                        f"Rotation reverse error: {(x - x_rec).abs().max():.6f}")


class TestInvertibleDiagonalLayer(unittest.TestCase):
    def test_forward_reverse_identity(self):
        from Model import InvertibleDiagonalLayer
        layer = InvertibleDiagonalLayer(nInput=4, nOutput=4)
        x = torch.randn(2, 4)
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4),
                        f"Diagonal reverse error: {(x - x_rec).abs().max():.6f}")


class TestInvertibleLinearLayer(unittest.TestCase):
    def test_forward_reverse_square(self):
        from Model import InvertibleLinearLayer
        layer = InvertibleLinearLayer(nInput=4, nOutput=4)
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


class TestInvertibleSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import InvertibleSigmaLayer
        layer = InvertibleSigmaLayer(nInput=4, nOutput=4)
        x = torch.randn(2, 4) * 0.3
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))

    def test_reverse_shape(self):
        from Model import InvertibleSigmaLayer
        layer = InvertibleSigmaLayer(nInput=4, nOutput=4)
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
# Simple path: BasicModel with conceptualOrder=1, symbolicOrder=1
# ---------------------------------------------------------------------------
def _make_simple_model(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nWords=16, nOutput=4):
    """Helper to create a BasicModel with ObjectEncoding set up for simple path.

    Sets both dims and codebook sizes (n* values) on TheObjectEncoding.
    Caller can override counts if needed.
    """
    from BasicModel import BasicModel, TheObjectEncoding
    TheObjectEncoding.nWhere = 0
    TheObjectEncoding.nWhen = 0
    TheObjectEncoding.objectSize = 0
    TheObjectEncoding.setInputDim(1)
    TheObjectEncoding.setPerceptDim(1)
    TheObjectEncoding.setConceptDim(1)
    TheObjectEncoding.setSymbolDim(0)
    TheObjectEncoding.setWordDim(1)
    TheObjectEncoding.setOutputDim(1)
    TheObjectEncoding.nInput = nInput
    TheObjectEncoding.nPercepts = nPercepts
    TheObjectEncoding.nConcepts = nConcepts
    TheObjectEncoding.nSymbols = nSymbols
    TheObjectEncoding.nWords = nWords
    TheObjectEncoding.nOutput = nOutput
    TheObjectEncoding.nObjects = 0  # reset for test isolation
    TheObjectEncoding.computeNObjects()
    return BasicModel()


class TestSimpleModelCreation(unittest.TestCase):

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_simple_model_creation(self):
        model = _make_simple_model()
        self.assertIsNotNone(model)

    def test_simple_model_traditional(self):
        """BasicModel (simple path) with ergodic=False produces valid output."""
        model = _make_simple_model(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
                       perceptPassThrough=True, symbolPassThrough=True,
                       reshape=True)
        x = torch.randn(2, 28*28, 1)  # batch of 2, flattened MNIST, dim=1
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_simple_model_ergodic(self):
        """BasicModel (simple path) with ergodic=True uses SigmaLayer path."""
        model = _make_simple_model(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, certainty=True, reshape=True)
        x = torch.randn(2, 28*28, 1)
        _, end_state, out = model.forward(x)
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


# ---------------------------------------------------------------------------
# Regression: Space shape contracts
# ---------------------------------------------------------------------------
class TestCanonicalSpaceShapes(unittest.TestCase):
    """Lock down tensor shapes for canonical Space subclasses."""

    def setUp(self):
        from BasicModel import (TheObjectEncoding, InputSpace,
                                ConceptualSpace, OutputSpace)
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4)
        TheObjectEncoding.setWordDim(1)
        TheObjectEncoding.nInput = 4
        TheObjectEncoding.nPercepts = 4
        TheObjectEncoding.nConcepts = 4
        TheObjectEncoding.nSymbols = 4
        TheObjectEncoding.nWords = 4
        TheObjectEncoding.nOutput = 4
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        self.B = 2  # batch

    def test_conceptual_space_forward_shape(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace(nIn, nOut)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.inputDim)
        x = torch.randn(self.B, nIn, inEmb)
        y = cs(x)
        outEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
        self.assertEqual(list(y.shape), [self.B, nOut, outEmb])

    def test_conceptual_space_reverse_shape(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace(nIn, nOut, reversible=True)
        outEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
        y = torch.randn(self.B, nOut, outEmb)
        x = cs.reverse(y)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.inputDim)
        self.assertEqual(list(x.shape), [self.B, nIn, inEmb])

    def test_output_space_forward_shape(self):
        from BasicModel import OutputSpace, TheObjectEncoding
        nIn, nOut = 4, 4
        os_ = OutputSpace(nIn, nOut)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.symbolDim)
        x = torch.randn(self.B, nIn, inEmb)
        y = os_(x)
        self.assertEqual(list(y.shape), [self.B, nOut, TheObjectEncoding.outputDim])


class TestSimpleModel(unittest.TestCase):
    """BasicModel (simple path) uses unified Space hierarchy with passThrough SymbolicSpace."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_simple_model_ergodic_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_simple_model_reverse_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, reversible=True, reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)


class TestVectorSetVariants(unittest.TestCase):
    """Lock down quantized vs unquantized VectorSet behavior."""

    def test_unquantized_passthrough(self):
        from BasicModel import VectorSet
        vs = VectorSet()
        vs.create(4, 4, 1, passThrough=True)
        x = torch.randn(2, 4, 1)
        y = vs.forward(x)
        self.assertTrue(torch.equal(x, y))

    def test_quantized_shape(self):
        from BasicModel import VectorSet, TheObjectEncoding
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.nInput = 4
        TheObjectEncoding.nPercepts = 4
        TheObjectEncoding.nConcepts = 4
        TheObjectEncoding.nSymbols = 4
        TheObjectEncoding.nWords = 0
        TheObjectEncoding.nOutput = 4
        TheObjectEncoding.computeNObjects()
        vs = VectorSet()
        vs.create(4, 4, 3, customVQ=False)
        vs.addVectors(nVec=4)
        x = torch.randn(2, 4, 3)
        y = vs.forward(x)
        # Output gains ObjectEncoding overhead (nWhere + nWhen)
        embeddingSize = 3 + TheObjectEncoding.objectSize
        self.assertEqual(list(y.shape), [2, 4, embeddingSize])


class TestModelEndToEnd(unittest.TestCase):
    """Lock down full model forward shapes and loss compatibility."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_simple_model_ergodic_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_reverse_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, reversible=True, reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_simple_model_loss_runs(self):
        """Verify forward + loss + backward doesn't crash."""
        from BasicModel import CertaintyWeightedCrossEntropy
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, certainty=True, reshape=True)
        x = torch.randn(2, 16, 1)
        target = torch.randn(2, 4)
        _, end_state, out = model.forward(x)
        loss_fn = CertaintyWeightedCrossEntropy()
        loss = loss_fn(out.squeeze(), target)
        loss.backward()
        # No crash = pass


class TestUniversalTrainingContract(unittest.TestCase):
    """All spaces expose getParameters() and paramUpdate()."""

    def test_space_has_training_contract(self):
        from BasicModel import Space
        s = Space([4, 8], [4, 8], 4)
        self.assertEqual(s.getParameters(), [])
        s.paramUpdate()  # should be a no-op, not crash

    def test_conceptual_space_has_training_contract(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4)
        TheObjectEncoding.nConcepts = 4
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        cs = ConceptualSpace(4, 4)
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        cs.paramUpdate()  # no crash


class TestSigmaLayerDeterministic(unittest.TestCase):
    """SigmaLayer(ergodic=False) behaves like LinearLayer + Tanh."""

    def test_deterministic_matches_linear_tanh(self):
        from Model import SigmaLayer, LinearLayer
        torch.manual_seed(42)
        nIn, nOut = 8, 4
        sigma = SigmaLayer(nIn, nOut, ergodic=False)
        sigma.train()

        # Build a matching LinearLayer + Tanh with same weights
        linear = LinearLayer(nIn, nOut, hasBias=True)
        with torch.no_grad():
            linear.W.copy_(sigma.layer.W)
            linear.bias.copy_(sigma.layer.bias)
        tanh = torch.nn.Tanh()

        x = torch.randn(2, nIn)
        y_sigma = sigma(x)
        y_manual = tanh(linear(x))
        self.assertTrue(torch.allclose(y_sigma, y_manual, atol=1e-6),
                        f"Deterministic SigmaLayer should match LinearLayer+Tanh")

    def test_deterministic_same_train_eval(self):
        from Model import SigmaLayer
        nIn, nOut = 8, 4
        sigma = SigmaLayer(nIn, nOut, ergodic=False)
        x = torch.randn(2, nIn)

        sigma.train()
        y_train = sigma(x).detach().clone()
        sigma.eval()
        y_eval = sigma(x).detach().clone()
        self.assertTrue(torch.allclose(y_train, y_eval, atol=1e-6),
                        "Non-ergodic mode should produce same output in train and eval")

    def test_non_ergodic_default(self):
        from Model import SigmaLayer
        sigma = SigmaLayer(nInput=8, nOutput=4)
        self.assertFalse(sigma.ergodic)


class TestCreateVectorSetQuantized(unittest.TestCase):
    """Space.createVectorSet supports both quantized and unquantized paths."""

    def test_quantized_creates_vectorset(self):
        from BasicModel import Space, VectorSet
        s = Space([4, 3], [4, 3], 4)
        s.createVectorSet(quantized=True)
        self.assertIsInstance(s.vectors(), VectorSet)

    def test_unquantized_creates_passthrough_vset(self):
        from BasicModel import Space, VectorSet
        s = Space([4, 3], [4, 3], 4)
        s.createVectorSet(quantized=False)
        self.assertIsInstance(s.vectors(), VectorSet)
        self.assertTrue(s.vectors().passThrough)

    def test_default_is_quantized(self):
        from BasicModel import Space, VectorSet
        s = Space([4, 3], [4, 3], 4)
        s.createVectorSet()
        self.assertIsInstance(s.vectors(), VectorSet)


class TestConceptualSpaceErgodic(unittest.TestCase):
    """ConceptualSpace with ergodic flag matches DerivedConceptualSpace behavior."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def _set_zero_object_encoding(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.perceptDim = 1
        TheObjectEncoding.conceptDim = 1
        TheObjectEncoding.symbolDim = 0
        TheObjectEncoding.wordDim = 1
        TheObjectEncoding.outputDim = 1
        TheObjectEncoding.nInput = 8
        TheObjectEncoding.nPercepts = 8
        TheObjectEncoding.nConcepts = 8
        TheObjectEncoding.nSymbols = 8
        TheObjectEncoding.nWords = 8
        TheObjectEncoding.nOutput = 8
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()

    def test_ergodic_forward_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace(nVec, nVec,
                             ergodic=True, quantized=False)
        x = torch.randn(2, nVec, nDim)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_non_ergodic_forward_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace(nVec, nVec,
                             ergodic=False, quantized=False)
        x = torch.randn(2, nVec, nDim)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_ergodic_flag_stored(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        cs_erg = ConceptualSpace(8, 8,
                                 ergodic=True, quantized=False)
        cs_det = ConceptualSpace(8, 8,
                                 ergodic=False, quantized=False)
        self.assertTrue(cs_erg.ergodic)
        self.assertFalse(cs_det.ergodic)

    def test_ergodic_reverse_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace(nVec, nVec,
                             ergodic=True, reversible=True, quantized=False)
        y = torch.randn(2, nVec, cDim)
        x = cs.reverse(y)
        self.assertEqual(list(x.shape), [2, nVec, nDim])

    def test_ergodic_exposes_params(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        cs = ConceptualSpace(8, 8,
                             ergodic=True, quantized=False)
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

    def test_canonical_forward_still_works(self):
        """Existing ConceptualSpace (with objectSize > 0) still works after changes."""
        from BasicModel import ConceptualSpace, TheObjectEncoding
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4)
        TheObjectEncoding.nConcepts = 4
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        nIn, nOut = 4, 4
        cs = ConceptualSpace(nIn, nOut)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.inputDim)
        x = torch.randn(2, nIn, inEmb)
        y = cs(x)
        outEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
        self.assertEqual(list(y.shape), [2, nOut, outEmb])


class TestInputSpaceUnquantized(unittest.TestCase):
    """InputSpace works with unquantized codebook (objectSize=0)."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_unquantized_forward_shape(self):
        from BasicModel import TheObjectEncoding, InputSpace
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = 8
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        nIn, nDim = 8, 1
        inp = InputSpace(nIn, nIn, quantized=False)
        x = torch.randn(2, nIn, nDim)
        y = inp(x)
        self.assertEqual(list(y.shape), [2, nIn, nDim])


class TestOutputSpaceZeroObjectSize(unittest.TestCase):
    """OutputSpace works with objectSize=0."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_forward_shape_zero_object_size(self):
        from BasicModel import TheObjectEncoding, OutputSpace
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        TheObjectEncoding.symbolDim = 1
        TheObjectEncoding.outputDim = 1
        TheObjectEncoding.nOutput = 3
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        nIn, nOut = 4, 3
        os_ = OutputSpace(nIn, nOut)
        x = torch.randn(2, nIn, 1)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, 1])

    def test_reverse_shape_zero_object_size(self):
        from BasicModel import TheObjectEncoding, OutputSpace
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        TheObjectEncoding.symbolDim = 1
        TheObjectEncoding.outputDim = 1
        TheObjectEncoding.nOutput = 3
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        nIn, nOut = 4, 3
        os_ = OutputSpace(nIn, nOut, reversible=True)
        y = torch.randn(2, nOut, 1)
        x = os_.reverse(y)
        self.assertEqual(list(x.shape), [2, nIn, 1])


class TestBaseModelFactory(unittest.TestCase):
    """BaseModel.from_config factory creates the correct model type."""

    def test_factory_creates_simple_model(self):
        from BasicModel import BaseModel, BasicModel, TheObjectEncoding
        orig_nWhere = TheObjectEncoding.nWhere
        orig_nWhen = TheObjectEncoding.nWhen
        orig_objectSize = TheObjectEncoding.objectSize
        xml = """<model>
  <architecture>
    <nInput>16</nInput>
    <nPercepts>16</nPercepts>
    <nConcepts>8</nConcepts>
    <nSymbols>8</nSymbols>
    <nOutput>4</nOutput>
    <inputDim>1</inputDim>
    <perceptDim>1</perceptDim>
    <conceptDim>1</conceptDim>
    <symbolDim>1</symbolDim>
    <outputDim>1</outputDim>
    <perceptPassThrough>true</perceptPassThrough>
    <symbolPassThrough>true</symbolPassThrough>
  </architecture>
</model>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml)
            path = f.name
        try:
            model, cfg = BaseModel.from_config(path)
            self.assertIsInstance(model, BasicModel)
            self.assertEqual(model.conceptualOrder, 1)
            self.assertEqual(model.symbolicOrder, 1)
        finally:
            os.unlink(path)
            TheObjectEncoding.nWhere = orig_nWhere
            TheObjectEncoding.nWhen = orig_nWhen
            TheObjectEncoding.objectSize = orig_objectSize

    def test_factory_creates_basic_model(self):
        from BasicModel import BaseModel, BasicModel as BM, TheObjectEncoding
        # BasicModel.create() needs non-zero encoding dimensions
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8,
                                        conceptDim=8, symbolDim=0, outputDim=4)
        # nSymbols must equal nConcepts (SymbolicSpace 1:1 mapping constraint),
        # and nPercepts must be 2*nConcepts (InvertiblePiLayer invertibility).
        xml = """<model>
  <architecture>
    <type>basic</type>
    <nWords>16</nWords>
    <conceptualOrder>2</conceptualOrder>
    <reversible>false</reversible>
  </architecture>
  <InputSpace><nActive>32</nActive><nDim>8</nDim></InputSpace>
  <PerceptualSpace><nActive>4</nActive><nDim>8</nDim><nVectors>8</nVectors></PerceptualSpace>
  <ConceptualSpace><nActive>2</nActive><nDim>8</nDim><nVectors>4</nVectors></ConceptualSpace>
  <SymbolicSpace><nActive>2</nActive><nDim>0</nDim></SymbolicSpace>
  <OutputSpace><nActive>2</nActive><nDim>4</nDim></OutputSpace>
</model>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml)
            path = f.name
        try:
            model, cfg = BaseModel.from_config(path)
            self.assertIsInstance(model, BM)
        finally:
            os.unlink(path)


class TestDataSourceBuffer(unittest.TestCase):
    """Data stores raw text as an immutable uint8 source buffer."""

    def test_source_buffer_created_for_text(self):
        from BasicModel import Data
        data = Data()
        data.load("xor")  # XOR uses text examples
        self.assertTrue(hasattr(data, 'train_source'))
        self.assertEqual(data.train_source.dtype, torch.uint8)

    def test_source_buffer_is_uint8(self):
        from BasicModel import Data
        data = Data()
        data.load("xor")
        self.assertEqual(data.train_source.dtype, torch.uint8)

    def test_no_source_buffer_for_numeric(self):
        """Numeric datasets should not have a source buffer."""
        from BasicModel import Data
        data = Data()
        data.load("mnist")
        self.assertIsNone(data.train_source)


class TestSymbolDimZeroPassthrough(unittest.TestCase):
    """symbolDim must be 0 when symbolic space is passthrough."""

    def test_passthrough_symbolic_space_has_zero_symbol_dim(self):
        """When symbolPassThrough=True, symbolDim should be 0 and
        embedding size should not be inflated by a symbol dimension."""
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        TheObjectEncoding.setInputDim(1)
        TheObjectEncoding.setPerceptDim(1)
        TheObjectEncoding.setConceptDim(1)
        TheObjectEncoding.setSymbolDim(0)
        TheObjectEncoding.setOutputDim(1)
        self.assertEqual(TheObjectEncoding.symbolDim, 0)
        self.assertEqual(TheObjectEncoding.getSymbolEmbedding(), 0)

    def test_objectencoding_zero_contribution_when_unused(self):
        """ObjectEncoding must not inflate tensor size when nWhere=0, nWhen=0."""
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        nDim = 10
        self.assertEqual(TheObjectEncoding.getEmbeddingSize(nDim), nDim)


class TestInputSpaceLexIntegration(unittest.TestCase):
    """InputSpace with text data creates a Lex instance, span table, and
    encodes spans as [nWhat + nWhere] via VectorSet codebook + ObjectEncoding."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def _make_text_data(self):
        """Create a minimal Data object with text examples and source buffer."""
        from BasicModel import Data
        data = Data()
        data.load("xor")
        return data

    def _make_input_space(self, tokenizer="grammatical"):
        """Create an InputSpace with model_type='lm' from XOR text data."""
        from BasicModel import InputSpace, TheObjectEncoding
        data = self._make_text_data()
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer=tokenizer)
        return inp, data

    def test_lex_created_on_init(self):
        """InputSpace with model_type='lm' creates a Lex instance."""
        from lex import Lex
        inp, _ = self._make_input_space()
        self.assertTrue(hasattr(inp, 'lex'))
        self.assertIsInstance(inp.lex, Lex)

    def test_span_table_created(self):
        """InputSpace stores a span table from Lex.encode()."""
        inp, _ = self._make_input_space()
        self.assertTrue(hasattr(inp, 'spans'))
        self.assertEqual(inp.spans.ndim, 2)
        self.assertEqual(inp.spans.shape[1], 3)  # (start, end, type)

    def test_span_table_shape(self):
        """Span table has one row per word token across all training text."""
        inp, _ = self._make_input_space()
        # XOR data: "hello world", "hello there", "loving world", "loving there"
        # Concatenated: "hello world hello there loving world loving there"
        # That's 8 words
        self.assertEqual(inp.spans.shape[0], 8)

    def test_forward_produces_correct_shape(self):
        """forward() with Lex path produces [batch, nInput, embeddingSize]."""
        inp, data = self._make_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        output = inp.forward(inputTensor)
        _, embSize = inp.getEmbeddedIO()
        self.assertEqual(list(output.shape), [batch_size, inp.outputShape[0], embSize])

    def test_lex_to_codebook_mapping_exists(self):
        """InputSpace builds a Lex token_id -> codebook index mapping."""
        inp, _ = self._make_input_space()
        self.assertTrue(hasattr(inp, 'lex_to_codebook'))
        # Every Lex token should have a mapping
        for word, token_id in inp.lex.vocab.items():
            self.assertIn(token_id, inp.lex_to_codebook)

    def test_nwhere_encodes_byte_offsets(self):
        """nWhere should encode span byte offsets, not sequential positions."""
        import math, numpy as np
        from BasicModel import TheObjectEncoding, PositionalEncoding
        inp, data = self._make_input_space()
        batch_size = 1
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        output = inp.forward(inputTensor)
        # Extract nWhere from the output
        embSize = output.shape[-1]
        where_idx = np.add([embSize, embSize], PositionalEncoding.index)
        p1 = output[0, 0, where_idx[0]].item()  # sin component of first word
        p2 = output[0, 0, where_idx[1]].item()  # cos component of first word
        # First word starts at byte offset 0
        div_term = TheObjectEncoding.where.div_term
        expected_p1 = math.sin(0 * div_term * div_term)
        expected_p2 = math.cos(0 * div_term * div_term)
        self.assertAlmostEqual(p1, expected_p1, places=5)
        self.assertAlmostEqual(p2, expected_p2, places=5)


class TestOutputSpaceTextReconstruction(unittest.TestCase):
    """OutputSpace can reconstruct text from symbolic vectors."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_numeric_output_unchanged(self):
        """Numeric OutputSpace should still produce [B, nOutput] tensor."""
        from BasicModel import TheObjectEncoding, OutputSpace
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        TheObjectEncoding.symbolDim = 1
        TheObjectEncoding.outputDim = 1
        TheObjectEncoding.nOutput = 3
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        nIn, nOut = 4, 3
        os_ = OutputSpace(nIn, nOut)
        x = torch.randn(2, nIn, 1)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, 1])
        # text_mode should be False for numeric data
        self.assertFalse(os_.text_mode)

    def test_text_mode_false_without_lex(self):
        """OutputSpace without lex info should have text_mode=False."""
        from BasicModel import TheObjectEncoding, OutputSpace
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8,
                                        conceptDim=8, symbolDim=0, outputDim=4)
        TheObjectEncoding.nOutput = 4
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        nIn, nOut = 4, 4
        os_ = OutputSpace(nIn, nOut)
        self.assertFalse(os_.text_mode)

    def test_set_text_mode_enables_reconstruction(self):
        """set_text_mode() stores codebook, words, and lex references."""
        from BasicModel import InputSpace, Data, OutputSpace, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="grammatical")
        # Create OutputSpace with the same embedding setup
        nOut = 4
        os_ = OutputSpace(nInput, nOut)
        os_.set_text_mode(inp)
        self.assertTrue(os_.text_mode)

    def test_reconstruct_from_known_vectors(self):
        """Given codebook vectors with nWhere, reconstruct_text should recover words at positions."""
        import math
        from BasicModel import (InputSpace, Data, OutputSpace, TheObjectEncoding,
                                PositionalEncoding)
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="grammatical")
        nOut = 4
        os_ = OutputSpace(nInput, nOut)
        os_.set_text_mode(inp)

        # Build synthetic vectors from known codebook entries with known nWhere
        codebook = inp.vectors()._emb.weight.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize
        nWhat = embSize - TheObjectEncoding.objectSize
        div_term = TheObjectEncoding.where.div_term

        # Pick first two words from the codebook
        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize])
        expected_words = []
        for i in range(nVec):
            vectors[0, i, :] = codebook[i]
            expected_words.append(words_list[i])
            # Set nWhere to encode byte offset i*6
            offset = i * 6
            pos = offset * div_term
            where_idx = np.add([embSize, embSize], PositionalEncoding.index)
            vectors[0, i, where_idx[0]] = math.sin(pos * div_term)
            vectors[0, i, where_idx[1]] = math.cos(pos * div_term)

        recovered_words, recovered_positions = os_.reconstruct_text(vectors)
        self.assertEqual(recovered_words[0], expected_words)

    def test_reconstruct_consecutive_no_nwhere(self):
        """When nWhere is zero, tokens are written consecutively."""
        from BasicModel import (InputSpace, Data, OutputSpace, TheObjectEncoding)
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="grammatical")
        nOut = 4
        os_ = OutputSpace(nInput, nOut)
        os_.set_text_mode(inp)

        # Build vectors with nWhere = 0 (all zeros)
        codebook = inp.vectors()._emb.weight.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize])
        expected_words = []
        for i in range(nVec):
            vectors[0, i, :embSize - TheObjectEncoding.objectSize] = \
                codebook[i, :embSize - TheObjectEncoding.objectSize]
            expected_words.append(words_list[i])
        # nWhere left as zero -> consecutive mode

        recovered_words, recovered_positions = os_.reconstruct_text(vectors)
        self.assertEqual(recovered_words[0], expected_words)

    def test_reconstruct_to_buffer(self):
        """reconstruct_text with to_buffer=True produces a string with positioned words."""
        import math
        from BasicModel import (InputSpace, Data, OutputSpace, TheObjectEncoding,
                                PositionalEncoding)
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="grammatical")
        nOut = 4
        os_ = OutputSpace(nInput, nOut)
        os_.set_text_mode(inp)

        # Build synthetic vectors with nWhere at known positions
        codebook = inp.vectors()._emb.weight.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize
        div_term = TheObjectEncoding.where.div_term
        where_idx = np.add([embSize, embSize], PositionalEncoding.index)

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize])
        # Word 0 at offset 0, word 1 at offset 6
        for i in range(nVec):
            vectors[0, i, :] = codebook[i]
            offset = i * 6
            pos = offset * div_term
            vectors[0, i, where_idx[0]] = math.sin(pos * div_term)
            vectors[0, i, where_idx[1]] = math.cos(pos * div_term)

        recovered_words, positions = os_.reconstruct_text(vectors)
        text = os_.reconstruct_buffer(vectors)
        # The buffer should contain words at byte offsets
        self.assertIsInstance(text[0], str)
        self.assertIn(words_list[0], text[0])
        self.assertIn(words_list[1], text[0])

    def test_forward_reverse_shapes_unchanged(self):
        """forward() and reverse() tensor shapes must not change with text_mode."""
        from BasicModel import (InputSpace, Data, OutputSpace, TheObjectEncoding)
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.symbolDim = 1
        TheObjectEncoding.outputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nOutput = 4
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="grammatical")
        nOut = 4
        os_ = OutputSpace(nInput, nOut, reversible=True)
        os_.set_text_mode(inp)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.symbolDim)
        x = torch.randn(2, nInput, inEmb)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, TheObjectEncoding.outputDim])
        # Reverse path should also be unchanged
        rev = os_.reverse(y)
        self.assertEqual(list(rev.shape), [2, nInput, inEmb])


class TestInputSpaceTextRoundTrip(unittest.TestCase):
    """InputSpace.reverse() must reconstruct text from latent state."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def _make_text_input_space(self):
        """Create an InputSpace with model_type='lm' from XOR text data."""
        from BasicModel import InputSpace, Data, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="grammatical")
        return inp, data

    def test_reverse_recovers_words(self):
        """forward -> reverse should recover the original words."""
        inp, data = self._make_text_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        # Decode what words we expect from the input bytes
        expected_words = []
        for b in range(batch_size):
            raw_bytes = inputTensor[b].squeeze().tolist()
            text = "".join(chr(int(c) & 0xFF) for c in raw_bytes).rstrip("\x00")
            expected_words.append(text.split())
        # Forward pass
        latent = inp.forward(inputTensor)
        # Reverse pass
        inp.reverse(latent)
        recovered = inp.reconstruct_text()
        for b in range(batch_size):
            # Only compare up to the number of words that fit in nVec
            nVec = inp.outputShape[0]
            exp = expected_words[b][:nVec]
            rec = recovered[b][:len(exp)]
            self.assertEqual(rec, exp,
                             f"Batch {b}: expected {exp}, got {rec}")

    def test_reverse_recovers_all_xor_examples(self):
        """All XOR training examples should round-trip through forward/reverse."""
        inp, data = self._make_text_input_space()
        all_inputs = data.train_input
        inputTensor = inp.prepInput(all_inputs)
        latent = inp.forward(inputTensor)
        inp.reverse(latent)
        recovered = inp.reconstruct_text()
        nVec = inp.outputShape[0]
        for b in range(len(all_inputs)):
            raw_bytes = inputTensor[b].squeeze().tolist()
            text = "".join(chr(int(c) & 0xFF) for c in raw_bytes).rstrip("\x00")
            exp = text.split()[:nVec]
            rec = recovered[b][:len(exp)]
            self.assertEqual(rec, exp,
                             f"Example {b}: expected {exp}, got {rec}")

    def test_reconstruct_text_joins_words(self):
        """reconstruct_text(join=True) returns joined strings."""
        inp, data = self._make_text_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        latent = inp.forward(inputTensor)
        inp.reverse(latent)
        joined = inp.reconstruct_text(join=True)
        self.assertIsInstance(joined[0], str)
        self.assertGreater(len(joined[0]), 0)

    def test_reverse_numeric_unchanged(self):
        """Numeric reverse path should still work exactly as before."""
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        TheObjectEncoding.setInputDim(1)
        TheObjectEncoding.setPerceptDim(1)
        TheObjectEncoding.setConceptDim(1)
        TheObjectEncoding.setSymbolDim(0)
        TheObjectEncoding.setOutputDim(1)
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = 8
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        from BasicModel import InputSpace
        nIn, nDim = 8, 1
        inp = InputSpace(nIn, nIn, quantized=False)
        x = torch.randn(2, nIn, nDim)
        y = inp.forward(x)
        result = inp.reverse(y)
        # Numeric path returns tensor, not text
        self.assertIsInstance(result, (torch.Tensor, list))


class TestTokenizerConfig(unittest.TestCase):
    """Tokenizer is selectable via config: traditional (word2vec) or grammatical (Lex)."""

    def test_traditional_tokenizer_no_lex(self):
        """tokenizer='traditional' should NOT create Lex instance."""
        from BasicModel import InputSpace, Data, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="traditional")
        self.assertFalse(hasattr(inp, 'lex'))

    def test_grammatical_tokenizer_creates_lex(self):
        """tokenizer='grammatical' should create Lex instance."""
        from BasicModel import InputSpace, Data, TheObjectEncoding
        from lex import Lex
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="grammatical")
        self.assertTrue(hasattr(inp, 'lex'))
        self.assertIsInstance(inp.lex, Lex)

    def test_traditional_keeps_reversible_dictionary(self):
        """tokenizer='traditional' creates Embedding for word2vec."""
        from BasicModel import InputSpace, Data, Embedding, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="traditional")
        self.assertIsInstance(inp.vectors(), Embedding)


class TestEmbeddingLexDelegation(unittest.TestCase):
    """Verify Embedding tokenize/tokenizeList delegate to Lex."""

    def test_tokenize_returns_word_list(self):
        """tokenize() returns list of word lists from byte tensor input."""
        from BasicModel import Embedding
        text = "the dog barks"
        byte_tensor = torch.tensor([ord(c) for c in text], dtype=torch.uint8)
        padded = torch.zeros(32, dtype=torch.uint8)
        padded[:len(byte_tensor)] = byte_tensor
        batch = padded.unsqueeze(0)  # [1, 32]

        emb = Embedding()
        result = emb.tokenize(batch)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ["the", "dog", "barks"])

    def test_tokenize_strips_punctuation(self):
        """tokenize() separates punctuation from words."""
        from BasicModel import Embedding
        text = "the dog barks."
        byte_tensor = torch.tensor([ord(c) for c in text], dtype=torch.uint8)
        padded = torch.zeros(32, dtype=torch.uint8)
        padded[:len(byte_tensor)] = byte_tensor
        batch = padded.unsqueeze(0)

        emb = Embedding()
        result = emb.tokenize(batch)
        # Lex splits "barks." into "barks" (WORD) + "." (SEPARATOR)
        # Only WORDs are returned
        self.assertIn("barks", result[0])
        self.assertNotIn("barks.", result[0])
        self.assertNotIn(".", result[0])

    def test_tokenize_list_builds_vocab(self):
        """tokenizeList() produces vocabulary via Lex."""
        from BasicModel import Embedding
        emb = Embedding()
        data = ["the dog barks", "the cat sits"]
        vocab = emb.tokenizeList(data)
        self.assertIn("dog", vocab)
        self.assertIn("cat", vocab)
        self.assertIn(" ", vocab)
        self.assertIn("\x00", vocab)


class TestInputSpaceParseEmbeddings(unittest.TestCase):

    def test_loads_parse_artifact(self):
        import tempfile, os
        from embed import WordVectors
        import numpy as np

        words = ["the", "dog", "barks", "cat", "sits"]
        vecs = np.random.randn(len(words), 20).astype(np.float32)
        wv = WordVectors(vecs, words)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            artifact_path = f.name
        wv.save(artifact_path)

        try:
            loaded = WordVectors.load(artifact_path)
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded._vectors.shape[1], 20)
            self.assertIn("dog", loaded)
        finally:
            os.unlink(artifact_path)


class TestXorForwardPass(unittest.TestCase):
    """Embedding-backed InputSpace handles xor.xml forward pass without assertion error."""

    def test_xor_forward_produces_output(self):
        """InputSpace with model_type='lm' can forward xor data through Embedding."""
        from BasicModel import InputSpace, Data, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="lm", pretrained=False, data=data,
                         tokenizer="traditional")
        inputTensor = inp.prepInput(data.train_input[:2])
        result = inp.forward(inputTensor)
        self.assertEqual(result.shape[0], 2)  # batch size
        self.assertEqual(result.shape[1], inp.outputShape[0])


class TestErgodicMnistReport(unittest.TestCase):
    """OutputSpace.forwardLinear.W accessible even with reversible=True."""

    def test_forward_layer_weight_accessible(self):
        """mnistReport can access the forward linear layer weight matrix."""
        from BasicModel import OutputSpace, TheObjectEncoding, LinearLayer
        # Create OutputSpace with reversible=True (the default from defaults.xml)
        TheObjectEncoding.setOutputDim(1)
        TheObjectEncoding.symbolDim = 1
        TheObjectEncoding.nOutput = 10
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        os_ = OutputSpace(10, 10, reversible=True)
        # The bug was: forwardLinear is a bound method, not a layer
        # After fix: we can get the layer via linear1
        fwd_layer = (os_.linear1 if hasattr(os_, 'linear1') else os_.forwardLinear)
        self.assertTrue(hasattr(fwd_layer, 'W'))
        self.assertIsInstance(fwd_layer, LinearLayer)


# ---------------------------------------------------------------------------
# TestModelTypeVariants — missing model type combinations
# ---------------------------------------------------------------------------
class TestModelTypeVariants(unittest.TestCase):
    """Exercise BasicModel.create() with configuration combinations not yet covered."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_invertible(self):
        """invertible=True with ergodic, reversible — forward + reverse."""
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                     invertible=True, ergodic=True, reversible=True,
                     perceptPassThrough=True, symbolPassThrough=True,
                     reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_has_norm(self):
        """hasNorm=True with ergodic — forward only."""
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                     hasNorm=True, ergodic=True,
                     perceptPassThrough=True, symbolPassThrough=True,
                     reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_conceptual_order_1(self):
        """conceptualOrder=2 — forward only (equal object counts).

        Higher-order cycles require a non-passthrough symbolic space so that
        symbolDim > 0 for the second perceptual/conceptual/symbolic spaces.
        """
        from BasicModel import TheObjectEncoding
        model = _make_simple_model(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4)
        TheObjectEncoding.setSymbolDim(1)
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
                     conceptualOrder=2, reversible=False,
                     perceptPassThrough=True, symbolPassThrough=False,
                     reshape=True)
        x = torch.randn(2, 8, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_symbolic_order_1(self):
        """symbolicOrder=2 — forward only (equal object counts).

        Higher-order cycles require a non-passthrough symbolic space so that
        symbolDim > 0 for the syntactic/symbolic spaces.
        """
        from BasicModel import TheObjectEncoding
        model = _make_simple_model(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4)
        TheObjectEncoding.setSymbolDim(1)
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8, nOutput=4,
                     symbolicOrder=2,
                     perceptPassThrough=True, symbolPassThrough=False,
                     reshape=True)
        x = torch.randn(2, 8, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_non_ergodic_reverse(self):
        """ergodic=False with reversible=True — forward + reverse."""
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                     ergodic=False, reversible=True,
                     perceptPassThrough=True, symbolPassThrough=True,
                     reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_percept_no_attention(self):
        """perceptHasAttention=False, perceptPassThrough=False — forward only."""
        model = _make_simple_model(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4)
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
                     perceptHasAttention=False,
                     perceptPassThrough=False, symbolPassThrough=True,
                     reshape=True)
        x = torch.randn(2, 8, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_concept_with_attention(self):
        """conceptHasAttention=True — forward only."""
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                     conceptHasAttention=True,
                     perceptPassThrough=True, symbolPassThrough=True,
                     reshape=True)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)


class TestReconstructionSymbols(unittest.TestCase):
    """Test that reconstruction symbols split correctly and enable reconstruction."""

    def _create_xor_model(self, nSymbols=3, nOutput=1):
        """Helper: create XOR model from XOR_exact.xml with given symbol/output counts.

        Also patches ConceptualSpace/nVectors to match nSymbols, because
        SymbolicSpace requires inputShape[0] == nVectors (1:1 concept→symbol mapping).
        """
        import tempfile
        import xml.etree.ElementTree as ET
        from BasicModel import BasicModel, TheData

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Patch autoload off
        auto = root.find("architecture/autoload")
        if auto is None:
            auto = ET.SubElement(root.find("architecture"), "autoload")
        auto.text = "false"

        # Patch symbol count (and concepts to match — SymbolicSpace requires nConcepts == nSymbols)
        sym_active = root.find("SymbolicSpace/nActive")
        if sym_active is not None:
            sym_active.text = str(nSymbols)
        sym_nvec = root.find("SymbolicSpace/nVectors")
        if sym_nvec is not None:
            sym_nvec.text = str(nSymbols)
        con_active = root.find("ConceptualSpace/nActive")
        if con_active is not None:
            con_active.text = str(nSymbols)

        # Patch output count
        out_active = root.find("OutputSpace/nActive")
        if out_active is not None:
            out_active.text = str(nOutput)
        out_nvec = root.find("OutputSpace/nVectors")
        if out_nvec is not None:
            out_nvec.text = str(nOutput)

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
        tree.write(tmp, xml_declaration=True)
        tmp.close()
        TheData.load("xor")
        m = BasicModel()
        m.create_from_config(tmp.name, data=TheData)
        os.unlink(tmp.name)
        return m

    def test_nOutputSymbols_computed(self):
        """nOutputSymbols and nReconSymbols are computed from nSymbols and nOutput."""
        m = self._create_xor_model(nSymbols=3, nOutput=1)
        self.assertEqual(m.nOutputSymbols, 1)
        self.assertEqual(m.nReconSymbols, 2)

    def test_no_recon_symbols_when_equal(self):
        """When nSymbols == nOutput, there are no reconstruction symbols."""
        m = self._create_xor_model(nSymbols=3, nOutput=3)
        self.assertEqual(m.nOutputSymbols, 3)
        self.assertEqual(m.nReconSymbols, 0)

    def test_forward_output_shape_unchanged(self):
        """Forward pass output shape is [batch, nOutput] regardless of recon symbols."""
        m = self._create_xor_model(nSymbols=3, nOutput=1)
        m.train(False)
        test_input, _ = m.inputSpace.getTestData()
        x = m.inputSpace.prepInput(test_input[:2])
        with torch.no_grad():
            forwardInput, symbols, outputPred = m.forward(x)
        # Output should have batch dim = 2
        self.assertEqual(outputPred.shape[0], 2)

    def test_recon_symbols_cached(self):
        """After forward(), model.recon_symbols has correct shape."""
        m = self._create_xor_model(nSymbols=3, nOutput=1)
        m.train(False)
        test_input, _ = m.inputSpace.getTestData()
        x = m.inputSpace.prepInput(test_input[:2])
        with torch.no_grad():
            forwardInput, symbols, outputPred = m.forward(x)
        # recon_symbols should be [batch, nReconSymbols, dim]
        self.assertIsNotNone(m.recon_symbols)
        self.assertEqual(m.recon_symbols.shape[0], 2)   # batch
        self.assertEqual(m.recon_symbols.shape[1], 2)    # nReconSymbols

    def test_reverse_uses_recon_symbols(self):
        """Reverse pass produces output with correct shape when recon symbols present."""
        m = self._create_xor_model(nSymbols=3, nOutput=1)
        m.train(False)
        test_input, _ = m.inputSpace.getTestData()
        x = m.inputSpace.prepInput(test_input[:2])
        with torch.no_grad():
            forwardInput, symbols, outputPred = m.forward(x)
            inputData, inputPred = m.reverse(symbols, outputPred)
        # Should not raise; inputData should have batch dimension
        self.assertEqual(inputData.shape[0], 2)

    def test_nsymbols_less_than_noutput_raises(self):
        """Creating a model with nSymbols < nOutput should raise."""
        with self.assertRaises(AssertionError):
            self._create_xor_model(nSymbols=2, nOutput=3)

    def test_xor_training_with_recon_symbols(self):
        """Training with recon symbols works end-to-end and output loss converges.

        Verifies:
        - Training loop runs without errors when recon symbols are active
        - Output loss converges to near-zero (XOR is learnable)
        - recon_symbols are populated on every forward pass
        - Recon symbols carry live gradients (grad_fn is not None)
        """
        torch.manual_seed(42)
        m = self._create_xor_model(nSymbols=3, nOutput=1)
        m.setAlpha(0.0)
        m.train(True)

        criterionOutput, criterionInput = m._getLossFn()
        optimizer = m.getOptimizer(lr=0.005)

        train_input, train_output = m.inputSpace.getTrainData()
        recon_symbols_seen = False

        for epoch in range(200):
            for i in range(0, len(train_input), 10):
                ib = train_input[i:i+10]
                ob = train_output[i:i+10]
                it = m.inputSpace.prepInput(ib)
                ot = m.outputSpace.prepOutput(ob)

                optimizer.zero_grad()
                forwardInput, symbols, outputPred = m.forward(it)
                lossOut = criterionOutput(outputPred.squeeze(), ot.squeeze())

                # Verify recon symbols are populated and have gradient connectivity
                if m.recon_symbols is not None:
                    recon_symbols_seen = True
                    self.assertIsNotNone(m.recon_symbols.grad_fn,
                                         "recon_symbols should have grad_fn (live computation graph)")

                inputData, inputPred = m.reverse(symbols, outputPred)
                lossIn = criterionInput(inputPred, forwardInput.squeeze())
                total = lossOut + lossIn

                total.backward()
                optimizer.step()

            m.inputSpace.shuffle()

        # Recon symbols should have been present during training
        self.assertTrue(recon_symbols_seen, "recon_symbols were never populated during training")

        # Output loss should converge — XOR is learnable regardless of recon symbols
        self.assertLess(lossOut.item(), 0.01,
                        f"Output loss ({lossOut.item():.4f}) should converge for XOR")

    def test_xor_perfect_reconstruction(self):
        """After training, all 4 XOR inputs reconstruct to the correct words.

        Uses XOR_exact.xml which configures PerceptualSpace with invertible=True
        and nActive=8 so that the non-naive InvertiblePiLayer path is exercised.
        The non-naive path uses SVD-based compute_Winverse() for numerically
        stable inversion (no pinv fallback).
        """
        from BasicModel import BasicModel, BasicModelFactory, TheData
        import xml.etree.ElementTree as ET

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Ensure autoload is off
        auto = root.find("architecture/autoload")
        if auto is None:
            auto = ET.SubElement(root.find("architecture"), "autoload")
        auto.text = "false"

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
        tree.write(tmp, xml_declaration=True)
        tmp.close()

        try:
            torch.manual_seed(42)
            TheData.load("xor")
            m = BasicModel()
            m.create_from_config(tmp.name, data=TheData)

            # Train for enough epochs to converge
            m.run(numEpochs=500, batchSize=10, lr=0.01)

            # Run a final evaluation pass with reverse
            test_input, test_output = m.inputSpace.getTestData()
            m.setAlpha(0.0)
            m.train(False)
            with torch.no_grad():
                m.runEpoch(test_input, test_output, lr=0, batchSize=len(test_input))

            # Check each input reconstructs to the correct words
            for i in range(len(test_input)):
                original = m._bytes_to_text(test_input[i])
                recon = m._bytes_to_text(m.inputSpace.reconstructed[i])
                self.assertEqual(
                    original.split(), recon.split(),
                    f"Input {i}: '{original}' reconstructed as '{recon}'"
                )
        finally:
            os.unlink(tmp.name)


class TestXor3dReversePass(unittest.TestCase):
    """XOR model with reshape=false + reversible=true (3D mode).

    InvertiblePiLayer doubles the sequence dimension, so nActive_percept
    must be 2*nActive_input.  This test verifies the model constructs and
    runs a forward+reverse pass without shape errors.
    """

    def test_construct_and_forward_reverse(self):
        from BasicModel import BasicModel, TheData
        import xml.etree.ElementTree as ET

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "xor_3d.xml")
        torch.manual_seed(42)
        TheData.load("xor")
        m = BasicModel()
        m.create_from_config(xml_path, data=TheData)

        # Forward pass
        test_input, test_output = m.inputSpace.getTestData()
        m.setAlpha(0.0)
        m.train(False)
        with torch.no_grad():
            m.runEpoch(test_input, test_output, lr=0, batchSize=len(test_input))

        # Verify no crash and shapes are consistent
        self.assertIsNotNone(m.inputSpace.reconstructed)


if __name__ == "__main__":
    unittest.main()
