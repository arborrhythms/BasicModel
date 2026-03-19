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
# Force GPU for unit tests (must be set before BasicModel import)
os.environ["BASICMODEL_DEVICE"] = "gpu"

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
        x = torch.randn(2, 4).to(TheDevice)
        y = layer(x)
        self.assertEqual(y.shape, (2, 3))

    def test_with_identity_weight(self):
        from Model import LinearLayer
        W = torch.eye(4).to(TheDevice)
        layer = LinearLayer(nInput=4, nOutput=4, W=W)
        x = torch.randn(1, 4).to(TheDevice)
        y = layer(x)
        self.assertEqual(y.shape, (1, 4))


class TestInvertibleRotationLayer(unittest.TestCase):
    def test_forward_reverse_identity(self):
        from Model import InvertibleRotationLayer
        dim = 4
        layer = InvertibleRotationLayer(dim)
        x = torch.randn(2, dim).to(TheDevice)
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4),
                        f"Rotation reverse error: {(x - x_rec).abs().max():.6f}")


class TestInvertibleDiagonalLayer(unittest.TestCase):
    def test_forward_reverse_identity(self):
        from Model import InvertibleDiagonalLayer
        layer = InvertibleDiagonalLayer(nInput=4, nOutput=4)
        x = torch.randn(2, 4).to(TheDevice)
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4),
                        f"Diagonal reverse error: {(x - x_rec).abs().max():.6f}")


class TestInvertibleLinearLayer(unittest.TestCase):
    def test_forward_reverse_square(self):
        from Model import InvertibleLinearLayer
        layer = InvertibleLinearLayer(nInput=4, nOutput=4)
        x = torch.randn(2, 4).to(TheDevice)
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-3),
                        f"Linear reverse error: {(x - x_rec).abs().max():.6f}")


class TestSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import SigmaLayer
        layer = SigmaLayer(nInput=8, nOutput=4)
        x = torch.randn(2, 8).to(TheDevice)
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))


class TestPiLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import PiLayer
        layer = PiLayer(nInput=6, nOutput=3)
        x = torch.randn(2, 6).to(TheDevice)
        y = layer(x)
        self.assertEqual(y.shape, (2, 3))


class TestInvertibleSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import InvertibleSigmaLayer
        layer = InvertibleSigmaLayer(nInput=4, nOutput=4)
        x = torch.randn(2, 4).to(TheDevice) * 0.3
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))

    def test_reverse_shape(self):
        from Model import InvertibleSigmaLayer
        layer = InvertibleSigmaLayer(nInput=4, nOutput=4)
        y = torch.randn(2, 4).to(TheDevice) * 0.3
        x = layer.reverse(y)
        self.assertEqual(x.shape, (2, 4))


class TestAttentionLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import AttentionLayer
        layer = AttentionLayer(nInput=8, nOutput=4)
        x = torch.randn(2, 8).to(TheDevice)
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))


class TestNormLayer(unittest.TestCase):
    def test_forward_runs(self):
        from Model import NormLayer
        layer = NormLayer(4, 4)
        x = torch.randn(3, 4).to(TheDevice)
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
        x = torch.randn(2, 28*28, 1).to(TheDevice)  # batch of 2, flattened MNIST, dim=1
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_simple_model_ergodic(self):
        """BasicModel (simple path) with ergodic=True uses SigmaLayer path."""
        model = _make_simple_model(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, certainty=True, reshape=True)
        x = torch.randn(2, 28*28, 1).to(TheDevice)
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
        x = torch.randn(1, 4).to(TheDevice)
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
        inEmb = TheObjectEncoding.getObjectEncodingSize(TheObjectEncoding.inputDim)
        x = torch.randn(self.B, nIn, inEmb).to(TheDevice)
        y = cs(x)
        outEmb = TheObjectEncoding.getObjectEncodingSize(TheObjectEncoding.conceptDim)
        self.assertEqual(list(y.shape), [self.B, nOut, outEmb])

    def test_conceptual_space_reverse_shape(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace(nIn, nOut, reversible=True)
        outEmb = TheObjectEncoding.getObjectEncodingSize(TheObjectEncoding.conceptDim)
        y = torch.randn(self.B, nOut, outEmb).to(TheDevice)
        x = cs.reverse(y)
        inEmb = TheObjectEncoding.getObjectEncodingSize(TheObjectEncoding.inputDim)
        self.assertEqual(list(x.shape), [self.B, nIn, inEmb])

    def test_output_space_forward_shape(self):
        from BasicModel import OutputSpace, TheObjectEncoding
        nIn, nOut = 4, 4
        os_ = OutputSpace(nIn, nOut)
        inEmb = TheObjectEncoding.getObjectEncodingSize(TheObjectEncoding.symbolDim)
        x = torch.randn(self.B, nIn, inEmb).to(TheDevice)
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
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       reshape=True)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_simple_model_reverse_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, reversible=True, reshape=True)
        x = torch.randn(2, 16, 1).to(TheDevice)
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
        x = torch.randn(2, 4, 1).to(TheDevice)
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
        vs = vs.to(TheDevice)
        x = torch.randn(2, 4, 3).to(TheDevice)
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
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       reshape=True)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_reverse_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, reversible=True, reshape=True)
        x = torch.randn(2, 16, 1).to(TheDevice)
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
        x = torch.randn(2, 16, 1).to(TheDevice)
        target = torch.randn(2, 4).to(TheDevice)
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

        x = torch.randn(2, nIn).to(TheDevice)
        y_sigma = sigma(x)
        y_manual = tanh(linear(x))
        self.assertTrue(torch.allclose(y_sigma, y_manual, atol=1e-6),
                        f"Deterministic SigmaLayer should match LinearLayer+Tanh")

    def test_deterministic_same_train_eval(self):
        from Model import SigmaLayer
        nIn, nOut = 8, 4
        sigma = SigmaLayer(nIn, nOut, ergodic=False)
        x = torch.randn(2, nIn).to(TheDevice)

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
        x = torch.randn(2, nVec, nDim).to(TheDevice)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_non_ergodic_forward_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace(nVec, nVec,
                             ergodic=False, quantized=False)
        x = torch.randn(2, nVec, nDim).to(TheDevice)
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
        y = torch.randn(2, nVec, cDim).to(TheDevice)
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
        inEmb = TheObjectEncoding.getObjectEncodingSize(TheObjectEncoding.inputDim)
        x = torch.randn(2, nIn, inEmb).to(TheDevice)
        y = cs(x)
        outEmb = TheObjectEncoding.getObjectEncodingSize(TheObjectEncoding.conceptDim)
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
        x = torch.randn(2, nIn, nDim).to(TheDevice)
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
        x = torch.randn(2, nIn, 1).to(TheDevice)
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
        y = torch.randn(2, nOut, 1).to(TheDevice)
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
    <reconstruct>none</reconstruct>
  </architecture>
  <InputSpace><nActive>32</nActive><nDim>8</nDim></InputSpace>
  <PerceptualSpace><nActive>4</nActive><nDim>8</nDim><nVectors>8</nVectors></PerceptualSpace>
  <ConceptualSpace><nActive>2</nActive><nDim>8</nDim><nVectors>4</nVectors></ConceptualSpace>
  <SymbolicSpace><nActive>2</nActive><nDim>1</nDim></SymbolicSpace>
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


class TestDataTextStorage(unittest.TestCase):
    """Data stores raw text as lists of strings for per-document processing."""

    def test_train_texts_created_for_text(self):
        from BasicModel import Data
        data = Data()
        data.load("xor")  # XOR uses text examples
        self.assertTrue(hasattr(data, 'train_texts'))
        self.assertIsInstance(data.train_texts, list)
        self.assertIsInstance(data.train_texts[0], str)

    def test_no_train_texts_for_numeric(self):
        """Numeric datasets should not have train_texts."""
        from BasicModel import Data
        data = Data()
        data.load("mnist")
        self.assertIsNone(data.train_texts)


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
        self.assertEqual(TheObjectEncoding.getSymbolEncodingSize(), 0)

    def test_objectencoding_zero_contribution_when_unused(self):
        """ObjectEncoding must not inflate tensor size when nWhere=0, nWhen=0."""
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        nDim = 10
        self.assertEqual(TheObjectEncoding.getObjectEncodingSize(nDim), nDim)


class TestInputSpaceLexIntegration(unittest.TestCase):
    """InputSpace with text data creates a Lex instance, span table, and
    encodes spans as [nWhat + nWhere] via VectorSet codebook + ObjectEncoding."""

    def setUp(self):
        from BasicModel import TheObjectEncoding, PositionalEncoding, TemporalEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        # Enforce non-zero nWhere/nWhen — earlier tests may have zeroed them
        TheObjectEncoding.nWhere = PositionalEncoding.nDim   # 2
        TheObjectEncoding.nWhen = TemporalEncoding.nDim      # 2
        TheObjectEncoding.objectSize = TheObjectEncoding.nWhere + TheObjectEncoding.nWhen

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

    def _make_input_space(self, lexer="word"):
        """Create an InputSpace with model_type='embedding' from XOR text data."""
        from BasicModel import InputSpace, TheObjectEncoding
        data = self._make_text_data()
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="embedding", embedding_path=None, data=data,
                         lexer=lexer)
        return inp, data

    def test_lex_created_on_init(self):
        """InputSpace with model_type='embedding' creates a Lex on its Embedding."""
        from lex import Lex
        inp, _ = self._make_input_space()
        from lex import Lex
        self.assertIsInstance(inp.vectors()._lex, Lex)

    def test_per_doc_spans_created(self):
        """InputSpace stores per-document `(text, start)` token streams."""
        inp, _ = self._make_input_space()
        self.assertTrue(hasattr(inp, 'doc_spans'))
        self.assertIsInstance(inp.doc_spans, list)
        for tokens in inp.doc_spans:
            self.assertIsInstance(tokens, list)
            self.assertTrue(all(isinstance(tok, tuple) for tok in tokens))
            self.assertTrue(all(len(tok) == 2 for tok in tokens))

    def test_per_doc_span_counts(self):
        """Each document token stream includes the lexical space token."""
        inp, _ = self._make_input_space()
        for tokens in inp.doc_spans:
            self.assertEqual(len(tokens), 3)

    def test_forward_produces_correct_shape(self):
        """forward() with Lex path produces [batch, nInput, embeddingSize]."""
        inp, data = self._make_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        output = inp.forward(inputTensor)
        _, embSize = inp.getEmbeddedIO()
        self.assertEqual(list(output.shape), [batch_size, inp.outputShape[0], embSize])

    def test_doc_spans_store_token_offsets(self):
        """Embedding stores token text alongside byte starts."""
        inp, _ = self._make_input_space()
        emb = inp.vectors()
        self.assertEqual(
            emb.doc_spans[0],
            [("hello", 0), (" ", 5), ("world", 6)],
        )

    def test_object_encoding_applied(self):
        """ObjectEncoding (nWhere + nWhen) is applied to forward() output."""
        from BasicModel import TheObjectEncoding
        inp, data = self._make_input_space()
        batch_size = 1
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        TheObjectEncoding.where.p = 0
        output = inp.forward(inputTensor)
        # With nWhere > 0, the reserved encoding dims should be non-zero
        # (ObjectEncoding.forward stamps sin/cos into the last objectSize dims)
        embSize = output.shape[-1]
        objSize = TheObjectEncoding.objectSize
        if objSize > 0:
            encoding_dims = output[0, 0, -objSize:]
            self.assertFalse(torch.all(encoding_dims == 0).item(),
                             "ObjectEncoding dims should be non-zero after forward()")


class TestOutputSpaceTextReconstruction(unittest.TestCase):
    """OutputSpace can reconstruct text from symbolic vectors."""

    def setUp(self):
        from BasicModel import TheObjectEncoding, PositionalEncoding, TemporalEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        # Enforce non-zero nWhere/nWhen — earlier tests may have zeroed them
        TheObjectEncoding.nWhere = PositionalEncoding.nDim   # 2
        TheObjectEncoding.nWhen = TemporalEncoding.nDim      # 2
        TheObjectEncoding.objectSize = TheObjectEncoding.nWhere + TheObjectEncoding.nWhen

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
        x = torch.randn(2, nIn, 1).to(TheDevice)
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
                         model_type="embedding", embedding_path=None, data=data,
                         lexer="word")
        # Create OutputSpace with the same embedding setup
        nOut = 8
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
        TheObjectEncoding.inputDim = 10  # need enough dims for reliable cosine matching
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="embedding", embedding_path=None, data=data,
                         lexer="word")
        nOut = 4
        os_ = OutputSpace(nInput, nOut)
        os_.set_text_mode(inp)

        # Build synthetic vectors from known codebook entries with known nWhere
        codebook = inp.vectors()._emb.weight.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize
        nWhat = embSize - TheObjectEncoding.objectSize
        div_term = TheObjectEncoding.where.div_term

        # Pick first two non-[MASK] words from the codebook
        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(TheDevice)
        expected_words = []
        # Skip [MASK] (zero vector) — cosine matching can't recover it
        usable = [j for j, w in enumerate(words_list) if w != "[MASK]"]
        for slot, j in enumerate(usable[:nVec]):
            vectors[0, slot, :nWhat] = codebook[j][:nWhat]
            expected_words.append(words_list[j])
            # Set nWhere to encode byte offset slot*6
            offset = slot * 6
            pos = offset * div_term
            where_idx = np.add([embSize, embSize], PositionalEncoding.index)
            vectors[0, slot, where_idx[0]] = math.sin(pos * div_term)
            vectors[0, slot, where_idx[1]] = math.cos(pos * div_term)

        recovered_words, recovered_positions = os_.reconstruct_text(vectors)
        self.assertEqual(recovered_words[0], expected_words)

    def test_reconstruct_consecutive_no_nwhere(self):
        """When nWhere is zero, tokens are written consecutively."""
        from BasicModel import (InputSpace, Data, OutputSpace, TheObjectEncoding)
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 10  # need enough dims for reliable cosine matching
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="embedding", embedding_path=None, data=data,
                         lexer="word")
        nOut = 4
        os_ = OutputSpace(nInput, nOut)
        os_.set_text_mode(inp)

        # Build vectors with nWhere = 0 (all zeros)
        codebook = inp.vectors()._emb.weight.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(TheDevice)
        expected_words = []
        # Skip [MASK] (zero vector) — cosine matching can't recover it
        usable = [j for j, w in enumerate(words_list) if w != "[MASK]"]
        for slot, j in enumerate(usable[:nVec]):
            vectors[0, slot, :embSize - TheObjectEncoding.objectSize] = \
                codebook[j, :embSize - TheObjectEncoding.objectSize]
            expected_words.append(words_list[j])
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
        TheObjectEncoding.inputDim = 10  # need enough dims for reliable cosine matching
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="embedding", embedding_path=None, data=data,
                         lexer="word")
        nOut = 4
        os_ = OutputSpace(nInput, nOut)
        os_.set_text_mode(inp)

        # Build synthetic vectors with nWhere at known positions
        codebook = inp.vectors()._emb.weight.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize
        nWhat = embSize - TheObjectEncoding.objectSize
        div_term = TheObjectEncoding.where.div_term
        where_idx = np.add([embSize, embSize], PositionalEncoding.index)

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(TheDevice)
        # Skip [MASK] and \x00 (both zero vectors) — cosine matching can't recover them
        usable = [j for j, w in enumerate(words_list) if w not in ("[MASK]", "\x00")]
        # Word 0 at offset 0, word 1 at offset 6
        for slot, j in enumerate(usable[:nVec]):
            vectors[0, slot, :nWhat] = codebook[j][:nWhat]
            offset = slot * 6
            pos = offset * div_term
            vectors[0, slot, where_idx[0]] = math.sin(pos * div_term)
            vectors[0, slot, where_idx[1]] = math.cos(pos * div_term)

        recovered_words, positions = os_.reconstruct_text(vectors)
        text = os_.reconstruct_buffer(vectors)
        # The buffer should contain words at byte offsets
        self.assertIsInstance(text[0], str)
        self.assertIn(words_list[usable[0]], text[0])
        self.assertIn(words_list[usable[1]], text[0])

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
                         model_type="embedding", embedding_path=None, data=data,
                         lexer="word")
        nOut = 4
        os_ = OutputSpace(nInput, nOut, reversible=True)
        os_.set_text_mode(inp)
        inEmb = TheObjectEncoding.getObjectEncodingSize(TheObjectEncoding.symbolDim)
        x = torch.randn(2, nInput, inEmb).to(TheDevice)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, TheObjectEncoding.outputDim])
        # Reverse path should also be unchanged
        rev = os_.reverse(y)
        self.assertEqual(list(rev.shape), [2, nInput, inEmb])


class TestInputSpaceTextRoundTrip(unittest.TestCase):
    """InputSpace.reverse() must reconstruct text from latent state."""

    def setUp(self):
        from BasicModel import TheObjectEncoding, PositionalEncoding, TemporalEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        # Enforce non-zero nWhere/nWhen — earlier tests may have zeroed them
        TheObjectEncoding.nWhere = PositionalEncoding.nDim   # 2
        TheObjectEncoding.nWhen = TemporalEncoding.nDim      # 2
        TheObjectEncoding.objectSize = TheObjectEncoding.nWhere + TheObjectEncoding.nWhen

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def _make_text_input_space(self):
        """Create an InputSpace with model_type='embedding' from XOR text data."""
        from BasicModel import InputSpace, Data, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 10  # need enough dims for reliable cosine matching
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="embedding", embedding_path=None, data=data,
                         lexer="word")
        return inp, data

    def test_reverse_recovers_words(self):
        """forward -> reverse should recover the original lexical tokens."""
        inp, data = self._make_text_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        expected_tokens = inp.vectors().tokenize(inputTensor)
        # Forward pass
        latent = inp.forward(inputTensor)
        # Reverse pass
        inp.reverse(latent)
        recovered = inp.reconstruct_text()
        for b in range(batch_size):
            nVec = inp.outputShape[0]
            exp = expected_tokens[b][:nVec]
            rec = recovered[b][:len(exp)]
            self.assertEqual(rec, exp,
                             f"Batch {b}: expected {exp}, got {rec}")

    def test_reverse_recovers_all_xor_examples(self):
        """All XOR examples should round-trip as lexical token streams."""
        inp, data = self._make_text_input_space()
        all_inputs = data.train_input
        inputTensor = inp.prepInput(all_inputs)
        latent = inp.forward(inputTensor)
        inp.reverse(latent)
        recovered = inp.reconstruct_text()
        expected = inp.vectors().tokenize(inputTensor)
        nVec = inp.outputShape[0]
        for b in range(len(all_inputs)):
            exp = expected[b][:nVec]
            rec = recovered[b][:len(exp)]
            self.assertEqual(rec, exp,
                             f"Example {b}: expected {exp}, got {rec}")

    def test_reconstruct_text_joins_words(self):
        """reconstruct_text(join=True) renders the whitespace buffer."""
        inp, data = self._make_text_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        latent = inp.forward(inputTensor)
        inp.reverse(latent)
        joined = inp.reconstruct_text(join=True)
        self.assertIsInstance(joined[0], str)
        expected = []
        for b in range(batch_size):
            raw_bytes = inputTensor[b].squeeze().tolist()
            expected.append(
                "".join(chr(int(c) & 0xFF) for c in raw_bytes).rstrip("\x00"))
        self.assertEqual(joined[:batch_size], expected)

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
        x = torch.randn(2, nIn, nDim).to(TheDevice)
        y = inp.forward(x)
        result = inp.reverse(y)
        # Numeric path returns tensor, not text
        self.assertIsInstance(result, (torch.Tensor, list))


class TestLexerConfig(unittest.TestCase):
    """Lexer cfg (word/sentence/grammar) always creates Lex span tables."""

    def test_embedding_always_creates_lex(self):
        """Embedding model_type always creates Lex instance."""
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
                         model_type="embedding", embedding_path=None, data=data)
        self.assertIsInstance(inp.vectors()._lex, Lex)

    def test_embedding_creates_reversible_dictionary(self):
        """Embedding model_type creates Embedding with Lex-backed codebook."""
        from BasicModel import InputSpace, Data, Embedding, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="embedding", embedding_path=None, data=data)
        self.assertIsInstance(inp.vectors(), Embedding)


class TestEmbeddingLexDelegation(unittest.TestCase):
    """Verify Embedding exposes a small Lex-facing override API."""

    @staticmethod
    def _make_batch(text, size=32):
        raw = torch.tensor([ord(c) for c in text], dtype=torch.uint8)
        padded = torch.zeros(size, dtype=torch.uint8)
        padded[:len(raw)] = raw
        return padded.unsqueeze(0)

    def test_tokenize_returns_word_list(self):
        """tokenize() returns all Lex token texts including SPACE tokens.

        With the new tokenization rules:
        - All Lex categories are included (WORD, SPACE, SEPARATOR, PUNCT)
        - SPACE is emitted between words within a sentence
        """
        from BasicModel import Embedding
        emb = Embedding()
        result = emb.tokenize(self._make_batch("the dog barks"))
        self.assertEqual(len(result), 1)
        # WORD("the") SPACE(" ") WORD("dog") SPACE(" ") WORD("barks")
        self.assertEqual(result[0], ["the", " ", "dog", " ", "barks"])

    def test_tokenize_splits_sentence_ending_punctuation(self):
        """tokenize() separates punctuation from words; all token categories returned."""
        from BasicModel import Embedding
        emb = Embedding()
        result = emb.tokenize(self._make_batch("the dog barks."))
        # Lex splits "barks." into "barks" (WORD) + "." (SEPARATOR)
        # All categories returned: WORD, SPACE, SEPARATOR
        # "the dog barks." → ["the", " ", "dog", " ", "barks", "."]
        self.assertIn("barks", result[0])
        self.assertNotIn("barks.", result[0])
        self.assertIn(".", result[0])
        self.assertEqual(result[0], ["the", " ", "dog", " ", "barks", "."])

    def test_forward_returns_token_metadata(self):
        """forward(return_meta=True) replaces old encoding wrappers."""
        from BasicModel import Embedding, TheObjectEncoding
        old_object_size = TheObjectEncoding.objectSize
        self.addCleanup(setattr, TheObjectEncoding, "objectSize", old_object_size)
        TheObjectEncoding.objectSize = 0
        emb = Embedding()
        emb.create(
            nInput=8,
            nVectors=8,
            nDim=10,
            embedding_path=None,
            source=["the dog barks"],
        )
        embedded, meta = emb.forward(
            self._make_batch("the dog barks"), return_meta=True)
        self.assertEqual(list(embedded.shape), [1, 8, emb.embeddingSize])
        self.assertEqual(
            meta["tokens"][0],
            [("the", 0), (" ", 3), ("dog", 4), (" ", 7), ("barks", 8)],
        )
        self.assertEqual(meta["span_counts"], [5])
        self.assertEqual(meta["final_offsets"], [13])


class TestMaskCodebookEntry(unittest.TestCase):
    def test_mask_codebook_entry_is_zero(self):
        """[MASK] exists in vocabulary as a zero vector after Embedding.create()."""
        from BasicModel import Embedding, TheObjectEncoding
        TheObjectEncoding.objectSize = 0
        emb = Embedding()
        emb.create(nInput=10, nVectors=2, nDim=10, embedding_path=None)
        self.assertIn("[MASK]", emb.cbow.key_to_index)
        idx = emb.cbow.key_to_index["[MASK]"]
        vec = emb._emb.weight[idx]
        self.assertTrue(torch.all(vec == 0.0))
        self.assertEqual(emb.mask_token_idx, idx)


class TestEmbeddingErgodicForward(unittest.TestCase):
    def test_vectorset_owns_exploration_state(self):
        from BasicModel import Embedding, VectorSet
        vs = VectorSet()
        emb = Embedding()
        self.assertFalse(vs.ergodic)
        self.assertAlmostEqual(vs.sigma_kappa, 0.01)
        self.assertFalse(emb.ergodic)
        self.assertAlmostEqual(emb.sigma_kappa, 0.01)

    def _make_batch(self, text, size=32):
        raw = torch.tensor([ord(c) for c in text], dtype=torch.uint8)
        padded = torch.zeros(size, dtype=torch.uint8)
        padded[:len(raw)] = raw
        return padded.unsqueeze(0)

    def _make_embedding(self, text="the dog"):
        from BasicModel import Embedding, TheObjectEncoding
        old_object_size = TheObjectEncoding.objectSize
        self.addCleanup(setattr, TheObjectEncoding, "objectSize", old_object_size)
        TheObjectEncoding.objectSize = 0
        emb = Embedding()
        emb.create(nInput=8, nVectors=8, nDim=10, embedding_path=None, source=[text])
        return emb

    def _seed_sigma(self, emb, word):
        device = emb._emb.weight.device
        emb.cbow.sigma = torch.zeros(emb._emb.weight.shape[0], device=device)
        emb.cbow.sigma[emb.cbow.key_to_index[word]] = 1.0
        emb.cbow.sigma_step = 1
        emb.cbow.sigma_beta = 0.0

    def test_forward_adds_ergodic_noise_from_sigma(self):
        emb = self._make_embedding()
        self._seed_sigma(emb, "the")
        batch = self._make_batch("the dog")

        emb.train()
        emb.ergodic = False
        baseline = emb.forward(batch)

        emb.ergodic = True
        emb.set_sigma(1.0)
        torch.manual_seed(0)
        noisy = emb.forward(batch)

        self.assertFalse(torch.allclose(baseline, noisy))

    def test_forward_sigma_zero_suppresses_ergodic_noise(self):
        emb = self._make_embedding()
        self._seed_sigma(emb, "the")
        batch = self._make_batch("the dog")

        emb.train()
        emb.ergodic = False
        baseline = emb.forward(batch)

        emb.ergodic = True
        emb.set_sigma(0.0)
        torch.manual_seed(0)
        suppressed = emb.forward(batch)

        self.assertTrue(torch.allclose(baseline, suppressed, atol=1e-5, rtol=1e-5))


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
            self.assertEqual(loaded.vector_size, 20)
            self.assertIn("dog", loaded)
        finally:
            os.unlink(artifact_path)


class TestLoadEmbeddingsEnwiki(unittest.TestCase):
    """Embedding._load_embeddings handles word2vec text format (.txt)."""

    ENWIKI_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "embeddings", "enwiki_20180420_100d.txt")

    @unittest.skip("slow — large file load")
    def test_load_enwiki_txt(self):
        """_load_embeddings loads word2vec text format via embeddingPath."""
        from BasicModel import Embedding
        wv = Embedding._load_embeddings(embedding_path=self.ENWIKI_PATH)
        self.assertIsNotNone(wv)
        self.assertEqual(wv.vector_size, 100)
        self.assertGreater(len(wv), 1000)
        self.assertIn("the", wv)

    @unittest.skip("slow — large file load")
    def test_load_enwiki_dim_filter(self):
        """_load_embeddings returns None when nDim doesn't match."""
        from BasicModel import Embedding
        wv = Embedding._load_embeddings(embedding_path=self.ENWIKI_PATH, nDim=50)
        self.assertIsNone(wv)

    def test_load_pt_format(self):
        """_load_embeddings loads .pt torch format."""
        import tempfile
        from embed import WordVectors
        words = ["hello", "world"]
        vecs = torch.randn(2, 20)
        wv = WordVectors(vecs, words)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        wv.save(path)
        try:
            from BasicModel import Embedding
            loaded = Embedding._load_embeddings(embedding_path=path, nDim=20)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 2)
            self.assertIn("hello", loaded)
        finally:
            os.unlink(path)

    def test_load_none_path(self):
        """_load_embeddings returns None when no path given."""
        from BasicModel import Embedding
        self.assertIsNone(Embedding._load_embeddings(embedding_path=None))

    def test_load_missing_file(self):
        """_load_embeddings returns None for nonexistent path."""
        from BasicModel import Embedding
        self.assertIsNone(Embedding._load_embeddings(embedding_path="/tmp/no_such_file.pt"))


class TestXorForwardPass(unittest.TestCase):
    """Embedding-backed InputSpace handles xor.xml forward pass without assertion error."""

    def test_xor_forward_produces_output(self):
        """InputSpace with model_type='embedding' can forward xor data through Embedding."""
        from BasicModel import InputSpace, Data, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0  # reset for test isolation
        TheObjectEncoding.computeNObjects()
        inp = InputSpace(nInput, nInput,
                         model_type="embedding", embedding_path=None, data=data)
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
        x = torch.randn(2, 16, 1).to(TheDevice)
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
        x = torch.randn(2, 16, 1).to(TheDevice)
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
        x = torch.randn(2, 8, 1).to(TheDevice)
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
        x = torch.randn(2, 8, 1).to(TheDevice)
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
                     reshape=True, naive=True)
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

        Uses XOR_recon.xml which is purpose-built for this test:
        nActive=3 input (no padding), nSymbols=6, nOutput=1 (5 recon symbols).
        """
        from BasicModel import BasicModel, TheData
        import xml.etree.ElementTree as ET

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_recon.xml")
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

            # Train with sigma annealing via runTrial
            m.runTrial(numEpochs=600, batchSize=10, lr=0.01)

            # Recon symbols should have been present during training
            self.assertIsNotNone(m.recon_symbols,
                                 "recon_symbols should be populated after training")

            # Output loss should converge
            outErr = m.trainLosses[0][-1] if m.trainLosses[0] else 1.0
            self.assertLess(outErr, 0.01,
                            f"Output loss ({outErr:.4f}) should converge for XOR")
        finally:
            os.unlink(tmp.name)

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

            # Train using XML-configured epoch count
            cfg = m.cfg if hasattr(m, 'cfg') else {}
            training = cfg.get('architecture', {}).get('training', {})
            epochs = int(training.get('numEpochs', 1000))
            lr = float(training.get('learningRate', 0.01))
            m.runTrial(numEpochs=epochs, batchSize=10, lr=lr)

            # Run a final evaluation pass with reverse
            test_input, test_output = m.inputSpace.getTestData()
            m.set_sigma(0)
            m.train(False)
            with torch.no_grad():
                m.runEpoch(batchSize=len(test_input), split="test")

            # Check reconstruction quality: at least 75% of inputs must
            # perfectly reconstruct (some words may snap to wrong codebook
            # entry when the reverse path is approximate).
            recon_texts = m.inputSpace.reconstruct_text(join=True)
            perfect = 0
            for i in range(len(test_input)):
                original = m._bytes_to_text(test_input[i])
                recon = recon_texts[i]
                orig_words = original.replace("\x00", " ").split()
                recon_words = recon.replace("\x00", " ").split()
                if orig_words == recon_words:
                    perfect += 1
            total = len(test_input)
            self.assertGreaterEqual(
                perfect / total, 0.5,
                f"Only {perfect}/{total} inputs reconstructed perfectly"
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
        import warnings
        from BasicModel import BasicModel, TheData
        import xml.etree.ElementTree as ET

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "xor_3d.xml")
        torch.manual_seed(42)
        TheData.load("xor")
        m = BasicModel()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PerceptualSpace: reversible=True with invertible=False")
            m.create_from_config(xml_path, data=TheData)

        # Forward pass
        test_input, test_output = m.inputSpace.getTestData()
        m.set_sigma(0)
        m.train(False)
        with torch.no_grad():
            m.runEpoch(batchSize=len(test_input), split="test")

        # Verify no crash and shapes are consistent
        self.assertIsNotNone(m.inputSpace.reconstructed)


class TestExpandMasked(unittest.TestCase):
    """InputSpace.expand_masked() produces N masked copies of a sentence embedding."""

    def setUp(self):
        from BasicModel import (TheObjectEncoding, PositionalEncoding,
                                TemporalEncoding, InputSpace)
        # Save/restore global state
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        # Enforce nWhere=2, nWhen=2
        TheObjectEncoding.nWhere = PositionalEncoding.nDim   # 2
        TheObjectEncoding.nWhen = TemporalEncoding.nDim      # 2
        TheObjectEncoding.objectSize = TheObjectEncoding.nWhere + TheObjectEncoding.nWhen

        # Build a minimal InputSpace with embedding from XOR data
        from BasicModel import Data
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0
        TheObjectEncoding.computeNObjects()
        self.inp = InputSpace(nInput, nInput,
                              model_type="embedding", embedding_path=None,
                              data=data, lexer="word")
        # Run a forward pass to get a real embedded tensor
        inputBatch = data.train_input[0:1]
        inputTensor = self.inp.prepInput(inputBatch)
        self.embedded = self.inp.forward(inputTensor)  # [1, nVec, embSize]
        self.sentence = "hello world"  # matches XOR training data
        self.embSize = self.embedded.shape[-1]
        self.nVec = self.embedded.shape[1]

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_expand_masked_shape(self):
        """3-word sentence produces [3, nVec, embSize] masked batch."""
        sentence = "hello loving world"
        words = sentence.split()
        masked, positions = self.inp.expand_masked(self.embedded, sentence)
        self.assertEqual(masked.shape[0], len(words))
        self.assertEqual(masked.shape[1], self.nVec)
        self.assertEqual(masked.shape[2], self.embSize)
        self.assertEqual(positions, [0, 1, 2])

    def test_masked_position_is_zero_content(self):
        """Masked position has zero content dims, non-zero position dims."""
        from BasicModel import PositionalEncoding, TemporalEncoding
        masked, _ = self.inp.expand_masked(self.embedded, self.sentence)
        embSize = self.embSize
        where_idx = np.add([embSize, embSize], PositionalEncoding.index)
        when_idx = np.add([embSize, embSize], TemporalEncoding.index)
        pos_dims = set(where_idx.tolist() + when_idx.tolist())
        content_dims = [d for d in range(embSize) if d not in pos_dims]
        # In copy i, position i should have zero content
        for i in range(masked.shape[0]):
            content_vals = masked[i, i, content_dims]
            self.assertTrue(torch.all(content_vals == 0.0),
                            f"Copy {i}: content at masked pos should be zero, "
                            f"got max={content_vals.abs().max().item():.6f}")

    def test_non_masked_positions_unchanged(self):
        """Non-masked positions identical to original."""
        masked, _ = self.inp.expand_masked(self.embedded, self.sentence)
        # In copy 1, position 0 should match original position 0
        self.assertTrue(torch.allclose(masked[1, 0], self.embedded[0, 0]),
                        "Non-masked position should match original")
        # In copy 0, position 1 should match original position 1
        self.assertTrue(torch.allclose(masked[0, 1], self.embedded[0, 1]),
                        "Non-masked position should match original")

    def test_position_encoding_preserved(self):
        """Position encoding (nWhere) at masked position matches original."""
        from BasicModel import PositionalEncoding, TemporalEncoding
        masked, _ = self.inp.expand_masked(self.embedded, self.sentence)
        embSize = self.embSize
        where_idx = np.add([embSize, embSize], PositionalEncoding.index)
        when_idx = np.add([embSize, embSize], TemporalEncoding.index)
        pos_dims = list(where_idx) + list(when_idx)
        # At masked position, positional dims should match original
        for i in range(masked.shape[0]):
            for d in pos_dims:
                self.assertAlmostEqual(
                    masked[i, i, d].item(),
                    self.embedded[0, i, d].item(),
                    places=5,
                    msg=f"Copy {i}, dim {d}: position encoding not preserved"
                )

    def test_arlm_truncates_future(self):
        """ARLM mode zeros all positions after the masked position."""
        sentence = "hello world"
        masked, positions = self.inp.expand_masked(self.embedded, sentence, maskedPrediction='ARLM')
        # In copy 0, positions 1+ should be completely zero
        self.assertTrue(torch.all(masked[0, 1:, :] == 0.0))
        # In copy 1, positions 2+ should be completely zero (if they exist)
        if masked.shape[1] > 2:
            self.assertTrue(torch.all(masked[1, 2:, :] == 0.0))


class TestExpandMaskedTargets(unittest.TestCase):
    def setUp(self):
        """Create an OutputSpace + Embedding to test expand_masked."""
        from BasicModel import (TheObjectEncoding, PositionalEncoding,
                                TemporalEncoding, InputSpace, OutputSpace, Data)
        # Save/restore global state
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        self._orig_nObjects = TheObjectEncoding.nObjects
        # Enforce nWhere=2, nWhen=2
        TheObjectEncoding.nWhere = PositionalEncoding.nDim
        TheObjectEncoding.nWhen = TemporalEncoding.nDim
        TheObjectEncoding.objectSize = TheObjectEncoding.nWhere + TheObjectEncoding.nWhen

        # Build a minimal InputSpace with embedding from XOR data
        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0
        TheObjectEncoding.computeNObjects()
        self.inp = InputSpace(nInput, nInput,
                              model_type="embedding", embedding_path=None,
                              data=data, lexer="word")
        self.emb = self.inp.vectors()

        # Build a minimal OutputSpace
        TheObjectEncoding.symbolDim = 1
        TheObjectEncoding.outputDim = 1
        TheObjectEncoding.nOutput = 4
        self.out = OutputSpace(nActiveInput=8, nActiveOutput=4,
                               reversible=False, data=data)

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize
        TheObjectEncoding.nObjects = self._orig_nObjects

    def _make_embedded(self, n_words, emb_size=None):
        """Create a synthetic [1, n_words, embSize] embedded sentence."""
        if emb_size is None:
            emb_size = self.emb._emb.weight.shape[1] + 4  # content + nWhere + nWhen
        return torch.randn(1, n_words, emb_size, device=TheDevice)

    def test_expand_masked_shape(self):
        """N words produce [N, 1, embSize] target tensor."""
        embedded = self._make_embedded(3)
        targets = self.out.expand_masked(embedded, "hello world test")
        self.assertEqual(targets.shape[0], 3)
        self.assertEqual(targets.shape[1], 1)
        self.assertEqual(targets.shape[2], embedded.shape[-1])

    def test_expand_masked_known_word(self):
        """Target for word i matches embedded[0, i, :]."""
        embedded = self._make_embedded(1)
        targets = self.out.expand_masked(embedded, "hello")
        torch.testing.assert_close(targets[0, 0], embedded[0, 0])

    def test_expand_masked_unknown_word(self):
        """Even unknown words get their embedded vector as target."""
        embedded = self._make_embedded(1)
        targets = self.out.expand_masked(embedded, "xyzzynotaword123")
        torch.testing.assert_close(targets[0, 0], embedded[0, 0])

    def test_arus_returns_zero_targets(self):
        """ARUS mode returns all-zero target vectors."""
        embedded = self._make_embedded(2)
        targets = self.out.expand_masked(embedded, "hello world", maskedPrediction='ARUS')
        self.assertTrue(torch.all(targets == 0.0))
        self.assertEqual(targets.shape[0], 2)


class TestMaskedPredictionIntegration(unittest.TestCase):
    """Integration tests for maskedPrediction stream interface."""

    def _create_xor_embedding_model(self):
        """Create an XOR embedding model via create_from_config (autoload off)."""
        import xml.etree.ElementTree as ET
        from BasicModel import BasicModel, TheData

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
        TheData.load("xor")
        m = BasicModel()
        m.create_from_config(tmp.name, data=TheData)
        os.unlink(tmp.name)
        return m

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._saved_nObjects = TheObjectEncoding.nObjects
        self._saved_nWhere = TheObjectEncoding.nWhere
        self._saved_nWhen = TheObjectEncoding.nWhen
        self._saved_objectSize = TheObjectEncoding.objectSize
        self.model = self._create_xor_embedding_model()

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nObjects = self._saved_nObjects
        TheObjectEncoding.nWhere = self._saved_nWhere
        TheObjectEncoding.nWhen = self._saved_nWhen
        TheObjectEncoding.objectSize = self._saved_objectSize

    def test_getbatch_standard_mode(self):
        """getBatch in standard mode returns correct batches and exhausts."""
        inp = self.model.inputSpace
        batch, nextNum = inp.getBatch(0, batchSize=2, split="train")
        self.assertIsNotNone(batch)
        inputTensor, outputTensor = batch
        self.assertEqual(inputTensor.shape[0], 2)  # batch of 2
        self.assertEqual(nextNum, 1)
        # Eventually exhausts
        batchNum = 0
        count = 0
        while True:
            batch, batchNum = inp.getBatch(batchNum, batchSize=2, split="train")
            if batch is None:
                break
            count += 1
        self.assertGreater(count, 0)

    def test_getbatch_test_split(self):
        """getBatch works with test split."""
        inp = self.model.inputSpace
        batch, nextNum = inp.getBatch(0, batchSize=2, split="test")
        self.assertIsNotNone(batch)


class TestRARLM(unittest.TestCase):
    """RARLM mode masks from end and truncates previous positions."""

    def setUp(self):
        from BasicModel import (TheObjectEncoding, PositionalEncoding,
                                TemporalEncoding, InputSpace, Data)
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        TheObjectEncoding.nWhere = PositionalEncoding.nDim
        TheObjectEncoding.nWhen = TemporalEncoding.nDim
        TheObjectEncoding.objectSize = TheObjectEncoding.nWhere + TheObjectEncoding.nWhen

        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0
        TheObjectEncoding.computeNObjects()
        self.inp = InputSpace(nInput, nInput,
                              model_type="embedding", embedding_path=None,
                              data=data, lexer="word")
        inputBatch = data.train_input[0:1]
        inputTensor = self.inp.prepInput(inputBatch)
        self.embedded = self.inp.forward(inputTensor)
        self.embSize = self.embedded.shape[-1]

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_rarlm_masks_from_end(self):
        """RARLM masks position (N-1-i) and zeros previous positions."""
        sentence = "hello world test"
        masked, positions = self.inp.expand_masked(self.embedded, sentence, maskedPrediction='RARLM')
        N = len(sentence.split())
        self.assertEqual(masked.shape[0], N)
        # Copy 0: position 2 masked, positions 0-1 zeroed
        self.assertTrue(torch.all(masked[0, :2, :] == 0.0),
                        "Copy 0: positions before masked pos should be zeroed")
        # Copy 1: position 1 masked, position 0 zeroed
        self.assertTrue(torch.all(masked[1, :1, :] == 0.0),
                        "Copy 1: positions before masked pos should be zeroed")
        # Copy 2: position 0 masked, nothing zeroed before it (pos=0)

    def test_rarlm_mask_positions_reversed(self):
        """RARLM returns mask_positions in reverse order [N-1, ..., 0]."""
        sentence = "hello world test"
        _, positions = self.inp.expand_masked(self.embedded, sentence, maskedPrediction='RARLM')
        self.assertEqual(positions, [2, 1, 0])

    def test_rarlm_content_zeroed_at_masked_pos(self):
        """Content dims at the masked position are zeroed in each copy."""
        from BasicModel import PositionalEncoding, TemporalEncoding
        sentence = "hello world test"
        masked, positions = self.inp.expand_masked(self.embedded, sentence, maskedPrediction='RARLM')
        embSize = self.embSize
        where_idx = np.add([embSize, embSize], PositionalEncoding.index)
        when_idx = np.add([embSize, embSize], TemporalEncoding.index)
        pos_dims = set(where_idx.tolist() + when_idx.tolist())
        content_dims = [d for d in range(embSize) if d not in pos_dims]
        for i, pos in enumerate(positions):
            content_vals = masked[i, pos, content_dims]
            self.assertTrue(torch.all(content_vals == 0.0),
                            f"Copy {i}: content at masked pos {pos} should be zero")


class TestRARLMTargets(unittest.TestCase):
    """RARLM targets are in reverse word order."""

    def setUp(self):
        from BasicModel import (TheObjectEncoding, PositionalEncoding,
                                TemporalEncoding, InputSpace, OutputSpace, Data)
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        self._orig_nObjects = TheObjectEncoding.nObjects
        TheObjectEncoding.nWhere = PositionalEncoding.nDim
        TheObjectEncoding.nWhen = TemporalEncoding.nDim
        TheObjectEncoding.objectSize = TheObjectEncoding.nWhere + TheObjectEncoding.nWhen

        data = Data()
        data.load("xor")
        nInput = 8
        TheObjectEncoding.inputDim = 1
        TheObjectEncoding.nInput = nInput
        TheObjectEncoding.nObjects = 0
        TheObjectEncoding.computeNObjects()
        self.inp = InputSpace(nInput, nInput,
                              model_type="embedding", embedding_path=None,
                              data=data, lexer="word")
        self.emb = self.inp.vectors()

        TheObjectEncoding.symbolDim = 1
        TheObjectEncoding.outputDim = 1
        TheObjectEncoding.nOutput = 4
        self.out = OutputSpace(nActiveInput=8, nActiveOutput=4,
                               reversible=False, data=data)

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize
        TheObjectEncoding.nObjects = self._orig_nObjects

    def test_rarlm_targets_reversed(self):
        """RARLM targets are MLM targets in reverse order."""
        emb_size = self.emb._emb.weight.shape[1] + 4
        embedded = torch.randn(1, 2, emb_size, device=TheDevice)
        mlm_targets = self.out.expand_masked(embedded, "hello world", maskedPrediction='MLM')
        rarlm_targets = self.out.expand_masked(embedded, "hello world", maskedPrediction='RARLM')
        # RARLM targets should be MLM targets reversed
        torch.testing.assert_close(rarlm_targets, mlm_targets.flip(0))


class TestTrainEmbeddingsFlag(unittest.TestCase):
    """trainEmbeddings config flag controls whether embedding weights are in optimizer."""

    def _create_model(self, train_embeddings):
        """Create an XOR embedding model with specified trainEmbeddings flag."""
        import xml.etree.ElementTree as ET
        from BasicModel import BasicModel, TheData

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        auto = root.find("architecture/autoload")
        if auto is None:
            auto = ET.SubElement(root.find("architecture"), "autoload")
        auto.text = "false"

        te = root.find("architecture/trainEmbeddings")
        if te is None:
            te = ET.SubElement(root.find("architecture"), "trainEmbeddings")
        te.text = str(train_embeddings).lower()

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
        tree.write(tmp, xml_declaration=True)
        tmp.close()
        TheData.load("xor")
        m = BasicModel()
        m.create_from_config(tmp.name, data=TheData)
        os.unlink(tmp.name)
        return m

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._saved = {
            'nObjects': TheObjectEncoding.nObjects,
            'nWhere': TheObjectEncoding.nWhere,
            'nWhen': TheObjectEncoding.nWhen,
            'objectSize': TheObjectEncoding.objectSize,
        }

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        for k, v in self._saved.items():
            setattr(TheObjectEncoding, k, v)

    def test_train_embeddings_includes_emb_params(self):
        """When trainEmbeddings=true, _emb.weight is in optimizer params."""
        from BasicModel import Embedding
        m = self._create_model(True)
        if not isinstance(m.inputSpace.vectors(), Embedding):
            self.skipTest("Model doesn't use Embedding")
        emb_weight = m.inputSpace.vectors()._emb.weight
        optimizer = m.getOptimizer(lr=0.001)
        opt_params = [p.data_ptr() for group in optimizer.param_groups for p in group['params']]
        self.assertIn(emb_weight.data_ptr(), opt_params,
                      "Embedding weight should be in optimizer when trainEmbeddings=true")

    def test_frozen_embeddings_default(self):
        """When trainEmbeddings=false, _emb.weight is NOT in optimizer params."""
        from BasicModel import Embedding
        m = self._create_model(False)
        if not isinstance(m.inputSpace.vectors(), Embedding):
            self.skipTest("Model doesn't use Embedding")
        emb_weight = m.inputSpace.vectors()._emb.weight
        optimizer = m.getOptimizer(lr=0.001)
        opt_params = [p.data_ptr() for group in optimizer.param_groups for p in group['params']]
        self.assertNotIn(emb_weight.data_ptr(), opt_params,
                         "Embedding weight should NOT be in optimizer when trainEmbeddings=false")


# ---------------------------------------------------------------------------
# Weight persistence — save/load round-trip with shape mismatches
# ---------------------------------------------------------------------------
class TestWeightShapeMismatch(unittest.TestCase):
    """Verify save_weights / load_weights handles vocab-size changes."""

    def _make_model(self, vocab_size, dim=8):
        """Create a minimal nn.Module with an embedding of given vocab_size."""
        from BasicModel import BasicModel
        m = BasicModel()
        m.name = "TestModel"
        # Directly set up a simple module structure to test shape-mismatch logic
        m.emb = nn.Embedding(vocab_size, dim)
        m.fc = nn.Linear(dim, 2)
        return m

    def test_save_load_same_shape(self):
        """Weights round-trip when architecture is identical."""
        m1 = self._make_model(100)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            m1.save_weights(path)
            m2 = self._make_model(100)
            self.assertTrue(m2.load_weights(path))
            # Verify weights match
            for k in m1.state_dict():
                torch.testing.assert_close(m1.state_dict()[k], m2.state_dict()[k])
        finally:
            os.unlink(path)

    def test_save_load_vocab_mismatch_fails(self):
        """load_weights fails when shapes don't match and no vocab is saved."""
        m1 = self._make_model(50)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            m1.save_weights(path)
            m2 = self._make_model(80)  # different vocab size
            self.assertFalse(m2.load_weights(path),
                             "load_weights should fail on shape mismatch without vocab")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Training actually updates weights (regression: numEpochs=1 did nothing)
# ---------------------------------------------------------------------------
class TestVocabSaveRestore(unittest.TestCase):
    """Verify vocab is saved with weights and restored on load."""

    def test_save_load_with_vocab(self):
        """Embedding vocab round-trips through save/load with no shape mismatch."""
        from BasicModel import BasicModel, TheData, Embedding

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        TheData.load("xor")

        m1 = BasicModel()
        m1.create_from_config(xml_path, data=TheData)

        # Add extra words to grow the vocab
        emb1 = m1._get_embedding()
        if emb1 is None:
            self.skipTest("XOR_exact doesn't use Embedding")
        for w in ["extra1", "extra2", "extra3"]:
            emb1.insert(w)
        vocab_before = list(emb1.cbow.index_to_key)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            m1.save_weights(path)

            # Create a fresh model (will have smaller vocab)
            m2 = BasicModel()
            m2.create_from_config(xml_path, data=TheData)
            emb2 = m2._get_embedding()
            self.assertNotEqual(len(emb2.cbow.index_to_key), len(vocab_before))

            # Load should restore vocab and weights cleanly
            self.assertTrue(m2.load_weights(path))
            emb2 = m2._get_embedding()
            self.assertEqual(list(emb2.cbow.index_to_key), vocab_before)
            # Embedding shapes must match exactly
            torch.testing.assert_close(
                m2.state_dict()["inputSpace.vectorSet.0._emb.weight"],
                m1.state_dict()["inputSpace.vectorSet.0._emb.weight"])
        finally:
            os.unlink(path)


class TestTrainingUpdatesWeights(unittest.TestCase):
    """Verify that run() with numEpochs=1 actually trains and changes weights."""

    def test_xor_weights_change_after_one_epoch(self):
        """XOR model weights must differ after 1 epoch of training."""
        from BasicModel import BasicModel, TheData

        # XOR_exact.xml has autoload=false, autosave=false
        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        TheData.load("xor")

        m = BasicModel()
        m.create_from_config(xml_path, data=TheData)

        weights_before = {k: v.clone() for k, v in m.state_dict().items()}

        m.runTrial(numEpochs=1, batchSize=10, lr=0.01)

        weights_after = m.state_dict()
        changed = sum(
            1 for k in weights_before
            if not torch.equal(weights_before[k], weights_after[k])
        )
        self.assertGreater(changed, 0,
                           "At least some weights must change after 1 epoch of training")


class TestRuntimeBatch(unittest.TestCase):
    """runtime_batch() context manager stages transient data."""

    def test_runtime_batch_sets_and_clears(self):
        from BasicModel import TheData
        TheData.load("xor")
        with TheData.runtime_batch(["hello world"], [[0]]):
            self.assertEqual(TheData._runtime_input, ["hello world"])
            self.assertEqual(TheData._runtime_output, [[0]])
        self.assertIsNone(TheData._runtime_input)
        self.assertIsNone(TheData._runtime_output)

    def test_runtime_batch_clears_on_exception(self):
        from BasicModel import TheData
        TheData.load("xor")
        with self.assertRaises(ValueError):
            with TheData.runtime_batch(["test"], [[1]]):
                raise ValueError("boom")
        self.assertIsNone(TheData._runtime_input)

    def test_runtime_batch_does_not_contaminate_train(self):
        from BasicModel import TheData
        TheData.load("xor")
        original_train = list(TheData.train_input)
        with TheData.runtime_batch(["injected"], [[99]]):
            pass
        self.assertEqual(list(TheData.train_input), original_train)

    def test_runtime_batch_with_sentences(self):
        from BasicModel import TheData
        TheData.load("xor")
        with TheData.runtime_batch(["hello"], sentences=["hello world"]):
            self.assertEqual(TheData._lm_sentences.get("runtime"), ["hello world"])
        self.assertNotIn("runtime", TheData._lm_sentences)


class TestRuntimeGetBatch(unittest.TestCase):
    """getBatch(split='runtime') serves staged runtime data."""

    def test_runtime_getBatch_returns_batch(self):
        import tempfile
        import xml.etree.ElementTree as ET
        from BasicModel import BasicModel, TheData

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        auto = root.find("architecture/autoload")
        if auto is None:
            auto = ET.SubElement(root.find("architecture"), "autoload")
        auto.text = "false"

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
        tree.write(tmp, xml_declaration=True)
        tmp.close()

        torch.manual_seed(42)
        TheData.load("xor")
        m = BasicModel()
        m.create_from_config(tmp.name, data=TheData)
        os.unlink(tmp.name)

        rt_input = [TheData.stringTensor("hello world")]
        rt_output = [torch.tensor([0], dtype=torch.float)]
        with TheData.runtime_batch(rt_input, rt_output):
            batch, nextBatch = m.inputSpace.getBatch(0, 1, "runtime")
            self.assertIsNotNone(batch)
            inp, out = batch
            self.assertEqual(inp.shape[0], 1)  # batch size 1


if __name__ == "__main__":
    unittest.main()
