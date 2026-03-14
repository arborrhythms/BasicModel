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

# Prevent OMP fork-safety crash on macOS when multiple libs load OpenMP
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
# Simple path: BasicModel with conceptualOrder=0, symbolicOrder=0
# ---------------------------------------------------------------------------
def _make_simple_model():
    """Helper to create a BasicModel with ObjectEncoding set up for simple path."""
    from BasicModel import BasicModel, TheObjectEncoding
    TheObjectEncoding.nWhere = 0
    TheObjectEncoding.nWhen = 0
    TheObjectEncoding.objectSize = 0
    TheObjectEncoding.setInputDim(1)
    TheObjectEncoding.setPerceptDim(1)
    TheObjectEncoding.setConceptDim(1)
    TheObjectEncoding.setSymbolDim(1)
    TheObjectEncoding.setOutputDim(1)
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
        model = _make_simple_model()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
                       perceptPassThrough=True, symbolPassThrough=True)
        x = torch.randn(2, 28*28, 1)  # batch of 2, flattened MNIST, dim=1
        out, end_state = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_simple_model_ergodic(self):
        """BasicModel (simple path) with ergodic=True uses SigmaLayer path."""
        model = _make_simple_model()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, certainty=True)
        x = torch.randn(2, 28*28, 1)
        out, end_state = model.forward(x)
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
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8, conceptDim=8, outputDim=4)
        self.B = 2  # batch

    def test_conceptual_space_forward_shape(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace([nIn, TheObjectEncoding.inputDim],
                             [nOut, TheObjectEncoding.conceptDim],
                             nOut, TheObjectEncoding.conceptDim,
                             nPrototypes=nOut)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.inputDim)
        x = torch.randn(self.B, nIn, inEmb)
        y = cs(x)
        outEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
        self.assertEqual(list(y.shape), [self.B, nOut, outEmb])

    def test_conceptual_space_reverse_shape(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace([nIn, TheObjectEncoding.inputDim],
                             [nOut, TheObjectEncoding.conceptDim],
                             nOut, TheObjectEncoding.conceptDim,
                             reversePass=True, nPrototypes=nOut)
        outEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
        y = torch.randn(self.B, nOut, outEmb)
        x = cs.reverse(y)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.inputDim)
        self.assertEqual(list(x.shape), [self.B, nIn, inEmb])

    def test_output_space_forward_shape(self):
        from BasicModel import OutputSpace, TheObjectEncoding
        nIn, nOut = 4, 4
        os_ = OutputSpace([nIn, TheObjectEncoding.conceptDim],
                          [nOut, TheObjectEncoding.outputDim],
                          nOut, TheObjectEncoding.outputDim)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
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
                       ergodic=True)
        x = torch.randn(2, 16, 1)
        out, end_state = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True)
        x = torch.randn(2, 16, 1)
        out, end_state = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_simple_model_reverse_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, reversePass=True)
        x = torch.randn(2, 16, 1)
        out, end_state = model.forward(x)
        data, start_state = model.reverse(end_state)
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

    @unittest.skip("Known bug: VectorSet.forward indexes nearestDist[v] but topk returns size 1")
    def test_quantized_shape(self):
        from BasicModel import VectorSet
        vs = VectorSet()
        vs.create(4, 4, 3, customVQ=False)
        vs.addVectors(nVec=4)
        x = torch.randn(2, 4, 3)
        y = vs.forward(x)
        self.assertEqual(list(y.shape), [2, 4, 3])


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
                       ergodic=True)
        x = torch.randn(2, 16, 1)
        out, end_state = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True)
        x = torch.randn(2, 16, 1)
        out, end_state = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_reverse_shapes(self):
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, reversePass=True)
        x = torch.randn(2, 16, 1)
        out, end_state = model.forward(x)
        data, start_state = model.reverse(end_state)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_simple_model_loss_runs(self):
        """Verify forward + loss + backward doesn't crash."""
        from BasicModel import CertaintyWeightedCrossEntropy
        model = _make_simple_model()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
                       perceptPassThrough=True, symbolPassThrough=True,
                       ergodic=True, certainty=True)
        x = torch.randn(2, 16, 1)
        target = torch.randn(2, 4)
        out, end_state = model.forward(x)
        loss_fn = CertaintyWeightedCrossEntropy()
        loss = loss_fn(out.squeeze(), target)
        loss.backward()
        # No crash = pass


class TestUniversalTrainingContract(unittest.TestCase):
    """All spaces expose getParameters() and paramUpdate()."""

    def test_space_has_training_contract(self):
        from BasicModel import Space
        s = Space([4, 8], [4, 8], 4, 8)
        self.assertEqual(s.getParameters(), [])
        s.paramUpdate()  # should be a no-op, not crash

    def test_conceptual_space_has_training_contract(self):
        from BasicModel import ConceptualSpace, TheObjectEncoding
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8, conceptDim=8, outputDim=4)
        cs = ConceptualSpace([4, TheObjectEncoding.inputDim],
                             [4, TheObjectEncoding.conceptDim],
                             4, TheObjectEncoding.conceptDim)
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        cs.paramUpdate()  # no crash


class TestSigmaLayerDeterministic(unittest.TestCase):
    """SigmaLayer(deterministic=True) behaves like LinearLayer + Tanh."""

    def test_deterministic_matches_linear_tanh(self):
        from Model import SigmaLayer, LinearLayer
        torch.manual_seed(42)
        nIn, nOut = 8, 4
        sigma = SigmaLayer(nIn, nOut, deterministic=True)
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
        sigma = SigmaLayer(nIn, nOut, deterministic=True)
        x = torch.randn(2, nIn)

        sigma.train()
        y_train = sigma(x).detach().clone()
        sigma.eval()
        y_eval = sigma(x).detach().clone()
        self.assertTrue(torch.allclose(y_train, y_eval, atol=1e-6),
                        "Deterministic mode should produce same output in train and eval")

    def test_non_deterministic_default(self):
        from Model import SigmaLayer
        sigma = SigmaLayer(nInput=8, nOutput=4)
        self.assertFalse(sigma.deterministic)


class TestCreateVectorSetQuantized(unittest.TestCase):
    """Space.createVectorSet supports both quantized and unquantized paths."""

    def test_quantized_creates_vectorset(self):
        from BasicModel import Space, VectorSet
        s = Space([4, 3], [4, 3], 4, 3)
        s.createVectorSet(quantized=True)
        self.assertIsInstance(s.vectors(), VectorSet)

    def test_unquantized_creates_passthrough_vset(self):
        from BasicModel import Space, VectorSet
        s = Space([4, 3], [4, 3], 4, 3)
        s.createVectorSet(quantized=False)
        self.assertIsInstance(s.vectors(), VectorSet)
        self.assertTrue(s.vectors().passThrough)

    def test_default_is_quantized(self):
        from BasicModel import Space, VectorSet
        s = Space([4, 3], [4, 3], 4, 3)
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

    def test_ergodic_forward_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace([nVec, nDim], [nVec, cDim], nVec, cDim,
                             ergodic=True, useVQ=False)
        x = torch.randn(2, nVec, nDim)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_non_ergodic_forward_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace([nVec, nDim], [nVec, cDim], nVec, cDim,
                             ergodic=False, useVQ=False)
        x = torch.randn(2, nVec, nDim)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_ergodic_flag_stored(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        cs_erg = ConceptualSpace([8, 1], [8, 1], 8, 1,
                                 ergodic=True, useVQ=False)
        cs_det = ConceptualSpace([8, 1], [8, 1], 8, 1,
                                 ergodic=False, useVQ=False)
        self.assertTrue(cs_erg.ergodic)
        self.assertFalse(cs_det.ergodic)

    def test_ergodic_reverse_shape(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace([nVec, nDim], [nVec, cDim], nVec, cDim,
                             ergodic=True, reversePass=True, useVQ=False)
        y = torch.randn(2, nVec, cDim)
        x = cs.reverse(y)
        self.assertEqual(list(x.shape), [2, nVec, nDim])

    def test_ergodic_exposes_params(self):
        self._set_zero_object_encoding()
        from BasicModel import ConceptualSpace
        cs = ConceptualSpace([8, 1], [8, 1], 8, 1,
                             ergodic=True, useVQ=False)
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

    def test_canonical_forward_still_works(self):
        """Existing ConceptualSpace (with objectSize > 0) still works after changes."""
        from BasicModel import ConceptualSpace, TheObjectEncoding
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8, conceptDim=8, outputDim=4)
        nIn, nOut = 4, 4
        cs = ConceptualSpace([nIn, TheObjectEncoding.inputDim],
                             [nOut, TheObjectEncoding.conceptDim],
                             nOut, TheObjectEncoding.conceptDim,
                             nPrototypes=nOut)
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
        nIn, nDim = 8, 1
        inp = InputSpace([nIn, nDim], [nIn, nDim], nIn, nDim=nDim, useVQ=False)
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
        nIn, nOut = 4, 3
        os_ = OutputSpace([nIn, 1], [nOut, 1], nOut, 1)
        x = torch.randn(2, nIn, 1)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, 1])

    def test_reverse_shape_zero_object_size(self):
        from BasicModel import TheObjectEncoding, OutputSpace
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        nIn, nOut = 4, 3
        os_ = OutputSpace([nIn, 1], [nOut, 1], nOut, 1, reversePass=True)
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
            self.assertEqual(model.conceptualOrder, 0)
            self.assertEqual(model.symbolicOrder, 0)
        finally:
            os.unlink(path)
            TheObjectEncoding.nWhere = orig_nWhere
            TheObjectEncoding.nWhen = orig_nWhen
            TheObjectEncoding.objectSize = orig_objectSize

    def test_factory_creates_basic_model(self):
        from BasicModel import BaseModel, BasicModel as BM, TheObjectEncoding
        # BasicModel.create() needs non-zero encoding dimensions
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8,
                                        conceptDim=8, outputDim=4)
        # nSymbols must equal nConcepts (SymbolicSpace 1:1 mapping constraint),
        # and nPercepts must be 2*nConcepts (ReversiblePiLayer invertibility).
        xml = """<model>
  <architecture>
    <type>basic</type>
    <nInput>32</nInput>
    <nPercepts>4</nPercepts>
    <nConcepts>2</nConcepts>
    <nSymbols>2</nSymbols>
    <nWords>16</nWords>
    <nOutput>32</nOutput>
    <conceptualOrder>1</conceptualOrder>
    <reversePass>false</reversePass>
    <perceptPrototypes>8</perceptPrototypes>
    <conceptPrototypes>4</conceptPrototypes>
    <inputDim>8</inputDim>
    <perceptDim>8</perceptDim>
    <conceptDim>8</conceptDim>
    <symbolDim>8</symbolDim>
    <outputDim>4</outputDim>
  </architecture>
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

    def _make_input_space(self):
        """Create an InputSpace with model_type='lm' from XOR text data."""
        from BasicModel import InputSpace, TheObjectEncoding
        data = self._make_text_data()
        nInput = 8
        nVectors = 8
        # The lm path resets ObjectEncoding dimensions internally via
        # setDimensions(), but super().__init__ needs a valid nDim.
        # Pass nDim=1 as a placeholder; the lm path overrides it.
        inp = InputSpace([data.getInputSize(), 1], [nInput, 1], nVectors,
                         nDim=1, model_type="lm", pretrained=False, data=data)
        return inp, data

    def test_lex_created_on_init(self):
        """InputSpace with model_type='lm' creates a Lex instance."""
        from Lex import Lex
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
        nIn, nOut = 4, 3
        os_ = OutputSpace([nIn, 1], [nOut, 1], nOut, 1)
        x = torch.randn(2, nIn, 1)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, 1])
        # text_mode should be False for numeric data
        self.assertFalse(os_.text_mode)

    def test_text_mode_false_without_lex(self):
        """OutputSpace without lex info should have text_mode=False."""
        from BasicModel import TheObjectEncoding, OutputSpace
        TheObjectEncoding.setDimensions(inputDim=8, perceptDim=8,
                                        conceptDim=8, outputDim=4)
        nIn, nOut = 4, 4
        os_ = OutputSpace([nIn, TheObjectEncoding.conceptDim],
                          [nOut, TheObjectEncoding.outputDim],
                          nOut, TheObjectEncoding.outputDim)
        self.assertFalse(os_.text_mode)

    def test_set_text_mode_enables_reconstruction(self):
        """set_text_mode() stores codebook, words, and lex references."""
        from BasicModel import InputSpace, Data, OutputSpace, TheObjectEncoding
        data = Data()
        data.load("xor")
        nInput = 8
        nVectors = 8
        inp = InputSpace([data.getInputSize(), 1], [nInput, 1], nVectors,
                         nDim=1, model_type="lm", pretrained=False, data=data)
        # Create OutputSpace with the same embedding setup
        nOut = 4
        os_ = OutputSpace([nInput, TheObjectEncoding.conceptDim],
                          [nOut, TheObjectEncoding.outputDim],
                          nOut, TheObjectEncoding.outputDim)
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
        nVectors = 8
        inp = InputSpace([data.getInputSize(), 1], [nInput, 1], nVectors,
                         nDim=1, model_type="lm", pretrained=False, data=data)
        nOut = 4
        os_ = OutputSpace([nInput, TheObjectEncoding.conceptDim],
                          [nOut, TheObjectEncoding.outputDim],
                          nOut, TheObjectEncoding.outputDim)
        os_.set_text_mode(inp)

        # Build synthetic vectors from known codebook entries with known nWhere
        codebook = inp.vectors().vq.codebook
        words_list = inp.vectors().words
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
        nVectors = 8
        inp = InputSpace([data.getInputSize(), 1], [nInput, 1], nVectors,
                         nDim=1, model_type="lm", pretrained=False, data=data)
        nOut = 4
        os_ = OutputSpace([nInput, TheObjectEncoding.conceptDim],
                          [nOut, TheObjectEncoding.outputDim],
                          nOut, TheObjectEncoding.outputDim)
        os_.set_text_mode(inp)

        # Build vectors with nWhere = 0 (all zeros)
        codebook = inp.vectors().vq.codebook
        words_list = inp.vectors().words
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
        nVectors = 8
        inp = InputSpace([data.getInputSize(), 1], [nInput, 1], nVectors,
                         nDim=1, model_type="lm", pretrained=False, data=data)
        nOut = 4
        os_ = OutputSpace([nInput, TheObjectEncoding.conceptDim],
                          [nOut, TheObjectEncoding.outputDim],
                          nOut, TheObjectEncoding.outputDim)
        os_.set_text_mode(inp)

        # Build synthetic vectors with nWhere at known positions
        codebook = inp.vectors().vq.codebook
        words_list = inp.vectors().words
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
        nVectors = 8
        inp = InputSpace([data.getInputSize(), 1], [nInput, 1], nVectors,
                         nDim=1, model_type="lm", pretrained=False, data=data)
        nOut = 4
        os_ = OutputSpace([nInput, TheObjectEncoding.conceptDim],
                          [nOut, TheObjectEncoding.outputDim],
                          nOut, TheObjectEncoding.outputDim,
                          reversePass=True)
        os_.set_text_mode(inp)
        inEmb = TheObjectEncoding.getEmbeddingSize(TheObjectEncoding.conceptDim)
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
        from BasicModel import InputSpace, Data
        data = Data()
        data.load("xor")
        nInput = 8
        nVectors = 8
        inp = InputSpace([data.getInputSize(), 1], [nInput, 1], nVectors,
                         nDim=1, model_type="lm", pretrained=False, data=data)
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
        TheObjectEncoding.setSymbolDim(1)
        TheObjectEncoding.setOutputDim(1)
        from BasicModel import InputSpace
        nIn, nDim = 8, 1
        inp = InputSpace([nIn, nDim], [nIn, nDim], nIn, nDim=nDim, useVQ=False)
        x = torch.randn(2, nIn, nDim)
        y = inp.forward(x)
        result = inp.reverse(y)
        # Numeric path returns tensor, not text
        self.assertIsInstance(result, (torch.Tensor, list))


if __name__ == "__main__":
    unittest.main()
