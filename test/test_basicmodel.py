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
# SimpleModel construction (renamed from DerivedModel, now in BasicModel.py)
# ---------------------------------------------------------------------------
class TestDerivedModel(unittest.TestCase):

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_derived_model_creation(self):
        from BasicModel import SimpleModel, DerivedModel
        model = SimpleModel()
        self.assertIsNotNone(model)
        # backward-compat alias
        self.assertIs(DerivedModel, SimpleModel)

    def test_derived_model_traditional(self):
        """SimpleModel with ergodic=False, certainty=False produces valid output."""
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic   = False
        model.certainty = False
        model.quantized = False
        model.create(nInput=28*28, nConcepts=20, nOutput=10)
        x = torch.randn(2, 28*28, 1)  # batch of 2, flattened MNIST, dim=1
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_derived_model_ergodic(self):
        """SimpleModel with ergodic=True uses SigmaLayer path."""
        from BasicModel import SimpleModel
        model = SimpleModel()
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


# ---------------------------------------------------------------------------
# Regression: Space shape contracts and SimpleModel
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
    """SimpleModel (renamed DerivedModel) uses unified Space hierarchy."""

    def setUp(self):
        from BasicModel import TheObjectEncoding
        self._orig_nWhere = TheObjectEncoding.nWhere
        self._orig_nWhen = TheObjectEncoding.nWhen
        self._orig_objectSize = TheObjectEncoding.objectSize
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_simple_model_ergodic_shapes(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic = True
        model.certainty = False
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(concepts.shape[0], 2)
        self.assertEqual(concepts.shape[1], 8)

    def test_simple_model_traditional_shapes(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic = False
        model.certainty = False
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_simple_model_reverse_shapes(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic = True
        model.certainty = False
        model.quantized = False
        model.reversePass = True
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        data, percepts = model.reverse(concepts)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)


class TestVectorSetVariants(unittest.TestCase):
    """Lock down quantized vs unquantized VectorSet behavior."""

    def test_unquantized_passthrough(self):
        from BasicModel import UnquantizedVSet
        vs = UnquantizedVSet()
        vs.create(4, 4, 1)
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
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0

    def tearDown(self):
        from BasicModel import TheObjectEncoding
        TheObjectEncoding.nWhere = self._orig_nWhere
        TheObjectEncoding.nWhen = self._orig_nWhen
        TheObjectEncoding.objectSize = self._orig_objectSize

    def test_derived_model_ergodic_shapes(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic = True
        model.certainty = False
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(concepts.shape[0], 2)
        self.assertEqual(concepts.shape[1], 8)

    def test_derived_model_traditional_shapes(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic = False
        model.certainty = False
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(concepts.shape[0], 2)
        self.assertEqual(concepts.shape[1], 8)

    def test_derived_model_reverse_shapes(self):
        from BasicModel import SimpleModel
        model = SimpleModel()
        model.ergodic = True
        model.certainty = False
        model.quantized = False
        model.reversePass = True
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        out, concepts = model.forward(x)
        data, percepts = model.reverse(concepts)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_derived_model_loss_runs(self):
        """Verify forward + loss + backward doesn't crash."""
        from BasicModel import SimpleModel, CertaintyWeightedCrossEntropy
        model = SimpleModel()
        model.ergodic = True
        model.certainty = True
        model.quantized = False
        model.create(nInput=16, nConcepts=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        target = torch.randn(2, 4)
        out, concepts = model.forward(x)
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

    def test_unquantized_creates_unquantized_vset(self):
        from BasicModel import Space, UnquantizedVSet
        s = Space([4, 3], [4, 3], 4, 3)
        s.createVectorSet(quantized=False)
        self.assertIsInstance(s.vectors(), UnquantizedVSet)

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


if __name__ == "__main__":
    unittest.main()
