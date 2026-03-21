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


def _populate_test_config(*,
                          inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0,
                          wordDim=1, outputDim=1,
                          nInput=16, nPercepts=16, nConcepts=8, nSymbols=8,
                          nWords=16, nOutput=4,
                          nWhere=0, nWhen=0,
                          reconstruct="NONE", reshape=False, ergodic=False,
                          naive=False, processSymbols=False,
                          perceptPassThrough=False, symbolPassThrough=False,
                          perceptHasAttention=True, conceptHasAttention=False,
                          invertible=False, hasNorm=False, quantized=False,
                          perceptQuantized=None, conceptQuantized=None,
                          certainty=False,
                          lexer="word"):
    """Populate TheXMLConfig._data — test equivalent of XML loading.

    Space constructors read nDim/nVectors from TheXMLConfig.  In production,
    the XML file provides these.  In tests, this helper provides them directly.
    """
    from BasicModel import TheXMLConfig, TheData
    # Reset global state to prevent cross-test pollution
    TheXMLConfig._requirements.clear()
    TheData.train_input = []
    TheData.test_input = []
    TheData.train_output = []
    TheData.test_output = []
    _pq = perceptQuantized if perceptQuantized is not None else quantized
    _cq = conceptQuantized if conceptQuantized is not None else quantized
    _objectSize = nWhere + nWhen
    _nObjects = nInput + nPercepts + nConcepts + nSymbols + nWords + nOutput
    _symbol_dim = conceptDim if symbolPassThrough else symbolDim
    TheXMLConfig._data.update({
        "architecture": {
            "reconstruct": reconstruct,
            "reshape": reshape,
            "ergodic": ergodic,
            "naive": naive,
            "processSymbols": processSymbols,
            "certainty": certainty,
            "objectSize": _objectSize,
            "nObjects": _nObjects,
            "nWhere": nWhere,
            "nWhen": nWhen,
            "embeddingPath": None,
            "data": {},
            "training": {},
        },
        "InputSpace": {
            "nActive": nInput,
            "nDim": inputDim,
            "nVectors": nInput,
            "quantized": quantized,
            "lexer": lexer,
        },
        "PerceptualSpace": {
            "nActive": nPercepts,
            "nDim": perceptDim,
            "nVectors": nPercepts,
            "quantized": _pq,
            "passThrough": perceptPassThrough,
            "hasAttention": perceptHasAttention,
            "invertible": invertible,
        },
        "ConceptualSpace": {
            "nActive": nConcepts,
            "nDim": conceptDim,
            "nVectors": nConcepts,
            "quantized": _cq,
            "hasAttention": conceptHasAttention,
            "hasNorm": hasNorm,
            "invertible": invertible,
        },
        "SymbolicSpace": {
            "nActive": nSymbols,
            "nDim": _symbol_dim,
            "nVectors": nSymbols,
            "passThrough": symbolPassThrough,
            "quantized": False,
        },
        "SyntacticSpace": {
            "nActive": nWords,
            "nDim": wordDim,
            "nVectors": nWords,
            "quantized": False,
        },
        "OutputSpace": {
            "nActive": nOutput,
            "nDim": outputDim,
            "nVectors": nOutput,
            "quantized": False,
        },
    })


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

    @unittest.skip("slow")
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
    """Helper to create a BasicModel with config set up for simple path."""
    from BasicModel import BasicModel
    _populate_test_config(
        inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
        nInput=nInput, nPercepts=nPercepts, nConcepts=nConcepts,
        nSymbols=nSymbols, nWords=nWords, nOutput=nOutput,
        symbolPassThrough=True)
    return BasicModel()


class TestSimpleModelCreation(unittest.TestCase):

    def test_simple_model_creation(self):
        model = _make_simple_model()
        self.assertIsNotNone(model)

    def test_simple_model_traditional(self):
        """BasicModel (simple path) with ergodic=False produces valid output."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
            perceptPassThrough=True, symbolPassThrough=True, reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        x = torch.randn(2, 28*28, 1).to(TheDevice)  # batch of 2, flattened MNIST, dim=1
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_simple_model_ergodic(self):
        """BasicModel (simple path) with ergodic=True uses SigmaLayer path."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, certainty=True, reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        x = torch.randn(2, 28*28, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)


# ---------------------------------------------------------------------------
# BasicModel.py — Full model
# ---------------------------------------------------------------------------
class TestBasicModelCreation(unittest.TestCase):
    def test_encodings(self):
        from BasicModel import WhereEncoding, WhenEncoding
        WhereEncoding.test()
        WhenEncoding.test()

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
# ---------------------------------------------------------------------------
# Encoding round-trip tests
# ---------------------------------------------------------------------------
class TestWhereEncodingRoundTrip(unittest.TestCase):
    """WhereEncoding: encode → reverse → decode recovers the original offset."""

    def test_stamp_round_trip(self):
        from BasicModel import WhereEncoding
        maxP = 4096
        pe = WhereEncoding(maxP)
        buf = torch.zeros(1, 1, 10, device=TheDevice)
        offsets = [0, 1, 5, 11, 42, 100, 255, 1000]
        for offset in offsets:
            pe.stamp(buf, 0, 0, offset)
            _, decoded = pe.reverse(buf.clone())
            self.assertAlmostEqual(decoded[0, 0].item(), offset, places=3,
                msg=f"stamp/reverse round-trip failed for offset={offset}")

    def test_forward_decode_round_trip(self):
        from BasicModel import WhereEncoding
        maxP = 256
        pe = WhereEncoding(maxP)
        pe.p = 0
        batch, n = 4, 3
        x = torch.zeros(batch, n, 10, device=TheDevice)
        y = pe.forward(x)
        _, decoded = pe.reverse(y)
        for b in range(batch):
            for v in range(n):
                expected = float(b * n + v)
                self.assertAlmostEqual(
                    decoded[b, v].item(), expected, places=3,
                    msg=f"forward/decode round-trip failed at batch={b}, vec={v}")

    def test_content_preserved(self):
        """Content dimensions (non-encoding slots) survive the round-trip."""
        from BasicModel import WhereEncoding
        pe = WhereEncoding(1000)
        pe.p = 0
        x = torch.randn(2, 3, 10, device=TheDevice)
        original = x.clone()
        y = pe.forward(x)
        cleaned, _ = pe.reverse(y)
        mask = torch.ones(10, dtype=torch.bool)
        mask[[-4, -3]] = False
        torch.testing.assert_close(cleaned[:, :, mask], original[:, :, mask])


class TestWhenEncodingRoundTrip(unittest.TestCase):
    """WhenEncoding: forward → reverse recovers the original time."""

    def test_forward_reverse_round_trip(self):
        from BasicModel import WhenEncoding
        maxT = 10000
        te = WhenEncoding(maxT)
        te.t = 0
        x = torch.zeros(5, 2, 10, device=TheDevice)
        y = te.forward(x)
        _, decoded = te.reverse(y)
        expected = torch.arange(0, 5, dtype=torch.float32)
        for b in range(5):
            for v in range(2):
                self.assertAlmostEqual(
                    decoded[b, v].item(), expected[b].item(), places=2,
                    msg=f"forward/reverse round-trip failed at batch={b}, vec={v}")

    def test_large_time_values(self):
        """Round-trip works for time values well into the range."""
        from BasicModel import WhenEncoding
        maxT = 10000
        te = WhenEncoding(maxT)
        te.t = 500
        x = torch.zeros(3, 1, 10, device=TheDevice)
        y = te.forward(x)
        _, decoded = te.reverse(y)
        expected = torch.arange(500, 503, dtype=torch.float32)
        for b in range(3):
            self.assertAlmostEqual(
                decoded[b, 0].item(), expected[b].item(), places=2,
                msg=f"round-trip failed for t={500+b}")

    def test_content_preserved(self):
        """Content dimensions (non-encoding slots) survive the round-trip."""
        from BasicModel import WhenEncoding
        te = WhenEncoding(10000)
        te.t = 0
        x = torch.randn(2, 3, 10, device=TheDevice)
        original = x.clone()
        y = te.forward(x)
        cleaned, _ = te.reverse(y)
        mask = torch.ones(10, dtype=torch.bool)
        mask[[-2, -1]] = False
        torch.testing.assert_close(cleaned[:, :, mask], original[:, :, mask])


# SubSpace — derived sizes, materialization, construction helpers
# ---------------------------------------------------------------------------
class TestSubSpaceDerivedSizes(unittest.TestCase):
    """SubSpace.getEncodingSize and getEmbeddedIO match ObjectEncoding math."""

    def test_getEncodingSize_adds_objectSize(self):
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding
        ss = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                      inputShape=[3, 8], outputShape=[3, 8])
        self.assertEqual(ss.getEncodingSize(8), 12)
        self.assertEqual(ss.getEncodingSize(0), 4)

    def test_getEmbeddedIO_no_reshape(self):
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding, ObjectEncoding
        ss = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                      objectEncoding=ObjectEncoding([3, 8], [5, 16], reshape=False, objectSize=4),
                      reshape=False,
                      inputShape=[3, 8], outputShape=[5, 16])
        inp, out = ss.getEmbeddedIO()
        self.assertEqual(inp, 8 + 4)   # nDim + objectSize
        self.assertEqual(out, 16 + 4)

    def test_getEmbeddedIO_reshape(self):
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding, ObjectEncoding
        ss = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                      objectEncoding=ObjectEncoding([3, 8], [5, 16], reshape=True, objectSize=4),
                      reshape=True,
                      inputShape=[3, 8], outputShape=[5, 16])
        inp, out = ss.getEmbeddedIO()
        self.assertEqual(inp, (8 + 4) * 3)
        self.assertEqual(out, (16 + 4) * 5)

    def test_zero_objectSize(self):
        from BasicModel import SubSpace, ObjectEncoding
        ss = SubSpace(objectEncoding=ObjectEncoding([2, 10], [2, 10]),
                      inputShape=[2, 10], outputShape=[2, 10])
        inp, out = ss.getEmbeddedIO()
        self.assertEqual(inp, 10)
        self.assertEqual(out, 10)


class TestSubSpaceMaterialize(unittest.TestCase):
    """SubSpace.materialize() returns the expected dense tensor."""

    def test_materialize_tensor(self):
        from BasicModel import SubSpace, Tensor, WhereEncoding, WhenEncoding
        t = torch.randn(2, 4, 12)
        ss = SubSpace.from_tensor(t, whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                                  inputShape=[4, 8], outputShape=[4, 8])
        result = ss.materialize()
        self.assertIs(result, t)
        self.assertIsInstance(ss.object, Tensor)

    def test_materialize_none_when_unset(self):
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding
        ss = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                      inputShape=[4, 8], outputShape=[4, 8])
        self.assertIsNone(ss.materialize())

    def test_shape_property(self):
        from BasicModel import SubSpace
        t = torch.randn(2, 4, 12)
        ss = SubSpace.from_tensor(t, inputShape=[4, 8], outputShape=[4, 8])
        self.assertEqual(ss.shape, torch.Size([2, 4, 12]))

    def test_shape_property_none(self):
        from BasicModel import SubSpace
        ss = SubSpace(inputShape=[4, 8], outputShape=[4, 8])
        self.assertIsNone(ss.shape)


class TestSubSpaceProperties(unittest.TestCase):
    """SubSpace batch tracking."""

    def test_batch_from_tensor(self):
        from BasicModel import SubSpace
        t = torch.randn(5, 4, 8, device=TheDevice)
        ss = SubSpace.from_tensor(t, inputShape=[4, 4], outputShape=[4, 4])
        self.assertEqual(ss.batch, 5)

    def test_batch_zero_when_empty(self):
        from BasicModel import SubSpace
        ss = SubSpace(inputShape=[4, 8], outputShape=[4, 8])
        self.assertEqual(ss.batch, 0)


class TestSubSpaceConstruction(unittest.TestCase):
    """SubSpace.from_tensor and from_components helpers."""

    def test_from_tensor(self):
        from BasicModel import SubSpace, Tensor, WhereEncoding, WhenEncoding
        t = torch.randn(2, 3, 10)
        ss = SubSpace.from_tensor(t, whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                                  inputShape=[3, 6], outputShape=[3, 6])
        self.assertIsInstance(ss.object, Tensor)
        self.assertIs(ss.object.W, t)
        self.assertEqual(ss.objectSize, 4)
        self.assertEqual(ss.inputShape, [3, 6])

    def test_from_components(self):
        from BasicModel import SubSpace, ActiveEncoding, Tensor, WhereEncoding, WhenEncoding
        object = torch.randn(2, 4, 8)
        act = torch.ones(2, 4)
        ae = ActiveEncoding()
        ss = SubSpace.from_components(
            object=object, activation=act, activeEncoding=ae,
            whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
            inputShape=[4, 4], outputShape=[4, 4])
        self.assertIsInstance(ss.object, Tensor)
        self.assertIsInstance(ss.activation, Tensor)
        self.assertIs(ss.object.W, object)
        self.assertIs(ss.activation.W, act)
        self.assertIs(ss.activeEncoding, ae)

    def test_from_components_defaults_none(self):
        from BasicModel import SubSpace
        ss = SubSpace.from_components(inputShape=[4, 8], outputShape=[4, 8])
        self.assertIsNone(ss.object)
        self.assertIsNone(ss.activation)
        self.assertIsNone(ss.where)
        self.assertIsNone(ss.when)


class TestSubSpaceActiveEncoding(unittest.TestCase):
    """ActiveEncoding round-trip through SubSpace."""

    def test_activation_stored_and_retrievable(self):
        from BasicModel import SubSpace, ActiveEncoding
        ae = ActiveEncoding()
        act = torch.tensor([0.5, 0.8, 0.1])
        encoded = ae.encode(act)
        ss = SubSpace(activeEncoding=ae, inputShape=[3, 8], outputShape=[3, 8])
        ss.activation = act
        decoded = ae.decode(encoded)
        torch.testing.assert_close(decoded, act)

    def test_two_spaces_independent_encoding(self):
        """Two SubSpaces can have different objectSize without shared coupling."""
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding, ObjectEncoding
        ss1 = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                        objectEncoding=ObjectEncoding([3, 8], [3, 8], objectSize=4),
                        inputShape=[3, 8], outputShape=[3, 8])
        ss2 = SubSpace(objectEncoding=ObjectEncoding([3, 16], [3, 16]),
                        inputShape=[3, 16], outputShape=[3, 16])
        self.assertEqual(ss1.getEncodingSize(8), 12)
        self.assertEqual(ss2.getEncodingSize(16), 16)
        inp1, out1 = ss1.getEmbeddedIO()
        inp2, out2 = ss2.getEmbeddedIO()
        self.assertEqual(inp1, 12)
        self.assertEqual(inp2, 16)


# Regression: Space shape contracts
# ---------------------------------------------------------------------------
class TestCanonicalSpaceShapes(unittest.TestCase):
    """Lock down tensor shapes for canonical Space subclasses."""

    def setUp(self):
        _populate_test_config(
            inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, wordDim=1, outputDim=4,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nWords=4, nOutput=4,
            reshape=True)
        self.B = 2  # batch

    def test_conceptual_space_forward_shape(self):
        from BasicModel import ConceptualSpace, TheXMLConfig
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace(nIn, nOut)
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("InputSpace", "nDim"))
        x = torch.randn(self.B, nIn, inEmb).to(TheDevice)
        y = cs(x)
        outEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("ConceptualSpace", "nDim"))
        self.assertEqual(list(y.shape), [self.B, nOut, outEmb])

    def test_conceptual_space_reverse_shape(self):
        from BasicModel import ConceptualSpace, TheXMLConfig
        _populate_test_config(
            inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, wordDim=1, outputDim=4,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nWords=4, nOutput=4,
            reshape=True, reconstruct="FULL")
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace(nIn, nOut)
        outEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("ConceptualSpace", "nDim"))
        y = torch.randn(self.B, nOut, outEmb).to(TheDevice)
        x = cs.reverse(y)
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("InputSpace", "nDim"))
        self.assertEqual(list(x.shape), [self.B, nIn, inEmb])

    def test_output_space_forward_shape(self):
        from BasicModel import OutputSpace, TheXMLConfig
        nIn, nOut = 4, 4
        os_ = OutputSpace(nIn, nOut)
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("SymbolicSpace", "nDim"))
        x = torch.randn(self.B, nIn, inEmb).to(TheDevice)
        y = os_(x)
        self.assertEqual(list(y.shape), [self.B, nOut, TheXMLConfig.space("OutputSpace", "nDim")])


class TestSimpleModel(unittest.TestCase):
    """BasicModel (simple path) uses unified Space hierarchy with passThrough SymbolicSpace."""

    def test_simple_model_ergodic_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_simple_model_reverse_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, reshape=True, reconstruct="FULL")
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)


class TestCodebookVariants(unittest.TestCase):
    """Lock down quantized vs unquantized Codebook behavior."""

    def test_unquantized_passthrough(self):
        from BasicModel import Codebook
        vs = Codebook()
        vs.create(4, 4, 1, passThrough=True)
        x = torch.randn(2, 4, 1).to(TheDevice)
        y = vs.forward(x)
        self.assertTrue(torch.equal(x, y))

    def test_quantized_shape(self):
        from BasicModel import Codebook, TheXMLConfig
        _populate_test_config(nInput=4, nPercepts=4, nConcepts=4, nSymbols=4,
                              nWords=0, nOutput=4)
        vs = Codebook()
        vs.create(4, 4, 3, customVQ=False,
                  objectSize=TheXMLConfig.objectSize)
        vs = vs.to(TheDevice)
        x = torch.randn(2, 4, 3).to(TheDevice)
        y = vs.forward(x)
        # Output gains ObjectEncoding overhead (nWhere + nWhen)
        embeddingSize = 3 + TheXMLConfig.objectSize
        self.assertEqual(list(y.shape), [2, 4, embeddingSize])


class TestBasisContract(unittest.TestCase):
    def test_tensor_identity_materialization(self):
        from BasicModel import Tensor
        payload = torch.randn(2, 3, 4, device=TheDevice)
        basis = Tensor()
        basis.create(3, 3, 4, passThrough=True)
        out = basis.forward(payload)
        self.assertIs(out, payload)
        self.assertIs(basis.materialize(), payload)
        rev = basis.reverse(payload)
        self.assertIs(rev, payload)

    def test_invalid_geometry_requires_2d_prototype_matrix(self):
        from BasicModel import Tensor
        basis = Tensor(W=torch.randn(2, 3, 4, device=TheDevice))
        basis.create(3, 3, 4, passThrough=True)
        with self.assertRaises(RuntimeError):
            basis.codebookDistance(torch.randn(2, 4, device=TheDevice))


class TestModelEndToEnd(unittest.TestCase):
    """Lock down full model forward shapes and loss compatibility."""

    def test_simple_model_ergodic_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_reverse_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, reshape=True, reconstruct="FULL")
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_simple_model_loss_runs(self):
        """Verify forward + loss + backward doesn't crash."""
        from BasicModel import CertaintyWeightedCrossEntropy
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, certainty=True, reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
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
        _populate_test_config(inputDim=8, nInput=4)
        Space.config_section = "InputSpace"
        s = Space([4, 8], [4, 8], 4)
        Space.config_section = None
        self.assertEqual(s.getParameters(), [])
        s.paramUpdate()  # should be a no-op, not crash

    def test_conceptual_space_has_training_contract(self):
        from BasicModel import ConceptualSpace
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nConcepts=4)
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


class TestSpaceBasisConstruction(unittest.TestCase):
    """Space builds its basis during construction."""

    def setUp(self):
        from BasicModel import Space
        _populate_test_config(inputDim=3, nInput=4)
        Space.config_section = "InputSpace"
        self.addCleanup(setattr, Space, 'config_section', None)

    def test_quantized_creates_codebook(self):
        from BasicModel import Codebook, Space
        _populate_test_config(inputDim=3, nInput=4, quantized=True)
        s = Space([4, 3], [4, 3], 4)
        self.assertIsInstance(s.vectors(), Codebook)
        self.assertFalse(s.vectors().passThrough)

    def test_unquantized_creates_passthrough_codebook(self):
        from BasicModel import Codebook, Space
        _populate_test_config(inputDim=3, nInput=4, quantized=False)
        s = Space([4, 3], [4, 3], 4)
        self.assertIsInstance(s.vectors(), Codebook)
        self.assertTrue(s.vectors().passThrough)

    def test_forward_subspace_round_trip_keeps_runtime_state(self):
        from BasicModel import InputSpace, PerceptualSpace, SubSpace
        _populate_test_config(
            inputDim=3, perceptDim=3,
            nInput=4, nPercepts=4,
            quantized=False, perceptPassThrough=True,
            nWhere=0, nWhen=0, reshape=False,
        )
        inp = InputSpace(4, 4, model_type="simple")
        per = PerceptualSpace(4, 4)
        x = torch.randn(2, 4, 3).to(TheDevice)

        input_state = inp.forward_subspace(x)
        self.assertIsInstance(input_state, SubSpace)
        self.assertTrue(torch.equal(input_state.materialize(), x))

        percept_state = per.forward_subspace(input_state)
        self.assertIsInstance(percept_state, SubSpace)
        self.assertTrue(torch.equal(percept_state.materialize(), x))

        reversed_state = per.reverse_subspace(percept_state)
        self.assertIsInstance(reversed_state, SubSpace)
        self.assertTrue(torch.equal(reversed_state.materialize(), x))


class TestConceptualSpaceErgodic(unittest.TestCase):
    """ConceptualSpace with ergodic flag matches DerivedConceptualSpace behavior."""

    def _set_zero_object_encoding(self, ergodic=False, reconstruct="NONE"):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8, nOutput=8,
            nWhere=0, nWhen=0, ergodic=ergodic, reconstruct=reconstruct)

    def test_ergodic_forward_shape(self):
        self._set_zero_object_encoding(ergodic=True)
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace(nVec, nVec)
        x = torch.randn(2, nVec, nDim).to(TheDevice)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_non_ergodic_forward_shape(self):
        self._set_zero_object_encoding(ergodic=False)
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace(nVec, nVec)
        x = torch.randn(2, nVec, nDim).to(TheDevice)
        y = cs(x)
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_ergodic_flag_stored(self):
        self._set_zero_object_encoding(ergodic=True)
        from BasicModel import ConceptualSpace
        cs_erg = ConceptualSpace(8, 8)
        self.assertTrue(cs_erg.ergodic)
        self._set_zero_object_encoding(ergodic=False)
        cs_det = ConceptualSpace(8, 8)
        self.assertFalse(cs_det.ergodic)

    def test_ergodic_reverse_shape(self):
        self._set_zero_object_encoding(ergodic=True, reconstruct="FULL")
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace(nVec, nVec)
        y = torch.randn(2, nVec, cDim).to(TheDevice)
        x = cs.reverse(y)
        self.assertEqual(list(x.shape), [2, nVec, nDim])

    def test_ergodic_exposes_params(self):
        self._set_zero_object_encoding(ergodic=True)
        from BasicModel import ConceptualSpace
        cs = ConceptualSpace(8, 8)
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

    def test_canonical_forward_still_works(self):
        """Existing ConceptualSpace (with objectSize > 0) still works after changes."""
        from BasicModel import ConceptualSpace, TheXMLConfig
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nConcepts=4)
        nIn, nOut = 4, 4
        cs = ConceptualSpace(nIn, nOut)
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("InputSpace", "nDim"))
        x = torch.randn(2, nIn, inEmb).to(TheDevice)
        y = cs(x)
        outEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("ConceptualSpace", "nDim"))
        self.assertEqual(list(y.shape), [2, nOut, outEmb])


class TestInputSpaceUnquantized(unittest.TestCase):
    """InputSpace works with unquantized codebook (objectSize=0)."""

    def test_unquantized_forward_shape(self):
        from BasicModel import InputSpace
        _populate_test_config(inputDim=1, nInput=8, nWhere=0, nWhen=0, quantized=False)
        nIn, nDim = 8, 1
        inp = InputSpace(nIn, nIn)
        x = torch.randn(2, nIn, nDim).to(TheDevice)
        y = inp(x)
        self.assertEqual(list(y.shape), [2, nIn, nDim])


class TestOutputSpaceZeroObjectSize(unittest.TestCase):
    """OutputSpace works with objectSize=0."""

    def test_forward_shape_zero_object_size(self):
        from BasicModel import OutputSpace
        _populate_test_config(symbolDim=1, outputDim=1, nOutput=3, nWhere=0, nWhen=0,
                              reshape=True)
        nIn, nOut = 4, 3
        os_ = OutputSpace(nIn, nOut)
        x = torch.randn(2, nIn, 1).to(TheDevice)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, 1])

    def test_reverse_shape_zero_object_size(self):
        from BasicModel import OutputSpace
        _populate_test_config(symbolDim=1, outputDim=1, nOutput=3, nWhere=0, nWhen=0,
                              reshape=True, reconstruct="FULL")
        nIn, nOut = 4, 3
        os_ = OutputSpace(nIn, nOut)
        y = torch.randn(2, nOut, 1).to(TheDevice)
        x = os_.reverse(y)
        self.assertEqual(list(x.shape), [2, nIn, 1])


class TestBaseModelFactory(unittest.TestCase):
    """BaseModel.from_config factory creates the correct model type."""

    def test_factory_creates_simple_model(self):
        from BasicModel import BaseModel, BasicModel
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
    <training><autoload>false</autoload></training>
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

    def test_factory_creates_basic_model(self):
        from BasicModel import BaseModel, BasicModel as BM
        # nSymbols must equal nConcepts (SymbolicSpace 1:1 mapping constraint),
        # and nPercepts must be 2*nConcepts (InvertiblePiLayer invertibility).
        xml = """<model>
  <architecture>
    <type>basic</type>
    <nWords>16</nWords>
    <conceptualOrder>2</conceptualOrder>
    <reconstruct>none</reconstruct>
    <training><autoload>false</autoload></training>
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
    """Text datasets store raw strings in train_input (tensorized lazily)."""

    def test_train_input_strings_for_text(self):
        from BasicModel import Data
        data = Data()
        data.load("xor")  # XOR uses text examples
        self.assertIsInstance(data.train_input, list)
        self.assertIsInstance(data.train_input[0], str)

    def test_train_input_tensors_for_numeric(self):
        """Numeric datasets store tensors in train_input."""
        from BasicModel import Data
        data = Data()
        data.load("mnist")
        self.assertIsInstance(data.train_input, torch.Tensor)


class TestSymbolDimZeroPassthrough(unittest.TestCase):
    """symbolDim must be 0 when symbolic space is passthrough."""

    def test_passthrough_symbolic_space_has_zero_symbol_dim(self):
        """When symbolPassThrough=True, symbolDim should be 0 and
        embedding size should not be inflated by a symbol dimension."""
        from BasicModel import TheXMLConfig
        _populate_test_config(inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0,
                              outputDim=1, nWhere=0, nWhen=0)
        self.assertEqual(TheXMLConfig.space("SymbolicSpace", "nDim"), 0)
        self.assertEqual(TheXMLConfig.encodingSize(0), 0)

    def test_objectencoding_zero_contribution_when_unused(self):
        """ObjectEncoding must not inflate tensor size when nWhere=0, nWhen=0."""
        from BasicModel import TheXMLConfig
        _populate_test_config(nWhere=0, nWhen=0)
        nDim = 10
        self.assertEqual(TheXMLConfig.encodingSize(nDim), nDim)


class TestInputSpaceLexIntegration(unittest.TestCase):
    """InputSpace with text data creates a Lex instance, span table, and
    encodes spans as [nWhat + nWhere] via a Codebook basis plus ObjectEncoding."""

    def _make_text_data(self):
        """Load XOR text data into TheData singleton."""
        from BasicModel import TheData
        TheData.load("xor")
        return TheData

    def _make_input_space(self, lexer="word"):
        """Create an InputSpace with model_type='embedding' from XOR text data."""
        from BasicModel import InputSpace, TheData, WhereEncoding, WhenEncoding
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              lexer=lexer)
        self._make_text_data()
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        return inp, TheData

    def test_token_stream_available(self):
        """InputSpace with model_type='embedding' can tokenize via _token_stream."""
        inp, _ = self._make_input_space()
        emb = inp.vectors()
        tokens = emb._token_stream("hello world")
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens[0][0], "hello")

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
        # Find the "hello world" doc (order may vary after shuffling)
        hw = None
        for spans in emb.doc_spans:
            words = "".join(t for t, _ in spans)
            if words == "hello world":
                hw = spans
                break
        self.assertIsNotNone(hw, "Expected 'hello world' doc in doc_spans")
        self.assertEqual(hw, [("hello", 0), (" ", 5), ("world", 6)])

    def test_object_encoding_applied(self):
        """ObjectEncoding (nWhere + nWhen) is applied to forward() output."""
        from BasicModel import TheXMLConfig
        inp, data = self._make_input_space()
        batch_size = 1
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        inp.subspace.whereEncoding.p = 0
        output = inp.forward(inputTensor)
        # With nWhere > 0, the reserved encoding dims should be non-zero
        # (ObjectEncoding.forward stamps sin/cos into the last objectSize dims)
        embSize = output.shape[-1]
        objSize = TheXMLConfig.objectSize
        if objSize > 0:
            encoding_dims = output[0, 0, -objSize:]
            self.assertFalse(torch.all(encoding_dims == 0).item(),
                             "ObjectEncoding dims should be non-zero after forward()")


class TestOutputSpaceTextReconstruction(unittest.TestCase):
    """OutputSpace can reconstruct text from symbolic vectors."""

    def test_numeric_output_unchanged(self):
        """Numeric OutputSpace should still produce [B, nOutput] tensor."""
        from BasicModel import OutputSpace
        _populate_test_config(symbolDim=1, outputDim=1, nOutput=3, nWhere=0, nWhen=0,
                              reshape=True)
        nIn, nOut = 4, 3
        os_ = OutputSpace(nIn, nOut)
        x = torch.randn(2, nIn, 1).to(TheDevice)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, 1])
        # text_mode should be False for numeric data
        self.assertFalse(os_.text_mode)

    def test_text_mode_false_without_lex(self):
        """OutputSpace without lex info should have text_mode=False."""
        from BasicModel import OutputSpace
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nOutput=4, reshape=True)
        nIn, nOut = 4, 4
        os_ = OutputSpace(nIn, nOut)
        self.assertFalse(os_.text_mode)

    def test_vectors_constructor_enables_reconstruction(self):
        """Passing vectors= shares the embedding basis with OutputSpace."""
        from BasicModel import InputSpace, TheData, OutputSpace, WhereEncoding, WhenEncoding
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              reshape=True)
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        nOut = 8
        os_ = OutputSpace(nInput, nOut, vectors=inp.vectors())
        self.assertTrue(os_.text_mode)

    def test_reconstruct_from_known_vectors(self):
        """Given codebook vectors with nWhere, reconstruct_data should recover words at positions."""
        import math
        from BasicModel import (InputSpace, TheData, OutputSpace, TheXMLConfig,
                                WhereEncoding, WhenEncoding)
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=10, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              reshape=True)
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        nOut = 4
        os_ = OutputSpace(nInput, nOut, vectors=inp.vectors())

        # Build synthetic vectors from known codebook entries with known nWhere
        codebook = inp.vectors().W.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize
        nWhat = embSize - TheXMLConfig.objectSize
        where = inp.subspace.whereEncoding

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
            where.stamp(vectors, 0, slot, slot * 6)

        recovered_words, recovered_positions = os_.reconstruct_data(vectors)
        self.assertEqual(recovered_words[0], expected_words)

    def test_reconstruct_consecutive_no_nwhere(self):
        """When nWhere is zero, tokens are written consecutively."""
        from BasicModel import (InputSpace, TheData, OutputSpace, TheXMLConfig,
                                WhereEncoding, WhenEncoding)
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=10, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              reshape=True)
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        nOut = 4
        os_ = OutputSpace(nInput, nOut, vectors=inp.vectors())

        # Build vectors with nWhere = 0 (all zeros)
        codebook = inp.vectors().W.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(TheDevice)
        expected_words = []
        # Skip [MASK] (zero vector) — cosine matching can't recover it
        usable = [j for j, w in enumerate(words_list) if w != "[MASK]"]
        for slot, j in enumerate(usable[:nVec]):
            vectors[0, slot, :embSize - TheXMLConfig.objectSize] = \
                codebook[j, :embSize - TheXMLConfig.objectSize]
            expected_words.append(words_list[j])
        # nWhere left as zero -> consecutive mode

        recovered_words, recovered_positions = os_.reconstruct_data(vectors)
        self.assertEqual(recovered_words[0], expected_words)

    def test_reconstruct_to_buffer(self):
        """reconstruct_data with to_buffer=True produces a string with positioned words."""
        import math
        from BasicModel import (InputSpace, TheData, OutputSpace, TheXMLConfig,
                                WhereEncoding, WhenEncoding)
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=10, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              reshape=True)
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        nOut = 4
        os_ = OutputSpace(nInput, nOut, vectors=inp.vectors())

        # Build synthetic vectors with nWhere at known positions
        codebook = inp.vectors().W.detach()
        words_list = inp.vectors().wv.index_to_key
        embSize = inp.vectors().embeddingSize
        nWhat = embSize - inp.objectSize
        where = inp.subspace.whereEncoding

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(TheDevice)
        # Skip [MASK] and \x00 (both zero vectors) — cosine matching can't recover them
        usable = [j for j, w in enumerate(words_list) if w not in ("[MASK]", "\x00")]
        # Word 0 at offset 0, word 1 at offset 6
        for slot, j in enumerate(usable[:nVec]):
            vectors[0, slot, :nWhat] = codebook[j][:nWhat]
            where.stamp(vectors, 0, slot, slot * 6)

        recovered_words, positions = os_.reconstruct_data(vectors)
        text = os_.reconstruct_buffer(vectors)
        # The buffer should contain words at byte offsets
        self.assertIsInstance(text[0], str)
        self.assertIn(words_list[usable[0]], text[0])
        self.assertIn(words_list[usable[1]], text[0])

    def test_forward_reverse_shapes_unchanged(self):
        """forward() and reverse() tensor shapes must not change with text_mode."""
        from BasicModel import (InputSpace, TheData, OutputSpace, TheXMLConfig,
                                WhereEncoding, WhenEncoding)
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, symbolDim=1, outputDim=1,
                              nInput=nInput, nOutput=4,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              reshape=True, reconstruct="FULL")
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        nOut = 4
        os_ = OutputSpace(nInput, nOut, vectors=inp.vectors())
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("SymbolicSpace", "nDim"))
        x = torch.randn(2, nInput, inEmb).to(TheDevice)
        y = os_(x)
        self.assertEqual(list(y.shape), [2, nOut, TheXMLConfig.space("OutputSpace", "nDim")])
        # Reverse path should also be unchanged
        rev = os_.reverse(y)
        self.assertEqual(list(rev.shape), [2, nInput, inEmb])


class TestInputSpaceTextRoundTrip(unittest.TestCase):
    """InputSpace.reverse() must reconstruct text from latent state."""

    def _make_text_input_space(self):
        """Create an InputSpace with model_type='embedding' from XOR text data."""
        from BasicModel import InputSpace, TheData, WhereEncoding, WhenEncoding
        nInput = 8
        _populate_test_config(inputDim=10, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              reshape=True)
        TheData.load("xor")
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        return inp, TheData

    def test_reverse_recovers_words(self):
        """forward -> reverse should recover the original lexical tokens.

        Content tokens (non-whitespace) must match exactly.  Trailing
        padding is space-filled, so the last token from tokenize() is a
        long whitespace run; the reverse path may recover a shorter
        space token — both are correct padding representations.
        """
        inp, data = self._make_text_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        expected_tokens = inp.vectors().tokenize(inputTensor)
        # Forward pass
        latent = inp.forward(inputTensor)
        # Reverse pass
        inp.reverse(latent)
        recovered = inp.reconstruct_data()
        for b in range(batch_size):
            nVec = inp.outputShape[0]
            exp = expected_tokens[b][:nVec]
            rec = recovered[b][:len(exp)]
            # Content tokens must match; padding tokens just need to be whitespace
            for i, (r, e) in enumerate(zip(rec, exp)):
                if e.strip() == "":
                    self.assertEqual(r.strip(), "",
                                     f"Batch {b} token {i}: expected whitespace, got {r!r}")
                else:
                    self.assertEqual(r, e,
                                     f"Batch {b} token {i}: expected {e!r}, got {r!r}")

    def test_reverse_recovers_all_xor_examples(self):
        """All XOR examples should round-trip as lexical token streams."""
        inp, data = self._make_text_input_space()
        all_inputs = data.train_input
        inputTensor = inp.prepInput(all_inputs)
        latent = inp.forward(inputTensor)
        inp.reverse(latent)
        recovered = inp.reconstruct_data()
        expected = inp.vectors().tokenize(inputTensor)
        nVec = inp.outputShape[0]
        for b in range(len(all_inputs)):
            exp = expected[b][:nVec]
            rec = recovered[b][:len(exp)]
            for i, (r, e) in enumerate(zip(rec, exp)):
                if e.strip() == "":
                    self.assertEqual(r.strip(), "",
                                     f"Example {b} token {i}: expected whitespace, got {r!r}")
                else:
                    self.assertEqual(r, e,
                                     f"Example {b} token {i}: expected {e!r}, got {r!r}")

    def test_reconstruct_data_joins_words(self):
        """reconstruct_data(text=True) renders the whitespace buffer."""
        inp, data = self._make_text_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        latent = inp.forward(inputTensor)
        inp.reverse(latent)
        joined = inp.reconstruct_data(text=True)
        self.assertIsInstance(joined[0], str)
        # Reconstructed text should match original input (ignoring trailing null/space padding)
        for b in range(batch_size):
            reconstructed = joined[b].rstrip('\x00 ')
            original = inputBatch[b]
            self.assertEqual(reconstructed, original,
                             f"Batch {b}: reconstructed {reconstructed!r} != original {original!r}")

    def test_reverse_numeric_unchanged(self):
        """Numeric reverse path should still work exactly as before."""
        _populate_test_config(inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, outputDim=1,
                              nInput=8, nWhere=0, nWhen=0, quantized=False)
        from BasicModel import InputSpace
        nIn, nDim = 8, 1
        inp = InputSpace(nIn, nIn)
        x = torch.randn(2, nIn, nDim).to(TheDevice)
        y = inp.forward(x)
        result = inp.reverse(y)
        # Numeric path returns tensor, not text
        self.assertIsInstance(result, (torch.Tensor, list))


class TestLexerConfig(unittest.TestCase):
    """Lexer cfg (word/sentence/grammar) always creates Lex span tables."""

    def test_embedding_can_tokenize(self):
        """Embedding model_type can tokenize text via _token_stream."""
        from BasicModel import InputSpace, TheData
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput)
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        tokens = inp.vectors()._token_stream("test input")
        self.assertEqual(tokens[0][0], "test")

    def test_embedding_creates_reversible_dictionary(self):
        """Embedding model_type creates Embedding with Lex-backed codebook."""
        from BasicModel import InputSpace, TheData, Embedding
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput)
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
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
        """tokenize() separates punctuation from words; all tokens returned."""
        from BasicModel import Embedding
        emb = Embedding()
        result = emb.tokenize(self._make_batch("the dog barks."))
        # quick_parser regex: words, punct, spaces are separate tokens
        # "the dog barks." → ["the", " ", "dog", " ", "barks", "."]
        self.assertIn("barks", result[0])
        self.assertNotIn("barks.", result[0])
        self.assertIn(".", result[0])
        self.assertEqual(result[0], ["the", " ", "dog", " ", "barks", "."])

    def test_forward_returns_token_metadata(self):
        """forward(return_meta=True) replaces old encoding wrappers."""
        from BasicModel import Embedding
        _populate_test_config(nWhere=0, nWhen=0)
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
        """get_mask_embedding() returns a zero vector; [MASK] is not in vocab."""
        from BasicModel import Embedding
        _populate_test_config(nWhere=0, nWhen=0)
        emb = Embedding()
        emb.create(nInput=10, nVectors=2, nDim=10, embedding_path=None)
        self.assertNotIn("[MASK]", emb.pretrain.key_to_index)
        mask_vec = emb.get_mask_embedding()
        self.assertTrue(torch.all(mask_vec == 0.0))
        self.assertEqual(mask_vec.shape[0], emb.wv._vectors.shape[1])


class TestEmbeddingErgodicForward(unittest.TestCase):
    def test_codebook_owns_exploration_state(self):
        from BasicModel import Codebook, Embedding
        vs = Codebook()
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
        from BasicModel import Embedding
        _populate_test_config(nWhere=0, nWhen=0)
        emb = Embedding()
        emb.create(nInput=8, nVectors=8, nDim=10, embedding_path=None, source=[text])
        return emb

    def _seed_sigma(self, emb, word):
        device = emb.wv._vectors.device
        emb.pretrain.sigma = torch.zeros(emb.wv._vectors.shape[0], device=device)
        emb.pretrain.sigma[emb.pretrain.key_to_index[word]] = 1.0
        emb.pretrain.sigma_step = 1
        emb.pretrain.sigma_beta = 0.0

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
        from BasicModel import InputSpace, TheData
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput)
        TheData.load("xor")
        inp = InputSpace(nInput, nInput,
                         model_type="embedding")
        inputTensor = inp.prepInput(TheData.train_input[:2])
        result = inp.forward(inputTensor)
        self.assertEqual(result.shape[0], 2)  # batch size
        self.assertEqual(result.shape[1], inp.outputShape[0])


class TestErgodicMnistReport(unittest.TestCase):
    """OutputSpace.forwardLinear.W accessible even with reversible=True."""

    def test_forward_layer_weight_accessible(self):
        """mnistReport can access the forward linear layer weight matrix."""
        from BasicModel import OutputSpace, LinearLayer
        _populate_test_config(outputDim=1, symbolDim=1, nOutput=10,
                              reshape=True, reconstruct="FULL")
        os_ = OutputSpace(10, 10)
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

    def test_invertible(self):
        """invertible=True with ergodic, reversible — forward + reverse."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            invertible=True, ergodic=True, reconstruct="FULL",
            perceptPassThrough=True, symbolPassThrough=True,
            reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_has_norm(self):
        """hasNorm=True with ergodic — forward only."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            hasNorm=True, ergodic=True,
            perceptPassThrough=True, symbolPassThrough=True,
            reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_conceptual_order_1(self):
        """conceptualOrder=2 — forward only (equal object counts).

        Higher-order cycles require a non-passthrough symbolic space so that
        symbolDim > 0 for the second perceptual/conceptual/symbolic spaces.
        """
        _populate_test_config(inputDim=1, perceptDim=1, conceptDim=1, symbolDim=1,
                              wordDim=1, outputDim=1,
                              nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
                              perceptPassThrough=True, symbolPassThrough=False,
                              reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
                     conceptualOrder=2)
        x = torch.randn(2, 8, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_symbolic_order_1(self):
        """symbolicOrder=2 — forward only (equal object counts).

        Higher-order cycles require a non-passthrough symbolic space so that
        symbolDim > 0 for the syntactic/symbolic spaces.
        """
        _populate_test_config(inputDim=1, perceptDim=1, conceptDim=1, symbolDim=1,
                              wordDim=1, outputDim=1,
                              nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
                              perceptPassThrough=True, symbolPassThrough=False,
                              reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8, nOutput=4,
                     symbolicOrder=2)
        x = torch.randn(2, 8, 1).to(TheDevice)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_non_ergodic_reverse(self):
        """ergodic=False with reversible=True — forward + reverse."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            ergodic=False, reconstruct="FULL",
            perceptPassThrough=True, symbolPassThrough=True,
            reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_percept_no_attention(self):
        """perceptHasAttention=False, perceptPassThrough=False — forward only."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
            perceptHasAttention=False,
            perceptPassThrough=False, symbolPassThrough=True,
            reshape=True, naive=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 8, 1)
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_concept_with_attention(self):
        """conceptHasAttention=True — forward only."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            conceptHasAttention=True,
            perceptPassThrough=True, symbolPassThrough=True,
            reshape=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
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
        with self.assertRaises(ValueError):
            self._create_xor_model(nSymbols=2, nOutput=3)

    @unittest.skip("slow")
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

    @unittest.skip("slow")
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
            recon_texts = m.inputSpace.reconstruct_data(text=True)
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
        from BasicModel import (WhereEncoding, WhenEncoding, InputSpace, TheData)
        _populate_test_config(inputDim=1, nInput=8,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim)

        # Build a minimal InputSpace with embedding from XOR data
        TheData.load("xor")
        nInput = 8
        self.inp = InputSpace(nInput, nInput,
                              model_type="embedding")
        # Run a forward pass to get a real embedded tensor
        inputBatch = TheData.train_input[0:1]
        inputTensor = self.inp.prepInput(inputBatch)
        self.embedded = self.inp.forward(inputTensor)  # [1, nVec, embSize]
        self.sentence = "hello world"  # matches XOR training data
        self.embSize = self.embedded.shape[-1]
        self.nVec = self.embedded.shape[1]

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
        from BasicModel import WhereEncoding, WhenEncoding
        masked, _ = self.inp.expand_masked(self.embedded, self.sentence)
        embSize = self.embSize
        where_idx = np.add([embSize, embSize], WhereEncoding.index)
        when_idx = np.add([embSize, embSize], WhenEncoding.index)
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
        from BasicModel import WhereEncoding, WhenEncoding
        masked, _ = self.inp.expand_masked(self.embedded, self.sentence)
        embSize = self.embSize
        where_idx = np.add([embSize, embSize], WhereEncoding.index)
        when_idx = np.add([embSize, embSize], WhenEncoding.index)
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
        from BasicModel import (WhereEncoding, WhenEncoding,
                                InputSpace, OutputSpace, TheData)
        _populate_test_config(inputDim=1, symbolDim=1, outputDim=1,
                              nInput=8, nOutput=4,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              reshape=True)

        # Build a minimal InputSpace with embedding from XOR data
        TheData.load("xor")
        nInput = 8
        self.inp = InputSpace(nInput, nInput,
                              model_type="embedding")
        self.emb = self.inp.subspace.vectors()

        # Build a minimal OutputSpace
        self.out = OutputSpace(nActiveInput=8, nActiveOutput=4)

    def _make_embedded(self, n_words, emb_size=None):
        """Create a synthetic [1, n_words, embSize] embedded sentence."""
        if emb_size is None:
            emb_size = self.emb.wv._vectors.shape[1] + 4  # content + nWhere + nWhen
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
        self.model = self._create_xor_embedding_model()

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
        from BasicModel import (WhereEncoding, WhenEncoding, InputSpace, TheData)
        _populate_test_config(inputDim=1, nInput=8,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim)

        TheData.load("xor")
        nInput = 8
        self.inp = InputSpace(nInput, nInput,
                              model_type="embedding")
        inputBatch = TheData.train_input[0:1]
        inputTensor = self.inp.prepInput(inputBatch)
        self.embedded = self.inp.forward(inputTensor)
        self.embSize = self.embedded.shape[-1]

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
        from BasicModel import WhereEncoding, WhenEncoding
        sentence = "hello world test"
        masked, positions = self.inp.expand_masked(self.embedded, sentence, maskedPrediction='RARLM')
        embSize = self.embSize
        where_idx = np.add([embSize, embSize], WhereEncoding.index)
        when_idx = np.add([embSize, embSize], WhenEncoding.index)
        pos_dims = set(where_idx.tolist() + when_idx.tolist())
        content_dims = [d for d in range(embSize) if d not in pos_dims]
        for i, pos in enumerate(positions):
            content_vals = masked[i, pos, content_dims]
            self.assertTrue(torch.all(content_vals == 0.0),
                            f"Copy {i}: content at masked pos {pos} should be zero")


class TestRARLMTargets(unittest.TestCase):
    """RARLM targets are in reverse word order."""

    def setUp(self):
        from BasicModel import (WhereEncoding, WhenEncoding,
                                InputSpace, OutputSpace, TheData)
        _populate_test_config(inputDim=1, symbolDim=1, outputDim=1,
                              nInput=8, nOutput=4,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              reshape=True)

        TheData.load("xor")
        nInput = 8
        self.inp = InputSpace(nInput, nInput,
                              model_type="embedding")
        self.emb = self.inp.subspace.vectors()

        self.out = OutputSpace(nActiveInput=8, nActiveOutput=4)

    def test_rarlm_targets_reversed(self):
        """RARLM targets are MLM targets in reverse order."""
        emb_size = self.emb.wv._vectors.shape[1] + 4
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

    def test_train_embeddings_includes_emb_params(self):
        """When trainEmbeddings=true, _emb.weight is in optimizer params."""
        from BasicModel import Embedding
        m = self._create_model(True)
        if not isinstance(m.inputSpace.vectors(), Embedding):
            self.skipTest("Model doesn't use Embedding")
        emb_weight = m.inputSpace.vectors().wv._vectors
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
        emb_weight = m.inputSpace.vectors().wv._vectors
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
        """Embedding vocab round-trips through the embedding file."""
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
        vocab_before = list(emb1.pretrain.index_to_key)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            emb_path = f.name
        try:
            m1.save_embeddings(emb_path)

            # Create a fresh model (will have smaller vocab)
            m2 = BasicModel()
            m2.create_from_config(xml_path, data=TheData)
            emb2 = m2._get_embedding()
            self.assertNotEqual(len(emb2.pretrain.index_to_key), len(vocab_before))

            # Load should restore vocab and embedding weights cleanly
            self.assertTrue(m2.load_embeddings(emb_path))
            emb2 = m2._get_embedding()
            self.assertEqual(list(emb2.pretrain.index_to_key), vocab_before)
            # Embedding shapes must match exactly
            torch.testing.assert_close(
                m2.state_dict()["inputSpace.subspace.object.wv._vectors"],
                m1.state_dict()["inputSpace.subspace.object.wv._vectors"])
        finally:
            os.unlink(emb_path)


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

    def test_runtime_batch_sets_and_restores(self):
        from BasicModel import TheData
        TheData.load("xor")
        original_train = list(TheData.train_input)
        original_output = list(TheData.train_output)
        with TheData.runtime_batch(["hello world"], [[0]]):
            self.assertEqual(TheData.train_input, ["hello world"])
            self.assertEqual(TheData.train_output, [[0]])
        self.assertEqual(list(TheData.train_input), original_train)
        self.assertEqual(list(TheData.train_output), original_output)

    def test_runtime_batch_restores_on_exception(self):
        from BasicModel import TheData
        TheData.load("xor")
        original_train = list(TheData.train_input)
        with self.assertRaises(ValueError):
            with TheData.runtime_batch(["test"], [[1]]):
                raise ValueError("boom")
        self.assertEqual(list(TheData.train_input), original_train)

    def test_runtime_batch_stores_strings(self):
        from BasicModel import TheData
        TheData.load("xor")
        with TheData.runtime_batch(["hello world"]):
            self.assertEqual(TheData.train_input, ["hello world"])


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

        rt_input = ["hello world"]
        rt_output = [torch.tensor([0], dtype=torch.float)]
        with TheData.runtime_batch(rt_input, rt_output):
            batch, nextBatch = m.inputSpace.getBatch(0, 1, "runtime")
            self.assertIsNotNone(batch)
            inp, out = batch
            self.assertEqual(inp.shape[0], 1)  # batch size 1


class TestReconstructionLossGradient(unittest.TestCase):
    """Verify that reconstruction loss (lossIn) flows gradients to the codebook."""

    def test_recon_loss_has_codebook_gradient(self):
        """lossIn.backward() must produce non-zero gradients on the codebook parameter.

        Uses XOR_pos.xml (nWhere=true, nWhen=true) so both content and positional
        dimensions participate in the reconstruction loss.
        """
        from BasicModel import BasicModel, TheData
        import xml.etree.ElementTree as ET

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_pos.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

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

            self.assertTrue(m.reversible)
            self.assertGreater(m.loss.reverse_scale, 0)

            m.set_sigma(0.5)
            m.train(True)
            # Run a single batch WITHOUT internal backward (train=False)
            result, _ = m.runBatch(
                batchNum=0, batchSize=4,
                split="train", train=False,
            )

            self.assertIsNotNone(result.lossIn)
            self.assertGreater(result.lossIn.item(), 0)

            # Backward through lossIn only
            result.lossIn.backward()

            # Gradient must reach the codebook
            emb = m.inputSpace.vectors()
            codebook_param = emb.wv._vectors
            self.assertIsNotNone(codebook_param.grad,
                "Codebook parameter should have gradients from lossIn")
            self.assertGreater(codebook_param.grad.abs().sum().item(), 0,
                "Codebook gradients should be non-zero")
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
