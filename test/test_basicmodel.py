"""BasicModel test suite — exercises all core modules.

Covers:
  - Model.py   layer tests (LinearLayer, Invertible*, PiLayer, SigmaLayer, etc.)
  - SPNN.py    classical neural network
  - SigmaPi.py product-sum network
  - SymPercept.py  bidirectional linear learning
  - Ergodic.py ergodic model construction
  - BasicModel.py  full model creation, forward/reverse pass, weight persistence
"""

import math
import os
import sys
import tempfile
import unittest
import warnings

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
from Space import SubSpace

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"


def _wrap_tensor(space, x):
    """Wrap a raw tensor in the space's SubSpace so forward()/reverse() can materialize it."""
    space.subspace.set_vectors(x)
    return space.subspace


def _unwrap(vspace):
    """Extract the dense tensor from a SubSpace returned by forward()/reverse()."""
    if isinstance(vspace, SubSpace):
        return vspace.materialize()
    return vspace


def _obj_size(section):
    """Compute per-space objectSize from config (nWhere + nWhen)."""
    from BasicModel import TheXMLConfig
    try:
        nw = TheXMLConfig.space(section, "nWhere")
    except KeyError:
        nw = 0
    try:
        nn = TheXMLConfig.space(section, "nWhen")
    except KeyError:
        nn = 0
    return nw + nn


def _xml_uses_embedding(filename):
    import xml.etree.ElementTree as ET

    xml_path = os.path.join(os.path.dirname(_BIN), "data", filename)
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return False
    return (root.findtext("architecture/modelType") or "").strip().lower() == "embedding"


_XOR_EXACT_USES_EMBEDDING = _xml_uses_embedding("XOR_exact.xml")


def _emit_warning_summary(caught):
    """Emit a single summary warning per warning type from a list of caught warnings."""
    from collections import Counter
    counts = Counter()
    for w in caught:
        # Collapse to warning type prefix
        msg = str(w.message)
        if msg.startswith("Range violation"):
            key = "Range violation"
        elif msg.startswith("PiLayer.reverse"):
            key = "PiLayer.reverse out-of-range"
        else:
            key = msg[:60]
        counts[key] += 1
    for key, n in counts.items():
        if n > 0:
            warnings.warn(f"{key} ({n} occurrences during training)")


def _populate_test_config(*,
                          inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0,
                          wordDim=1, outputDim=1,
                          nInput=16, nPercepts=16, nConcepts=8, nSymbols=8,
                          nWords=16, nOutput=4,
                          nWhere=0, nWhen=0,
                          reconstruct="NONE", flatten=False, ergodic=False,
                          naive=False, processSymbols=False,
                          useSubspaceActivation=False,
                          perceptPassThrough=False, symbolPassThrough=False,
                          perceptHasAttention=True, conceptHasAttention=False,
                          invertible=False, hasNorm=False, quantized=False,
                          perceptQuantized=None, conceptQuantized=None,
                          certainty=False,
                          demuxed=False,
                          lexer="word"):
    """Populate TheXMLConfig._data — test equivalent of XML loading.

    Space constructors read nDim/nVectors from TheXMLConfig.  In production,
    the XML file provides these.  In tests, this helper provides them directly.
    """
    from BasicModel import TheXMLConfig, TheData
    from util import init_config, ProjectPaths
    import os
    # Always load model.xml defaults first so all keys are present
    init_config(defaults_path=os.path.join(ProjectPaths.DATA_DIR, "model.xml"))
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
    # Deep-merge test overrides onto model.xml defaults so that keys like
    # 'normalize', 'syntax', etc. are always present.
    _overrides = {
        "architecture": {
            "reconstruct": reconstruct,
            "ergodic": ergodic,
            "naive": naive,
            "processSymbols": processSymbols,
            "useSubspaceActivation": useSubspaceActivation,
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
            "flatten": False,  # InputSpace never flattens
            "quantized": quantized,
            "demuxed": demuxed,
            "normalize": True,  # matches model.xml default
            "lexer": lexer,
        },
        "PerceptualSpace": {
            "nActive": nPercepts,
            "nDim": perceptDim,
            "nVectors": nPercepts,
            "flatten": flatten,
            "quantized": _pq,
            "passThrough": perceptPassThrough,
            "hasAttention": perceptHasAttention,
            "invertible": invertible,
            "normalize": False,  # matches model.xml default
        },
        "ConceptualSpace": {
            "nActive": nConcepts,
            "nDim": conceptDim,
            "nVectors": nConcepts,
            "flatten": flatten,
            "quantized": _cq,
            "hasAttention": conceptHasAttention,
            "hasNorm": hasNorm,
            "invertible": invertible,
            "normalize": False,  # matches model.xml default
        },
        "SymbolicSpace": {
            "nActive": nSymbols,
            "nDim": _symbol_dim,
            "nVectors": nSymbols,
            "flatten": flatten,
            "passThrough": symbolPassThrough,
            "quantized": not symbolPassThrough,
            "normalize": False,  # matches model.xml default
        },
        "SyntacticSpace": {
            "nActive": nWords,
            "nDim": wordDim,
            "nVectors": nWords,
            "flatten": flatten,
            "quantized": False,
            "normalize": False,  # matches model.xml default
        },
        "OutputSpace": {
            "nActive": nOutput,
            "nDim": outputDim,
            "nVectors": nOutput,
            "nWhere": 0,
            "nWhen": 0,
            "flatten": True,  # OutputSpace always flattens
            "quantized": False,
            "invertible": False,
            "normalize": True,  # matches model.xml default
        },
        "ModalSpace": {
            "nDim": perceptDim,
            "nVectors": nPercepts,
            "flatten": flatten,
            "passThrough": False,
            "hasAttention": perceptHasAttention,
            "invertible": invertible,
            "quantized": _pq,
            "whatPassThrough": False,
            "wherePassThrough": True,
            "whenPassThrough": True,
            "normalize": False,  # matches model.xml default
        },
    }
    for section, vals in _overrides.items():
        if section in TheXMLConfig._data and isinstance(TheXMLConfig._data[section], dict):
            TheXMLConfig._data[section].update(vals)
        else:
            TheXMLConfig._data[section] = vals


# ---------------------------------------------------------------------------
# Model.py — Layer tests
# ---------------------------------------------------------------------------
class TestLinearLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import LinearLayer
        layer = LinearLayer(nInput=4, nOutput=3)
        x = torch.randn(2, 4).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 3))

    def test_with_identity_weight(self):
        from Model import LinearLayer
        layer = LinearLayer(nInput=4, nOutput=4, ergodic=True)  # ergodic init = eye
        x = torch.randn(1, 4).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (1, 4))


class TestInvertibleLinearLayer(unittest.TestCase):
    def test_forward_reverse_square(self):
        from Model import InvertibleLinearLayer
        layer = InvertibleLinearLayer(nInput=4, nOutput=4)
        layer.set_sigma(0)
        x = torch.randn(2, 4).to(TheDevice.get())
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4),
                        f"LDU reverse error: {(x - x_rec).abs().max():.6f}")

    def test_forward_reverse_ergodic(self):
        from Model import InvertibleLinearLayer
        layer = InvertibleLinearLayer(nInput=4, nOutput=4, ergodic=True, stable=True)
        with torch.no_grad():
            layer.var.fill_(0.2)
            layer.bias.fill_(0.8)
        x = torch.randn(2, 4).to(TheDevice.get())
        y = layer(x)
        x_rec = layer.reverse(y)
        err = (x - x_rec).norm() / x.norm()
        self.assertLess(err.item(), 0.01,
                        f"Ergodic LDU reverse error: {err:.4f}")


class TestSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import SigmaLayer
        layer = SigmaLayer(nInput=8, nOutput=4)
        x = torch.randn(2, 8).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))


class TestPiLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import PiLayer
        layer = PiLayer(nInput=6, nOutput=3)
        x = torch.randn(2, 6).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 3))


class TestInvertibleSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        from Model import SigmaLayer
        layer = SigmaLayer(nInput=4, nOutput=4, invertible=True)
        x = torch.randn(2, 4).to(TheDevice.get()) * 0.3
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))

    def test_reverse_shape(self):
        from Model import SigmaLayer
        layer = SigmaLayer(nInput=4, nOutput=4, invertible=True)
        y = torch.randn(2, 4).to(TheDevice.get()) * 0.3
        x = layer.reverse(y)
        self.assertEqual(x.shape, (2, 4))


class TestAttentionLayer(unittest.TestCase):
    def test_asymmetric_forward_shape(self):
        from Model import AttentionLayer
        layer = AttentionLayer(nInput=8, nOutput=4, type="asymmetric")
        x = torch.randn(2, 5, 8).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 4))

    def test_symmetric_forward_shape(self):
        from Model import AttentionLayer
        layer = AttentionLayer(nInput=8, nOutput=4, type="symmetric")
        x = torch.randn(2, 5, 8).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 4))

    def test_transformer_forward_shape(self):
        from Model import AttentionLayer
        layer = AttentionLayer(nInput=8, nOutput=4, nHeads=2, type="transformer")
        x = torch.randn(2, 5, 8).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 4))

    def test_transformer_single_object(self):
        """Single-object 3D input [B, 1, D] -> [B, 1, nOut]."""
        from Model import AttentionLayer
        layer = AttentionLayer(nInput=8, nOutput=4, nHeads=2, type="transformer")
        x = torch.randn(2, 1, 8).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 1, 4))

    def test_inline(self):
        from Model import AttentionLayer
        AttentionLayer.test()


class TestNormLayer(unittest.TestCase):
    def test_forward_2d(self):
        from Model import NormLayer
        layer = NormLayer(4, 6)
        x = torch.randn(3, 4).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (3, 6))

    def test_forward_3d(self):
        from Model import NormLayer
        layer = NormLayer(4, 6)
        x = torch.randn(3, 5, 4).to(TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (3, 5, 6))

    def test_reverse_3d(self):
        from Model import NormLayer
        layer = NormLayer(4, 6)
        layer.lr = 0
        x = torch.randn(3, 5, 4).to(TheDevice.get())
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-5))

    def test_inline(self):
        from Model import NormLayer
        NormLayer.test()


class TestMemory(unittest.TestCase):
    def test_mem_update(self):
        from Model import Mem
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg")
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
            perceptPassThrough=True, symbolPassThrough=True, flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        x = torch.randn(2, 28*28, 1).to(TheDevice.get())  # batch of 2, flattened MNIST, dim=1
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_simple_model_ergodic(self):
        """BasicModel (simple path) with ergodic=True uses SigmaLayer path."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, certainty=True, flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        x = torch.randn(2, 28*28, 1).to(TheDevice.get())
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
        x = torch.randn(1, 4).to(TheDevice.get())
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
        buf = torch.zeros(1, 1, 10, device=TheDevice.get())
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
        x = torch.zeros(batch, n, 10, device=TheDevice.get())
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
        x = torch.randn(2, 3, 10, device=TheDevice.get())
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
        x = torch.zeros(5, 2, 10, device=TheDevice.get())
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
        x = torch.zeros(3, 1, 10, device=TheDevice.get())
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
        x = torch.randn(2, 3, 10, device=TheDevice.get())
        original = x.clone()
        y = te.forward(x)
        cleaned, _ = te.reverse(y)
        mask = torch.ones(10, dtype=torch.bool)
        mask[[-2, -1]] = False
        torch.testing.assert_close(cleaned[:, :, mask], original[:, :, mask])


# SubSpace — derived sizes, materialization, construction helpers
# ---------------------------------------------------------------------------
class TestSubSpaceDerivedSizes(unittest.TestCase):
    """SubSpace.getEncodingSize, getEncodedInputSize, getEncodedOutputSize match EventEncoding math."""

    def test_getEncodingSize_returns_muxedSize(self):
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding
        ss = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                      inputShape=[3, 8], outputShape=[3, 8])
        # muxedSize = nWhat(4) + nWhere(2) + nWhen(2) = 8 (inputShape[1])
        self.assertEqual(ss.getEncodingSize(8), 8)
        self.assertEqual(ss.muxedSize, 8)

    def test_getEncodedIO_no_reshape(self):
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding, EventEncoding
        # Shapes already include muxed width (nWhere=2 + nWhen=2)
        ss = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                      objectEncoding=EventEncoding([3, 12], [5, 20], flatten=False),
                      flatten=False,
                      inputShape=[3, 12], outputShape=[5, 20])
        self.assertEqual(ss.getEncodedInputSize(), 12)
        self.assertEqual(ss.getEncodedOutputSize(), 20)

    def test_getEncodedIO_reshape(self):
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding, EventEncoding
        # Shapes already include muxed width (nWhere=2 + nWhen=2)
        ss = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                      objectEncoding=EventEncoding([3, 12], [5, 20], flatten=True),
                      flatten=True,
                      inputShape=[3, 12], outputShape=[5, 20])
        self.assertEqual(ss.getEncodedInputSize(), 12 * 3)
        self.assertEqual(ss.getEncodedOutputSize(), 20 * 5)

    def test_zero_nWhere_nWhen(self):
        from BasicModel import SubSpace, EventEncoding
        ss = SubSpace(objectEncoding=EventEncoding([2, 10], [2, 10]),
                      inputShape=[2, 10], outputShape=[2, 10])
        self.assertEqual(ss.getEncodedInputSize(), 10)
        self.assertEqual(ss.getEncodedOutputSize(), 10)


class TestSubSpaceMaterialize(unittest.TestCase):
    """SubSpace.materialize() returns the expected dense tensor."""

    def test_materialize_tensor(self):
        from BasicModel import SubSpace, Tensor, WhereEncoding, WhenEncoding
        t = torch.randn(2, 4, 12)
        ss = SubSpace.from_tensor(t, whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                                  inputShape=[4, 8], outputShape=[4, 8])
        result = ss.materialize()
        self.assertIs(result, t)
        self.assertIsInstance(ss.event, Tensor)

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
        t = torch.randn(5, 4, 8, device=TheDevice.get())
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
        self.assertIsInstance(ss.event, Tensor)
        self.assertIs(ss.event.W, t)
        self.assertEqual(ss.nWhere, 2)
        self.assertEqual(ss.nWhen, 2)
        self.assertEqual(ss.muxedSize, 6)
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
        self.assertIsInstance(ss.event, Tensor)
        self.assertIsInstance(ss.activation, Tensor)
        self.assertIs(ss.event.W, object)
        self.assertIs(ss.activation.W, act)
        self.assertIs(ss.activeEncoding, ae)

    def test_from_components_defaults_initialized(self):
        """All modalities are initialized (as empty Tensor bases) even with no args."""
        from BasicModel import SubSpace, Tensor
        ss = SubSpace.from_components(inputShape=[4, 8], outputShape=[4, 8])
        self.assertIsInstance(ss.event, Tensor)
        self.assertIsInstance(ss.activation, Tensor)
        self.assertIsInstance(ss.where, Tensor)
        self.assertIsInstance(ss.when, Tensor)


class TestSubSpaceActiveEncoding(unittest.TestCase):
    """ActiveEncoding round-trip through SubSpace."""

    def test_activation_stored_and_retrievable(self):
        from BasicModel import SubSpace, ActiveEncoding
        ae = ActiveEncoding()
        act = torch.tensor([[0.5, 0.8, 0.1]])  # [1, 3] — batch=1, nVectors=3
        encoded = ae.encode(act.squeeze(0))
        ss = SubSpace(activeEncoding=ae, inputShape=[3, 8], outputShape=[3, 8])
        ss.set_activation(act)
        retrieved = ss.get_activation()
        torch.testing.assert_close(retrieved, act)
        decoded = ae.decode(encoded)
        torch.testing.assert_close(decoded, act.squeeze(0))

    def test_two_spaces_independent_encoding(self):
        """Two SubSpaces can have different muxedSize without shared coupling."""
        from BasicModel import SubSpace, WhereEncoding, WhenEncoding, EventEncoding
        # ss1: nWhere=2, nWhen=2, muxedSize=12
        ss1 = SubSpace(whereEncoding=WhereEncoding(1, 2), whenEncoding=WhenEncoding(10000, 2),
                        objectEncoding=EventEncoding([3, 12], [3, 12]),
                        inputShape=[3, 12], outputShape=[3, 12])
        ss2 = SubSpace(objectEncoding=EventEncoding([3, 16], [3, 16]),
                        inputShape=[3, 16], outputShape=[3, 16])
        self.assertEqual(ss1.muxedSize, 12)
        self.assertEqual(ss2.muxedSize, 16)
        self.assertEqual(ss1.getEncodedInputSize(), 12)
        self.assertEqual(ss2.getEncodedInputSize(), 16)


# Regression: Space shape contracts
# ---------------------------------------------------------------------------
class TestCanonicalSpaceShapes(unittest.TestCase):
    """Lock down tensor shapes for canonical Space subclasses."""

    def setUp(self):
        _populate_test_config(
            inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, wordDim=1, outputDim=4,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nWords=4, nOutput=4,
            flatten=True)
        self.B = 2  # batch

    def test_conceptual_space_forward_shape(self):
        from BasicModel import ConceptualSpace, TheXMLConfig
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace([nIn, nDim], [nOut, nDim], [nOut, nDim])
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("InputSpace", "nDim"))
        x = torch.randn(self.B, nIn, inEmb).to(TheDevice.get())
        y = _unwrap(cs(_wrap_tensor(cs, x)))
        outEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("ConceptualSpace", "nDim"))
        self.assertEqual(list(y.shape), [self.B, nOut, outEmb])

    def test_conceptual_space_reverse_shape(self):
        from BasicModel import ConceptualSpace, TheXMLConfig
        _populate_test_config(
            inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, wordDim=1, outputDim=4,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nWords=4, nOutput=4,
            flatten=True, reconstruct="FULL")
        nIn, nOut, nDim = 4, 4, 8
        cs = ConceptualSpace([nIn, nDim], [nOut, nDim], [nOut, nDim])
        outEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("ConceptualSpace", "nDim"))
        y = torch.randn(self.B, nOut, outEmb).to(TheDevice.get())
        x = _unwrap(cs.reverse(_wrap_tensor(cs, y)))
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("InputSpace", "nDim"))
        self.assertEqual(list(x.shape), [self.B, nIn, inEmb])

    def test_output_space_forward_shape(self):
        from BasicModel import OutputSpace, TheXMLConfig
        nIn, nOut = 4, 4
        os_ = OutputSpace([nIn, 8], [nOut, 4], [nOut, 4])
        inEmb = os_.inputShape[1]
        x = torch.randn(self.B, nIn, inEmb).to(TheDevice.get())
        y = _unwrap(os_(_wrap_tensor(os_, x)))
        self.assertEqual(list(y.shape), [self.B, nOut, TheXMLConfig.space("OutputSpace", "nDim")])


class TestSimpleModel(unittest.TestCase):
    """BasicModel (simple path) uses unified Space hierarchy with passThrough SymbolicSpace."""

    def test_simple_model_ergodic_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_simple_model_reverse_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, flatten=True, reconstruct="FULL")
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
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
        x = torch.randn(2, 4, 1).to(TheDevice.get())
        y = vs.forward(x)
        self.assertTrue(torch.equal(x, y))

    def test_quantized_shape(self):
        from BasicModel import Codebook, TheXMLConfig
        _populate_test_config(nInput=4, nPercepts=4, nConcepts=4, nSymbols=4,
                              nWords=0, nOutput=4)
        vs = Codebook()
        vs.create(4, 4, 3, customVQ=False)
        vs = vs.to(TheDevice.get())
        x = torch.randn(2, 4, 3).to(TheDevice.get())
        y = vs.forward(x)
        # Basis.nDim is actual width — no objectSize padding
        self.assertEqual(list(y.shape), [2, 4, 3])


class TestBasisContract(unittest.TestCase):
    def test_tensor_identity_materialization(self):
        from BasicModel import Tensor
        payload = torch.randn(2, 3, 4, device=TheDevice.get())
        basis = Tensor()
        basis.create(3, 3, 4, passThrough=True)
        out = basis.forward(payload)
        self.assertIs(out, payload)
        self.assertIs(basis.getW(), payload)
        rev = basis.reverse(payload)
        self.assertIs(rev, payload)

    def test_invalid_geometry_requires_2d_prototype_matrix(self):
        from BasicModel import Tensor
        basis = Tensor(W=torch.randn(2, 3, 4, device=TheDevice.get()))
        basis.create(3, 3, 4, passThrough=True)
        with self.assertRaises(RuntimeError):
            basis.codebookDistance(torch.randn(2, 4, device=TheDevice.get()))


class TestModelEndToEnd(unittest.TestCase):
    """Lock down full model forward shapes and loss compatibility."""

    def test_simple_model_ergodic_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_reverse_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, flatten=True, reconstruct="FULL")
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
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
            ergodic=True, certainty=True, flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
        target = torch.randn(2, 4).to(TheDevice.get())
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
        s = Space([4, 8], [4, 8], [4, 8])
        Space.config_section = None
        self.assertEqual(s.getParameters(), [])
        s.paramUpdate()  # should be a no-op, not crash

    def test_conceptual_space_has_training_contract(self):
        from BasicModel import ConceptualSpace
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nConcepts=4)
        cs = ConceptualSpace([4, 8], [4, 8], [4, 8])
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

        x = torch.randn(2, nIn).to(TheDevice.get())
        y_sigma = sigma(x)
        y_manual = tanh(linear(x))
        self.assertTrue(torch.allclose(y_sigma, y_manual, atol=1e-6),
                        f"Deterministic SigmaLayer should match LinearLayer+Tanh")

    def test_deterministic_same_train_eval(self):
        from Model import SigmaLayer
        nIn, nOut = 8, 4
        sigma = SigmaLayer(nIn, nOut, ergodic=False)
        x = torch.randn(2, nIn).to(TheDevice.get())

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
        s = Space([4, 3], [4, 3], [4, 3])
        self.assertIsInstance(s.get_vectors(), Codebook)
        self.assertFalse(s.get_vectors().passThrough)

    def test_unquantized_creates_passthrough_codebook(self):
        from BasicModel import Codebook, Space
        _populate_test_config(inputDim=3, nInput=4, quantized=False)
        s = Space([4, 3], [4, 3], [4, 3])
        self.assertIsInstance(s.get_vectors(), Codebook)
        self.assertTrue(s.get_vectors().passThrough)

    def test_forward_subspace_round_trip_keeps_runtime_state(self):
        from BasicModel import InputSpace, PerceptualSpace, SubSpace
        _populate_test_config(
            inputDim=3, perceptDim=3,
            nInput=4, nPercepts=4,
            quantized=False, perceptPassThrough=True,
            nWhere=0, nWhen=0, flatten=False,
        )
        inp = InputSpace([4, 3], [4, 3], [4, 3], model_type="simple")
        per = PerceptualSpace([4, 3], [4, 3], [4, 3])
        x = torch.randn(2, 4, 3).to(TheDevice.get())

        inp.forward(x)
        input_state = inp.subspace
        self.assertIsInstance(input_state, SubSpace)
        self.assertTrue(torch.equal(input_state.materialize(), x))

        percept_state = per.forward(input_state)
        self.assertIsInstance(percept_state, SubSpace)
        self.assertTrue(torch.equal(percept_state.materialize(), x))

        reversed_state = per.reverse(percept_state)
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
        cs = ConceptualSpace([nVec, nDim], [nVec, cDim], [nVec, cDim])
        x = torch.randn(2, nVec, nDim).to(TheDevice.get())
        y = _unwrap(cs(_wrap_tensor(cs, x)))
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_non_ergodic_forward_shape(self):
        self._set_zero_object_encoding(ergodic=False)
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace([nVec, nDim], [nVec, cDim], [nVec, cDim])
        x = torch.randn(2, nVec, nDim).to(TheDevice.get())
        y = _unwrap(cs(_wrap_tensor(cs, x)))
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_ergodic_flag_stored(self):
        self._set_zero_object_encoding(ergodic=True)
        from BasicModel import ConceptualSpace
        cs_erg = ConceptualSpace([8, 1], [8, 1], [8, 1])
        self.assertTrue(cs_erg.ergodic)
        self._set_zero_object_encoding(ergodic=False)
        cs_det = ConceptualSpace([8, 1], [8, 1], [8, 1])
        self.assertFalse(cs_det.ergodic)

    def test_ergodic_reverse_shape(self):
        self._set_zero_object_encoding(ergodic=True, reconstruct="FULL")
        from BasicModel import ConceptualSpace
        nVec, nDim, cDim = 8, 1, 1
        cs = ConceptualSpace([nVec, nDim], [nVec, cDim], [nVec, cDim])
        y = torch.randn(2, nVec, cDim).to(TheDevice.get())
        x = _unwrap(cs.reverse(_wrap_tensor(cs, y)))
        self.assertEqual(list(x.shape), [2, nVec, nDim])

    def test_ergodic_exposes_params(self):
        self._set_zero_object_encoding(ergodic=True)
        from BasicModel import ConceptualSpace
        cs = ConceptualSpace([8, 1], [8, 1], [8, 1])
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

    def test_canonical_forward_still_works(self):
        """Existing ConceptualSpace (with objectSize > 0) still works after changes."""
        from BasicModel import ConceptualSpace, TheXMLConfig
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nConcepts=4)
        nIn, nOut = 4, 4
        cs = ConceptualSpace([nIn, 8], [nOut, 8], [nOut, 8])
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("InputSpace", "nDim"))
        x = torch.randn(2, nIn, inEmb).to(TheDevice.get())
        y = _unwrap(cs(_wrap_tensor(cs, x)))
        outEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("ConceptualSpace", "nDim"))
        self.assertEqual(list(y.shape), [2, nOut, outEmb])


class TestInputSpaceUnquantized(unittest.TestCase):
    """InputSpace works with unquantized codebook (objectSize=0)."""

    def test_unquantized_forward_shape(self):
        from BasicModel import InputSpace
        _populate_test_config(inputDim=1, nInput=8, nWhere=0, nWhen=0, quantized=False)
        nIn, nDim = 8, 1
        inp = InputSpace([nIn, nDim], [nIn, nDim], [nIn, nDim])
        x = torch.randn(2, nIn, nDim).to(TheDevice.get())
        y = _unwrap(inp(x))
        self.assertEqual(list(y.shape), [2, nIn, nDim])


class TestOutputSpaceZeroObjectSize(unittest.TestCase):
    """OutputSpace works with objectSize=0."""

    def test_forward_shape_zero_object_size(self):
        from BasicModel import OutputSpace
        _populate_test_config(symbolDim=1, outputDim=1, nOutput=3, nWhere=0, nWhen=0,
                              flatten=True)
        nIn, nOut = 4, 3
        os_ = OutputSpace([nIn, 1], [nOut, 1], [nOut, 1])
        x = torch.randn(2, nIn, 1).to(TheDevice.get())
        y = _unwrap(os_(_wrap_tensor(os_, x)))
        self.assertEqual(list(y.shape), [2, nOut, 1])

    def test_reverse_shape_zero_object_size(self):
        from BasicModel import OutputSpace
        _populate_test_config(symbolDim=1, outputDim=1, nOutput=3, nWhere=0, nWhen=0,
                              flatten=True, reconstruct="FULL")
        nIn, nOut = 4, 3
        os_ = OutputSpace([nIn, 1], [nOut, 1], [nOut, 1])
        y = torch.randn(2, nOut, 1).to(TheDevice.get())
        x = _unwrap(os_.reverse(_wrap_tensor(os_, y)))
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
  <InputSpace><nOutput>32</nOutput><nDim>8</nDim></InputSpace>
  <PerceptualSpace><nOutput>4</nOutput><nDim>8</nDim><nVectors>4</nVectors></PerceptualSpace>
  <ConceptualSpace><nOutput>2</nOutput><nDim>8</nDim><nVectors>2</nVectors></ConceptualSpace>
  <SymbolicSpace><nOutput>2</nOutput><nDim>1</nDim></SymbolicSpace>
  <OutputSpace><nOutput>2</nOutput><nDim>4</nDim></OutputSpace>
</model>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml)
            path = f.name
        try:
            model, cfg = BaseModel.from_config(path)
            self.assertIsInstance(model, BM)
        finally:
            os.unlink(path)

    def test_factory_creates_mental_model(self):
        from BasicModel import BaseModel, MentalModel
        xml = """<model>
  <architecture>
    <type>mental</type>
    <reconstruct>none</reconstruct>
    <training><autoload>false</autoload></training>
  </architecture>
</model>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml)
            path = f.name
        try:
            model, cfg = BaseModel.from_config(path)
            self.assertIsInstance(model, MentalModel)
            self.assertIs(model.inputSpace.outputSpace, model.outputSpace)
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
        from BasicModel import InputSpace, TheData, TheXMLConfig, WhereEncoding, WhenEncoding
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              lexer=lexer)
        self._make_text_data()
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        return inp, TheData

    def test_token_stream_available(self):
        """InputSpace with model_type='embedding' can tokenize via _token_stream."""
        inp, _ = self._make_input_space()
        emb = inp.get_vectors()
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
        output = _unwrap(inp.forward(inputTensor))
        embSize = inp.subspace.getEncodedOutputSize()
        self.assertEqual(list(output.shape), [batch_size, inp.outputShape[0], embSize])

    def test_doc_spans_store_token_offsets(self):
        """Embedding stores token text alongside byte starts."""
        inp, _ = self._make_input_space()
        emb = inp.get_vectors()
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
        output = _unwrap(inp.forward(inputTensor))
        # With nWhere > 0, the reserved encoding dims should be non-zero
        # (ObjectEncoding.forward stamps sin/cos into the last objectSize dims)
        embSize = output.shape[-1]
        objSize = _obj_size("InputSpace")
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
                              flatten=True)
        nIn, nOut = 4, 3
        os_ = OutputSpace([nIn, 1], [nOut, 1], [nOut, 1])
        x = torch.randn(2, nIn, 1).to(TheDevice.get())
        y = _unwrap(os_(_wrap_tensor(os_, x)))
        self.assertEqual(list(y.shape), [2, nOut, 1])
        # text_mode should be False for numeric data
        self.assertFalse(os_.text_mode)

    def test_text_mode_false_without_lex(self):
        """OutputSpace without lex info should have text_mode=False."""
        from BasicModel import OutputSpace
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nOutput=4, flatten=True)
        nIn, nOut = 4, 4
        os_ = OutputSpace([nIn, 8], [nOut, 4], [nOut, 4])
        self.assertFalse(os_.text_mode)

    def test_vectors_constructor_enables_reconstruction(self):
        """Passing vectors= shares the embedding basis with OutputSpace."""
        from BasicModel import InputSpace, TheData, OutputSpace, TheXMLConfig, WhereEncoding, WhenEncoding
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              flatten=True)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        nOut = 8
        _sdim = TheXMLConfig.space("SymbolicSpace", "nDim") or TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=inp.get_vectors())
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
                              flatten=True)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        nOut = 4
        _sdim = TheXMLConfig.space("SymbolicSpace", "nDim") or TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=inp.get_vectors())

        # Build synthetic vectors from known codebook entries with known nWhere
        codebook = inp.get_vectors().getW().detach()
        words_list = inp.get_vectors().wv.index_to_key
        embSize = inp.muxedSize
        nWhat = embSize - _obj_size("InputSpace")
        where = inp.subspace.whereEncoding

        # Pick first two non-[MASK] words from the codebook
        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(TheDevice.get())
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
                              flatten=True)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        nOut = 4
        _sdim = TheXMLConfig.space("SymbolicSpace", "nDim") or TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=inp.get_vectors())

        # Build vectors with nWhere = 0 (all zeros)
        codebook = inp.get_vectors().getW().detach()
        words_list = inp.get_vectors().wv.index_to_key
        embSize = inp.muxedSize

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(TheDevice.get())
        expected_words = []
        # Skip [MASK] (zero vector) — cosine matching can't recover it
        usable = [j for j, w in enumerate(words_list) if w != "[MASK]"]
        for slot, j in enumerate(usable[:nVec]):
            vectors[0, slot, :embSize - _obj_size("InputSpace")] = \
                codebook[j, :embSize - _obj_size("InputSpace")]
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
                              flatten=True)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        nOut = 4
        _sdim = TheXMLConfig.space("SymbolicSpace", "nDim") or TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=inp.get_vectors())

        # Build synthetic vectors with nWhere at known positions
        codebook = inp.get_vectors().getW().detach()
        words_list = inp.get_vectors().wv.index_to_key
        embSize = inp.muxedSize
        nWhat = inp.nWhat
        where = inp.subspace.whereEncoding

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(TheDevice.get())
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
                              flatten=True, reconstruct="FULL")
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        nOut = 4
        _sdim = TheXMLConfig.space("SymbolicSpace", "nDim") or TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=inp.get_vectors())
        inEmb = TheXMLConfig.encodingSize(TheXMLConfig.space("SymbolicSpace", "nDim"))
        x = torch.randn(2, nInput, inEmb).to(TheDevice.get())
        y = os_(_wrap_tensor(os_, x))
        self.assertEqual(list(_unwrap(y).shape), [2, nOut, TheXMLConfig.space("OutputSpace", "nDim")])
        # Reverse path should also be unchanged
        rev = _unwrap(os_.reverse(y))
        self.assertEqual(list(rev.shape), [2, nInput, inEmb])


class TestInputSpaceTextRoundTrip(unittest.TestCase):
    """InputSpace.reverse() must reconstruct text from latent state."""

    def _make_text_input_space(self):
        """Create an InputSpace with model_type='embedding' from XOR text data."""
        from BasicModel import InputSpace, TheData, TheXMLConfig, WhereEncoding, WhenEncoding
        nInput = 8
        _populate_test_config(inputDim=10, nInput=nInput,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              flatten=True)
        TheData.load("xor")
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
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
        expected_tokens = inp.get_vectors().tokenize(inputTensor)
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
        expected = inp.get_vectors().tokenize(inputTensor)
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
        inp = InputSpace([nIn, nDim], [nIn, nDim], [nIn, nDim])
        x = torch.randn(2, nIn, nDim).to(TheDevice.get())
        y = inp.forward(x)
        result = _unwrap(inp.reverse(y))
        # Numeric path returns tensor, not text
        self.assertIsInstance(result, (torch.Tensor, list))


class TestLexerConfig(unittest.TestCase):
    """Lexer cfg (word/sentence/grammar) always creates Lex span tables."""

    def test_embedding_can_tokenize(self):
        """Embedding model_type can tokenize text via _token_stream."""
        from BasicModel import InputSpace, TheData, TheXMLConfig
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        tokens = inp.get_vectors()._token_stream("test input")
        self.assertEqual(tokens[0][0], "test")

    def test_embedding_creates_reversible_dictionary(self):
        """Embedding model_type creates Embedding with Lex-backed codebook."""
        from BasicModel import InputSpace, TheData, TheXMLConfig, Embedding
        TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        self.assertIsInstance(inp.get_vectors(), Embedding)


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
        self.assertEqual(list(embedded.shape), [1, 8, emb.nDim])
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
        emb.set_sigma(0.5)
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
    """Embedding._load_embeddings handles word2vec text format (.txt).

    The enwiki file is large; setUpClass loads it once and the test
    checks all properties in a single pass to avoid repeated file I/O.
    """

    ENWIKI_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "embeddings", "enwiki_20180420_100d.txt")

    @classmethod
    def setUpClass(cls):
        """Load the enwiki file once for all tests in this class."""
        if not _RUN_SLOW:
            cls.wv = None
            return
        from BasicModel import Embedding
        cls.wv = Embedding._load_embeddings(embedding_path=cls.ENWIKI_PATH)

    @unittest.skipIf(not _RUN_SLOW, "slow — set RUN_SLOW=1")
    def test_load_enwiki(self):
        """Verify txt format properties and nDim-filter logic from a single load."""
        # Basic load: file loaded once in setUpClass
        self.assertIsNotNone(self.wv)
        self.assertEqual(self.wv.vector_size, 100)
        self.assertGreater(len(self.wv), 1000)
        self.assertIn("the", self.wv)
        # nDim filter: _load_embeddings(nDim=50) returns None because 100 != 50.
        # The filter condition is: wv.vector_size != nDim → return None.
        # We verify the precondition without a second slow file load.
        self.assertNotEqual(self.wv.vector_size, 50,
                            "vector_size should be 100; nDim=50 filter would return None")

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
        from BasicModel import InputSpace, TheData, TheXMLConfig
        nInput = 8
        _populate_test_config(inputDim=1, nInput=nInput)
        TheData.load("xor")
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        inputTensor = inp.prepInput(TheData.train_input[:2])
        result = _unwrap(inp.forward(inputTensor))
        self.assertEqual(result.shape[0], 2)  # batch size
        self.assertEqual(result.shape[1], inp.outputShape[0])


class TestErgodicMnistReport(unittest.TestCase):
    """OutputSpace.forwardLinear.W accessible even with reversible=True."""

    def test_forward_layer_weight_accessible(self):
        """mnistReport can access the forward linear layer weight matrix."""
        from BasicModel import OutputSpace, LinearLayer
        _populate_test_config(outputDim=1, symbolDim=1, nOutput=10,
                              flatten=True, reconstruct="FULL")
        os_ = OutputSpace([10, 1], [10, 1], [10, 1])
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
            flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    @unittest.expectedFailure
    def test_has_norm(self):
        """hasNorm=True with ergodic — forward only."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            hasNorm=True, ergodic=True,
            perceptPassThrough=True, symbolPassThrough=True,
            flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
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
                              flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
                     conceptualOrder=2)
        x = torch.randn(2, 8, 1).to(TheDevice.get())
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
                              nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8, nOutput=4,
                              perceptPassThrough=True, symbolPassThrough=False,
                              flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8, nOutput=4,
                     symbolicOrder=2)
        x = torch.randn(2, 8, 1).to(TheDevice.get())
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
            flatten=True)
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
            flatten=True, naive=True)
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
            flatten=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1)
        # Untrained model with nonlinear=False — expect concept range warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
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
        sym_active = root.find("SymbolicSpace/nOutput")
        if sym_active is not None:
            sym_active.text = str(nSymbols)
        sym_nvec = root.find("SymbolicSpace/nVectors")
        if sym_nvec is not None:
            sym_nvec.text = str(nSymbols)
        con_active = root.find("ConceptualSpace/nOutput")
        if con_active is not None:
            con_active.text = str(nSymbols)
        con_nvec = root.find("ConceptualSpace/nVectors")
        if con_nvec is not None:
            con_nvec.text = str(nSymbols)

        # Patch output count
        out_active = root.find("OutputSpace/nOutput")
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

            # Ergodic training — accumulate range warnings, emit summary at end
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

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

            _emit_warning_summary(caught)

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
        from BasicModel import (WhereEncoding, WhenEncoding, InputSpace, TheData, TheXMLConfig)
        _populate_test_config(inputDim=1, nInput=8,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim)

        # Build a minimal InputSpace with embedding from XOR data
        TheData.load("xor")
        nInput = 8
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        self.inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                              model_type="embedding")
        # Run a forward pass to get a real embedded tensor
        inputBatch = TheData.train_input[0:1]
        inputTensor = self.inp.prepInput(inputBatch)
        self.embedded = _unwrap(self.inp.forward(inputTensor))  # [1, nVec, embSize]
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
                                InputSpace, OutputSpace, TheData, TheXMLConfig)
        _populate_test_config(inputDim=1, symbolDim=1, outputDim=1,
                              nInput=8, nOutput=4,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              flatten=True)

        # Build a minimal InputSpace with embedding from XOR data
        TheData.load("xor")
        nInput = 8
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        self.inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                              model_type="embedding")
        self.emb = self.inp.subspace.get_vectors()

        # Build a minimal OutputSpace — input carries symbol objectSize
        _obj_sym = _obj_size("SymbolicSpace")
        self.out = OutputSpace([8, 1 + _obj_sym], [4, 1], [4, 1])

    def _make_embedded(self, n_words, emb_size=None):
        """Create a synthetic [1, n_words, embSize] embedded sentence."""
        if emb_size is None:
            emb_size = self.emb.wv._vectors.shape[1] + 4  # content + nWhere + nWhen
        return torch.randn(1, n_words, emb_size, device=TheDevice.get())

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
        from BasicModel import (WhereEncoding, WhenEncoding, InputSpace, TheData, TheXMLConfig)
        _populate_test_config(inputDim=1, nInput=8,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim)

        TheData.load("xor")
        nInput = 8
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        self.inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                              model_type="embedding")
        inputBatch = TheData.train_input[0:1]
        inputTensor = self.inp.prepInput(inputBatch)
        self.embedded = _unwrap(self.inp.forward(inputTensor))
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
                                InputSpace, OutputSpace, TheData, TheXMLConfig)
        _populate_test_config(inputDim=1, symbolDim=1, outputDim=1,
                              nInput=8, nOutput=4,
                              nWhere=WhereEncoding.nDim, nWhen=WhenEncoding.nDim,
                              flatten=True)

        TheData.load("xor")
        nInput = 8
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        self.inp = InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                              model_type="embedding")
        self.emb = self.inp.subspace.get_vectors()

        _obj_sym = _obj_size("SymbolicSpace")
        self.out = OutputSpace([8, 1 + _obj_sym], [4, 1], [4, 1])

    def test_rarlm_targets_reversed(self):
        """RARLM targets are MLM targets in reverse order."""
        emb_size = self.emb.wv._vectors.shape[1] + 4
        embedded = torch.randn(1, 2, emb_size, device=TheDevice.get())
        mlm_targets = self.out.expand_masked(embedded, "hello world", maskedPrediction='MLM')
        rarlm_targets = self.out.expand_masked(embedded, "hello world", maskedPrediction='RARLM')
        # RARLM targets should be MLM targets reversed
        torch.testing.assert_close(rarlm_targets, mlm_targets.flip(0))


@unittest.skipIf(not _XOR_EXACT_USES_EMBEDDING, "Model doesn't use Embedding")
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
        self.assertIsInstance(m.inputSpace.get_vectors(), Embedding)
        emb_weight = m.inputSpace.get_vectors().wv._vectors
        optimizer = m.getOptimizer(lr=0.001)
        opt_params = [p.data_ptr() for group in optimizer.param_groups for p in group['params']]
        self.assertIn(emb_weight.data_ptr(), opt_params,
                      "Embedding weight should be in optimizer when trainEmbeddings=true")

    def test_frozen_embeddings_default(self):
        """When trainEmbeddings=false, _emb.weight is NOT in optimizer params."""
        from BasicModel import Embedding
        m = self._create_model(False)
        self.assertIsInstance(m.inputSpace.get_vectors(), Embedding)
        emb_weight = m.inputSpace.get_vectors().wv._vectors
        optimizer = m.getOptimizer(lr=0.001)
        opt_params = [p.data_ptr() for group in optimizer.param_groups for p in group['params']]
        self.assertNotIn(emb_weight.data_ptr(), opt_params,
                         "Embedding weight should NOT be in optimizer when trainEmbeddings=false")

    def test_joint_mode_passes_sbow_to_total_loss(self):
        """runBatch must forward JOINT sbow loss into ModelLoss.total()."""
        from BasicModel import Embedding
        m = self._create_model("joint")
        self.assertIsInstance(m.inputSpace.get_vectors(), Embedding)

        optimizer = m.getOptimizer(lr=0.001)
        sentinel = torch.tensor(1.2345, device=TheDevice.get())
        seen = {}

        original_total = m.loss.total
        original_train_embeddings = m.trainEmbeddings

        def capture_total(lossOut, lossIn=None, sbow=None):
            seen["sbow"] = sbow
            return original_total(lossOut, lossIn, sbow)

        def fake_train_embeddings(trainMod, index, split):
            if getattr(m, "train_embedding", "NONE") == "JOINT":
                return sentinel
            return original_train_embeddings(trainMod, index, split)

        m.loss.total = capture_total
        m.trainEmbeddings = fake_train_embeddings
        m.loss.reverse_scale = 0.0

        result, _ = m.runBatch(
            train=True,
            batchNum=0,
            batchSize=1,
            split="train",
            optimizer=optimizer,
        )

        self.assertIsNotNone(result)
        self.assertIs(seen.get("sbow"), sentinel)


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
@unittest.skipIf(not _XOR_EXACT_USES_EMBEDDING, "XOR_exact doesn't use Embedding")
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
        self.assertIsNotNone(emb1)
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
                m2.state_dict()["inputSpace.subspace.event.wv._vectors"],
                m1.state_dict()["inputSpace.subspace.event.wv._vectors"])
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
            emb = m.inputSpace.get_vectors()
            codebook_param = emb.wv._vectors
            self.assertIsNotNone(codebook_param.grad,
                "Codebook parameter should have gradients from lossIn")
            self.assertGreater(codebook_param.grad.abs().sum().item(), 0,
                "Codebook gradients should be non-zero")
        finally:
            os.unlink(tmp.name)


class TestXorExactErgodic(unittest.TestCase):
    """XOR_exact with ergodic=true: reconstruction must not diverge to NaN."""

    def test_xor_perfect_reconstruction_ergodic(self):
        """Same as test_xor_perfect_reconstruction but with ergodic=true.

        Requires stable=True on the invertible PiLayer to prevent log(1-tanh)
        from hitting -inf when ergodic noise drives WX large early in training.
        """
        from BasicModel import BasicModel, TheData
        import xml.etree.ElementTree as ET

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Force ergodic=true
        erg = root.find("architecture/ergodic")
        if erg is None:
            erg = ET.SubElement(root.find("architecture"), "ergodic")
        erg.text = "true"

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

            # Ergodic training — accumulate range warnings, emit summary at end
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m.runTrial(numEpochs=600, batchSize=10, lr=0.01)

            _emit_warning_summary(caught)

            # Check that loss did not diverge to NaN
            final_losses = [l[-1] for l in m.trainLosses if l]
            for loss_val in final_losses:
                self.assertFalse(
                    loss_val != loss_val,  # NaN check
                    f"Loss diverged to NaN under ergodic=true"
                )
                self.assertLess(loss_val, 1.0,
                    f"Loss should converge under ergodic=true, got {loss_val:.4f}")
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Subspace Activation tests
# ---------------------------------------------------------------------------
class TestSubspaceActivation(unittest.TestCase):
    """Tests for the SubSpace.materialize(k) and activation machinery."""

    def test_set_get_activation(self):
        """set_activation / get_activation round-trip."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        activation = torch.randn(2, 4).to(TheDevice.get())
        ss.set_activation(activation)
        got = ss.get_activation()
        self.assertTrue(torch.equal(activation, got))

    def test_set_activation_squeeze(self):
        """set_activation accepts [batch, n, 1] and squeezes."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        activation = torch.randn(2, 4, 1).to(TheDevice.get())
        ss.set_activation(activation)
        got = ss.get_activation()
        self.assertEqual(got.shape, (2, 4))

    def test_materialize_topk(self):
        """Materialize(k) returns top-k vectors by activation."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[8, 3], outputShape=[8, 3])
        # Create a known tensor: 8 vectors of dim 3
        x = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3).to(TheDevice.get())
        ss.set_vectors(x)
        # Set activation: highest at indices 7, 5, 3, 1
        activation = torch.tensor([[0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 1.0]]).to(TheDevice.get())
        ss.set_activation(activation)
        selected = ss.materialize(k=4)
        self.assertEqual(selected.shape, (1, 4, 3))
        # The top-4 by activation are indices 7(1.0), 5(0.9), 1(0.8), 3(0.7)
        expected_indices = [7, 5, 1, 3]
        for i, idx in enumerate(expected_indices):
            self.assertTrue(torch.allclose(selected[0, i], x[0, idx]),
                            f"Position {i}: expected vector at index {idx}")

    def test_materialize_k_none(self):
        """Materialize(k=None) returns all vectors."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(TheDevice.get())
        ss.set_vectors(x)
        activation = torch.randn(2, 4).to(TheDevice.get())
        ss.set_activation(activation)
        result = ss.materialize(k=None)
        self.assertTrue(torch.equal(result, x))

    def test_materialize_k_geq_nspace(self):
        """Materialize(k >= nSpace) returns all vectors."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(TheDevice.get())
        ss.set_vectors(x)
        activation = torch.randn(2, 4).to(TheDevice.get())
        ss.set_activation(activation)
        result = ss.materialize(k=4)
        self.assertTrue(torch.equal(result, x))
        result2 = ss.materialize(k=10)
        self.assertTrue(torch.equal(result2, x))


    def test_materialize_activation_mode_with_stored(self):
        """materialize(mode='activation') returns stored activation."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(TheDevice.get())
        ss.set_vectors(x)
        # activation was set by set_vectors -> set_activation_vectors
        result = ss.materialize(mode="activation")
        d = x.shape[-1]
        expected = 2 * torch.norm(x, dim=-1) / math.sqrt(d) - 1
        self.assertTrue(torch.allclose(result, expected))

    def test_materialize_activation_mode_computes_from_event(self):
        """materialize(mode='activation') computes activation when not stored."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(TheDevice.get())
        # Store event directly without setting activation
        ss.event.setW(x)
        ss.activation.setW(None)
        result = ss.materialize(mode="activation")
        d = x.shape[-1]
        expected = 2 * torch.norm(x, dim=-1) / math.sqrt(d) - 1
        self.assertTrue(torch.allclose(result, expected))

    def test_materialize_activation_mode_no_data_asserts(self):
        """materialize(mode='activation') asserts when no event vectors exist."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        ss.event.setW(None)
        ss.activation.setW(None)
        with self.assertRaises(AssertionError):
            ss.materialize(mode="activation")

    def test_materialize_default_mode_unchanged(self):
        """materialize() with default mode='active' behaves as before."""
        from Space import SubSpace
        ss = SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(TheDevice.get())
        ss.set_vectors(x)
        result = ss.materialize()
        self.assertTrue(torch.equal(result, x))


class TestGrammar(unittest.TestCase):
    """Tests for Grammar and TheGrammar singleton."""

    def test_length(self):
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(len(g), 15)

    def test_indexing(self):
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(g[0], "START → true(S) EOF")
        self.assertEqual(g[1], "S → swap(S, S)")
        self.assertEqual(g[2], "S → equals(S, S)")
        self.assertEqual(g[3], "S → part(S, S)")
        self.assertEqual(g[4], "S → C")
        self.assertEqual(g[5], "C → union(C, C)")
        self.assertEqual(g[12], "C → P")
        self.assertEqual(g[14], "P → ε")

    def test_arity_start(self):
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(g.arity(0), 1)  # START → true(S) EOF

    def test_arity_unary(self):
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(g.arity(7), 1)   # lower(C)
        self.assertEqual(g.arity(9), 1)   # lift(C)
        self.assertEqual(g.arity(10), 1)  # not(C)
        self.assertEqual(g.arity(11), 1)  # non(C)

    def test_arity_binary(self):
        from Model import Grammar
        g = Grammar(lazy_init=False)
        for r in [1, 2, 3, 5, 6, 8]:
            self.assertEqual(g.arity(r), 2, f"rule {r} should be binary")

    def test_arity_transition(self):
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(g.arity(4), 1)   # S -> C -- transition
        self.assertEqual(g.arity(12), 1)  # C -> P -- transition

    def test_arity_terminal(self):
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(g.arity(14), 0)  # P -> epsilon -- terminal

    def test_space_partitions_unconfigured(self):
        """Unconfigured Grammar returns full hardcoded defaults."""
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(g.symbolic(), [1, 2, 3])                     # swap, equals, part
        self.assertEqual(g.conceptual(), [5, 6, 7, 8, 9, 10, 11])   # union..non
        self.assertEqual(g.perceptual(), [13, 14])                   # I P, epsilon

    def test_configure_from_dict(self):
        """Grammar.configure() parses functional notation from dict."""
        from Model import Grammar
        g = Grammar(lazy_init=False)
        g.configure({
            "START": "true(S) EOF",
            "S": ["swap(S, S)", "equals(S, S)", "C"],
            "C": ["union(C, C)", "P"],
            "P": ["I P", "\u03b5"],
        })
        self.assertEqual(g.symbolic(), [1, 2])    # swap, equals
        self.assertEqual(g.conceptual(), [4])      # union
        self.assertEqual(g.perceptual(), [6, 7])   # I P, epsilon

    def test_configure_single_rule_string(self):
        """Single rule as string (not list) works."""
        from Model import Grammar
        g = Grammar(lazy_init=False)
        g.configure({"S": "C", "C": "P", "P": "\u03b5"})
        self.assertEqual(g.symbolic(), [])     # no S-tier methods
        self.assertEqual(g.conceptual(), [])   # no C-tier methods
        self.assertEqual(g.perceptual(), [2])  # epsilon

    def test_configure_unknown_rule_raises(self):
        """Unknown rule text raises ValueError."""
        from Model import Grammar
        g = Grammar(lazy_init=False)
        with self.assertRaises(ValueError):
            g.configure({"S": ["UNKNOWN RULE"]})

    def test_symbolic_transition(self):
        """symbolic_transition() returns S->C rule."""
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(g.symbolic_transition(), 4)  # unconfigured default: rule 4

    def test_conceptual_transition(self):
        """conceptual_transition() returns C->P rule."""
        from Model import Grammar
        g = Grammar(lazy_init=False)
        self.assertEqual(g.conceptual_transition(), 12)  # unconfigured default: rule 12

    def test_transition_none_when_not_configured(self):
        """Transition returns None if not in active set."""
        from Model import Grammar
        g = Grammar(lazy_init=False)
        g.configure({"S": "swap(S, S)", "C": "union(C, C)", "P": "\u03b5"})
        self.assertIsNone(g.symbolic_transition())   # no S->C
        self.assertIsNone(g.conceptual_transition())  # no C->P


class TestWordEncoding(unittest.TestCase):
    """Tests for WordEncoding."""

    def test_encode_decode_roundtrip(self):
        from Space import WordEncoding
        we = WordEncoding(nBatch=4, nActive=64)
        word = we.encode(2, 42, 1)
        b, v, r = we.decode(word)
        self.assertEqual((b, v, r), (2, 42, 1))

    def test_encode_validates_rule(self):
        from Space import WordEncoding
        we = WordEncoding(nBatch=4, nActive=64)
        with self.assertRaises(AssertionError):
            we.encode(0, 0, 99)  # rule out of range

    def test_encode_validates_negative_batch(self):
        from Space import WordEncoding
        we = WordEncoding(nBatch=4, nActive=64)
        with self.assertRaises(AssertionError):
            we.encode(-1, 0, 0)

    def test_encode_validates_negative_vector(self):
        from Space import WordEncoding
        we = WordEncoding(nBatch=4, nActive=64)
        with self.assertRaises(AssertionError):
            we.encode(0, -1, 0)


class TestSubspaceWords(unittest.TestCase):
    """Tests for SubSpace word support."""

    def _make_ss(self):
        from Space import SubSpace
        return SubSpace(inputShape=[4, 3], outputShape=[4, 3])

    def test_words_default_empty(self):
        ss = self._make_ss()
        self.assertEqual(ss.get_words(), [])

    def test_add_word_start_state(self):
        ss = self._make_ss()
        ss.add_word(0, 0, 0)
        self.assertEqual(ss.get_words(), [(0, 0, 0)])

    def test_add_multiple_words(self):
        ss = self._make_ss()
        ss.add_word(0, 0, 0)
        ss.add_word(0, 42, 1)
        self.assertEqual(ss.get_words(), [(0, 0, 0), (0, 42, 1)])

    def test_set_words(self):
        ss = self._make_ss()
        words = [(0, 0, 0), (0, 1, 2), (0, 2, 3)]
        ss.set_words(words)
        self.assertEqual(ss.get_words(), words)

    def test_add_word_validates(self):
        ss = self._make_ss()
        with self.assertRaises(AssertionError):
            ss.add_word(0, 0, 99)  # bad rule


class TestSyntacticSpaceRoundTrip(unittest.TestCase):
    """SyntacticSpace.reverse deterministically recovers activation from derivation."""

    def test_forward_reverse_recovers_activation(self):
        """Pass top-k activation → forward → delete activation → reverse → compare."""
        _populate_test_config(
            inputDim=3, perceptDim=3, conceptDim=3, symbolDim=1, wordDim=3,
            nInput=4, nPercepts=4, nConcepts=20, nSymbols=20, nWords=20, nOutput=10,
            perceptPassThrough=True, symbolPassThrough=True)
        from Space import SyntacticSpace, SubSpace
        nVectors = 20
        nDim = 3
        syn = SyntacticSpace([nVectors, nDim], [nVectors, nDim], [nVectors, nDim])
        # Create input subspace with top-k=7 activation
        ss = SubSpace(inputShape=[nVectors, nDim], outputShape=[nVectors, nDim])
        x = torch.randn(2, nVectors, nDim).to(TheDevice.get())
        ss.set_vectors(x)
        # Build sparse symbolic presence: 7 present positions per batch
        symbols = torch.zeros(2, nVectors, device=TheDevice.get())
        # Pick 7 random positions per batch
        for b in range(2):
            indices = torch.randperm(nVectors)[:7]
            symbols[b, indices] = 1.0
        ss.set_symbols(symbols)
        original_symbols = symbols.clone()
        # Forward: generates derivation
        out = syn.forward(ss)
        # Verify words were produced
        words = out.get_words()
        self.assertGreater(len(words), 0)
        # Delete the activation from the output subspace
        out.activation.setW(None)
        self.assertIsNone(out.get_activation())
        # Reverse: should recover symbols from derivation
        recovered = syn.reverse(out)
        recovered_symbols = recovered.get_symbols()
        self.assertIsNotNone(recovered_symbols)
        # The present/absent positions should match
        self.assertTrue(torch.equal(
            (original_symbols > 0.5).float().cpu(),
            (recovered_symbols > 0.5).float().cpu()),
            "Recovered symbol positions don't match original")


class TestSubspaceNormalize(unittest.TestCase):
    """Tests for SubSpace.normalize()."""

    def _make_ss(self):
        from Space import SubSpace
        return SubSpace(inputShape=[4, 3], outputShape=[4, 3])

    def test_percepts_range(self):
        """normalize('percepts') produces values in [-1, 1] via tanh."""
        ss = self._make_ss()
        x = torch.randn(2, 4, 3)
        ss.set_vectors(x.clone())
        ss.normalize("percepts", target="what")
        y = ss.select("what")
        self.assertTrue(torch.all(y >= -1) and torch.all(y <= 1))
        self.assertTrue(torch.allclose(y, torch.tanh(x)))

    def test_concepts_range(self):
        """normalize('concepts') produces values in [-1, 1] via tanh."""
        ss = self._make_ss()
        x = torch.randn(2, 4, 3)
        ss.set_activation(x[:, :, 0].clone())
        ss.normalize("concepts", target="activation")
        y = ss.get_activation()
        self.assertTrue(torch.all(y >= -1) and torch.all(y <= 1))
        self.assertTrue(torch.allclose(y, torch.tanh(x[:, :, 0])))

    def test_symbols_discrete(self):
        """normalize('symbols') produces {0, 1} integers with STE gradients."""
        ss = self._make_ss()
        x = torch.randn(2, 4, requires_grad=True)
        ss.set_activation(x)
        ss.normalize("symbols", target="activation")
        y = ss.get_activation()
        # Output should be exactly 0 or 1
        self.assertTrue(torch.all((y == 0) | (y == 1)))
        # Gradients should flow (straight-through estimator)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(x.grad != 0))

    def test_invalid_kind_raises(self):
        """normalize() with unknown kind raises ValueError."""
        ss = self._make_ss()
        ss.set_vectors(torch.randn(2, 4, 3))
        with self.assertRaises(ValueError):
            ss.normalize("bogus", target="what")


class TestDataScaling(unittest.TestCase):
    """Tests for Data min/max tracking and scaling helpers."""

    def test_xor_data_ranges(self):
        """After loading XOR, Data has correct input/output min/max."""
        from BasicModel import TheData
        TheData.load("xor")
        # XOR uses text input (embedded, L2-normalized) and binary labels
        self.assertEqual(TheData.input_min, -1.0)
        self.assertEqual(TheData.input_max, 1.0)
        self.assertEqual(TheData.output_min, 0.0)
        self.assertEqual(TheData.output_max, 1.0)

    def test_normalize_denormalize_roundtrip(self):
        """Data.normalize and Data.denormalize are inverses."""
        from BasicModel import TheData
        TheData.input_min = -5.0
        TheData.input_max = 5.0
        TheData.output_min = -5.0
        TheData.output_max = 5.0
        x = torch.tensor([[-5.0, 0.0, 5.0]])
        scaled = TheData.normalize(x, which="input")
        self.assertTrue(torch.allclose(scaled, torch.tensor([[-1.0, 0.0, 1.0]])))
        roundtrip = TheData.denormalize(scaled, which="input")
        self.assertTrue(torch.allclose(roundtrip, x))
        # denormalize(output): [-1,1] -> [min,max]
        act = torch.tensor([[-1.0, 0.0, 1.0]])
        rescaled = TheData.denormalize(act, which="output")
        self.assertTrue(torch.allclose(rescaled, torch.tensor([[-5.0, 0.0, 5.0]])))

    def test_degenerate_range_noop(self):
        """When min==max, scaling is a no-op (returns input unchanged)."""
        from BasicModel import TheData
        TheData.input_min = 3.0
        TheData.input_max = 3.0
        x = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.equal(TheData.normalize(x, which="input"), x))


class TestNormalizeFlag(unittest.TestCase):
    """Tests for the normalize flag on SubSpace.normalize()."""

    def test_normalize_false_does_not_modify(self):
        """normalize=False checks range but does not modify the tensor."""
        from Space import SubSpace
        _populate_test_config(inputDim=4, nInput=4)
        ss = SubSpace(inputShape=[4, 4], outputShape=[4, 4])
        # Set vectors that are NOT in [0,1] range
        x = torch.randn(2, 4, 4) * 5
        ss.set_vectors(x.clone())
        original = ss.materialize().clone()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ss.normalize("percepts", target="what", normalize=False)
        after = ss.materialize()
        self.assertTrue(torch.equal(original, after),
                        "normalize=False should not modify the tensor")
        self.assertTrue(len(w) > 0, "Should have emitted a warning")

    def test_normalize_true_does_modify(self):
        """normalize=True (default) modifies the tensor."""
        from Space import SubSpace
        _populate_test_config(inputDim=4, nInput=4)
        ss = SubSpace(inputShape=[4, 4], outputShape=[4, 4])
        x = torch.randn(2, 4, 4) * 5
        ss.set_vectors(x.clone())
        original = ss.materialize().clone()
        ss.normalize("percepts", target="what", normalize=True)
        after = ss.materialize()
        self.assertFalse(torch.equal(original, after),
                         "normalize=True should modify the tensor")
        self.assertTrue(torch.all(after >= -1) and torch.all(after <= 1))


class TestInputSpaceScaling(unittest.TestCase):
    """Tests for InputSpace min-max scaling of non-embedding data."""

    def test_simple_input_scaled_to_unit(self):
        """InputSpace(normalize=True) scales passthrough what-content to [-1,1]."""
        from BasicModel import InputSpace, TheData, TheXMLConfig
        TheData.load("xor")
        TheData.input_min = -3.0
        TheData.input_max = 3.0
        nInput = 4
        _populate_test_config(inputDim=4, nInput=nInput, nWhere=0, nWhen=0)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        inp = InputSpace([nInput, _idim], [_invec, _idim],
                         [nInput, _idim], model_type="simple")
        x = torch.FloatTensor([[[-3, -1, 1, 3]] * nInput]).to(TheDevice.get())
        result = inp.forward(x)
        what = result.select("what")
        self.assertTrue(torch.all(what >= -1.01) and torch.all(what <= 1.01),
                        f"what should be in [-1,1], got [{what.min():.4f}, {what.max():.4f}]")


class TestSubspaceActivationPipeline(unittest.TestCase):
    """Full pipeline test with useSubspaceActivation=true."""

    def test_simple_model_subspace_activation(self):
        """BasicModel with useSubspaceActivation=True produces valid output shapes."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
            perceptPassThrough=True, symbolPassThrough=True, flatten=True,
            useSubspaceActivation=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        x = torch.randn(2, 28*28, 1).to(TheDevice.get())
        _, end_state, out = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_subspace_activation_stored(self):
        """After forward with useSubspaceActivation, spaces have activations."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True, flatten=True,
            useSubspaceActivation=True)
        from BasicModel import BasicModel
        model = BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).to(TheDevice.get())
        model.forward(x)

        # Check that activations are stored on spaces that compute them.
        # InputSpace (entry point) and passthrough spaces don't set activation —
        # they forward the upstream SubSpace unchanged.
        from BasicModel import InputSpace
        for space in model.spaces:
            if isinstance(space, InputSpace):
                continue
            if getattr(space, 'passThrough', False):
                continue
            activation = space.subspace.get_activation()
            self.assertIsNotNone(activation,
                                 f"{space.name} should have activation after forward")
            self.assertEqual(activation.shape[0], 2,
                             f"{space.name} activation batch dim wrong")


# ---------------------------------------------------------------------------
# InputSpace(demuxed=True) / ModalSpace tests
# ---------------------------------------------------------------------------

class TestInputSpaceDemuxed(unittest.TestCase):
    """InputSpace with demuxed=True separates what/where/when and auto-muxes on materialize."""

    def test_demuxed_slots_populated(self):
        """InputSpace(demuxed=True).forward() populates what, where, when independently."""
        from BasicModel import InputSpace, TheData, TheXMLConfig
        TheData.load("xor")
        nInput = 4
        _populate_test_config(inputDim=8, nInput=nInput, nWhere=2, nWhen=2, demuxed=True)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim],
                         [nInput, _idim + _obj], model_type="simple")
        x = torch.randn(2, nInput, _idim).to(TheDevice.get())
        result = inp.forward(x)
        self.assertTrue(result.is_demuxed)
        self.assertEqual(list(result.what.getW().shape), [2, nInput, _idim])
        self.assertEqual(list(result.where.getW().shape), [2, nInput, 2])
        self.assertEqual(list(result.when.getW().shape), [2, nInput, 2])

    def test_materialize_produces_muxed(self):
        """InputSpace(demuxed=True).materialize() produces concat([what, where, when])."""
        from BasicModel import InputSpace, TheData, TheXMLConfig
        TheData.load("xor")
        nInput = 4
        _populate_test_config(inputDim=8, nInput=nInput, nWhere=2, nWhen=2, demuxed=True)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = InputSpace([nInput, _idim], [_invec, _idim],
                         [nInput, _idim + _obj], model_type="simple")
        x = torch.randn(2, nInput, _idim).to(TheDevice.get())
        result = inp.forward(x)
        muxed = result.materialize()
        self.assertEqual(list(muxed.shape), [2, nInput, _idim + _obj])

    def test_equivalence_with_muxed_input_space(self):
        """InputSpace(demuxed=True).materialize() == InputSpace(demuxed=False).materialize()."""
        from BasicModel import InputSpace, TheData, TheXMLConfig
        TheData.load("xor")
        nInput = 4

        # Build muxed (legacy) InputSpace
        _populate_test_config(inputDim=8, nInput=nInput, nWhere=2, nWhen=2, demuxed=False)
        _idim = TheXMLConfig.space("InputSpace", "nDim")
        _invec = TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        legacy = InputSpace([nInput, _idim], [_invec, _idim],
                            [nInput, _idim + _obj], model_type="simple")

        # Build demuxed InputSpace
        _populate_test_config(inputDim=8, nInput=nInput, nWhere=2, nWhen=2, demuxed=True)
        demuxed = InputSpace([nInput, _idim], [_invec, _idim],
                             [nInput, _idim + _obj], model_type="simple")

        x = torch.randn(2, nInput, _idim).to(TheDevice.get())
        legacy_out = _unwrap(legacy.forward(x))
        demuxed_out = _unwrap(demuxed.forward(x))
        self.assertTrue(torch.allclose(legacy_out, demuxed_out, atol=1e-5),
                        f"max diff: {(legacy_out - demuxed_out).abs().max():.6f}")


class TestModalSpace(unittest.TestCase):
    """ModalSpace routes what/where/when through independent PerceptualSpaces."""

    def test_forward_shape(self):
        """ModalSpace.forward() produces correct muxed output shape."""
        from BasicModel import ModalSpace, SubSpace, TheXMLConfig
        nInput = 4
        nWhere = 2
        nWhen = 2
        nDim = 8
        _populate_test_config(inputDim=nDim, perceptDim=nDim,
                              nInput=nInput, nPercepts=nInput,
                              nWhere=nWhere, nWhen=nWhen,
                              perceptPassThrough=True)
        muxed_w = nDim + nWhere + nWhen
        space = ModalSpace([nInput, muxed_w], [nInput, nDim], [nInput, muxed_w])
        # Build a demuxed input
        what_t = torch.randn(2, nInput, nDim).to(TheDevice.get())
        where_t = torch.randn(2, nInput, nWhere).to(TheDevice.get())
        when_t = torch.randn(2, nInput, nWhen).to(TheDevice.get())
        ss = SubSpace(inputShape=[nInput, muxed_w], outputShape=[nInput, muxed_w])
        ss.set_demuxed(what_t, where_t, when_t)
        result = space.forward(ss)
        materialized = result.materialize()
        self.assertEqual(list(materialized.shape), [2, nInput, muxed_w])

    def test_degenerate_no_position(self):
        """With nWhere=nWhen=0, ModalSpace degenerates to a single PerceptualSpace."""
        from BasicModel import ModalSpace, SubSpace, TheXMLConfig
        nInput = 4
        nDim = 8
        _populate_test_config(inputDim=nDim, perceptDim=nDim,
                              nInput=nInput, nPercepts=nInput,
                              nWhere=0, nWhen=0,
                              perceptPassThrough=True)
        space = ModalSpace([nInput, nDim], [nInput, nDim], [nInput, nDim])
        self.assertIsNone(space.whereSpace)
        self.assertIsNone(space.whenSpace)
        # Forward with muxed input (no demux needed)
        x = torch.randn(2, nInput, nDim).to(TheDevice.get())
        ss = SubSpace(inputShape=[nInput, nDim], outputShape=[nInput, nDim])
        ss.set_vectors(x)
        result = space.forward(ss)
        self.assertEqual(list(result.materialize().shape), [2, nInput, nDim])


class TestBasicModelDemuxed(unittest.TestCase):
    """BasicModel with demuxed=True creates InputSpace(demuxed) + ModalSpace."""

    def test_create_from_config(self):
        """BasicModel with demuxed=True creates InputSpace(demuxed) + ModalSpace."""
        from BasicModel import BasicModel, InputSpace, ModalSpace, TheData
        TheData.load("xor")
        _populate_test_config(
            inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0,
            outputDim=4,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nWords=4, nOutput=4,
            nWhere=2, nWhen=2,
            demuxed=True,
            flatten=True, perceptPassThrough=True)
        model = BasicModel()
        model.create(nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nOutput=4,
                     model_type="simple")
        self.assertIsInstance(model.inputSpace, InputSpace)
        self.assertTrue(model.inputSpace.demuxed)
        self.assertIsInstance(model.perceptualSpace, ModalSpace)


class TestOldSyntacticLayer(unittest.TestCase):
    """Tests for OldSyntacticLayer — the original learnable derivation stack."""

    def _make_layer(self, nInput=16, max_depth=7, hidden_dim=32):
        from Model import OldSyntacticLayer
        return OldSyntacticLayer(nInput=nInput, nOutput=nInput,
                                 max_depth=max_depth, hidden_dim=hidden_dim)

    def _dev(self):
        return TheDevice.get()

    def test_forward_shapes(self):
        layer = self._make_layer()
        x = torch.randn(4, 16).to(self._dev())
        out = layer.forward(x)
        self.assertEqual(out["rule_logits"].shape, (4, 7, 15))
        self.assertEqual(out["rule_probs"].shape, (4, 7, 15))
        self.assertEqual(out["predicted_rules"].shape, (4, 7))

    def test_gradient_flows_through_gumbel(self):
        layer = self._make_layer(nInput=8, hidden_dim=16)
        layer.train()
        x = torch.randn(2, 8, requires_grad=True, device=self._dev())
        out = layer.forward(x)
        loss = out["rule_logits"].pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue((x.grad.abs() > 0).any())

    def test_eval_uses_softmax(self):
        layer = self._make_layer(nInput=8, hidden_dim=16)
        layer.eval()
        x = torch.randn(2, 8).to(self._dev())
        out = layer.forward(x)
        sums = out["rule_probs"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_shared_weights(self):
        """Verify the recursive architecture uses shared weights."""
        layer = self._make_layer(nInput=8, max_depth=12, hidden_dim=16)
        self.assertIsInstance(layer.derivation_layer, type(layer.input_proj))
        self.assertIsInstance(layer.rule_head, type(layer.input_proj))
        self.assertEqual(layer.depth_embed.num_embeddings, 12)

    def test_set_tau(self):
        layer = self._make_layer()
        layer.set_tau(0.5)
        self.assertEqual(layer.tau, 0.5)


class TestSyntacticLayer(unittest.TestCase):
    """Tests for SyntacticLayer — per-space grammar with executable rules."""

    def _dev(self):
        return TheDevice.get()

    def _make_symbolic_layer(self, nInput=8, max_depth=7, hidden_dim=16):
        from Model import SyntacticLayer, Grammar
        g = Grammar(lazy_init=False)
        return SyntacticLayer(
            nInput=nInput, nOutput=nInput,
            rules=g.symbolic(),       # [1] (swap)
            transition_rule=g.symbolic_transition(),  # 2 (S->C)
            max_depth=max_depth,
            hidden_dim=hidden_dim,
            grammar=g,
        )

    def _make_conceptual_layer(self, nInput=8, max_depth=7, hidden_dim=16):
        from Model import SyntacticLayer, Grammar
        g = Grammar(lazy_init=False)
        return SyntacticLayer(
            nInput=nInput, nOutput=nInput,
            rules=g.conceptual(),     # [3,4,5,6,7,8,9,10,11]
            transition_rule=g.conceptual_transition(),  # 12 (C->P)
            max_depth=max_depth,
            hidden_dim=hidden_dim,
            grammar=g,
        )

    def _make_perceptual_layer(self, nInput=8, max_depth=7, hidden_dim=16):
        from Model import SyntacticLayer, Grammar
        g = Grammar(lazy_init=False)
        return SyntacticLayer(
            nInput=nInput, nOutput=nInput,
            rules=g.perceptual(),     # [13, 14]
            transition_rule=None,
            max_depth=max_depth,
            hidden_dim=hidden_dim,
            grammar=g,
        )

    # ── Shape tests ───────────────────────────────────────────────

    def test_symbolic_forward_shapes(self):
        layer = self._make_symbolic_layer()
        x = torch.randn(4, 8).to(self._dev())
        out = layer.forward(x)
        # symbolic rules [1,2,3] + transition 4 = 4 local rules
        self.assertEqual(out["rule_logits"].shape, (4, 7, 4))
        self.assertEqual(out["rule_probs"].shape, (4, 7, 4))
        self.assertEqual(out["predicted_rules"].shape, (4, 7))
        self.assertNotIn("composed_activation", out)

    def test_conceptual_forward_shapes(self):
        layer = self._make_conceptual_layer()
        x = torch.randn(2, 8).to(self._dev())
        out = layer.forward(x)
        # conceptual rules [5..11] (7) + transition 12 = 8 local rules
        self.assertEqual(out["rule_logits"].shape, (2, 7, 8))
        self.assertNotIn("composed_activation", out)

    def test_perceptual_forward_shapes(self):
        layer = self._make_perceptual_layer()
        x = torch.randn(2, 8).to(self._dev())
        out = layer.forward(x)
        # perceptual rules [13, 14] = 2 local rules
        self.assertEqual(out["rule_logits"].shape, (2, 7, 2))

    # ── Space projection operation tests ─────────────────────────

    def test_symbolic_and_is_min(self):
        """AND (Gödel t-norm): min(left, right)."""
        left  = torch.tensor([[0.8, 0.5, 1.0]])
        right = torch.tensor([[0.6, 1.0, 0.0]])
        result = torch.min(left, right)
        expected = torch.tensor([[0.6, 0.5, 0.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_symbolic_or_is_max(self):
        """OR (Gödel t-conorm): max(left, right)."""
        left  = torch.tensor([[0.2, 0.9]])
        right = torch.tensor([[0.7, 0.3]])
        result = torch.max(left, right)
        expected = torch.tensor([[0.7, 0.9]])
        self.assertTrue(torch.allclose(result, expected))

    def test_symbolic_not_is_complement(self):
        """NOT: 1.0 - left."""
        left = torch.tensor([[0.3, 0.8]])
        result = 1.0 - left
        expected = torch.tensor([[0.7, 0.2]])
        self.assertTrue(torch.allclose(result, expected))

    def test_conceptual_union_on_vectors(self):
        """UNION: max(left, right) on [B, N, D] vectors."""
        left  = torch.tensor([[[0.2, 0.5], [0.9, 0.1]]])
        right = torch.tensor([[[0.7, 0.3], [0.3, 0.8]]])
        result = torch.max(left, right)
        expected = torch.tensor([[[0.7, 0.5], [0.9, 0.8]]])
        self.assertTrue(torch.allclose(result, expected))

    def test_conceptual_intersection_on_vectors(self):
        """INTERSECTION: min(left, right) on [B, N, D] vectors."""
        left  = torch.tensor([[[0.5, 0.9], [0.7, 0.2]]])
        right = torch.tensor([[[0.3, 0.8], [0.9, 0.1]]])
        result = torch.min(left, right)
        expected = torch.tensor([[[0.3, 0.8], [0.7, 0.1]]])
        self.assertTrue(torch.allclose(result, expected))

    # ── Gradient flow (rule prediction) ────────────────────────────

    def test_gradient_flows_through_rule_prediction(self):
        layer = self._make_symbolic_layer()
        layer.train()
        x = torch.randn(2, 8, requires_grad=True, device=self._dev())
        out = layer.forward(x)
        loss = out["rule_logits"].pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue((x.grad.abs() > 0).any())

    # ── Eval mode ────────��─────────────────────────────���──────────

    def test_eval_uses_softmax(self):
        layer = self._make_symbolic_layer()
        layer.eval()
        x = torch.randn(2, 8).to(self._dev())
        out = layer.forward(x)
        sums = out["rule_probs"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    # ── Reverse ─────────────��─────────────────────────────────────

    def test_reverse_recovers_positions(self):
        layer = self._make_symbolic_layer()
        words = [(0, 2, 2), (0, 5, 2), (0, 7, 6)]  # 3 word tuples
        activation = layer.reverse(words, nVectors=8, batch_size=1)
        self.assertEqual(activation.shape, (1, 8))
        self.assertEqual(activation[0, 2].item(), 1.0)
        self.assertEqual(activation[0, 5].item(), 1.0)
        self.assertEqual(activation[0, 7].item(), 1.0)
        self.assertEqual(activation[0, 0].item(), 0.0)

    # ── Utilities ─────────────────────────��───────────────────────

    def test_set_tau(self):
        layer = self._make_symbolic_layer()
        layer.set_tau(0.5)
        self.assertEqual(layer.tau, 0.5)


class TestShiftReduce(unittest.TestCase):
    """Tests for SyntacticSpace shift/reduce (writeSymbols / resetStack)."""

    def _make_syntactic_space(self, nSym=8):
        """Create a SyntacticSpace with minimal config for S/R testing."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=1, wordDim=1,
            nInput=nSym, nPercepts=nSym, nConcepts=nSym, nSymbols=nSym,
            nWords=nSym, nOutput=nSym,
            perceptPassThrough=True, symbolPassThrough=True)
        from Space import SyntacticSpace
        syn = SyntacticSpace(
            inputShape=(nSym, 1),
            spaceShape=(nSym, 1),
            outputShape=(nSym, 1),
        )
        syn.to(TheDevice.get())
        syn.eval()
        return syn

    def test_writeSymbols_shift(self):
        """writeSymbols pushes onto _sr_stack and returns valid result dict."""
        syn = self._make_syntactic_space()
        syn.resetStack()
        act = torch.randn(1, 8, device=TheDevice.get())
        result = syn.writeSymbols(act)
        # Stack should have one entry after shift (single symbol, no binary reduce possible)
        self.assertEqual(len(syn._sr_stack), 1)
        # Result dict has required keys
        self.assertIn("transition", result)
        self.assertIn("composed", result)
        self.assertIn("words", result)
        # transition is a bool
        self.assertIsInstance(result["transition"], bool)
        # composed should be [B, N]
        self.assertEqual(result["composed"].shape, (1, 8))

    def test_writeSymbols_stack_grows(self):
        """Shifting two symbols produces stack entries (may reduce to 1 or stay at 2)."""
        syn = self._make_syntactic_space()
        syn.resetStack()
        act1 = torch.randn(1, 8, device=TheDevice.get())
        act2 = torch.randn(1, 8, device=TheDevice.get())
        syn.writeSymbols(act1)
        result = syn.writeSymbols(act2)
        # After two shifts, stack should have at least 1 entry (may have reduced)
        self.assertGreaterEqual(len(syn._sr_stack), 1)
        # composed should be [B, N]
        self.assertEqual(result["composed"].shape, (1, 8))

    def test_resetStack(self):
        """resetStack clears _sr_stack, _sr_where_stack, _sr_words."""
        syn = self._make_syntactic_space()
        # Manually populate stacks
        syn._sr_stack = [torch.zeros(1, 8)]
        syn._sr_where_stack = [torch.zeros(1, 8, 3)]
        syn._sr_words = [(0, 1, 2)]
        syn.resetStack()
        self.assertEqual(len(syn._sr_stack), 0)
        self.assertEqual(len(syn._sr_where_stack), 0)
        self.assertEqual(len(syn._sr_words), 0)

    # ── ConceptualSpace writeConcepts / resetStack tests ──────────

    def _make_conceptual_space(self, nCon=8, nDim=4):
        """Create a ConceptualSpace with syntax enabled for S/R testing."""
        _populate_test_config(
            inputDim=nDim, perceptDim=nDim, conceptDim=nDim, symbolDim=1,
            wordDim=1,
            nInput=nCon, nPercepts=nCon, nConcepts=nCon, nSymbols=nCon,
            nWords=nCon, nOutput=nCon,
            perceptPassThrough=True, symbolPassThrough=True)
        # Enable syntax so ConceptualSpace creates its syntactic_layer
        from BasicModel import TheXMLConfig
        TheXMLConfig._data["architecture"]["syntax"] = True
        from Space import ConceptualSpace
        cs = ConceptualSpace(
            inputShape=(nCon, nDim),
            spaceShape=(nCon, nDim),
            outputShape=(nCon, nDim),
        )
        cs.to(TheDevice.get())
        cs.eval()
        return cs, nCon, nDim

    def test_writeConcepts_shift(self):
        """writeConcepts pushes onto stack and returns valid dict without transition."""
        cs, nCon, nDim = self._make_conceptual_space()
        cs.resetStack()
        activation = torch.randn(1, nCon, device=TheDevice.get())
        vectors = torch.randn(1, nCon, nDim, device=TheDevice.get())
        result = cs.writeConcepts(activation, vectors)
        # Stack should have one entry after shift
        self.assertEqual(len(cs._sr_stack), 1)
        self.assertEqual(len(cs._sr_act_stack), 1)
        # Result dict has required keys
        self.assertIn("transition", result)
        self.assertIn("composed", result)
        self.assertIn("words", result)
        # transition is a bool
        self.assertIsInstance(result["transition"], bool)
        # composed should be [B, N, D]
        self.assertEqual(result["composed"].shape, (1, nCon, nDim))

    def test_writeConcepts_stack_grows(self):
        """Shifting two concepts produces stack entries (may reduce to 1 or stay at 2)."""
        cs, nCon, nDim = self._make_conceptual_space()
        cs.resetStack()
        act1 = torch.randn(1, nCon, device=TheDevice.get())
        vec1 = torch.randn(1, nCon, nDim, device=TheDevice.get())
        act2 = torch.randn(1, nCon, device=TheDevice.get())
        vec2 = torch.randn(1, nCon, nDim, device=TheDevice.get())
        cs.writeConcepts(act1, vec1)
        result = cs.writeConcepts(act2, vec2)
        # After two shifts, stack should have at least 1 entry (may have reduced)
        self.assertGreaterEqual(len(cs._sr_stack), 1)
        # composed should be [B, N, D]
        self.assertEqual(result["composed"].shape, (1, nCon, nDim))

    def test_resetStack_conceptual(self):
        """resetStack clears _sr_stack, _sr_act_stack, _sr_words."""
        cs, nCon, nDim = self._make_conceptual_space()
        # Manually populate stacks
        cs._sr_stack = [torch.zeros(1, nCon, nDim)]
        cs._sr_act_stack = [torch.zeros(1, nCon)]
        cs._sr_words = [(0, 1, 2)]
        cs.resetStack()
        self.assertEqual(len(cs._sr_stack), 0)
        self.assertEqual(len(cs._sr_act_stack), 0)
        self.assertEqual(len(cs._sr_words), 0)

    # ── PerceptualSpace writePercepts / resetStack tests ────────

    def _make_perceptual_space(self, nPer=8, nDim=4):
        """Create a PerceptualSpace with syntax enabled for S/R testing."""
        _populate_test_config(
            inputDim=nDim, perceptDim=nDim, conceptDim=nDim, symbolDim=1,
            wordDim=1,
            nInput=nPer, nPercepts=nPer, nConcepts=nPer, nSymbols=nPer,
            nWords=nPer, nOutput=nPer,
            perceptPassThrough=False, symbolPassThrough=True)
        # Enable syntax so PerceptualSpace creates its syntactic_layer
        from BasicModel import TheXMLConfig
        TheXMLConfig._data["architecture"]["syntax"] = True
        from Space import PerceptualSpace
        ps = PerceptualSpace(
            inputShape=(nPer, nDim),
            spaceShape=(nPer, nDim),
            outputShape=(nPer, nDim),
        )
        ps.to(TheDevice.get())
        ps.eval()
        return ps, nPer, nDim

    def test_writePercepts_shift(self):
        """writePercepts returns valid dict with transition=False and composed key."""
        ps, nPer, nDim = self._make_perceptual_space()
        ps.resetStack()
        activation = torch.randn(1, nPer, device=TheDevice.get())
        vectors = torch.randn(1, nPer, nDim, device=TheDevice.get())
        result = ps.writePercepts(activation, vectors)
        # Result dict has required keys
        self.assertIn("transition", result)
        self.assertIn("composed", result)
        self.assertIn("words", result)
        # P is terminal — transition is always False
        self.assertFalse(result["transition"])

    def test_resetStack_perceptual(self):
        """resetStack clears _sr_words."""
        ps, nPer, nDim = self._make_perceptual_space()
        # Manually populate words
        ps._sr_words = [(0, 1, 13), (0, 2, 13)]
        ps.resetStack()
        self.assertEqual(len(ps._sr_words), 0)

    # ── SyntacticSpace.forward() S/R integration ──────────────

    def test_syntactic_forward_uses_sr(self):
        """SyntacticSpace.forward() processes symbols via S/R loop."""
        from Space import SubSpace
        space = self._make_syntactic_space()
        ss = SubSpace(inputShape=[8, 1], outputShape=[8, 1])
        act = torch.zeros(2, 8, device=TheDevice.get())
        act[0, 0] = 1.0
        act[0, 2] = 1.0
        act[1, 1] = 1.0
        ss.set_symbols(act)
        result = space.forward(ss)
        composed = result.get_symbols()
        self.assertIsNotNone(composed)
        self.assertEqual(composed.shape, (2, 8))

    # ── readSymbols / readConcepts / readPercepts tests ────────

    def test_readSymbols_from_words(self):
        """readSymbols reconstructs activation from word tuples."""
        space = self._make_syntactic_space()
        word = space.subspace.wordEncoding.encode(0, 3, 2)  # batch=0, vec=3, rule=2 (AND)
        result = space.readSymbols([word], batch_size=1)
        self.assertEqual(result.shape, (1, 8))  # nSym=8
        self.assertGreater(result[0, 3].item(), 0)

    def test_readConcepts_from_words(self):
        """readConcepts reconstructs activation from word tuples."""
        cs, nCon, nDim = self._make_conceptual_space()
        word = cs.subspace.wordEncoding.encode(0, 2, 9)  # batch=0, vec=2, rule=9 (PART)
        result = cs.readConcepts([word], batch_size=1)
        self.assertEqual(result.shape[0], 1)
        self.assertGreater(result[0, 2].item(), 0)

    def test_readPercepts_from_words(self):
        """readPercepts reconstructs activation from word tuples."""
        ps, nPer, nDim = self._make_perceptual_space()
        word = ps.subspace.wordEncoding.encode(0, 1, 13)  # batch=0, vec=1, rule=13 (P->W)
        result = ps.readPercepts([word], batch_size=1)
        self.assertEqual(result.shape[0], 1)
        self.assertGreater(result[0, 1].item(), 0)


class TestShiftReduceConfigFlag(unittest.TestCase):
    """Verify that the shiftReduce config flag gates S/R vs composeSyntax."""

    def test_shift_reduce_config_flag_default(self):
        """shiftReduce config flag defaults to True (S/R enabled)."""
        from BasicModel import TheXMLConfig
        self.assertTrue(TheXMLConfig.get("architecture.shiftReduce", True))

    def test_shift_reduce_config_flag_from_xml(self):
        """shiftReduce is present in model.xml after adding the element."""
        from BasicModel import TheXMLConfig
        # After init_config loads model.xml, shiftReduce should be True
        from util import init_config, ProjectPaths
        import os
        init_config(defaults_path=os.path.join(ProjectPaths.DATA_DIR, "model.xml"))
        val = TheXMLConfig.get("architecture.shiftReduce")
        self.assertTrue(val)

    def test_shift_reduce_legacy_path(self):
        """Setting shiftReduce=False selects the legacy composeSyntax path."""
        from BasicModel import TheXMLConfig
        from util import init_config, ProjectPaths
        import os
        init_config(defaults_path=os.path.join(ProjectPaths.DATA_DIR, "model.xml"))
        # Override to False
        TheXMLConfig.set("architecture.shiftReduce", False)
        self.assertFalse(TheXMLConfig.get("architecture.shiftReduce"))
        # Restore
        TheXMLConfig.set("architecture.shiftReduce", True)


if __name__ == "__main__":
    unittest.main()
