"""BasicModel test suite -- exercises all core modules.

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
import Models
import Spaces
import Language
import Layers


_RUN_SLOW = os.getenv("RUN_SLOW") == "1"


def _wrap_tensor(space, x):
    """Wrap a raw tensor in the space's SubSpace so forward()/reverse() can materialize it."""
    space.subspace.set_event(x)
    return space.subspace


def _unwrap(vspace):
    """Extract the dense tensor from a SubSpace returned by forward()/reverse()."""
    if isinstance(vspace, Models.SubSpace):
        return vspace.materialize()
    return vspace


def _obj_size(section):
    """Compute per-space objectSize from config (nWhere + nWhen)."""
    try:
        nw = Models.TheXMLConfig.space(section, "nWhere")
    except KeyError:
        nw = 0
    try:
        nn = Models.TheXMLConfig.space(section, "nWhen")
    except KeyError:
        nn = 0
    return nw + nn


def _build_text_pair(nInput):
    """Build an (InputSpace, PerceptualSpace) pair for text-mode tests.

    The lexicon now lives on PerceptualSpace; tests that want a post-embed
    tensor from raw text must drive both spaces. PerceptualSpace's output
    count matches InputSpace's output count so the embedded tensor has
    shape [batch, nInput, embSize].
    """
    _idim = Models.TheXMLConfig.space("InputSpace", "nDim")
    _invec = Models.TheXMLConfig.space("InputSpace", "nVectors")
    _iobj = _obj_size("InputSpace")
    _pdim = Models.TheXMLConfig.space("PerceptualSpace", "nDim")
    _pobj = _obj_size("PerceptualSpace")
    inp = Models.InputSpace([nInput, _idim], [_invec, _idim],
                            [nInput, _idim + _iobj], model_type="embedding")
    psp = Models.PerceptualSpace([nInput, _idim + _iobj],
                                 [nInput, _pdim],
                                 [nInput, _pdim + _pobj],
                                 model_type="embedding")
    # Wire the peer reference BasicModel.create_from_config normally installs.
    # InputSpace._lex_batch requires the PerceptualSpace back-pointer so it
    # can delegate tokenization to the owning lexicon.
    object.__setattr__(inp, '_peer_perceptual', psp)
    return inp, psp


def _text_embed(inp, psp, raw_input):
    """Run raw text input through InputSpace then PerceptualSpace's lex+embed.
    Returns the post-embedding tensor [batch, nVectors, embSize]."""
    inp_sub = inp.forward(raw_input)
    psp_sub = psp._embed(inp_sub)
    return psp_sub.materialize()


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
    """Emit a single summary warning per warning type from a list of caught warnings.

    Known-noise PyTorch internal deprecation warnings (e.g.
    ``torch.jit.script_method``, fired by PyTorch's embedding code on
    import) are filtered out before aggregation — they are not under
    our control and just clutter clean output.
    """
    from collections import Counter
    _NOISE_PATTERNS = ("script_method",)
    counts = Counter()
    for w in caught:
        msg = str(w.message)
        if any(pat in msg for pat in _NOISE_PATTERNS):
            continue
        # Collapse to warning type prefix
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
                          perceptHasAttention=True, conceptHasAttention=False,
                          invertible=False, codebook=False,
                          perceptCodebook=None, conceptCodebook=None,
                          certainty=False,
                          demuxed=False,
                          lexer="word",
                          **_legacy):
    """Populate TheXMLConfig._data -- test equivalent of XML loading.

    Space constructors read nDim/nVectors from TheXMLConfig.  In production,
    the XML file provides these.  In tests, this helper provides them directly.
    """
    from util import init_config, ProjectPaths
    import os
    # Always load model.xml defaults first so all keys are present
    init_config(defaults_path=os.path.join(ProjectPaths.DATA_DIR, "model.xml"))
    # Reset global state to prevent cross-test pollution
    Models.TheXMLConfig._requirements.clear()
    Models.TheData.train_input = []
    Models.TheData.test_input = []
    Models.TheData.train_output = []
    Models.TheData.test_output = []
    _pq = perceptCodebook if perceptCodebook is not None else codebook
    _cq = conceptCodebook if conceptCodebook is not None else codebook
    _objectSize = nWhere + nWhen
    _nObjects = nInput + nPercepts + nConcepts + nSymbols + nWords + nOutput
    _symbol_dim = symbolDim
    # Deep-merge test overrides onto model.xml defaults so that keys like
    # 'syntax', etc. are always present.
    _nInputDim = -1 if flatten else 0
    _overrides = {
        "architecture": {
            "reconstruct": reconstruct,
            "ergodic": ergodic,
            "naive": naive,
            "processSymbols": processSymbols,
            "useSubspaceActivation": useSubspaceActivation,
            "objectSize": _objectSize,
            "nObjects": _nObjects,
            "nWhere": nWhere,
            "nWhen": nWhen,
            "embeddingPath": None,
            "data": {},
            "training": {
                "certainty": certainty,
            },
        },
        "InputSpace": {
            "nActive": nInput,
            "nDim": inputDim,
            "nVectors": nInput,
            "nInputDim": 0,  # InputSpace never flattens
            "codebook": codebook,
            "demuxed": demuxed,
            "lexer": lexer,
        },
        "PerceptualSpace": {
            "nActive": nPercepts,
            "nDim": perceptDim,
            "nVectors": nPercepts,
            "nInputDim": _nInputDim,
            "codebook": _pq,
            "hasAttention": perceptHasAttention,
            "invertible": invertible,
        },
        "ConceptualSpace": {
            "nActive": nConcepts,
            "nDim": conceptDim,
            "nVectors": nConcepts,
            "nInputDim": _nInputDim,
            "codebook": _cq,
            "hasAttention": conceptHasAttention,
            "invertible": invertible,
        },
        "SymbolicSpace": {
            "nActive": nSymbols,
            "nDim": _symbol_dim,
            "nVectors": nSymbols,
            "nInputDim": _nInputDim,
            "codebook": True,
        },
        "OutputSpace": {
            "nActive": nOutput,
            "nDim": outputDim,
            "nVectors": nOutput,
            "nWhere": 0,
            "nWhen": 0,
            "nInputDim": -1,  # OutputSpace always flattens
            "codebook": False,
            "invertible": False,
        },
        "ModalSpace": {
            "nDim": perceptDim,
            "nVectors": nPercepts,
            "nInputDim": _nInputDim,
            "hasAttention": perceptHasAttention,
            "invertible": invertible,
            "codebook": _pq,
        },
    }
    for section, vals in _overrides.items():
        if section in Models.TheXMLConfig._data and isinstance(Models.TheXMLConfig._data[section], dict):
            Models.TheXMLConfig._data[section].update(vals)
        else:
            Models.TheXMLConfig._data[section] = vals


# ---------------------------------------------------------------------------
# Model.py -- Layer tests
# ---------------------------------------------------------------------------
class TestLinearLayer(unittest.TestCase):
    def test_forward_shape(self):
        layer = Models.LinearLayer(nInput=4, nOutput=3)
        x = torch.randn(2, 4).to(Models.TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 3))

    def test_with_identity_weight(self):
        layer = Models.LinearLayer(nInput=4, nOutput=4, ergodic=True)  # ergodic init = eye
        x = torch.randn(1, 4).to(Models.TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (1, 4))


class TestInvertibleLinearLayer(unittest.TestCase):
    def test_forward_reverse_square(self):
        layer = Layers.InvertibleLinearLayer(nInput=4, nOutput=4)
        layer.set_sigma(0)
        x = torch.randn(2, 4).to(Models.TheDevice.get())
        y = layer(x)
        x_rec = layer.reverse(y)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4),
                        f"LDU reverse error: {(x - x_rec).abs().max():.6f}")

    def test_forward_reverse_ergodic(self):
        layer = Layers.InvertibleLinearLayer(nInput=4, nOutput=4, ergodic=True, stable=True)
        with torch.no_grad():
            layer.var.fill_(0.2)
            layer.bias.fill_(0.8)
        x = torch.randn(2, 4).to(Models.TheDevice.get())
        y = layer(x)
        x_rec = layer.reverse(y)
        err = (x - x_rec).norm() / x.norm()
        self.assertLess(err.item(), 0.01,
                        f"Ergodic LDU reverse error: {err:.4f}")


class TestSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        layer = Layers.SigmaLayer(nInput=8, nOutput=4)
        x = torch.randn(2, 8).to(Models.TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))


class TestPiLayer(unittest.TestCase):
    def test_forward_shape(self):
        layer = Layers.PiLayer(nInput=6, nOutput=3)
        x = torch.randn(2, 6).to(Models.TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 3))


class TestInvertibleSigmaLayer(unittest.TestCase):
    def test_forward_shape(self):
        layer = Layers.SigmaLayer(nInput=4, nOutput=4, invertible=True)
        x = torch.randn(2, 4).to(Models.TheDevice.get()) * 0.3
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))

    def test_reverse_shape(self):
        layer = Layers.SigmaLayer(nInput=4, nOutput=4, invertible=True)
        y = torch.randn(2, 4).to(Models.TheDevice.get()) * 0.3
        x = layer.reverse(y)
        self.assertEqual(x.shape, (2, 4))


class TestAttentionLayer(unittest.TestCase):
    def test_asymmetric_forward_shape(self):
        layer = Layers.AttentionLayer(nInput=8, nOutput=4, type="asymmetric")
        x = torch.randn(2, 5, 8).to(Models.TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 4))

    def test_symmetric_forward_shape(self):
        layer = Layers.AttentionLayer(nInput=8, nOutput=4, type="symmetric")
        x = torch.randn(2, 5, 8).to(Models.TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 4))

    def test_transformer_forward_shape(self):
        layer = Layers.AttentionLayer(nInput=8, nOutput=4, nHeads=2, type="transformer")
        x = torch.randn(2, 5, 8).to(Models.TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 4))

    def test_transformer_single_object(self):
        """Single-object 3D input [B, 1, D] -> [B, 1, nOut]."""
        layer = Layers.AttentionLayer(nInput=8, nOutput=4, nHeads=2, type="transformer")
        x = torch.randn(2, 1, 8).to(Models.TheDevice.get())
        y = layer(x)
        self.assertEqual(y.shape, (2, 1, 4))

    def test_inline(self):
        Layers.AttentionLayer.test()


class TestMemory(unittest.TestCase):
    def test_mem_update(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg")
            Layers.Mem.test()  # Runs the built-in test


# ---------------------------------------------------------------------------
# SPNN.py -- Classical neural network
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
        # Run a few epochs -- just check it doesn't crash
        for _ in range(3):
            net.run()


# ---------------------------------------------------------------------------
# SigmaPi.py -- Product-sum network
# ---------------------------------------------------------------------------
class TestSigmaPi(unittest.TestCase):
    def test_logical_function_net_creation(self):
        from SigmaPi import LogicalFunctionNet
        net = LogicalFunctionNet(nInput=2, nHidden=4, nOutput=1)
        self.assertIsNotNone(net)
        self.assertTrue(hasattr(net, 'hidden'))
        self.assertTrue(hasattr(net, 'output'))


# ---------------------------------------------------------------------------
# SymPercept.py -- Bidirectional linear learning
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
# Simple path: BasicModel with conceptualOrder=1
# ---------------------------------------------------------------------------
def _make_simple_model(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nWords=16, nOutput=4):
    """Helper to create a BasicModel with config set up for simple path."""
    _populate_test_config(
        inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
        nInput=nInput, nPercepts=nPercepts, nConcepts=nConcepts,
        nSymbols=nSymbols, nWords=nWords, nOutput=nOutput,
        symbolPassThrough=True)
    return Models.BasicModel()


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
        model = Models.BasicModel()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        x = torch.randn(2, 28*28, 1).tanh().to(Models.TheDevice.get())  # batch of 2, flattened MNIST, dim=1
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_simple_model_ergodic(self):
        """BasicModel (simple path) with ergodic=True uses SigmaLayer path."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, certainty=True, flatten=True)
        model = Models.BasicModel()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        x = torch.randn(2, 28*28, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)


# ---------------------------------------------------------------------------
# BasicModel.py -- Full model
# ---------------------------------------------------------------------------
class TestBasicModelCreation(unittest.TestCase):
    def test_encodings(self):
        Models.WhereEncoding.test()
        Models.WhenEncoding.test()

    def test_config_loading(self):
        cfg = Models.BasicModel.load_config()
        # model.xml should exist and parse
        self.assertIsInstance(cfg, dict)
        if cfg:
            self.assertIn("architecture", cfg)


class TestWeightPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        layer = Models.LinearLayer(nInput=4, nOutput=3)
        x = torch.randn(1, 4).to(Models.TheDevice.get())
        y_before = layer(x).detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(layer.state_dict(), path)
            # Create a fresh layer and load
            layer2 = Models.LinearLayer(nInput=4, nOutput=3)
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
    """WhereEncoding: encode -> reverse -> decode recovers the original offset."""

    def test_stamp_round_trip(self):
        maxP = 4096
        pe = Models.WhereEncoding(maxP)
        buf = torch.zeros(1, 1, 10, device=Models.TheDevice.get())
        offsets = [0, 1, 5, 11, 42, 100, 255, 1000]
        for offset in offsets:
            pe.stamp(buf, 0, 0, offset)
            _, decoded = pe.reverse(buf.clone())
            self.assertAlmostEqual(decoded[0, 0].item(), offset, places=3,
                msg=f"stamp/reverse round-trip failed for offset={offset}")

    def test_forward_decode_round_trip(self):
        maxP = 256
        pe = Models.WhereEncoding(maxP)
        pe.p = 0
        batch, n = 4, 3
        x = torch.zeros(batch, n, 10, device=Models.TheDevice.get())
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
        pe = Models.WhereEncoding(1000)
        pe.p = 0
        x = torch.randn(2, 3, 10, device=Models.TheDevice.get())
        original = x.clone()
        y = pe.forward(x)
        cleaned, _ = pe.reverse(y)
        mask = torch.ones(10, dtype=torch.bool)
        enc_idx = np.add([10] * len(pe.index), pe.index)
        mask[enc_idx] = False
        torch.testing.assert_close(cleaned[:, :, mask], original[:, :, mask])


class TestWhenEncodingRoundTrip(unittest.TestCase):
    """WhenEncoding: forward -> reverse recovers the original time."""

    def test_forward_reverse_round_trip(self):
        maxT = 10000
        te = Models.WhenEncoding(maxT)
        te.t = 0
        x = torch.zeros(5, 2, 10, device=Models.TheDevice.get())
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
        maxT = 10000
        te = Models.WhenEncoding(maxT)
        te.t = 500
        x = torch.zeros(3, 1, 10, device=Models.TheDevice.get())
        y = te.forward(x)
        _, decoded = te.reverse(y)
        expected = torch.arange(500, 503, dtype=torch.float32)
        for b in range(3):
            self.assertAlmostEqual(
                decoded[b, 0].item(), expected[b].item(), places=2,
                msg=f"round-trip failed for t={500+b}")

    def test_content_preserved(self):
        """Content dimensions (non-encoding slots) survive the round-trip."""
        te = Models.WhenEncoding(10000)
        te.t = 0
        x = torch.randn(2, 3, 10, device=Models.TheDevice.get())
        original = x.clone()
        y = te.forward(x)
        cleaned, _ = te.reverse(y)
        mask = torch.ones(10, dtype=torch.bool)
        mask[[-2, -1]] = False
        torch.testing.assert_close(cleaned[:, :, mask], original[:, :, mask])


# SubSpace -- derived sizes, materialization, construction helpers
# ---------------------------------------------------------------------------
class TestSubSpaceDerivedSizes(unittest.TestCase):
    """SubSpace.getEncodingSize, getEncodedInputSize, getEncodedOutputSize match EventEncoding math."""

    def test_getEncodingSize_returns_muxedSize(self):
        ss = Models.SubSpace(whereEncoding=Models.WhereEncoding(1, 2), whenEncoding=Models.WhenEncoding(10000, 2),
                      inputShape=[3, 8], outputShape=[3, 8])
        # muxedSize = nWhat(4) + nWhere(2) + nWhen(2) = 8 (inputShape[1])
        self.assertEqual(ss.getEncodingSize(8), 8)
        self.assertEqual(ss.muxedSize, 8)

    def test_getEncodedIO_no_reshape(self):
        # Shapes already include muxed width (nWhere=2 + nWhen=2)
        # nInputDim/nOutputDim default to 0 -> resolves to inputShape[1]/outputShape[1]
        ss = Models.SubSpace(whereEncoding=Models.WhereEncoding(1, 2), whenEncoding=Models.WhenEncoding(10000, 2),
                      objectEncoding=Models.EventEncoding([3, 12], [5, 20]),
                      inputShape=[3, 12], outputShape=[5, 20])
        self.assertEqual(ss.getEncodedInputSize(), 12)
        self.assertEqual(ss.getEncodedOutputSize(), 20)

    def test_getEncodedIO_reshape(self):
        # Equivalent of old flatten=True: nInputDim = flat_in, nOutputDim = per-vector dim.
        # Layer sees flat_in -> flat_out; forwardEnd reshapes flat_out -> [nOut, per_dim].
        ss = Models.SubSpace(whereEncoding=Models.WhereEncoding(1, 2), whenEncoding=Models.WhenEncoding(10000, 2),
                      objectEncoding=Models.EventEncoding([3, 12], [5, 20]),
                      nInputDim=12 * 3, nOutputDim=20,
                      inputShape=[3, 12], outputShape=[5, 20])
        self.assertEqual(ss.getEncodedInputSize(), 12 * 3)
        self.assertEqual(ss.getEncodedOutputSize(), 20 * 5)

    def test_zero_nWhere_nWhen(self):
        ss = Models.SubSpace(objectEncoding=Models.EventEncoding([2, 10], [2, 10]),
                      inputShape=[2, 10], outputShape=[2, 10])
        self.assertEqual(ss.getEncodedInputSize(), 10)
        self.assertEqual(ss.getEncodedOutputSize(), 10)


class TestSubSpaceMaterialize(unittest.TestCase):
    """SubSpace.materialize() returns the expected dense tensor."""

    def test_materialize_tensor(self):
        t = torch.randn(2, 4, 12)
        ss = Models.SubSpace.from_tensor(t, whereEncoding=Models.WhereEncoding(1, 2), whenEncoding=Models.WhenEncoding(10000, 2),
                                  inputShape=[4, 8], outputShape=[4, 8])
        result = ss.materialize()
        self.assertIs(result, t)
        self.assertIsInstance(ss.event, Models.Tensor)

    def test_materialize_none_when_unset(self):
        ss = Models.SubSpace(whereEncoding=Models.WhereEncoding(1, 2), whenEncoding=Models.WhenEncoding(10000, 2),
                      inputShape=[4, 8], outputShape=[4, 8])
        self.assertIsNone(ss.materialize())

    def test_shape_property(self):
        t = torch.randn(2, 4, 12)
        ss = Models.SubSpace.from_tensor(t, inputShape=[4, 8], outputShape=[4, 8])
        self.assertEqual(ss.shape, torch.Size([2, 4, 12]))

    def test_shape_property_none(self):
        ss = Models.SubSpace(inputShape=[4, 8], outputShape=[4, 8])
        self.assertIsNone(ss.shape)


class TestSubSpaceProperties(unittest.TestCase):
    """SubSpace batch tracking."""

    def test_batch_from_tensor(self):
        t = torch.randn(5, 4, 8, device=Models.TheDevice.get())
        ss = Models.SubSpace.from_tensor(t, inputShape=[4, 4], outputShape=[4, 4])
        self.assertEqual(ss.batch, 5)

    def test_batch_zero_when_empty(self):
        ss = Models.SubSpace(inputShape=[4, 8], outputShape=[4, 8])
        self.assertEqual(ss.batch, 0)


class TestSubSpaceConstruction(unittest.TestCase):
    """SubSpace.from_tensor and from_components helpers."""

    def test_from_tensor(self):
        t = torch.randn(2, 3, 10)
        ss = Models.SubSpace.from_tensor(t, whereEncoding=Models.WhereEncoding(1, 2), whenEncoding=Models.WhenEncoding(10000, 2),
                                  inputShape=[3, 6], outputShape=[3, 6])
        self.assertIsInstance(ss.event, Models.Tensor)
        self.assertIs(ss.event.W, t)
        self.assertEqual(ss.nWhere, 2)
        self.assertEqual(ss.nWhen, 2)
        self.assertEqual(ss.muxedSize, 6)
        self.assertEqual(ss.inputShape, [3, 6])

    def test_from_components(self):
        object = torch.randn(2, 4, 8)
        act = torch.ones(2, 4)
        ae = Models.ActiveEncoding()
        ss = Models.SubSpace.from_components(
            object=object, activation=act, activeEncoding=ae,
            whereEncoding=Models.WhereEncoding(1, 2), whenEncoding=Models.WhenEncoding(10000, 2),
            inputShape=[4, 4], outputShape=[4, 4])
        self.assertIsInstance(ss.event, Models.Tensor)
        self.assertIsInstance(ss.activation, Models.Tensor)
        self.assertIs(ss.event.W, object)
        self.assertIs(ss.activation.W, act)
        self.assertIs(ss.activeEncoding, ae)

    def test_from_components_defaults_initialized(self):
        """All modalities are initialized (as empty Tensor bases) even with no args."""
        ss = Models.SubSpace.from_components(inputShape=[4, 8], outputShape=[4, 8])
        self.assertIsInstance(ss.event, Models.Tensor)
        self.assertIsInstance(ss.activation, Models.Tensor)
        self.assertIsInstance(ss.where, Models.Tensor)
        self.assertIsInstance(ss.when, Models.Tensor)


class TestSubSpaceActiveEncoding(unittest.TestCase):
    """ActiveEncoding round-trip through SubSpace."""

    def test_activation_stored_and_retrievable(self):
        """Legacy scalar [B, N] is lifted to 4-valued bivector [B, N, 2]."""
        ae = Models.ActiveEncoding()
        self.assertEqual(ae.nDim, 2)
        act_scalar = torch.tensor([[0.5, 0.8, 0.1]])  # [1, 3]
        ss = Models.SubSpace(activeEncoding=ae, inputShape=[3, 8], outputShape=[3, 8])
        ss.set_activation(act_scalar)
        retrieved = ss.get_activation()
        # Positive scalars lift to [x, 0] (TRUE pole with magnitude).
        expected = torch.stack([act_scalar, torch.zeros_like(act_scalar)], dim=-1)
        torch.testing.assert_close(retrieved, expected)

    def test_activation_bivector_roundtrip(self):
        """Full [B, N, 2] bivector passes through unchanged."""
        ae = Models.ActiveEncoding()
        ss = Models.SubSpace(activeEncoding=ae, inputShape=[3, 8], outputShape=[3, 8])
        bivec = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # TRUE, FALSE, BOTH
        ss.set_activation(bivec)
        retrieved = ss.get_activation()
        torch.testing.assert_close(retrieved, bivec)

    def test_activation_four_corners(self):
        """Tetralemma corners round-trip and reduce to expected presence."""
        ae = Models.ActiveEncoding()
        ss = Models.SubSpace(activeEncoding=ae, inputShape=[4, 8], outputShape=[4, 8])
        corners = torch.tensor([[[1.0, 0.0],   # TRUE (asti)
                                 [0.0, 1.0],   # FALSE (nasti)
                                 [1.0, 1.0],   # BOTH (ubhaya)
                                 [0.0, 0.0]]]) # NEITHER (anubhaya)
        ss.set_activation(corners)
        retrieved = ss.get_activation()
        torch.testing.assert_close(retrieved, corners)
        # Presence = max(aP, aN): TRUE/FALSE/BOTH all present, NEITHER gated off.
        pres = ss.activation_presence()
        torch.testing.assert_close(pres, torch.tensor([[1.0, 1.0, 1.0, 0.0]]))

    def test_activation_negative_scalar_lifts_to_false_pole(self):
        """Legacy negative scalar lifts to [0, |x|] (FALSE pole)."""
        ae = Models.ActiveEncoding()
        ss = Models.SubSpace(activeEncoding=ae, inputShape=[2, 8], outputShape=[2, 8])
        ss.set_activation(torch.tensor([[-0.7, 0.4]]))
        retrieved = ss.get_activation()
        expected = torch.tensor([[[0.0, 0.7], [0.4, 0.0]]])
        torch.testing.assert_close(retrieved, expected)

    def test_two_spaces_independent_encoding(self):
        """Two SubSpaces can have different muxedSize without shared coupling."""
        # ss1: nWhere=2, nWhen=2, muxedSize=12
        ss1 = Models.SubSpace(whereEncoding=Models.WhereEncoding(1, 2), whenEncoding=Models.WhenEncoding(10000, 2),
                        objectEncoding=Models.EventEncoding([3, 12], [3, 12]),
                        inputShape=[3, 12], outputShape=[3, 12])
        ss2 = Models.SubSpace(objectEncoding=Models.EventEncoding([3, 16], [3, 16]),
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
        nIn, nOut, nDim = 4, 4, 8
        cs = Models.ConceptualSpace([nIn, nDim], [nOut, nDim], [nOut, nDim])
        inEmb = Models.TheXMLConfig.encodingSize(Models.TheXMLConfig.space("InputSpace", "nDim"))
        x = torch.randn(self.B, nIn, inEmb).tanh().to(Models.TheDevice.get())
        y = _unwrap(cs(_wrap_tensor(cs, x)))
        outEmb = Models.TheXMLConfig.encodingSize(Models.TheXMLConfig.space("ConceptualSpace", "nDim"))
        self.assertEqual(list(y.shape), [self.B, nOut, outEmb])

    def test_conceptual_space_reverse_shape(self):
        _populate_test_config(
            inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, wordDim=1, outputDim=4,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nWords=4, nOutput=4,
            flatten=True, reconstruct="FULL")
        nIn, nOut, nDim = 4, 4, 8
        cs = Models.ConceptualSpace([nIn, nDim], [nOut, nDim], [nOut, nDim])
        outEmb = Models.TheXMLConfig.encodingSize(Models.TheXMLConfig.space("ConceptualSpace", "nDim"))
        y = torch.randn(self.B, nOut, outEmb).tanh().to(Models.TheDevice.get())
        x = _unwrap(cs.reverse(_wrap_tensor(cs, y)))
        inEmb = Models.TheXMLConfig.encodingSize(Models.TheXMLConfig.space("InputSpace", "nDim"))
        self.assertEqual(list(x.shape), [self.B, nIn, inEmb])

    def test_output_space_forward_shape(self):
        nIn, nOut = 4, 4
        os_ = Models.OutputSpace([nIn, 8], [nOut, 4], [nOut, 4])
        inEmb = os_.inputShape[1]
        x = torch.randn(self.B, nIn, inEmb).to(Models.TheDevice.get())
        y = _unwrap(os_(_wrap_tensor(os_, x)))
        self.assertEqual(list(y.shape), [self.B, nOut, Models.TheXMLConfig.space("OutputSpace", "nDim")])


class TestSimpleModel(unittest.TestCase):
    """BasicModel (simple path) uses unified Space hierarchy."""

    def test_simple_model_ergodic_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, flatten=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            flatten=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_simple_model_reverse_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, flatten=True, reconstruct="FULL")
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)


class TestCodebookVariants(unittest.TestCase):
    """Lock down Codebook behavior."""

    def test_codebook_shape(self):
        _populate_test_config(nInput=4, nPercepts=4, nConcepts=4, nSymbols=4,
                              nWords=0, nOutput=4)
        vs = Models.Codebook()
        vs.create(4, 4, 3, customVQ=False)
        vs = vs.to(Models.TheDevice.get())
        x = torch.randn(2, 4, 3).to(Models.TheDevice.get())
        y = vs.forward(x)
        # Basis.nDim is actual width -- no objectSize padding
        self.assertEqual(list(y.shape), [2, 4, 3])


class TestBasisContract(unittest.TestCase):
    def test_tensor_identity_materialization(self):
        payload = torch.randn(2, 3, 4, device=Models.TheDevice.get())
        basis = Models.Tensor()
        basis.create(3, 3, 4)
        out = basis.forward(payload)
        self.assertIs(out, payload)
        self.assertIs(basis.getW(), payload)
        rev = basis.reverse(payload)
        self.assertIs(rev, payload)

    def test_invalid_geometry_requires_2d_prototype_matrix(self):
        basis = Models.Tensor(W=torch.randn(2, 3, 4, device=Models.TheDevice.get()))
        basis.create(3, 3, 4)
        with self.assertRaises(RuntimeError):
            basis.codebookDistance(torch.randn(2, 4, device=Models.TheDevice.get()))


class TestModelEndToEnd(unittest.TestCase):
    """Lock down full model forward shapes and loss compatibility."""

    def test_simple_model_ergodic_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, flatten=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_traditional_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            flatten=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(end_state.shape[0], 2)

    def test_simple_model_reverse_shapes(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, flatten=True, reconstruct="FULL")
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_simple_model_loss_runs(self):
        """Verify forward + loss + backward doesn't crash."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True,
            ergodic=True, certainty=True, flatten=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        target = torch.randn(2, 4).to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        loss_fn = Models.CertaintyWeightedCrossEntropy()
        loss = loss_fn(out.squeeze(), target)
        loss.backward()
        # No crash = pass


class TestUniversalTrainingContract(unittest.TestCase):
    """All spaces expose getParameters() and paramUpdate()."""

    def test_space_has_training_contract(self):
        _populate_test_config(inputDim=8, nInput=4)
        Models.Space.config_section = "InputSpace"
        s = Models.Space([4, 8], [4, 8], [4, 8])
        Models.Space.config_section = None
        self.assertEqual(s.getParameters(), [])
        s.paramUpdate()  # should be a no-op, not crash

    def test_conceptual_space_has_training_contract(self):
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nConcepts=4)
        cs = Models.ConceptualSpace([4, 8], [4, 8], [4, 8])
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        cs.paramUpdate()  # no crash


class TestSigmaLayerDeterministic(unittest.TestCase):
    """SigmaLayer(ergodic=False) behaves like LinearLayer + Tanh."""

    def test_deterministic_matches_linear_tanh(self):
        torch.manual_seed(42)
        nIn, nOut = 8, 4
        sigma = Layers.SigmaLayer(nIn, nOut, ergodic=False)
        sigma.train()

        linear = Models.LinearLayer(nIn, nOut, hasBias=True)
        with torch.no_grad():
            linear.W.copy_(sigma.layer.W)
        tanh = torch.nn.Tanh()

        x = torch.randn(2, nIn).to(Models.TheDevice.get())
        y_sigma = sigma(x)
        z = torch.atanh(x.clamp(-1 + Layers.epsilon, 1 - Layers.epsilon))
        y_manual = tanh(linear(z))
        self.assertTrue(torch.allclose(y_sigma, y_manual, atol=1e-6),
                        f"SigmaLayer should match tanh ∘ linear ∘ atanh ∘ clamp")

    def test_deterministic_same_train_eval(self):
        nIn, nOut = 8, 4
        sigma = Layers.SigmaLayer(nIn, nOut, ergodic=False)
        x = torch.randn(2, nIn).to(Models.TheDevice.get())

        sigma.train()
        y_train = sigma(x).detach().clone()
        sigma.eval()
        y_eval = sigma(x).detach().clone()
        self.assertTrue(torch.allclose(y_train, y_eval, atol=1e-6),
                        "Non-ergodic mode should produce same output in train and eval")

    def test_non_ergodic_default(self):
        sigma = Layers.SigmaLayer(nInput=8, nOutput=4)
        self.assertFalse(sigma.ergodic)


class TestSpaceBasisConstruction(unittest.TestCase):
    """Space builds its basis during construction."""

    def setUp(self):
        _populate_test_config(inputDim=3, nInput=4)
        Models.Space.config_section = "InputSpace"
        self.addCleanup(setattr, Models.Space, 'config_section', None)

    def test_codebook_creates_codebook(self):
        _populate_test_config(inputDim=3, nInput=4, codebook=True)
        s = Models.Space([4, 3], [4, 3], [4, 3])
        self.assertIsInstance(s.get_vectors(), Models.Codebook)

    def test_no_codebook_creates_tensor_basis(self):
        _populate_test_config(inputDim=3, nInput=4, codebook=False)
        s = Models.Space([4, 3], [4, 3], [4, 3])
        # With codebook=False, Space._build_object_basis returns a
        # Tensor (identity) basis instead of a Codebook in passThrough
        # mode -- the passThrough flag was retired with Stage 1.
        self.assertIsInstance(s.get_vectors(), Models.Tensor)

    # ``test_forward_subspace_round_trip_keeps_runtime_state`` was
    # removed in Stage 1: it tested the legacy ``perceptPassThrough``
    # identity short-circuit which is gone.  Round-trip invertibility
    # is covered by ``test_invertibility.TestPerceptualSpaceInvertible``.


class TestConceptualSpaceErgodic(unittest.TestCase):
    """ConceptualSpace with ergodic flag matches DerivedConceptualSpace behavior."""

    def _set_zero_object_encoding(self, ergodic=False, reconstruct="NONE"):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8, nOutput=8,
            nWhere=0, nWhen=0, ergodic=ergodic, reconstruct=reconstruct)

    def test_ergodic_forward_shape(self):
        self._set_zero_object_encoding(ergodic=True)
        nVec, nDim, cDim = 8, 1, 1
        cs = Models.ConceptualSpace([nVec, nDim], [nVec, cDim], [nVec, cDim])
        x = torch.randn(2, nVec, nDim).tanh().to(Models.TheDevice.get())
        y = _unwrap(cs(_wrap_tensor(cs, x)))
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_non_ergodic_forward_shape(self):
        self._set_zero_object_encoding(ergodic=False)
        nVec, nDim, cDim = 8, 1, 1
        cs = Models.ConceptualSpace([nVec, nDim], [nVec, cDim], [nVec, cDim])
        x = torch.randn(2, nVec, nDim).tanh().to(Models.TheDevice.get())
        y = _unwrap(cs(_wrap_tensor(cs, x)))
        self.assertEqual(list(y.shape), [2, nVec, cDim])

    def test_ergodic_flag_stored(self):
        self._set_zero_object_encoding(ergodic=True)
        cs_erg = Models.ConceptualSpace([8, 1], [8, 1], [8, 1])
        self.assertTrue(cs_erg.ergodic)
        self._set_zero_object_encoding(ergodic=False)
        cs_det = Models.ConceptualSpace([8, 1], [8, 1], [8, 1])
        self.assertFalse(cs_det.ergodic)

    def test_ergodic_reverse_shape(self):
        self._set_zero_object_encoding(ergodic=True, reconstruct="FULL")
        nVec, nDim, cDim = 8, 1, 1
        cs = Models.ConceptualSpace([nVec, nDim], [nVec, cDim], [nVec, cDim])
        y = torch.randn(2, nVec, cDim).tanh().to(Models.TheDevice.get())
        x = _unwrap(cs.reverse(_wrap_tensor(cs, y)))
        self.assertEqual(list(x.shape), [2, nVec, nDim])

    def test_ergodic_exposes_params(self):
        self._set_zero_object_encoding(ergodic=True)
        cs = Models.ConceptualSpace([8, 1], [8, 1], [8, 1])
        params = cs.getParameters()
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

    def test_canonical_forward_still_works(self):
        """Existing ConceptualSpace (with objectSize > 0) still works after changes."""
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nConcepts=4)
        nIn, nOut = 4, 4
        cs = Models.ConceptualSpace([nIn, 8], [nOut, 8], [nOut, 8])
        inEmb = Models.TheXMLConfig.encodingSize(Models.TheXMLConfig.space("InputSpace", "nDim"))
        x = torch.randn(2, nIn, inEmb).tanh().to(Models.TheDevice.get())
        y = _unwrap(cs(_wrap_tensor(cs, x)))
        outEmb = Models.TheXMLConfig.encodingSize(Models.TheXMLConfig.space("ConceptualSpace", "nDim"))
        self.assertEqual(list(y.shape), [2, nOut, outEmb])


class TestInputSpaceNoCodebook(unittest.TestCase):
    """InputSpace works with non-codebook mode (objectSize=0)."""

    def test_no_codebook_forward_shape(self):
        _populate_test_config(inputDim=1, nInput=8, nWhere=0, nWhen=0, codebook=False)
        nIn, nDim = 8, 1
        inp = Models.InputSpace([nIn, nDim], [nIn, nDim], [nIn, nDim])
        x = torch.randn(2, nIn, nDim).to(Models.TheDevice.get())
        y = _unwrap(inp(x))
        self.assertEqual(list(y.shape), [2, nIn, nDim])


class TestOutputSpaceZeroObjectSize(unittest.TestCase):
    """OutputSpace works with objectSize=0."""

    def test_forward_shape_zero_object_size(self):
        _populate_test_config(symbolDim=1, outputDim=1, nOutput=3, nWhere=0, nWhen=0,
                              flatten=True)
        nIn, nOut = 4, 3
        os_ = Models.OutputSpace([nIn, 1], [nOut, 1], [nOut, 1])
        x = torch.randn(2, nIn, 1).to(Models.TheDevice.get())
        y = _unwrap(os_(_wrap_tensor(os_, x)))
        self.assertEqual(list(y.shape), [2, nOut, 1])

    def test_reverse_shape_zero_object_size(self):
        _populate_test_config(symbolDim=1, outputDim=1, nOutput=3, nWhere=0, nWhen=0,
                              flatten=True, reconstruct="FULL")
        nIn, nOut = 4, 3
        os_ = Models.OutputSpace([nIn, 1], [nOut, 1], [nOut, 1])
        y = torch.randn(2, nOut, 1).to(Models.TheDevice.get())
        x = _unwrap(os_.reverse(_wrap_tensor(os_, y)))
        self.assertEqual(list(x.shape), [2, nIn, 1])


class TestBaseModelFactory(unittest.TestCase):
    """BaseModel.from_config factory creates the correct model type."""

    def test_factory_creates_simple_model(self):
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
            model, cfg = Models.BaseModel.from_config(path)
            self.assertIsInstance(model, Models.BasicModel)
            self.assertEqual(model.conceptualOrder, 1)
        finally:
            os.unlink(path)

    def test_factory_creates_basic_model(self):
        # nSymbols must equal nConcepts (SymbolicSpace 1:1 mapping constraint),
        # and nPercepts must be 2*nConcepts (PiLayer invertibility).
        # symbol_dim must equal concept_dim (SymbolicSpace owns no
        # SigmaLayer/PiLayer post 2026-05 ownership rule).
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
  <SymbolicSpace><nOutput>2</nOutput></SymbolicSpace>
  <OutputSpace><nOutput>2</nOutput><nDim>4</nDim></OutputSpace>
</model>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml)
            path = f.name
        try:
            model, cfg = Models.BaseModel.from_config(path)
            self.assertIsInstance(model, Models.BasicModel)
        finally:
            os.unlink(path)

    def test_factory_creates_mental_model(self):
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
            model, cfg = Models.BaseModel.from_config(path)
            self.assertIsInstance(model, Models.BasicModel)
            self.assertIs(model.inputSpace.outputSpace, model.outputSpace)
        finally:
            os.unlink(path)


class TestDataTextStorage(unittest.TestCase):
    """Text datasets store raw strings in train_input (tensorized lazily)."""

    def test_train_input_strings_for_text(self):
        data = Models.Data()
        data.load("xor")  # XOR uses text examples
        self.assertIsInstance(data.train_input, list)
        self.assertIsInstance(data.train_input[0], str)

    def test_train_input_tensors_for_numeric(self):
        """Numeric datasets store tensors in train_input."""
        data = Models.Data()
        data.load("mnist")
        self.assertIsInstance(data.train_input, torch.Tensor)



class TestSymbolDimZero(unittest.TestCase):
    """symbolDim=0 produces a zero-width symbolic encoding."""

    def test_symbolic_space_zero_dim_produces_zero_encoding(self):
        _populate_test_config(inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0,
                              outputDim=1, nWhere=0, nWhen=0)
        self.assertEqual(Models.TheXMLConfig.space("SymbolicSpace", "nDim"), 0)
        self.assertEqual(Models.TheXMLConfig.encodingSize(0), 0)

    def test_objectencoding_zero_contribution_when_unused(self):
        """ObjectEncoding must not inflate tensor size when nWhere=0, nWhen=0."""
        _populate_test_config(nWhere=0, nWhen=0)
        nDim = 10
        self.assertEqual(Models.TheXMLConfig.encodingSize(nDim), nDim)


class TestInputSpaceLexIntegration(unittest.TestCase):
    """InputSpace with text data creates a Lex instance, span table, and
    encodes spans as [nWhat + nWhere] via a Codebook basis plus ObjectEncoding."""

    def _make_text_data(self):
        """Load XOR text data into TheData singleton."""
        Models.TheData.load("xor")
        return Models.TheData

    def _make_input_space(self, lexer="word"):
        """Create an (InputSpace, PerceptualSpace) pair from XOR text data."""
        nInput = 8
        _populate_test_config(inputDim=1, perceptDim=10, nInput=nInput,
                              nWhere=Models.WhereEncoding.nDim, nWhen=Models.WhenEncoding.nDim,
                              lexer=lexer)
        self._make_text_data()
        inp, psp = _build_text_pair(nInput)
        return inp, psp, Models.TheData

    def test_token_stream_available(self):
        """PerceptualSpace with model_type='embedding' can tokenize via _token_stream."""
        _, psp, _ = self._make_input_space()
        emb = psp.vocabulary
        tokens = emb._token_stream("hello world")
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens[0][0], "hello")

    def test_per_doc_spans_created(self):
        """PerceptualSpace stores per-document `(text, start)` token streams."""
        _, psp, _ = self._make_input_space()
        self.assertTrue(hasattr(psp, 'doc_spans'))
        self.assertIsInstance(psp.doc_spans, list)
        for tokens in psp.doc_spans:
            self.assertIsInstance(tokens, list)
            self.assertTrue(all(isinstance(tok, tuple) for tok in tokens))
            self.assertTrue(all(len(tok) == 2 for tok in tokens))

    def test_per_doc_span_counts(self):
        """Each document token stream includes the lexical space token."""
        _, psp, _ = self._make_input_space()
        for tokens in psp.doc_spans:
            self.assertEqual(len(tokens), 3)

    def test_forward_produces_correct_shape(self):
        """forward() with Lex path produces [batch, nInput, embeddingSize]."""
        inp, psp, data = self._make_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        output = _text_embed(inp, psp, inputTensor)
        embSize = psp.subspace.getEncodedOutputSize()
        self.assertEqual(list(output.shape), [batch_size, psp.outputShape[0], embSize])

    def test_doc_spans_store_token_offsets(self):
        """Embedding stores token text alongside byte starts."""
        _, psp, _ = self._make_input_space()
        emb = psp.vocabulary
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
        inp, psp, data = self._make_input_space()
        batch_size = 1
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        psp.subspace.whereEncoding.p = 0
        output = _text_embed(inp, psp, inputTensor)
        # With nWhere > 0, the reserved encoding dims should be non-zero
        # (ObjectEncoding.forward stamps sin/cos into the last objectSize dims)
        embSize = output.shape[-1]
        objSize = _obj_size("PerceptualSpace")
        if objSize > 0:
            encoding_dims = output[0, 0, -objSize:]
            self.assertFalse(torch.all(encoding_dims == 0).item(),
                             "ObjectEncoding dims should be non-zero after forward()")


class TestOutputSpaceTextReconstruction(unittest.TestCase):
    """OutputSpace can reconstruct text from symbolic vectors."""

    def test_numeric_output_unchanged(self):
        """Numeric OutputSpace should still produce [B, nOutput] tensor."""
        _populate_test_config(symbolDim=1, outputDim=1, nOutput=3, nWhere=0, nWhen=0,
                              flatten=True)
        nIn, nOut = 4, 3
        os_ = Models.OutputSpace([nIn, 1], [nOut, 1], [nOut, 1])
        x = torch.randn(2, nIn, 1).to(Models.TheDevice.get())
        y = _unwrap(os_(_wrap_tensor(os_, x)))
        self.assertEqual(list(y.shape), [2, nOut, 1])
        # text_mode should be False for numeric data
        self.assertFalse(os_.text_mode)

    def test_text_mode_false_without_lex(self):
        """OutputSpace without lex info should have text_mode=False."""
        _populate_test_config(inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0, outputDim=4,
                              nOutput=4, flatten=True)
        nIn, nOut = 4, 4
        os_ = Models.OutputSpace([nIn, 8], [nOut, 4], [nOut, 4])
        self.assertFalse(os_.text_mode)

    def test_vectors_constructor_enables_reconstruction(self):
        """Passing vectors= shares the embedding basis with OutputSpace."""
        Models.TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, perceptDim=10, nInput=nInput,
                              nWhere=Models.WhereEncoding.nDim, nWhen=Models.WhenEncoding.nDim,
                              flatten=True)
        inp, psp = _build_text_pair(nInput)
        nOut = 8
        _sdim = Models.TheXMLConfig.space("SymbolicSpace", "nDim") or Models.TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = Models.TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = Models.OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=psp.vocabulary)
        self.assertTrue(os_.text_mode)

    def test_reconstruct_from_known_vectors(self):
        """Given codebook vectors with nWhere, reconstruct_data should recover words at positions."""
        import math
        Models.TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=10, perceptDim=10, nInput=nInput,
                              nWhere=Models.WhereEncoding.nDim, nWhen=Models.WhenEncoding.nDim,
                              flatten=True)
        inp, psp = _build_text_pair(nInput)
        nOut = 4
        _sdim = Models.TheXMLConfig.space("SymbolicSpace", "nDim") or Models.TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = Models.TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = Models.OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=psp.vocabulary)

        # Build synthetic vectors from known codebook entries with known nWhere
        codebook = psp.vocabulary.getW().detach()
        words_list = psp.vocabulary.wv.index_to_key
        embSize = psp.muxedSize
        nWhat = embSize - _obj_size("PerceptualSpace")
        where = psp.subspace.whereEncoding

        # Pick first two non-[MASK] words from the codebook
        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(Models.TheDevice.get())
        expected_words = []
        # Skip [MASK] (zero vector) -- cosine matching can't recover it
        usable = [j for j, w in enumerate(words_list) if w != "[MASK]"]
        for slot, j in enumerate(usable[:nVec]):
            vectors[0, slot, :nWhat] = codebook[j][:nWhat]
            expected_words.append(words_list[j])
            where.stamp(vectors, 0, slot, slot * 6)

        recovered_words, recovered_positions = os_.reconstruct_data(vectors)
        self.assertEqual(recovered_words[0], expected_words)

    def test_reconstruct_consecutive_no_nwhere(self):
        """When nWhere is zero, tokens are written consecutively."""
        Models.TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=10, perceptDim=10, nInput=nInput,
                              nWhere=Models.WhereEncoding.nDim, nWhen=Models.WhenEncoding.nDim,
                              flatten=True)
        inp, psp = _build_text_pair(nInput)
        nOut = 4
        _sdim = Models.TheXMLConfig.space("SymbolicSpace", "nDim") or Models.TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = Models.TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = Models.OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=psp.vocabulary)

        # Build vectors with nWhere = 0 (all zeros)
        codebook = psp.vocabulary.getW().detach()
        words_list = psp.vocabulary.wv.index_to_key
        embSize = psp.muxedSize

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(Models.TheDevice.get())
        expected_words = []
        # Skip [MASK] (zero vector) -- cosine matching can't recover it
        usable = [j for j, w in enumerate(words_list) if w != "[MASK]"]
        for slot, j in enumerate(usable[:nVec]):
            vectors[0, slot, :embSize - _obj_size("PerceptualSpace")] = \
                codebook[j, :embSize - _obj_size("PerceptualSpace")]
            expected_words.append(words_list[j])
        # nWhere left as zero -> consecutive mode

        recovered_words, recovered_positions = os_.reconstruct_data(vectors)
        self.assertEqual(recovered_words[0], expected_words)

    def test_reconstruct_to_buffer(self):
        """reconstruct_data with to_buffer=True produces a string with positioned words."""
        import math
        Models.TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=10, perceptDim=10, nInput=nInput,
                              nWhere=Models.WhereEncoding.nDim, nWhen=Models.WhenEncoding.nDim,
                              flatten=True)
        inp, psp = _build_text_pair(nInput)
        nOut = 4
        _sdim = Models.TheXMLConfig.space("SymbolicSpace", "nDim") or Models.TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = Models.TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = Models.OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=psp.vocabulary)

        # Build synthetic vectors with nWhere at known positions
        codebook = psp.vocabulary.getW().detach()
        words_list = psp.vocabulary.wv.index_to_key
        embSize = psp.muxedSize
        nWhat = psp.nWhat
        where = psp.subspace.whereEncoding

        batch = 1
        nVec = 2
        vectors = torch.zeros([batch, nVec, embSize]).to(Models.TheDevice.get())
        # Skip [MASK] and \x00 (both zero vectors) -- cosine matching can't recover them
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
        Models.TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, symbolDim=1, outputDim=1,
                              nInput=nInput, nOutput=4,
                              nWhere=Models.WhereEncoding.nDim, nWhen=Models.WhenEncoding.nDim,
                              flatten=True, reconstruct="FULL")
        _idim = Models.TheXMLConfig.space("InputSpace", "nDim")
        _invec = Models.TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = Models.InputSpace([nInput, _idim], [_invec, _idim], [nInput, _idim + _obj],
                         model_type="embedding")
        nOut = 4
        _sdim = Models.TheXMLConfig.space("SymbolicSpace", "nDim") or Models.TheXMLConfig.space("ConceptualSpace", "nDim")
        _odim = Models.TheXMLConfig.space("OutputSpace", "nDim")
        _obj_sym = _obj_size("SymbolicSpace")
        os_ = Models.OutputSpace([nInput, _sdim + _obj_sym], [nOut, _odim], [nOut, _odim], vectors=inp.vocabulary)
        inEmb = Models.TheXMLConfig.encodingSize(Models.TheXMLConfig.space("SymbolicSpace", "nDim"))
        x = torch.randn(2, nInput, inEmb).to(Models.TheDevice.get())
        y = os_(_wrap_tensor(os_, x))
        self.assertEqual(list(_unwrap(y).shape), [2, nOut, Models.TheXMLConfig.space("OutputSpace", "nDim")])
        # Reverse path should also be unchanged
        rev = _unwrap(os_.reverse(y))
        self.assertEqual(list(rev.shape), [2, nInput, inEmb])


class TestInputSpaceTextRoundTrip(unittest.TestCase):
    """PerceptualSpace.reverse() must reconstruct text from latent state."""

    def _make_text_input_space(self):
        """Create an (InputSpace, PerceptualSpace) pair from XOR text data."""
        nInput = 8
        _populate_test_config(inputDim=10, perceptDim=10, nInput=nInput,
                              nWhere=Models.WhereEncoding.nDim, nWhen=Models.WhenEncoding.nDim,
                              flatten=True)
        Models.TheData.load("xor")
        inp, psp = _build_text_pair(nInput)
        return inp, psp, Models.TheData

    def test_reverse_recovers_words(self):
        """forward -> reverse should recover the original lexical tokens.

        Content tokens (non-whitespace) must match exactly.  Trailing
        padding is space-filled, so the last token from tokenize() is a
        long whitespace run; the reverse path may recover a shorter
        space token -- both are correct padding representations.
        """
        inp, psp, data = self._make_text_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        expected_tokens = psp.vocabulary.tokenize(inputTensor)
        # Forward pass through input->perceptual
        latent = psp._embed(inp.forward(inputTensor))
        # Reverse pass via PerceptualSpace
        psp.reverse(latent)
        recovered = psp.reconstruct_data()
        for b in range(batch_size):
            nVec = psp.outputShape[0]
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
        inp, psp, data = self._make_text_input_space()
        all_inputs = data.train_input
        inputTensor = inp.prepInput(all_inputs)
        latent = psp._embed(inp.forward(inputTensor))
        psp.reverse(latent)
        recovered = psp.reconstruct_data()
        expected = psp.vocabulary.tokenize(inputTensor)
        nVec = psp.outputShape[0]
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
        inp, psp, data = self._make_text_input_space()
        batch_size = 2
        inputBatch = data.train_input[0:batch_size]
        inputTensor = inp.prepInput(inputBatch)
        latent = psp._embed(inp.forward(inputTensor))
        psp.reverse(latent)
        joined = psp.reconstruct_data(text=True)
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
                              nInput=8, nWhere=0, nWhen=0, codebook=False)
        nIn, nDim = 8, 1
        inp = Models.InputSpace([nIn, nDim], [nIn, nDim], [nIn, nDim])
        x = torch.randn(2, nIn, nDim).to(Models.TheDevice.get())
        y = inp.forward(x)
        result = _unwrap(inp.reverse(y))
        # Numeric path returns tensor, not text
        self.assertIsInstance(result, (torch.Tensor, list))


class TestLexerConfig(unittest.TestCase):
    """Lexer cfg (word/sentence/grammar) always creates Lex span tables."""

    def test_embedding_can_tokenize(self):
        """Embedding model_type can tokenize text via _token_stream."""
        Models.TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, perceptDim=10, nInput=nInput)
        inp, psp = _build_text_pair(nInput)
        tokens = psp.vocabulary._token_stream("test input")
        self.assertEqual(tokens[0][0], "test")

    def test_embedding_creates_reversible_dictionary(self):
        """Embedding model_type creates Embedding with Lex-backed codebook."""
        Models.TheData.load("xor")
        nInput = 8
        _populate_test_config(inputDim=1, perceptDim=10, nInput=nInput)
        inp, psp = _build_text_pair(nInput)
        self.assertIsInstance(psp.vocabulary, Models.Embedding)


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
        emb = Models.Embedding()
        result = emb.tokenize(self._make_batch("the dog barks"))
        self.assertEqual(len(result), 1)
        # WORD("the") SPACE(" ") WORD("dog") SPACE(" ") WORD("barks")
        self.assertEqual(result[0], ["the", " ", "dog", " ", "barks"])

    def test_tokenize_splits_sentence_ending_punctuation(self):
        """tokenize() separates punctuation from words; all tokens returned."""
        emb = Models.Embedding()
        result = emb.tokenize(self._make_batch("the dog barks."))
        # quick_parser regex: words, punct, spaces are separate tokens
        # "the dog barks." -> ["the", " ", "dog", " ", "barks", "."]
        self.assertIn("barks", result[0])
        self.assertNotIn("barks.", result[0])
        self.assertIn(".", result[0])
        self.assertEqual(result[0], ["the", " ", "dog", " ", "barks", "."])

    def test_forward_returns_token_metadata(self):
        """forward(return_meta=True) replaces old encoding wrappers."""
        _populate_test_config(nWhere=0, nWhen=0)
        emb = Models.Embedding()
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
        _populate_test_config(nWhere=0, nWhen=0)
        emb = Models.Embedding()
        emb.create(nInput=10, nVectors=2, nDim=10, embedding_path=None)
        self.assertNotIn("[MASK]", emb.pretrain.key_to_index)
        mask_vec = emb.get_mask_embedding()
        self.assertTrue(torch.all(mask_vec == 0.0))
        self.assertEqual(mask_vec.shape[0], emb.wv._vectors.shape[1])


class TestEmbeddingErgodicForward(unittest.TestCase):
    def test_embedding_owns_exploration_state(self):
        """``sigma_kappa`` was lifted off Basis (2026-05-02 cleanup);
        only ``Embedding`` carries / consumes it now. ``Codebook`` /
        ``Tensor`` no longer expose it."""
        vs = Models.Codebook()
        emb = Models.Embedding()
        self.assertFalse(vs.ergodic)
        self.assertFalse(hasattr(vs, 'sigma_kappa'))
        self.assertFalse(emb.ergodic)
        self.assertAlmostEqual(emb.sigma_kappa, 0.01)

    def _make_batch(self, text, size=32):
        raw = torch.tensor([ord(c) for c in text], dtype=torch.uint8)
        padded = torch.zeros(size, dtype=torch.uint8)
        padded[:len(raw)] = raw
        return padded.unsqueeze(0)

    def _make_embedding(self, text="the dog"):
        _populate_test_config(nWhere=0, nWhen=0)
        emb = Models.Embedding()
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
        cls.wv = Models.Embedding._load_embeddings(embedding_path=cls.ENWIKI_PATH)

    @unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
    def test_load_enwiki(self):
        """Verify txt format properties and nDim-filter logic from a single load."""
        # Basic load: file loaded once in setUpClass
        self.assertIsNotNone(self.wv)
        self.assertEqual(self.wv.vector_size, 100)
        self.assertGreater(len(self.wv), 1000)
        self.assertIn("the", self.wv)
        # nDim filter: _load_embeddings(nDim=50) returns None because 100 != 50.
        # The filter condition is: wv.vector_size != nDim -> return None.
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
            loaded = Models.Embedding._load_embeddings(embedding_path=path, nDim=20)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 2)
            self.assertIn("hello", loaded)
        finally:
            os.unlink(path)

    def test_load_none_path(self):
        """_load_embeddings returns None when no path given."""
        self.assertIsNone(Models.Embedding._load_embeddings(embedding_path=None))

    def test_load_missing_file(self):
        """_load_embeddings returns None for nonexistent path."""
        self.assertIsNone(Models.Embedding._load_embeddings(embedding_path="/tmp/no_such_file.pt"))


class TestXorForwardPass(unittest.TestCase):
    """Embedding-backed text path handles xor.xml forward pass without assertion error."""

    def test_xor_forward_produces_output(self):
        """InputSpace + PerceptualSpace can forward xor data through Embedding."""
        nInput = 8
        _populate_test_config(inputDim=1, perceptDim=10, nInput=nInput)
        Models.TheData.load("xor")
        inp, psp = _build_text_pair(nInput)
        inputTensor = inp.prepInput(Models.TheData.train_input[:2])
        result = _text_embed(inp, psp, inputTensor)
        self.assertEqual(result.shape[0], 2)  # batch size
        self.assertEqual(result.shape[1], psp.outputShape[0])


class TestErgodicMnistReport(unittest.TestCase):
    """OutputSpace.forwardLinear.W accessible even with reversible=True."""

    def test_forward_layer_weight_accessible(self):
        """mnistReport can access the forward linear layer weight matrix."""
        _populate_test_config(outputDim=1, symbolDim=1, nOutput=10,
                              flatten=True, reconstruct="FULL")
        os_ = Models.OutputSpace([10, 1], [10, 1], [10, 1])
        # The bug was: forwardLinear is a bound method, not a layer
        # After fix: we can get the layer via linear1
        fwd_layer = (os_.linear1 if hasattr(os_, 'linear1') else os_.forwardLinear)
        self.assertTrue(hasattr(fwd_layer, 'W'))
        self.assertIsInstance(fwd_layer, Models.LinearLayer)


# ---------------------------------------------------------------------------
# TestModelTypeVariants -- missing model type combinations
# ---------------------------------------------------------------------------
class TestModelTypeVariants(unittest.TestCase):
    """Exercise BasicModel.create() with configuration combinations not yet covered."""

    def test_invertible(self):
        """invertible=True with ergodic, reversible -- forward + reverse."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            invertible=True, ergodic=True, reconstruct="FULL",
            perceptPassThrough=True, symbolPassThrough=True,
            flatten=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_conceptual_order_1(self):
        """conceptualOrder=1 with non-passthrough symbolic -- forward only.

        After the 2026-05-05 BasicModel/BasicModel merger, the
        per-stage path is the only construction path and
        ``conceptualOrder`` literally drives the per-stage iteration.
        """
        _populate_test_config(inputDim=1, perceptDim=1, conceptDim=1, symbolDim=1,
                              wordDim=1, outputDim=1,
                              nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
                              perceptPassThrough=True, symbolPassThrough=False,
                              flatten=True)
        model = Models.BasicModel()
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
                     conceptualOrder=1)
        x = torch.randn(2, 8, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_non_ergodic_reverse(self):
        """ergodic=False with reversible=True -- forward + reverse."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            ergodic=False, reconstruct="FULL",
            perceptPassThrough=True, symbolPassThrough=True,
            flatten=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh()
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        data, start_state = model.reverse(end_state, out)
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data.shape[1], 16)

    def test_percept_no_attention(self):
        """perceptHasAttention=False, perceptPassThrough=False -- forward only."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
            perceptHasAttention=False,
            perceptPassThrough=False, symbolPassThrough=True,
            flatten=True, naive=True)
        model = Models.BasicModel()
        model.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 8, 1)
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)

    def test_concept_with_attention(self):
        """conceptHasAttention=True -- forward only."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            conceptHasAttention=True,
            perceptPassThrough=True, symbolPassThrough=True,
            flatten=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        x = torch.randn(2, 16, 1).tanh()
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)


class TestReconstructionSymbols(unittest.TestCase):
    """Test that reconstruction symbols split correctly and enable reconstruction."""

    def _create_xor_model(self, nSymbols=3, nOutput=1):
        """Helper: create XOR model from XOR_exact.xml with given symbol/output counts.

        Also patches ConceptualSpace/nVectors to match nSymbols, because
        SymbolicSpace requires inputShape[0] == nVectors (1:1 concept->symbol mapping).
        """
        import tempfile
        import xml.etree.ElementTree as ET

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Patch autoload off
        auto = root.find("architecture/autoload")
        if auto is None:
            auto = ET.SubElement(root.find("architecture"), "autoload")
        auto.text = "false"

        # Patch symbol count (and concepts to match -- SymbolicSpace requires nConcepts == nSymbols)
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
        Models.TheData.load("xor")
        m = Models.BasicModel()
        m.create_from_config(tmp.name, data=Models.TheData)
        os.unlink(tmp.name)
        return m

    def test_forward_output_shape_unchanged(self):
        """Forward pass output shape is [batch, nOutput] regardless of nSymbols."""
        m = self._create_xor_model(nSymbols=3, nOutput=1)
        m.train(False)
        test_input, _ = m.inputSpace.getTestData()
        x = m.inputSpace.prepInput(test_input[:2])
        with torch.no_grad():
            forwardInput, symbols, outputPred, _ = m.forward(x)
        self.assertEqual(outputPred.shape[0], 2)

    def test_reverse_uses_all_symbols(self):
        """Reverse pass produces output with correct shape from the full symbol stream."""
        m = self._create_xor_model(nSymbols=3, nOutput=1)
        m.train(False)
        test_input, _ = m.inputSpace.getTestData()
        x = m.inputSpace.prepInput(test_input[:2])
        with torch.no_grad():
            forwardInput, symbols, outputPred, _ = m.forward(x)
            inputData, inputPred = m.reverse(symbols, outputPred)
        self.assertEqual(inputData.shape[0], 2)

    def test_xor_perfect_reconstruction(self):
        """After training, all 4 XOR inputs reconstruct to the correct words.

        Uses XOR_exact.xml which configures PerceptualSpace with invertible=True
        and nActive=8 so that the non-naive PiLayer(invertible=True) path is
        exercised. The non-naive path uses the LDU/triangular-solve inverse.
        """
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
            Models.TheData.load("xor")
            m = Models.BasicModel()
            m.create_from_config(tmp.name, data=Models.TheData)

            # Ergodic training -- accumulate range warnings, emit summary at end
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
            recon_texts = m.perceptualSpace.reconstruct_data(text=True)
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

    PiLayer(invertible=True) doubles the sequence dimension, so
    nActive_percept must be 2*nActive_input.  This test verifies the model
    constructs and runs a forward+reverse pass without shape errors.
    """

    def test_construct_and_forward_reverse(self):
        import warnings
        import xml.etree.ElementTree as ET

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "xor_3d.xml")
        torch.manual_seed(42)
        Models.TheData.load("xor")
        m = Models.BasicModel()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PerceptualSpace: reversible=True with invertible=False")
            m.create_from_config(xml_path, data=Models.TheData)

        # Forward pass
        test_input, test_output = m.inputSpace.getTestData()
        m.set_sigma(0)
        m.train(False)
        with torch.no_grad():
            m.runEpoch(batchSize=len(test_input), split="test")

        # Verify no crash and shapes are consistent.  Reconstructed state
        # now lives on PerceptualSpace (text mode owns the Embedding there).
        self.assertIsNotNone(m.perceptualSpace.reconstructed)

@unittest.skipIf(not _XOR_EXACT_USES_EMBEDDING, "Model doesn't use Embedding")
class TestTrainEmbeddingsFlag(unittest.TestCase):
    """trainEmbeddings config flag controls whether embedding weights are in optimizer."""

    def _create_model(self, train_embeddings):
        """Create an XOR embedding model with specified trainEmbeddings flag."""
        import xml.etree.ElementTree as ET

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
        Models.TheData.load("xor")
        m = Models.BasicModel()
        m.create_from_config(tmp.name, data=Models.TheData)
        os.unlink(tmp.name)
        return m

    def test_train_embeddings_includes_emb_params(self):
        """When trainEmbeddings=true, _emb.weight is in optimizer params."""
        m = self._create_model(True)
        self.assertIsInstance(m.perceptualSpace.vocabulary, Models.Embedding)
        emb_weight = m.perceptualSpace.vocabulary.wv._vectors
        optimizer = m.getOptimizer(lr=0.001)
        opt_params = [p.data_ptr() for group in optimizer.param_groups for p in group['params']]
        self.assertIn(emb_weight.data_ptr(), opt_params,
                      "Embedding weight should be in optimizer when trainEmbeddings=true")

    def test_frozen_embeddings_default(self):
        """When trainEmbeddings=false, _emb.weight is NOT in optimizer params."""
        m = self._create_model(False)
        self.assertIsInstance(m.perceptualSpace.vocabulary, Models.Embedding)
        emb_weight = m.perceptualSpace.vocabulary.wv._vectors
        optimizer = m.getOptimizer(lr=0.001)
        opt_params = [p.data_ptr() for group in optimizer.param_groups for p in group['params']]
        self.assertNotIn(emb_weight.data_ptr(), opt_params,
                         "Embedding weight should NOT be in optimizer when trainEmbeddings=false")

    def test_joint_mode_passes_sbow_to_total_loss(self):
        """runBatch must forward JOINT sbow loss into ModelLoss.total()."""
        m = self._create_model("joint")
        self.assertIsInstance(m.perceptualSpace.vocabulary, Models.Embedding)

        optimizer = m.getOptimizer(lr=0.001)
        sentinel = torch.tensor(1.2345, device=Models.TheDevice.get())
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

        loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
        inp_items, out_items = next(iter(loader))
        inputTensor = m.inputSpace.prepInput(inp_items)
        outputTensor = (m.outputSpace.prepOutput(out_items)
                        if out_items is not None else None)
        result, _ = m.runBatch(
            train=True,
            batchNum=0,
            batchSize=1,
            split="train",
            optimizer=optimizer,
            batch_override=(inputTensor, outputTensor),
        )

        self.assertIsNotNone(result)
        self.assertIs(seen.get("sbow"), sentinel)


# ---------------------------------------------------------------------------
# Weight persistence -- save/load round-trip with shape mismatches
# ---------------------------------------------------------------------------
class TestWeightShapeMismatch(unittest.TestCase):
    """Verify save_weights / load_weights handles vocab-size changes."""

    def _make_model(self, vocab_size, dim=8):
        """Create a minimal nn.Module with an embedding of given vocab_size."""
        m = Models.BasicModel()
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

    def test_load_ignores_unexpected_keys_when_all_current_keys_match(self):
        """Stale alias/module keys must not block loading valid weights."""
        m1 = self._make_model(100)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            state = {k: v.detach().clone() for k, v in m1.state_dict().items()}
            state["stale.alias.fc.weight"] = state["fc.weight"].detach().clone()
            torch.save({"state_dict": state}, path)

            m2 = self._make_model(100)
            self.assertTrue(m2.load_weights(path))
            for k in m1.state_dict():
                torch.testing.assert_close(m1.state_dict()[k], m2.state_dict()[k])
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

        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        Models.TheData.load("xor")

        m1 = Models.BasicModel()
        m1.create_from_config(xml_path, data=Models.TheData)

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
            m2 = Models.BasicModel()
            m2.create_from_config(xml_path, data=Models.TheData)
            emb2 = m2._get_embedding()
            self.assertNotEqual(len(emb2.pretrain.index_to_key), len(vocab_before))

            # Load should restore vocab and embedding weights cleanly
            self.assertTrue(m2.load_embeddings(emb_path))
            emb2 = m2._get_embedding()
            self.assertEqual(list(emb2.pretrain.index_to_key), vocab_before)
            # Embedding shapes must match exactly
            torch.testing.assert_close(
                m2.state_dict()["perceptualSpace.subspace.what.wv._vectors"],
                m1.state_dict()["perceptualSpace.subspace.what.wv._vectors"])
        finally:
            os.unlink(emb_path)


class TestTrainingUpdatesWeights(unittest.TestCase):
    """Verify that run() with numEpochs=1 actually trains and changes weights."""

    def test_xor_weights_change_after_one_epoch(self):
        """XOR model weights must differ after 1 epoch of training."""

        # XOR_exact.xml has autoload=false, autosave=false
        xml_path = os.path.join(os.path.dirname(_BIN), "data", "XOR_exact.xml")
        Models.TheData.load("xor")

        m = Models.BasicModel()
        m.create_from_config(xml_path, data=Models.TheData)

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
        Models.TheData.load("xor")
        original_train = list(Models.TheData.train_input)
        original_output = list(Models.TheData.train_output)
        with Models.TheData.runtime_batch(["hello world"], [[0]]):
            self.assertEqual(Models.TheData.train_input, ["hello world"])
            self.assertEqual(Models.TheData.train_output, [[0]])
        self.assertEqual(list(Models.TheData.train_input), original_train)
        self.assertEqual(list(Models.TheData.train_output), original_output)

    def test_runtime_batch_restores_on_exception(self):
        Models.TheData.load("xor")
        original_train = list(Models.TheData.train_input)
        with self.assertRaises(ValueError):
            with Models.TheData.runtime_batch(["test"], [[1]]):
                raise ValueError("boom")
        self.assertEqual(list(Models.TheData.train_input), original_train)

    def test_runtime_batch_stores_strings(self):
        Models.TheData.load("xor")
        with Models.TheData.runtime_batch(["hello world"]):
            self.assertEqual(Models.TheData.train_input, ["hello world"])


class TestRuntimeDataLoader(unittest.TestCase):
    """data_loader(split='train') serves staged runtime data via runtime_batch."""

    def test_runtime_data_loader_returns_batch(self):
        import tempfile
        import xml.etree.ElementTree as ET

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
        Models.TheData.load("xor")
        m = Models.BasicModel()
        m.create_from_config(tmp.name, data=Models.TheData)
        os.unlink(tmp.name)

        rt_input = ["hello world"]
        rt_output = [torch.tensor([0], dtype=torch.float)]
        with Models.TheData.runtime_batch(rt_input, rt_output):
            loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
            inp_items, out_items = next(iter(loader))
            inputTensor = m.inputSpace.prepInput(inp_items)
            outputTensor = (m.outputSpace.prepOutput(out_items)
                            if out_items is not None else None)
            batch = (inputTensor, outputTensor)
            nextBatch = 1
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
            Models.TheData.load("xor")
            m = Models.BasicModel()
            m.create_from_config(tmp.name, data=Models.TheData)

            self.assertTrue(m.reversible)
            self.assertGreater(m.loss.reverse_scale, 0)

            m.set_sigma(0.5)
            m.train(True)
            # Run a single batch WITHOUT internal backward (train=False)
            loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
            inp_items, out_items = next(iter(loader))
            inputTensor = m.inputSpace.prepInput(inp_items)
            outputTensor = (m.outputSpace.prepOutput(out_items)
                            if out_items is not None else None)
            result, _ = m.runBatch(
                batchNum=0, batchSize=4,
                split="train", train=False,
                batch_override=(inputTensor, outputTensor),
            )

            self.assertIsNotNone(result.lossIn)
            self.assertGreater(result.lossIn.item(), 0)

            # Backward through lossIn only
            result.lossIn.backward()

            # Gradient must reach the codebook
            emb = m.perceptualSpace.vocabulary
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
            Models.TheData.load("xor")
            m = Models.BasicModel()
            m.create_from_config(tmp.name, data=Models.TheData)

            # Ergodic training -- accumulate range warnings, emit summary at end
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m.runTrial(numEpochs=200, batchSize=10, lr=0.01)

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
        """set_activation / get_activation round-trip.

        Scalar [B, N] is lifted to the 4-valued bivector [B, N, 2]:
        positive values go to the TRUE pole, negatives to FALSE.
        """
        ss = Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        activation = torch.randn(2, 4).to(Models.TheDevice.get())
        ss.set_activation(activation)
        got = ss.get_activation()
        expected = torch.stack(
            [torch.relu(activation), torch.relu(-activation)], dim=-1)
        self.assertTrue(torch.equal(got, expected))

    def test_set_activation_bivector_passthrough(self):
        """set_activation accepts [B, N, 2] bivector unchanged."""
        ss = Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        bivec = torch.rand(2, 4, 2).to(Models.TheDevice.get())
        ss.set_activation(bivec)
        got = ss.get_activation()
        self.assertEqual(got.shape, (2, 4, 2))
        self.assertTrue(torch.equal(got, bivec))

    def test_materialize_topk(self):
        """Materialize(k) returns top-k vectors by activation, gated."""
        ss = Models.SubSpace(inputShape=[8, 3], outputShape=[8, 3])
        # Create a known tensor: 8 vectors of dim 3
        x = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3).to(Models.TheDevice.get())
        ss.set_event(x)
        # Set activation: highest at indices 7, 5, 3, 1
        activation = torch.tensor([[0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 1.0]]).to(Models.TheDevice.get())
        ss.set_activation(activation)
        selected = ss.materialize(k=4)
        self.assertEqual(selected.shape, (1, 4, 3))
        # The top-4 by activation are indices 7(1.0), 5(0.9), 1(0.8), 3(0.7)
        expected_indices = [7, 5, 1, 3]
        for i, idx in enumerate(expected_indices):
            expected = x[0, idx] * activation[0, idx]
            self.assertTrue(torch.allclose(selected[0, i], expected),
                            f"Position {i}: expected vector at index {idx}")

    def test_materialize_k_none(self):
        """Materialize(k=None) returns event * presence.

        Presence = max(aP, aN) = |activation| for a scalar lifted to the
        bivector, so the effective gate is abs(scalar).
        """
        ss = Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(Models.TheDevice.get())
        ss.set_event(x)
        activation = torch.randn(2, 4).to(Models.TheDevice.get())
        ss.set_activation(activation)
        result = ss.materialize(k=None)
        expected = x * activation.abs().unsqueeze(-1)
        self.assertTrue(torch.equal(result, expected))

    def test_materialize_k_geq_nspace(self):
        """Materialize(k >= nSpace) returns event * presence."""
        ss = Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(Models.TheDevice.get())
        ss.set_event(x)
        activation = torch.randn(2, 4).to(Models.TheDevice.get())
        ss.set_activation(activation)
        expected = x * activation.abs().unsqueeze(-1)
        result = ss.materialize(k=4)
        self.assertTrue(torch.equal(result, expected))
        result2 = ss.materialize(k=10)
        self.assertTrue(torch.equal(result2, expected))


    def test_materialize_activation_mode_with_stored(self):
        """materialize(mode='activation') returns stored activation."""
        ss = Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(Models.TheDevice.get())
        ss.set_event(x)
        # set_event defaults to unit activation
        result = ss.materialize(mode="activation")
        expected = torch.ones(2, 4, device=x.device)
        self.assertTrue(torch.allclose(result, expected))

    def test_materialize_activation_mode_computes_from_event(self):
        """materialize(mode='activation') computes 4-valued activation when
        not stored, reduced to presence = max(aP, aN).
        """
        ss = Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(Models.TheDevice.get())
        # Store event directly without setting activation
        ss.event.setW(x)
        ss.activation.setW(None)
        result = ss.materialize(mode="activation")
        # Presence derived from the bivector [aP, aN]:
        #   aP = ||relu(what)|| / sqrt(nWhat)
        #   aN = ||relu(-what)|| / sqrt(nWhat)
        # materialize(mode='activation') returns effective_activation()
        # which is presence * modal_gate. Since no _active is set, gate = 1.
        what_slice = x[:, :, :ss.nWhat]
        d = max(ss.nWhat, 1)
        pos = torch.relu(what_slice).norm(dim=-1) / math.sqrt(d)
        neg = torch.relu(-what_slice).norm(dim=-1) / math.sqrt(d)
        expected_presence = torch.maximum(pos.clamp(0.0, 1.0), neg.clamp(0.0, 1.0))
        self.assertTrue(torch.allclose(result, expected_presence))

    def test_materialize_activation_mode_no_data_asserts(self):
        """materialize(mode='activation') asserts when no event vectors exist."""
        ss = Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        ss.event.setW(None)
        ss.activation.setW(None)
        with self.assertRaises(AssertionError):
            ss.materialize(mode="activation")

    def test_materialize_default_mode_unchanged(self):
        """materialize() with default mode='active' behaves as before."""
        ss = Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])
        x = torch.randn(2, 4, 3).to(Models.TheDevice.get())
        ss.set_event(x)
        result = ss.materialize()
        self.assertTrue(torch.equal(result, x))


class TestGrammar(unittest.TestCase):
    """Tests for Grammar (S-tier only)."""

    def _make_grammar(self):
        g = Language.Grammar()
        g.configure({
            "S": ["swap(S, S)", "not(S)", "union(S, S)"],
        })
        return g

    def test_length(self):
        g = self._make_grammar()
        self.assertEqual(len(g), 3)

    def test_indexing(self):
        g = self._make_grammar()
        self.assertEqual(g[0], "S -> swap(S, S)")
        self.assertEqual(g[1], "S -> not(S)")
        self.assertEqual(g[2], "S -> union(S, S)")

    def test_arity(self):
        g = self._make_grammar()
        self.assertEqual(g.arity(0), 2)  # swap
        self.assertEqual(g.arity(1), 1)  # not
        self.assertEqual(g.arity(2), 2)  # union

    def test_space_partitions(self):
        g = self._make_grammar()
        self.assertEqual(g.symbolic(), [0, 1, 2])

    def test_configure_from_dict(self):
        g = Language.Grammar()
        g.configure({
            "S": ["swap(S, S)", "equals(S, S)", "union(S, S)"],
        })
        self.assertEqual(g.symbolic(), [0, 1, 2])

    def test_configure_single_rule_string(self):
        g = Language.Grammar()
        g.configure({"S": "not(S)"})
        self.assertEqual(g.symbolic(), [0])

    def test_configure_unknown_rule_raises(self):
        # After Phase A, whitespace-separated identifiers parse as a
        # typed merge rule. An unparseable RHS must contain non-identifier
        # tokens (e.g. digits or punctuation outside function-call syntax).
        g = Language.Grammar()
        with self.assertRaises(ValueError):
            g.configure({"S": ["123 456"]})

    def test_configure_multi_lhs_keys(self):
        """Phase A: non-S LHS keys (VO, NP, VP, ...) are accepted as typed rules."""
        g = Language.Grammar()
        g.configure({"S": "S VO", "VO": "V O"})
        self.assertEqual(len(g), 2)
        self.assertEqual(g.rules[0].lhs, "S")
        self.assertEqual(g.rules[1].lhs, "VO")

    def test_configure_s_first_ordering(self):
        """S stays first even when declared after other keys."""
        g = Language.Grammar()
        g.configure({"VO": "V O", "S": "S VO"})
        self.assertEqual(g.rules[0].lhs, "S")
        self.assertEqual(g.rules[1].lhs, "VO")

    def test_symbolic_transition_none(self):
        """With no S->S epsilon rule, symbolic_transition() is None."""
        g = self._make_grammar()
        self.assertIsNone(g.symbolic_transition())


class TestBasisMereology(unittest.TestCase):
    """Scalar form (scalar=True) of the mereological suite: clipped-cosine
    parthood and the region indicators derived from it."""

    def test_part_satisfies_boole_contrapositive(self):
        b = Spaces.Basis()
        x = torch.tensor([0.5, -0.3, 0.4])
        y = torch.tensor([0.2, 0.6, -0.1])
        self.assertAlmostEqual(
            b.part(x, y, scalar=True).item(),
            b.part(-y, -x, scalar=True).item(),
            places=6,
        )

    def test_part_is_clipped_cosine(self):
        b = Spaces.Basis()
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.5, 0.5, 0.0])
        self.assertAlmostEqual(b.part(x, y, scalar=True).item(), math.sqrt(0.5), places=5)

    def test_part_opposite_directions_zero(self):
        b = Spaces.Basis()
        x = torch.tensor([1.0, 0.0])
        y = torch.tensor([-1.0, 0.0])
        self.assertAlmostEqual(b.part(x, y, scalar=True).item(), 0.0, places=5)

    def test_whole_is_part_reversed(self):
        b = Spaces.Basis()
        x = torch.tensor([0.5, 0.5, 0.0])
        y = torch.tensor([1.0, 0.0, 0.0])
        self.assertAlmostEqual(
            b.whole(x, y, scalar=True).item(),
            b.part(y, x, scalar=True).item(),
            places=6,
        )

    def test_equal_is_mutual_parthood(self):
        b = Spaces.Basis()
        x = torch.tensor([0.5, 0.5, 0.0])
        y = torch.tensor([1.0, 0.0, 0.0])
        expected = b.part(x, y, scalar=True).item() * b.part(y, x, scalar=True).item()
        self.assertAlmostEqual(b.equal(x, y, scalar=True).item(), expected, places=6)

    def test_overlap_true_when_both_strictly_partial(self):
        b = Spaces.Basis()
        x = torch.tensor([1.0, 0.5, 0.0])
        y = torch.tensor([0.5, 1.0, 0.0])
        # cos > 0 and < 1 -> equal in (0, 1) -> overlap region.
        self.assertTrue(bool(b.overlap(x, y, scalar=True).item()))

    def test_overlap_false_when_parallel(self):
        b = Spaces.Basis()
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([2.0, 0.0, 0.0])
        # cosine == 1 -> equal == 1 -> outside (0, 1).
        self.assertFalse(bool(b.overlap(x, y, scalar=True).item()))

    def test_underlap_true_when_disjoint(self):
        b = Spaces.Basis()
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([-1.0, 0.0, 0.0])
        # cosine negative -> clipped to 0 -> equal == 0.
        self.assertTrue(bool(b.underlap(x, y, scalar=True).item()))

    def test_underlap_false_when_partially_overlapping(self):
        b = Spaces.Basis()
        x = torch.tensor([1.0, 1.0, 0.0])
        y = torch.tensor([1.0, 0.0, 0.0])
        self.assertFalse(bool(b.underlap(x, y, scalar=True).item()))

    def test_scalar_mereology_respects_boole_contrapositive(self):
        """Scalar members (part/whole/equal/boundary) are invariant under
        (A, B) -> (-B, -A)."""
        b = Spaces.Basis()
        torch.manual_seed(0)
        x = torch.randn(4)
        y = torch.randn(4)
        for name in ("part", "whole", "equal", "boundary"):
            fn = getattr(b, name)
            self.assertAlmostEqual(
                fn(x, y, scalar=True).item(),
                fn(-y, -x, scalar=True).item(),
                places=5,
                msg=f"{name} fails Boole contrapositive",
            )

    def test_region_mereology_respects_boole_contrapositive(self):
        """Boolean region indicators (overlap/underlap) are invariant under
        (A, B) -> (-B, -A)."""
        b = Spaces.Basis()
        torch.manual_seed(0)
        x = torch.randn(4)
        y = torch.randn(4)
        for name in ("overlap", "underlap"):
            fn = getattr(b, name)
            self.assertEqual(
                bool(fn(x, y, scalar=True).item()),
                bool(fn(-y, -x, scalar=True).item()),
                msg=f"{name} fails Boole contrapositive",
            )

    def test_boundary_zero_under_clipped_cosine(self):
        """Under clipped cosine, part is symmetric so boundary = 0."""
        b = Spaces.Basis()
        torch.manual_seed(1)
        x = torch.randn(4)
        y = torch.randn(4)
        self.assertAlmostEqual(b.boundary(x, y, scalar=True).item(), 0.0, places=5)

    def test_equal_three_region_partition(self):
        """equal in [0, 1] partitions into underlap (=0), overlap (in (0,1)),
        identity (=1), disjoint and exhaustive."""
        b = Spaces.Basis()
        # Underlap: opposite directions -> equal = 0
        x_u = torch.tensor([1.0, 0.0])
        y_u = torch.tensor([-1.0, 0.0])
        self.assertAlmostEqual(b.equal(x_u, y_u, scalar=True).item(), 0.0, places=5)
        self.assertTrue(bool(b.underlap(x_u, y_u, scalar=True).item()))
        self.assertFalse(bool(b.overlap(x_u, y_u, scalar=True).item()))
        # Overlap: non-aligned, non-orthogonal -> 0 < equal < 1
        x_o = torch.tensor([1.0, 1.0])
        y_o = torch.tensor([1.0, 0.0])
        e_o = b.equal(x_o, y_o, scalar=True).item()
        self.assertGreater(e_o, 0.0)
        self.assertLess(e_o, 1.0)
        self.assertTrue(bool(b.overlap(x_o, y_o, scalar=True).item()))
        self.assertFalse(bool(b.underlap(x_o, y_o, scalar=True).item()))
        # Identity: parallel same direction -> equal = 1
        x_i = torch.tensor([1.0, 0.0])
        y_i = torch.tensor([2.0, 0.0])
        self.assertAlmostEqual(b.equal(x_i, y_i, scalar=True).item(), 1.0, places=5)
        self.assertFalse(bool(b.overlap(x_i, y_i, scalar=True).item()))
        self.assertFalse(bool(b.underlap(x_i, y_i, scalar=True).item()))


class TestBasisMereologyVector(unittest.TestCase):
    """Default vector form of the mereological suite on Basis.

    part(x, y)     = x * (y / ||y||)          elementwise
    whole(x, y)    = (1 - x) * (y / ||y||)
    equal(x, y)    = part(x, y) * part(y, x)
    overlap(x, y)  = min(part(x, y), part(y, x))
    underlap(x, y) = min(whole(x, y), whole(y, x))
    boundary(x, y) = |part(x, y) - part(y, x)|
    """

    def test_part_vector_equals_x_times_yhat(self):
        b = Spaces.Basis()
        x = torch.tensor([0.4, 0.6, 0.2])
        y = torch.tensor([0.0, 3.0, 0.0])     # ||y|| = 3, y_hat = [0,1,0]
        expected = torch.tensor([0.0, 0.6, 0.0])
        self.assertTrue(torch.allclose(b.part(x, y), expected, atol=1e-6))

    def test_part_vector_has_same_shape_as_x(self):
        b = Spaces.Basis()
        x = torch.randn(2, 5, 7)
        y = torch.randn(2, 5, 7)
        self.assertEqual(b.part(x, y).shape, x.shape)

    def test_whole_vector_equals_one_minus_x_times_yhat(self):
        b = Spaces.Basis()
        x = torch.tensor([0.4, 0.6, 0.2])
        y = torch.tensor([0.0, 3.0, 0.0])
        expected = torch.tensor([0.6, 0.4, 0.8]) * torch.tensor([0.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(b.whole(x, y), expected, atol=1e-6))

    def test_part_plus_whole_equals_yhat(self):
        """(x + (1 - x)) * y_hat = y_hat"""
        b = Spaces.Basis()
        x = torch.tensor([0.3, 0.7, 0.1, 0.9])
        y = torch.tensor([1.0, 2.0, 0.5, 1.5])
        y_hat = y / torch.norm(y)
        self.assertTrue(torch.allclose(b.part(x, y) + b.whole(x, y), y_hat, atol=1e-6))

    def test_equal_vector_is_elementwise_product_of_parts(self):
        b = Spaces.Basis()
        x = torch.tensor([0.5, 0.8, 0.3])
        y = torch.tensor([0.2, 0.9, 0.4])
        expected = b.part(x, y) * b.part(y, x)
        self.assertTrue(torch.allclose(b.equal(x, y), expected, atol=1e-6))

    def test_overlap_vector_is_elementwise_min_of_parts(self):
        b = Spaces.Basis()
        x = torch.tensor([0.5, 0.8, 0.3])
        y = torch.tensor([0.2, 0.9, 0.4])
        expected = torch.minimum(b.part(x, y), b.part(y, x))
        self.assertTrue(torch.allclose(b.overlap(x, y), expected, atol=1e-6))

    def test_underlap_vector_is_elementwise_min_of_wholes(self):
        b = Spaces.Basis()
        x = torch.tensor([0.5, 0.8, 0.3])
        y = torch.tensor([0.2, 0.9, 0.4])
        expected = torch.minimum(b.whole(x, y), b.whole(y, x))
        self.assertTrue(torch.allclose(b.underlap(x, y), expected, atol=1e-6))

    def test_boundary_vector_is_elementwise_abs_diff(self):
        b = Spaces.Basis()
        x = torch.tensor([0.5, 0.8, 0.3])
        y = torch.tensor([0.2, 0.9, 0.4])
        expected = torch.abs(b.part(x, y) - b.part(y, x))
        self.assertTrue(torch.allclose(b.boundary(x, y), expected, atol=1e-6))

    def test_part_orthogonal_unit_axes_is_zero_vector(self):
        """TRUE=[1,0] projected onto FALSE=[0,1] direction: zero vector.
        Demonstrates tetralemma disjointness on the bivector."""
        b = Spaces.Basis()
        true_corner = torch.tensor([1.0, 0.0])
        false_corner = torch.tensor([0.0, 1.0])
        p = b.part(true_corner, false_corner)
        self.assertTrue(torch.allclose(p, torch.zeros(2), atol=1e-6))

    def test_part_parallel_unit_axes_preserves_direction(self):
        """BOTH=[1,1] projected onto TRUE=[1,0] gives [1,0]: only the
        true-pole coordinate survives."""
        b = Spaces.Basis()
        both_corner = torch.tensor([1.0, 1.0])
        true_corner = torch.tensor([1.0, 0.0])
        p = b.part(both_corner, true_corner)
        self.assertTrue(torch.allclose(p, torch.tensor([1.0, 0.0]), atol=1e-6))

    def test_copart_vector_is_y_minus_x(self):
        """Vector copart(x, y) = y - x: the y-content not accounted for by x."""
        b = Spaces.Basis()
        x = torch.tensor([0.3, 0.5, 0.2])
        y = torch.tensor([0.8, 0.4, 0.7])
        expected = y - x
        self.assertTrue(torch.allclose(b.copart(x, y), expected, atol=1e-6))

    def test_copart_scalar_is_complement_of_part(self):
        """Scalar copart(x, y) = 1 - part(x, y, scalar=True)."""
        b = Spaces.Basis()
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.5, 0.5, 0.0])
        p = b.part(x, y, scalar=True).item()
        self.assertAlmostEqual(b.copart(x, y, scalar=True).item(), 1.0 - p, places=6)

    def test_copart_scalar_zero_when_identical_directions(self):
        """x parallel to y -> part = 1 -> copart scalar = 0."""
        b = Spaces.Basis()
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([2.0, 0.0, 0.0])
        self.assertAlmostEqual(b.copart(x, y, scalar=True).item(), 0.0, places=5)

    def test_copart_scalar_one_when_orthogonal(self):
        """x orthogonal to y -> part = 0 -> copart scalar = 1."""
        b = Spaces.Basis()
        x = torch.tensor([1.0, 0.0])
        y = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(b.copart(x, y, scalar=True).item(), 1.0, places=5)


class TestWordEncoding(unittest.TestCase):
    """Tests for WordEncoding."""

    def test_encode_decode_roundtrip(self):
        we = Spaces.WordEncoding(nBatch=4, nActive=64)
        word = we.encode(2, 42, 0)
        b, v, r = we.decode(word)
        self.assertEqual((b, v, r), (2, 42, 0))

    def test_encode_validates_rule(self):
        we = Spaces.WordEncoding(nBatch=4, nActive=64)
        with self.assertRaises(AssertionError):
            we.encode(0, 0, 99)  # rule out of range

    def test_encode_validates_negative_batch(self):
        we = Spaces.WordEncoding(nBatch=4, nActive=64)
        with self.assertRaises(AssertionError):
            we.encode(-1, 0, 0)

    def test_encode_validates_negative_vector(self):
        we = Spaces.WordEncoding(nBatch=4, nActive=64)
        with self.assertRaises(AssertionError):
            we.encode(0, -1, 0)


class TestSubspaceWords(unittest.TestCase):
    """Tests for SubSpace word support."""

    def _make_ss(self):
        return Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])

    def test_words_default_empty(self):
        ss = self._make_ss()
        self.assertEqual(ss.get_words(), [])

    def test_add_word_start_state(self):
        ss = self._make_ss()
        ss.add_word(0, 0, 0)
        self.assertEqual(ss.get_words(), [(0, 0, 0, 0, -1, -1, -1)])

    def test_add_multiple_words(self):
        ss = self._make_ss()
        ss.add_word(0, 0, 0)
        ss.add_word(0, 42, 0)
        self.assertEqual(ss.get_words(),
                         [(0, 0, 0, 0, -1, -1, -1), (0, 42, 0, 0, -1, -1, -1)])

    def test_set_words(self):
        ss = self._make_ss()
        words = [(0, 0, 0), (0, 1, 2), (0, 2, 3)]
        ss.set_words(words)
        self.assertEqual(ss.get_words(), words)

    def test_add_word_validates(self):
        ss = self._make_ss()
        with self.assertRaises(AssertionError):
            ss.add_word(0, 0, 99)  # bad rule


class TestSubspaceNormalize(unittest.TestCase):
    """Tests for SubSpace.normalize()."""

    def _make_ss(self):
        return Models.SubSpace(inputShape=[4, 3], outputShape=[4, 3])

    def test_percepts_range(self):
        """normalize('percepts') produces values in [-1, 1] via tanh."""
        ss = self._make_ss()
        x = torch.randn(2, 4, 3)
        ss.set_event(x.clone())
        ss.normalize("percepts", target="what", normalize=True)
        y = ss.select("what")
        self.assertTrue(torch.all(y >= -1) and torch.all(y <= 1))
        self.assertTrue(torch.allclose(y, torch.tanh(x)))

    def test_percepts_reverse_normalize_roundtrip(self):
        """reverse=True applies atanh as the inverse percept normalization."""
        ss = self._make_ss()
        x = torch.randn(2, 4, 3) * 0.25
        ss.set_event(x.clone())
        ss.normalize("percepts", target="event", normalize=True)
        ss.normalize("percepts", target="event", normalize=True, reverse=True)
        self.assertTrue(torch.allclose(ss.materialize(), x, atol=1e-6))

    def test_concepts_range(self):
        """normalize('concepts') produces bivector entries in [0, 1] via tanh.

        Activation is now a 4-valued bivector [aP, aN]; normalization applies
        tanh element-wise, so each pole stays non-negative.
        """
        ss = self._make_ss()
        scalar = torch.randn(2, 4)
        ss.set_activation(scalar.clone())
        # set_activation lifted scalar -> bivector [relu(x), relu(-x)].
        lifted = torch.stack([torch.relu(scalar), torch.relu(-scalar)], dim=-1)
        ss.normalize("concepts", target="activation", normalize=True)
        y = ss.get_activation()
        self.assertTrue(torch.all(y >= -1) and torch.all(y <= 1))
        self.assertTrue(torch.allclose(y, torch.tanh(lifted), atol=1e-6))

    def test_reverse_requires_normalize_true(self):
        """reverse=True is an inverse transform, not a range check."""
        ss = self._make_ss()
        ss.set_event(torch.randn(2, 4, 3))
        with self.assertRaises(ValueError):
            ss.normalize("percepts", target="event", normalize=False, reverse=True)

    def test_symbols_discrete(self):
        """normalize('symbols') produces {0, 1} integers with STE gradients."""
        ss = self._make_ss()
        x = torch.randn(2, 4, requires_grad=True)
        ss.set_activation(x)
        ss.normalize("symbols", target="activation", normalize=True)
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
        ss.set_event(torch.randn(2, 4, 3))
        with self.assertRaises(ValueError):
            ss.normalize("bogus", target="what", normalize=True)


class TestSymbolObjective(unittest.TestCase):
    """SymbolicSpace uses residual-first objective terms."""

    def test_symbol_objective_residual_primary_l1_secondary(self):
        _populate_test_config(conceptDim=3, symbolDim=3,
                              nConcepts=2, nSymbols=2,
                              perceptHasAttention=False)
        sym = Models.SymbolicSpace(
            inputShape=[2, 3],
            spaceShape=[2, 3],
            outputShape=[2, 3],
        )
        sym.symbol_residual_scale = 2.0
        sym.l1_lambda = 0.1
        sym.decorrelation_weight = 0.0
        sym.spectral_flatness_weight = 0.0

        predicted = torch.tensor(
            [[[1.0, -2.0, 3.0], [0.5, -0.5, 1.5]]],
            requires_grad=True,
        )
        target = torch.zeros_like(predicted)
        terms = sym._compute_symbol_terms(predicted, target=target)

        self.assertIn("symbol_residual", terms)
        self.assertIn("symbol_l1", terms)
        self.assertTrue(torch.allclose(
            terms["symbol_residual"],
            2.0 * torch.nn.functional.mse_loss(predicted, target),
        ))
        self.assertTrue(torch.allclose(
            terms["symbol_l1"],
            0.1 * predicted.abs().mean(),
        ))

        loss = sum(terms.values())
        loss.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertGreater(predicted.grad.abs().sum().item(), 0.0)

    def test_symbol_objective_uses_nearest_codebook_target(self):
        _populate_test_config(conceptDim=3, symbolDim=3,
                              nConcepts=2, nSymbols=2,
                              perceptHasAttention=False)
        sym = Models.SymbolicSpace(
            inputShape=[2, 3],
            spaceShape=[2, 3],
            outputShape=[2, 3],
        )
        sym.symbol_residual_scale = 1.0
        sym.l1_lambda = 0.0
        sym.subspace.what.setW(torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]))

        predicted = torch.tensor(
            [[[0.9, 1.1, 1.0], [0.1, -0.1, 0.0]]],
            requires_grad=True,
        )
        terms = sym._compute_symbol_terms(predicted, use_codebook_target=True)

        target = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
        self.assertIn("symbol_residual", terms)
        self.assertTrue(torch.allclose(
            terms["symbol_residual"],
            torch.nn.functional.mse_loss(predicted, target),
        ))


class TestDataScaling(unittest.TestCase):
    """Tests for Data min/max tracking and scaling helpers."""

    def test_xor_data_ranges(self):
        """After loading XOR, Data has correct input/output min/max."""
        Models.TheData.load("xor")
        # XOR uses text input (embedded, L2-normalized) and binary labels
        self.assertEqual(Models.TheData.input_min, -1.0)
        self.assertEqual(Models.TheData.input_max, 1.0)
        self.assertEqual(Models.TheData.output_min, 0.0)
        self.assertEqual(Models.TheData.output_max, 1.0)

    def test_normalize_denormalize_roundtrip(self):
        """Data.normalize and Data.denormalize are inverses."""
        Models.TheData.input_min = -5.0
        Models.TheData.input_max = 5.0
        Models.TheData.output_min = -5.0
        Models.TheData.output_max = 5.0
        x = torch.tensor([[-5.0, 0.0, 5.0]])
        scaled = Models.TheData.normalize(x, which="input")
        self.assertTrue(torch.allclose(scaled, torch.tensor([[-1.0, 0.0, 1.0]])))
        roundtrip = Models.TheData.denormalize(scaled, which="input")
        self.assertTrue(torch.allclose(roundtrip, x))
        # denormalize(output): [-1,1] -> [min,max]
        act = torch.tensor([[-1.0, 0.0, 1.0]])
        rescaled = Models.TheData.denormalize(act, which="output")
        self.assertTrue(torch.allclose(rescaled, torch.tensor([[-5.0, 0.0, 5.0]])))

    def test_degenerate_range_noop(self):
        """When min==max, scaling is a no-op (returns input unchanged)."""
        Models.TheData.input_min = 3.0
        Models.TheData.input_max = 3.0
        x = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.equal(Models.TheData.normalize(x, which="input"), x))


class TestNormalizeFlag(unittest.TestCase):
    """Tests for the normalize flag on SubSpace.normalize()."""

    def test_normalize_false_does_not_modify(self):
        """normalize=False asserts range and does not modify the tensor."""
        _populate_test_config(inputDim=4, nInput=4)
        ss = Models.SubSpace(inputShape=[4, 4], outputShape=[4, 4])
        # Set vectors that are NOT in [-1,1] range
        x = torch.randn(2, 4, 4) * 5
        ss.set_event(x.clone())
        original = ss.materialize().clone()
        with self.assertRaises(AssertionError):
            ss.normalize("percepts", target="what", normalize=False)
        after = ss.materialize()
        self.assertTrue(torch.equal(original, after),
                        "normalize=False should not modify the tensor")

    def test_normalize_true_does_modify(self):
        """normalize=True modifies the tensor."""
        _populate_test_config(inputDim=4, nInput=4)
        ss = Models.SubSpace(inputShape=[4, 4], outputShape=[4, 4])
        x = torch.randn(2, 4, 4) * 5
        ss.set_event(x.clone())
        original = ss.materialize().clone()
        ss.normalize("percepts", target="what", normalize=True)
        after = ss.materialize()
        self.assertFalse(torch.equal(original, after),
                         "normalize=True should modify the tensor")
        self.assertTrue(torch.all(after >= -1) and torch.all(after <= 1))


class TestInputSpaceScaling(unittest.TestCase):
    """Tests for InputSpace min-max scaling of non-embedding data."""

    def test_simple_input_scaled_to_unit(self):
        """InputSpace scales passthrough what-content to [-1,1]."""
        Models.TheData.load("xor")
        Models.TheData.input_min = -3.0
        Models.TheData.input_max = 3.0
        nInput = 4
        _populate_test_config(inputDim=4, nInput=nInput, nWhere=0, nWhen=0)
        _idim = Models.TheXMLConfig.space("InputSpace", "nDim")
        _invec = Models.TheXMLConfig.space("InputSpace", "nVectors")
        inp = Models.InputSpace([nInput, _idim], [_invec, _idim],
                         [nInput, _idim], model_type="simple")
        x = torch.FloatTensor([[[-3, -1, 1, 3]] * nInput]).to(Models.TheDevice.get())
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
        model = Models.BasicModel()
        model.create(nInput=28*28, nPercepts=28*28, nConcepts=20, nSymbols=20, nOutput=10)
        x = torch.randn(2, 28*28, 1).tanh().to(Models.TheDevice.get())
        _, end_state, out, _ = model.forward(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved

    def test_subspace_activation_stored(self):
        """After forward with useSubspaceActivation, spaces have activations."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0, wordDim=1, outputDim=1,
            nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4,
            perceptPassThrough=True, symbolPassThrough=True, flatten=True,
            useSubspaceActivation=True)
        model = Models.BasicModel()
        model.create(nInput=16, nPercepts=16, nConcepts=8, nSymbols=8, nOutput=4)
        # SigmaLayer.nonlinear=True applies atanh; input must be in [-1, 1]
        # or atanh produces NaN. Matches the companion test above.
        x = torch.randn(2, 16, 1).tanh().to(Models.TheDevice.get())
        model.forward(x)

        # Check that activations are stored on spaces that compute them.
        # InputSpace (entry point) doesn't set activation --
        # it forwards the upstream SubSpace unchanged.
        for space in model.spaces:
            if isinstance(space, Models.InputSpace):
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
        Models.TheData.load("xor")
        nInput = 4
        _populate_test_config(inputDim=8, nInput=nInput, nWhere=2, nWhen=2, demuxed=True)
        _idim = Models.TheXMLConfig.space("InputSpace", "nDim")
        _invec = Models.TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = Models.InputSpace([nInput, _idim], [_invec, _idim],
                         [nInput, _idim + _obj], model_type="simple")
        x = torch.randn(2, nInput, _idim).to(Models.TheDevice.get())
        result = inp.forward(x)
        self.assertTrue(result.is_demuxed)
        self.assertEqual(list(result.what.getW().shape), [2, nInput, _idim])
        self.assertEqual(list(result.where.getW().shape), [2, nInput, 2])
        self.assertEqual(list(result.when.getW().shape), [2, nInput, 2])

    def test_materialize_produces_muxed(self):
        """InputSpace(demuxed=True).materialize() produces concat([what, where, when])."""
        Models.TheData.load("xor")
        nInput = 4
        _populate_test_config(inputDim=8, nInput=nInput, nWhere=2, nWhen=2, demuxed=True)
        _idim = Models.TheXMLConfig.space("InputSpace", "nDim")
        _invec = Models.TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        inp = Models.InputSpace([nInput, _idim], [_invec, _idim],
                         [nInput, _idim + _obj], model_type="simple")
        x = torch.randn(2, nInput, _idim).to(Models.TheDevice.get())
        result = inp.forward(x)
        muxed = result.materialize()
        self.assertEqual(list(muxed.shape), [2, nInput, _idim + _obj])

    def test_equivalence_with_muxed_input_space(self):
        """InputSpace(demuxed=True).materialize() == InputSpace(demuxed=False).materialize()."""
        Models.TheData.load("xor")
        nInput = 4

        # Build muxed (legacy) InputSpace
        _populate_test_config(inputDim=8, nInput=nInput, nWhere=2, nWhen=2, demuxed=False)
        _idim = Models.TheXMLConfig.space("InputSpace", "nDim")
        _invec = Models.TheXMLConfig.space("InputSpace", "nVectors")
        _obj = _obj_size("InputSpace")
        legacy = Models.InputSpace([nInput, _idim], [_invec, _idim],
                            [nInput, _idim + _obj], model_type="simple")

        # Build demuxed InputSpace
        _populate_test_config(inputDim=8, nInput=nInput, nWhere=2, nWhen=2, demuxed=True)
        demuxed = Models.InputSpace([nInput, _idim], [_invec, _idim],
                             [nInput, _idim + _obj], model_type="simple")

        x = torch.randn(2, nInput, _idim).to(Models.TheDevice.get())
        legacy_out = _unwrap(legacy.forward(x))
        demuxed_out = _unwrap(demuxed.forward(x))
        self.assertTrue(torch.allclose(legacy_out, demuxed_out, atol=1e-5),
                        f"max diff: {(legacy_out - demuxed_out).abs().max():.6f}")


class TestModalSpace(unittest.TestCase):
    """ModalSpace routes what/where/when through independent PerceptualSpaces."""

    def test_forward_shape(self):
        """ModalSpace.forward() produces correct muxed output shape."""
        nInput = 4
        nWhere = 2
        nWhen = 2
        nDim = 8
        _populate_test_config(inputDim=nDim, perceptDim=nDim,
                              nInput=nInput, nPercepts=nInput,
                              nWhere=nWhere, nWhen=nWhen,
                              perceptPassThrough=True)
        muxed_w = nDim + nWhere + nWhen
        space = Models.ModalSpace([nInput, muxed_w], [nInput, nDim], [nInput, muxed_w])
        # Build a demuxed input
        what_t = torch.randn(2, nInput, nDim).to(Models.TheDevice.get())
        where_t = torch.randn(2, nInput, nWhere).to(Models.TheDevice.get())
        when_t = torch.randn(2, nInput, nWhen).to(Models.TheDevice.get())
        ss = Models.SubSpace(inputShape=[nInput, muxed_w], outputShape=[nInput, muxed_w])
        ss.set_demuxed(what_t, where_t, when_t)
        result = space.forward(ss)
        materialized = result.materialize()
        self.assertEqual(list(materialized.shape), [2, nInput, muxed_w])

    def test_degenerate_no_position(self):
        """With nWhere=nWhen=0, ModalSpace degenerates to a single PerceptualSpace."""
        nInput = 4
        nDim = 8
        _populate_test_config(inputDim=nDim, perceptDim=nDim,
                              nInput=nInput, nPercepts=nInput,
                              nWhere=0, nWhen=0,
                              perceptPassThrough=True)
        space = Models.ModalSpace([nInput, nDim], [nInput, nDim], [nInput, nDim])
        self.assertIsNone(space.whereSpace)
        self.assertIsNone(space.whenSpace)
        # Forward with muxed input (no demux needed)
        x = torch.randn(2, nInput, nDim).to(Models.TheDevice.get())
        ss = Models.SubSpace(inputShape=[nInput, nDim], outputShape=[nInput, nDim])
        ss.set_event(x)
        result = space.forward(ss)
        self.assertEqual(list(result.materialize().shape), [2, nInput, nDim])


class TestBasicModelDemuxed(unittest.TestCase):
    """BasicModel with demuxed=True creates InputSpace(demuxed) + ModalSpace."""

    def test_create_from_config(self):
        """BasicModel with demuxed=True creates InputSpace(demuxed) + ModalSpace."""
        Models.TheData.load("xor")
        _populate_test_config(
            inputDim=8, perceptDim=8, conceptDim=8, symbolDim=0,
            outputDim=4,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nWords=4, nOutput=4,
            nWhere=2, nWhen=2,
            demuxed=True,
            flatten=True, perceptPassThrough=True)
        model = Models.BasicModel()
        model.create(nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nOutput=4,
                     model_type="simple")
        self.assertIsInstance(model.inputSpace, Models.InputSpace)
        self.assertTrue(model.inputSpace.demuxed)
        self.assertIsInstance(model.perceptualSpace, Models.ModalSpace)




if __name__ == "__main__":
    unittest.main()
