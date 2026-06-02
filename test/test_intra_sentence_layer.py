"""Unit tests for IntraSentenceLayer (Task 2, STM serial/parallel modes).

IntraSentenceLayer is the in-STM autoregressive predictor: a combined
PI-then-Sigma fold with NO intermediate tanh (the Sigma body is built
``nonlinear=False``). The PI body lifts STM slots into the working width
and the raw-linear Sigma collapses them (serial: cross-slot sum-fold ->
one idea ``[B, D]``; parallel: per-slot map -> ``[B, N, D]``). A soft
routing distribution conditions the Sigma output as an additive bias;
``routing=None`` skips it so the layer is usable before the per-word
router is wired (Task 4).

Asserts:
  * Serial forward returns [B, D], finite.
  * Parallel forward returns [B, N, D], finite.
  * A non-None routing changes the serial output (bias is applied).
  * The parallel/invertible roundtrip reverse(forward(x)) ~= x.
  * ConceptualSpace owns ``intraSentenceLayer`` and ``Reset()`` reaches
    it without error.

See: doc/plans (STM serial/parallel modes), bin/Layers.py
(IntraSentenceLayer), bin/Spaces.py (ConceptualSpace.__init__).
"""
import os
import sys
import unittest
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

from Layers import IntraSentenceLayer, PiLayer, SigmaLayer
from util import TheDevice

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG_PATH = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS_PATH = os.path.join(_DATA_DIR, "model.xml")


def _make_layer(concept_dim=6, stm_capacity=8, routing_dim=5,
                working_dim=None):
    """Build an IntraSentenceLayer pinned to exploit mode (sigma=0) so
    the invertible-roundtrip assertions are deterministic."""
    layer = IntraSentenceLayer(
        concept_dim=concept_dim,
        stm_capacity=stm_capacity,
        routing_dim=routing_dim,
        working_dim=working_dim,
    )
    layer.set_sigma(0)
    layer.eval()
    return layer.to(TheDevice.get())


class TestIntraSentenceLayerShapes(unittest.TestCase):
    """Forward output shapes + finiteness in both regimes."""

    def test_serial_forward_shape(self):
        B, K, D = 4, 7, 6
        layer = _make_layer(concept_dim=D)
        x = torch.randn(B, K, D, device=TheDevice.get()).tanh()
        out = layer.forward(x, routing=None, parallel=False)
        self.assertEqual(tuple(out.shape), (B, D),
                         "serial forward must collapse the K slots to [B, D]")
        self.assertTrue(torch.isfinite(out).all(), "serial output must be finite")

    def test_parallel_forward_shape(self):
        B, N, D = 3, 5, 6
        layer = _make_layer(concept_dim=D)
        x = torch.randn(B, N, D, device=TheDevice.get()).tanh()
        out = layer.forward(x, routing=None, parallel=True)
        self.assertEqual(tuple(out.shape), (B, N, D),
                         "parallel forward must be per-slot -> [B, N, D]")
        self.assertTrue(torch.isfinite(out).all(), "parallel output must be finite")

    def test_no_intermediate_activation(self):
        # The defining requirement: the Sigma body has nonlinear=False
        # so there is genuinely no tanh interposed before the Sigma
        # matmul. The PI body keeps its log-domain nonlinearity.
        layer = _make_layer()
        self.assertIsInstance(layer.pi, PiLayer)
        self.assertIsInstance(layer.sigma, SigmaLayer)
        self.assertFalse(layer.sigma.nonlinear,
                         "Sigma must be built nonlinear=False (no intermediate tanh).")
        self.assertTrue(layer.pi.nonlinear,
                        "PI keeps its log-domain boundary nonlinearity.")
        # Sublayers reachable by the param cascade (appended to ModuleList).
        ids = {id(p) for p in layer.getParameters()}
        self.assertTrue(any(id(p) in ids for p in layer.pi.parameters()))
        self.assertTrue(any(id(p) in ids for p in layer.sigma.parameters()))


class TestIntraSentenceLayerRouting(unittest.TestCase):
    """The routing distribution is actually injected as a Sigma bias."""

    def test_routing_changes_serial_output(self):
        B, K, D, R = 4, 7, 6, 5
        layer = _make_layer(concept_dim=D, routing_dim=R)
        # Make the routing projection non-trivial so the bias is visible
        # (zero-init Linear bias + small random weight would still be
        # detectable, but force a clearly non-zero bias to be safe).
        with torch.no_grad():
            layer.routing_proj.weight.normal_(0.0, 1.0)
            layer.routing_proj.bias.normal_(0.0, 1.0)
        x = torch.randn(B, K, D, device=TheDevice.get()).tanh()
        routing = torch.randn(B, R, device=TheDevice.get())
        out_none = layer.forward(x, routing=None, parallel=False)
        out_routed = layer.forward(x, routing=routing, parallel=False)
        self.assertEqual(tuple(out_routed.shape), (B, D))
        self.assertFalse(
            torch.allclose(out_none, out_routed, atol=1e-6),
            "passing a non-None routing must change the serial output "
            "(additive Sigma bias is applied).")
        # The difference must equal exactly the projected routing bias.
        expected_delta = layer.routing_proj(routing)
        self.assertTrue(
            torch.allclose(out_routed - out_none, expected_delta, atol=1e-5),
            "routing delta must equal the projected bias.")

    def test_routing_changes_parallel_output(self):
        B, N, D, R = 3, 5, 6, 5
        layer = _make_layer(concept_dim=D, routing_dim=R)
        with torch.no_grad():
            layer.routing_proj.weight.normal_(0.0, 1.0)
            layer.routing_proj.bias.normal_(0.0, 1.0)
        x = torch.randn(B, N, D, device=TheDevice.get()).tanh()
        routing = torch.randn(B, R, device=TheDevice.get())
        out_none = layer.forward(x, routing=None, parallel=True)
        out_routed = layer.forward(x, routing=routing, parallel=True)
        self.assertFalse(
            torch.allclose(out_none, out_routed, atol=1e-6),
            "routing bias must broadcast over the slot axis in parallel mode.")


class TestIntraSentenceLayerRoundtrip(unittest.TestCase):
    """Parallel/per-slot path is invertible (square sublayers)."""

    def test_parallel_roundtrip_no_routing(self):
        B, N, D = 3, 5, 6
        layer = _make_layer(concept_dim=D)
        x = (torch.rand(B, N, D, device=TheDevice.get()) * 2 - 1)
        y = layer.forward(x, routing=None, parallel=True)
        x_rec = layer.reverse(y, routing=None, parallel=True)
        self.assertEqual(tuple(x_rec.shape), (B, N, D))
        # Square invertible PI->Sigma: roundtrip is exact up to the LDU
        # inverse tolerance. The repo's invertible Pi/Sigma roundtrip
        # tests admit rel-err ~0.1 (test_invertibility.py); use an
        # absolute atol on [-1,1] data of 2e-1 (the lift/lower learned
        # roundtrip convention) -- here it is much tighter in practice.
        err = (x - x_rec).abs().max().item()
        self.assertLess(err, 2e-1,
                        f"parallel roundtrip should be tight; max|err|={err:.2e}")

    def test_parallel_roundtrip_with_routing(self):
        # Subtracting the additive routing bias must recover the same
        # pre-bias state, so the roundtrip is unaffected by routing.
        B, N, D, R = 3, 5, 6, 5
        layer = _make_layer(concept_dim=D, routing_dim=R)
        with torch.no_grad():
            layer.routing_proj.weight.normal_(0.0, 0.5)
            layer.routing_proj.bias.normal_(0.0, 0.5)
        x = (torch.rand(B, N, D, device=TheDevice.get()) * 2 - 1)
        routing = torch.randn(B, R, device=TheDevice.get())
        y = layer.forward(x, routing=routing, parallel=True)
        x_rec = layer.reverse(y, routing=routing, parallel=True)
        err = (x - x_rec).abs().max().item()
        self.assertLess(err, 2e-1,
                        f"routed roundtrip should recover x; max|err|={err:.2e}")

    def test_serial_reverse_shape(self):
        # Serial collapse is many-to-one: reverse is APPROXIMATE and
        # fans the recovered fold across k = stm_capacity - 1 slots.
        B, K, D, cap = 4, 7, 6, 8
        layer = _make_layer(concept_dim=D, stm_capacity=cap)
        x = torch.randn(B, K, D, device=TheDevice.get()).tanh()
        pred = layer.forward(x, routing=None, parallel=False)
        recon = layer.reverse(pred, routing=None, parallel=False)
        # Serial reverse fans the recovered fold across k = cap - 1 slots
        # and returns [B, k, D].
        self.assertEqual(tuple(recon.shape), (B, cap - 1, D),
                         "serial reverse defaults k = stm_capacity - 1 -> [B, k, D]")
        self.assertTrue(torch.isfinite(recon).all())

    def test_intra_loss_is_mse(self):
        B, D = 4, 6
        layer = _make_layer(concept_dim=D)
        pred = torch.randn(B, D, device=TheDevice.get())
        target = torch.randn(B, D, device=TheDevice.get())
        loss = layer.intra_loss(pred, target)
        expected = torch.nn.functional.mse_loss(pred, target)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(loss.dim(), 0, "intra_loss returns a scalar")


class TestConceptualSpaceOwnership(unittest.TestCase):
    """ConceptualSpace builds + owns the layer, and Reset reaches it."""

    @classmethod
    def setUpClass(cls):
        import Models
        import Language
        from util import init_config
        init_config(path=_CONFIG_PATH, defaults_path=_DEFAULTS_PATH)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cls.model, _ = Models.BasicModel.from_config(_CONFIG_PATH)
        cls.model.eval()

    def test_space_owns_intra_layer(self):
        cs = self.model.conceptualSpaces[0]
        self.assertTrue(hasattr(cs, "intraSentenceLayer"),
                        "ConceptualSpace must own intraSentenceLayer.")
        self.assertIsInstance(cs.intraSentenceLayer, IntraSentenceLayer)

    def test_intra_layer_in_cascade(self):
        cs = self.model.conceptualSpaces[0]
        # Appended to self.layers so the Start/Reset cascade reaches it.
        self.assertIn(cs.intraSentenceLayer, list(cs.layers),
                      "intraSentenceLayer must be in ConceptualSpace.layers.")

    def test_intra_loss_weight_read(self):
        cs = self.model.conceptualSpaces[0]
        self.assertTrue(hasattr(cs, "intra_loss_weight"))
        # Default knob value is 0.1 (model.xml).
        self.assertAlmostEqual(float(cs.intra_loss_weight), 0.1, places=6)

    def test_reset_reaches_layer(self):
        cs = self.model.conceptualSpaces[0]
        # Reset cascade must not error with the layer in self.layers.
        cs.Reset()
        # The layer's own Reset is a structural no-op but must be callable.
        cs.intraSentenceLayer.Reset()

    def test_dims_match_space(self):
        cs = self.model.conceptualSpaces[0]
        layer = cs.intraSentenceLayer
        self.assertEqual(layer.concept_dim, cs.concept_dim,
                         "layer concept_dim must match the space's.")
        self.assertEqual(layer.stm_capacity, cs.stm_capacity,
                         "layer stm_capacity must reuse the space's STM source.")

    def test_routing_dim_is_n_rules(self):
        # The routing width must now be the grammar's rule-vocabulary
        # size (n_rules), NOT the old concept_dim placeholder. This is
        # the dim WordSubSpace.routing_state.rule_probs is emitted at and
        # the dim routing_proj projects from.
        import Language
        cs = self.model.conceptualSpaces[0]
        n_rules = len(Language.TheGrammar.rule_table)
        self.assertGreater(n_rules, 0, "grammar rule_table must be populated.")
        self.assertEqual(
            cs._intra_routing_dim, n_rules,
            "ConceptualSpace._intra_routing_dim must be len(rule_table).")
        self.assertEqual(
            cs.intraSentenceLayer.routing_proj.in_features, n_rules,
            "routing_proj must project from the n_rules rule distribution.")


class TestRuleConditionedPredictor(unittest.TestCase):
    """End-to-end: after a serial forward the per-word router populates a
    dense ``rule_probs`` and the intra-sentence predictor is genuinely
    rule-conditioned (the bias fires by exactly ``routing_proj(rule_probs)``).
    """

    @classmethod
    def setUpClass(cls):
        import Models
        import Language
        from util import init_config, init_device
        init_device("cpu")
        init_config(path=_CONFIG_PATH, defaults_path=_DEFAULTS_PATH)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model, _ = Models.BasicModel.from_config(_CONFIG_PATH)
        Models.TheData.load("xor")
        cls.model.eval()

    def _run_one_forward(self):
        model = self.model
        loader = model.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x = model.inputSpace.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                model.forward(x)

    def test_routing_state_built_with_rule_probs(self):
        import Language
        self._run_one_forward()
        ws = self.model.wordSubSpace
        rs = getattr(ws, "routing_state", None)
        self.assertIsNotNone(rs, "compose must build wordSubSpace.routing_state.")
        # current_rules dict contract preserved (ADDITIVE, not replaced).
        self.assertIsInstance(ws.current_rules, dict,
                              "current_rules must stay a dict (SS dispatch).")
        rp = rs.rule_probs
        self.assertTrue(torch.is_tensor(rp),
                        "routing_state.rule_probs must be a tensor after compose.")
        n_rules = len(Language.TheGrammar.rule_table)
        self.assertEqual(int(rp.shape[-1]), n_rules,
                         "rule_probs last dim must be n_rules.")
        # Distribution: each fired row sums to 1 (or is an all-zero row).
        row_sums = rp.sum(dim=1)
        for s in row_sums.tolist():
            self.assertTrue(abs(s - 1.0) < 1e-5 or abs(s) < 1e-9,
                            f"each rule_probs row must sum to 1 or 0; got {s}")
        self.assertTrue(torch.isfinite(rp).all(), "rule_probs must be finite.")

    def test_intra_routing_for_predict_returns_real_tensor(self):
        import Language
        self._run_one_forward()
        cs = self.model.conceptualSpace
        r = cs._intra_routing_for_predict()
        self.assertIsNotNone(
            r, "_intra_routing_for_predict must return a real tensor after a "
               "serial forward (not None).")
        self.assertTrue(torch.is_tensor(r))
        n_rules = len(Language.TheGrammar.rule_table)
        self.assertEqual(int(r.shape[-1]), n_rules,
                         "returned routing must be [B, n_rules].")

    def test_bias_fires_on_real_model(self):
        # The defining acceptance check: with the real rule_probs from a
        # serial forward, the predictor output MUST change vs routing=None
        # by EXACTLY the projected routing bias.
        self._run_one_forward()
        cs = self.model.conceptualSpace
        layer = cs.intraSentenceLayer
        # Force a clearly non-zero projection so the delta is unambiguous.
        with torch.no_grad():
            layer.routing_proj.weight.normal_(0.0, 1.0)
            layer.routing_proj.bias.normal_(0.0, 1.0)
        routing = cs._intra_routing_for_predict()
        self.assertIsNotNone(routing)
        snap = cs.stm.snapshot()
        self.assertIsNotNone(snap)
        ctx = snap[:, :-1] if snap.shape[1] >= 2 else snap
        with torch.no_grad():
            out_none = layer.forward(ctx, routing=None, parallel=False)
            out_routed = layer.forward(ctx, routing=routing, parallel=False)
            expected_delta = layer.routing_proj(routing)
        self.assertFalse(
            torch.allclose(out_none, out_routed, atol=1e-6),
            "rule_probs must change the predictor output (bias must fire).")
        self.assertTrue(
            torch.allclose(out_routed - out_none, expected_delta, atol=1e-5),
            "predictor delta must equal exactly routing_proj(rule_probs).")


if __name__ == "__main__":
    unittest.main()
