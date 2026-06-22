"""Gradient-bearing ``rule_probs``: the intra-sentence predictor's routing
bias must backprop into the signal router's scorer parameters.

Before this change, ``SymbolSubSpace._synthesize_rule_probs`` scattered unit
mass onto the SELECTED (hard, argmax) rule-ids -- a DETACHED distribution,
so ``IntraSentenceLayer.routing_proj(rule_probs)`` could not train the
router. The fix (``_synthesize_rule_probs_soft``) aggregates the router's
SOFT per-space_role marginals cached in ``LanguageLayer._last_space_role_routings``
(``unary action_probs`` apply columns + ``binary reduce_marginal_op``
summed over all reduction rounds) into a global ``[B, n_rules]`` tensor
that keeps a graph back to the router's anchor scorers
(``copy_anchor`` / ``apply_anchor`` on the unary layer, ``reduce_anchor``
on the binary layer).

Proof points (the deliverable):
  1. On a FULL-ROUTER grammar (``MM_xor_loopback.xml``, which fires
     ``LanguageLayer.compose``), a grad-enabled ``compose`` makes
     ``routing_state.rule_probs.requires_grad`` True with a live graph.
  2. A loss through a non-uniform projection of ``rule_probs`` (the shape
     of the predictor's ``routing_proj`` bias) backprops NON-ZERO grad
     into the router's unary scorer anchors.
  3. A bare router with no space_role mask (so the binary reduce path actually
     carries mass) backprops NON-ZERO grad into BOTH the unary
     ``apply_anchor`` AND the binary ``reduce_anchor`` -- the complete
     unary+binary marginal -> rule_probs -> router-scorer path.
  4. The default-only fast path (no router; ``_last_space_role_routings`` empty)
     still dispatches to the (detached) hard scatter without error.

See: bin/Language.py ``SymbolSubSpace._synthesize_rule_probs`` /
``_synthesize_rule_probs_soft`` / ``_map_op_columns``; the NaN-safe
``_masked_softmax_lastdim`` (fully space_role-masked reduce rows).
"""
import os
import sys
import types
import unittest
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import torch.nn as nn

import Language
from Language import LanguageLayer, SymbolSubSpace

_DATA_DIR = os.path.join(_PROJECT, "data")
_ROUTER_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")   # full router
_DEFAULT_CONFIG = os.path.join(_DATA_DIR, "model.xml")            # default-only
_DEFAULTS_PATH = os.path.join(_DATA_DIR, "model.xml")


def _build_model(config_path):
    """Construct a BasicModel from ``config_path`` on CPU, grammar reset."""
    import Models
    from util import init_config, init_device
    init_device("cpu")
    init_config(path=config_path, defaults_path=_DEFAULTS_PATH)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model, _ = Models.BasicModel.from_config(config_path)
    model.eval()
    return model


class TestSoftRuleProbsGradReachesRouter(unittest.TestCase):
    """The soft ``rule_probs`` path carries gradient predictor-loss ->
    routing_proj -> rule_probs -> router marginals -> router scorer params.
    """

    def test_rule_probs_requires_grad_on_full_router(self):
        """A grad-enabled ``compose`` on the full-router grammar makes
        ``routing_state.rule_probs`` a LIVE differentiable tensor."""
        model = _build_model(_ROUTER_CONFIG)
        ss = model.symbolSpace
        self.assertIsNotNone(ss, "router config must have a symbolSpace.")
        self.assertFalse(
            ss._grammar_is_default_only,
            "MM_xor_loopback must be a FULL-ROUTER grammar (compose fires).")
        D = ss.languageLayer.feature_dim
        x = torch.randn(2, 4, D, requires_grad=True)
        ss.compose(x)
        rp = ss.routing_state.rule_probs
        self.assertTrue(torch.is_tensor(rp))
        self.assertTrue(
            rp.requires_grad,
            "soft rule_probs must require grad (router params are trainable).")
        self.assertIsNotNone(
            rp.grad_fn, "soft rule_probs must carry a graph back to the router.")
        self.assertTrue(torch.isfinite(rp).all(), "rule_probs must be finite.")

    def test_grad_reaches_unary_anchors_via_projection(self):
        """A loss through a NON-UNIFORM projection of ``rule_probs`` (the
        predictor's routing_proj bias shape) backprops non-zero grad into
        the router's unary scorer anchors.

        Two test-setup controls make the proof unambiguous:
          * ``rule_probs.sum()`` is degenerate -- each L1-normalized row
            sums to a constant 1 -- so the bias is projected by a
            non-uniform map, exactly as the real
            ``IntraSentenceLayer.routing_proj`` does.
          * The unary action scorer is a straight-through softmax; with
            random anchors it can SATURATE to one-hot, where
            ``d(softmax)/d(logits) ~= 0`` and the (real, connected)
            gradient is incidentally ~0. We zero the unary anchors so the
            action posterior is non-degenerate and the gradient magnitude
            is a clean, seed-robust non-zero -- the analogue of
            ``test_bias_fires_on_real_model`` forcing a non-trivial
            ``routing_proj``. The graph CONNECTIVITY (the deliverable) is
            asserted unconditionally by
            ``test_rule_probs_requires_grad_on_full_router``.
        """
        model = _build_model(_ROUTER_CONFIG)
        ss = model.symbolSpace
        ll = ss.languageLayer
        # Space-role-free fold: the grammar collapses to a SINGLE reduction space_role
        # (no hardcoded S/C/P). Resolve whichever key the unary layer was
        # registered under -- there is exactly one -- and assert the
        # router exposes the APPLY/COPY anchor scorers on it.
        unary_keys = list(ll._unary_layers.keys())
        self.assertEqual(
            len(unary_keys), 1,
            f"space_role-free fold expects one unary reduction space_role; "
            f"got {unary_keys}.")
        ul = ll._unary_layers[unary_keys[0]]
        self.assertTrue(
            hasattr(ul, "apply_anchor") and hasattr(ul, "copy_anchor"),
            "unary reduction layer must expose the router's APPLY/COPY "
            "anchor scorers.")
        # De-saturate the straight-through action softmax so the gradient
        # path magnitude is not masked by an incidental one-hot posterior.
        with torch.no_grad():
            ul.apply_anchor.zero_()
            ul.copy_anchor.zero_()
        for p in (ul.apply_anchor, ul.copy_anchor):
            p.grad = None
        torch.manual_seed(0)
        D = ll.feature_dim
        x = torch.randn(2, 4, D, requires_grad=True)
        ss.compose(x)
        rp = ss.routing_state.rule_probs
        self.assertTrue(rp.requires_grad and rp.grad_fn is not None)
        # Mimic routing_proj: [n_rules] -> [concept_dim] non-uniform map.
        proj = nn.Linear(int(rp.shape[1]), 5, bias=True)
        loss = proj(rp).pow(2).sum()
        loss.backward()
        apply_g = ul.apply_anchor.grad
        self.assertIsNotNone(
            apply_g, "unary apply_anchor must receive grad from rule_probs.")
        self.assertGreater(
            float(apply_g.abs().sum()), 0.0,
            "gradient must reach the unary apply_anchor (the router's "
            "APPLY scorer) through rule_probs.")

    def test_grad_reaches_unary_and_binary_anchors_bare_router(self):
        """On a bare router with NO space_role mask, both the unary
        ``apply_anchor`` and the binary ``reduce_anchor`` receive non-zero
        grad -- the complete unary + binary marginal -> rule_probs ->
        router-scorer path. (The MM_xor_loopback grammar space_role-masks its
        SS-space_role reduce ops against the CS-initialized position space_role, so the
        reduce column is structurally zero there; an unspace_role-masked router
        exercises the binary path.)"""
        # rule_table must be populated for n_rules > 0; init the grammar
        # via the router config (3 rules: not / intersection / union).
        model = _build_model(_ROUTER_CONFIG)
        n_rules = len(Language.TheGrammar.rule_table)
        self.assertGreaterEqual(n_rules, 3, "grammar must have >= 3 rules.")

        router = LanguageLayer(
            n_input=4, n_output=4, hidden_dim=16, feature_dim=4,
            max_depth=3, temperature=1.0)

        class _NotOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return -self.p(x)

        class _AndOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Linear(4, 4, bias=False)

            def forward(self, a, b):
                return self.p(a) * self.p(b)

        class _OrOp(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Linear(4, 4, bias=False)

            def forward(self, a, b):
                return (self.p(a) + self.p(b)).clamp(-1.0, 1.0)

        # NO op_space_roles -> no space_role mask -> the reduce path carries mass.
        router.attach_unary_ops(ops=[_NotOp()], rule_ids=[0], space_role="SS")
        router.attach_layer_ops(
            ops=[_AndOp(), _OrOp()], rule_ids=[1, 2], space_role="SS")

        # Bind the SymbolSubSpace aggregator methods to a stand-in carrying
        # this router (the methods only need ``self.languageLayer``).
        stub = types.SimpleNamespace(languageLayer=router)
        stub._synthesize_rule_probs_soft = types.MethodType(
            SymbolSubSpace._synthesize_rule_probs_soft, stub)
        stub._map_op_columns = types.MethodType(
            SymbolSubSpace._map_op_columns, stub)

        torch.manual_seed(1)
        x = torch.randn(2, 4, 4, requires_grad=True)
        router.compose(x, word_space=None)
        rp = stub._synthesize_rule_probs_soft(router._last_space_role_routings, 2)
        self.assertTrue(torch.is_tensor(rp) and rp.requires_grad)
        self.assertTrue(torch.isfinite(rp).all())
        # All three rules should carry mass here (no space_role mask).
        col_mass = rp.sum(dim=0)
        self.assertGreater(
            float(col_mass[1] + col_mass[2]), 0.0,
            "binary reduce ops must carry mass in the unspace_role-masked router.")

        ul = router._unary_layers["SS"]
        bl = router._binary_layers["SS"]
        for p in (ul.apply_anchor, bl.reduce_anchor):
            p.grad = None
        proj = nn.Linear(int(rp.shape[1]), 5, bias=True)
        proj(rp).pow(2).sum().backward()

        self.assertIsNotNone(ul.apply_anchor.grad)
        self.assertGreater(
            float(ul.apply_anchor.grad.abs().sum()), 0.0,
            "grad must reach the unary apply_anchor.")
        self.assertIsNotNone(bl.reduce_anchor.grad)
        self.assertGreater(
            float(bl.reduce_anchor.grad.abs().sum()), 0.0,
            "grad must reach the binary reduce_anchor (the router's REDUCE "
            "scorer) through the reduce_marginal_op aggregation.")

    def test_default_only_fallback_is_detached(self):
        """The default-only fast path (no router; ``_last_space_role_routings``
        empty) dispatches to the HARD scatter and returns a finite,
        DETACHED ``rule_probs`` without error (the fallback is preserved
        exactly)."""
        model = _build_model(_DEFAULT_CONFIG)
        ss = model.symbolSpace
        self.assertIsNotNone(ss, "default config must have a symbolSpace.")
        self.assertTrue(
            ss._grammar_is_default_only,
            "model.xml must be a default-only grammar (no router).")
        self.assertFalse(
            bool(getattr(ss.languageLayer, "_last_space_role_routings", {})),
            "default-only path must leave _last_space_role_routings empty.")
        rbt = ss._default_compose_rules()
        with torch.enable_grad():
            rp = ss._synthesize_rule_probs(rbt, batch_size=2)
        self.assertTrue(torch.is_tensor(rp), "hard fallback must return a tensor.")
        self.assertFalse(
            rp.requires_grad,
            "default-only hard scatter must be DETACHED (no router to train).")
        self.assertTrue(torch.isfinite(rp).all(),
                        "hard fallback rule_probs must be finite.")


if __name__ == "__main__":
    unittest.main()
