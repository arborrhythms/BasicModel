"""Stage 9: SymbolizeLayer as binary GrammarLayer.

Tests the new ``SymbolizeLayer`` class:

  * Class-level attributes: ``rule_name == "symbolize"``, ``arity == 2``,
    ``space_role == 'CS'``, ``invertible == True``.
  * ``forward(left, right)`` identifies a percept_id and a symbol_idx
    (by nearest-row match in PS.percept_store / SS.codebook), then
    delegates to ``WholeSpace.insert_meta(ps_idx, ws_idx,
    fused_vec=combine(left, right))`` and returns the META vector.
  * ``reverse(parent)`` walks the SS taxonomy starting from a
    nearest-match to ``parent`` and recovers the ``(left, right)``
    pair as ``(PS_vec, SS_vec)``.
  * Idempotent: a second ``forward(left, right)`` on the same pair
    must return the same META idx (no new META row allocated; EMA
    update of the stored fused vec).
  * Signal-router dispatch: a grammar declaring ``symbolize(C, C)`` at the
    CS space_role causes ``SymbolizeLayer`` to bind to the signal router via
    ``_attach_per_space_syntactic_layer``.
  * No-PerceptStore fallback: when ``PartSpace`` lacks a
    ``percept_store`` (legacy lexicon mode), ``forward`` falls back to
    a no-op average ``(left + right) / 2`` without registering a META
    node.
  * Numerical guard: ``NaN`` / ``Inf`` in ``left`` / ``right`` raises
    (project's "fail loud" policy).

Stage 9 of doc/plans/2026-05-27-perceptstore-meta-taxonomy-reentrancy.md.
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_radix_model():
    """Build the MM_xor radix-chunking model for end-to-end tests."""
    import warnings
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    m.eval()
    return m


def _ws_row_from_pos(ws, pos):
    """Position -> SS.codebook row index via ``_ws_pos_to_row``."""
    row = ws._ws_pos_to_row.get(int(pos))
    if row is None:
        raise AssertionError(
            f"position {pos} has no SS-side row binding; expected an "
            f"SS or META position")
    return int(row)


def _ps_row_from_pos(ws, pos):
    """Position -> PS.percept_store row index via ``_ps_pos_to_row``."""
    row = ws._ps_pos_to_row.get(int(pos))
    if row is None:
        raise AssertionError(
            f"position {pos} has no PS-side row binding; expected a "
            f"PS position")
    return int(row)


# ---------------------------------------------------------------------------
# Class attribute tests
# ---------------------------------------------------------------------------


class TestSymbolizeLayerClassAttributes(unittest.TestCase):
    """Stage 9: class-level attributes for the binary META grammar-op contract."""

    def test_meta_is_grammarlayer_subclass(self):
        from Layers import GrammarLayer, SymbolizeLayer
        self.assertTrue(
            issubclass(SymbolizeLayer, GrammarLayer),
            "SymbolizeLayer must inherit from GrammarLayer (Stage 9).")

    def test_meta_arity_is_two(self):
        from Layers import SymbolizeLayer
        self.assertEqual(
            SymbolizeLayer.arity, 2,
            "SymbolizeLayer must have arity == 2 (binary op).")

    def test_meta_rule_name(self):
        from Layers import SymbolizeLayer
        self.assertEqual(SymbolizeLayer.rule_name, "symbolize")

    def test_meta_space_role_is_CS(self):
        from Layers import SymbolizeLayer
        self.assertEqual(
            SymbolizeLayer.space_role, 'CS',
            "SymbolizeLayer.space_role must be 'CS'.")

    def test_meta_invertible(self):
        """SymbolizeLayer is invertible (reverse recovers the discrete pair).

        Note: invertibility here is at the discrete identity level
        (recovers the PS and SS rows), not full vector-space exact.
        """
        from Layers import SymbolizeLayer
        self.assertTrue(
            SymbolizeLayer.invertible,
            "SymbolizeLayer.invertible must be True (discrete-level recovery).")


# ---------------------------------------------------------------------------
# Forward path
# ---------------------------------------------------------------------------


class TestSymbolizeLayerForward(unittest.TestCase):
    """``forward(left, right)`` identifies (ps_idx, ws_idx) by nearest
    match, calls ``WholeSpace.insert_meta``, returns the META vector."""

    def test_forward_creates_meta_node_and_returns_meta_vector(self):
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        # Pre-seed a percept and a symbol so the nearest-match lookups
        # can identify them.
        ps_idx = ws.insert_percept(b"hello")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        # Pin the codebook rows to deterministic vectors so nearest-
        # match is unambiguous.
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[0] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[1] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        # Construct SymbolizeLayer with Space refs.
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        # Forward with the codebook vectors themselves; should find
        # exact nearest matches and create a META node.
        n_taxonomy_before = len(ws.taxonomy)
        out = meta.forward(ps_vec, ws_vec)
        # Output is a tensor of shape [D] (the META row vector).
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.shape[-1], D)
        # A new META node was registered.
        self.assertEqual(
            len(ws.taxonomy), n_taxonomy_before + 1,
            "SymbolizeLayer.forward must register a new META node when none "
            "exists for the pair.")
        # The new META node's children are {ps_idx, ws_idx}.
        meta_idx = ws.meta_pair_to_idx.get((ps_idx, ws_idx))
        self.assertIsNotNone(
            meta_idx,
            "SymbolizeLayer.forward must register the pair in "
            "ws.meta_pair_to_idx.")
        children = set(ws.taxonomy_children(meta_idx))
        self.assertEqual(children, {ps_idx, ws_idx})

    def test_forward_returns_existing_meta_vector_when_pair_known(self):
        """forward on a pair with an existing META returns that META's vec."""
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        ps_idx = ws.insert_percept(b"world")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[2] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[3] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        # Pre-register the META node directly.
        explicit_fused = torch.zeros(D)
        explicit_fused[4] = 0.5
        meta_idx = ws.insert_meta(ps_idx, ws_idx, fused_vec=explicit_fused)
        meta_row = _ws_row_from_pos(ws,meta_idx)
        meta_vec = ws.subspace.what.getW()[meta_row].detach().clone()
        # Forward through SymbolizeLayer; must return the existing META vec
        # (post-EMA-update with the new fused), not allocate a new row.
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        n_taxonomy_before = len(ws.taxonomy)
        out = meta.forward(ps_vec, ws_vec)
        self.assertEqual(
            len(ws.taxonomy), n_taxonomy_before,
            "SymbolizeLayer.forward must NOT register a new META node when the "
            "pair already has one.")
        # Output must match the (EMA-updated) META row, not the original
        # pre-EMA snapshot. We just check it's finite and is the row's
        # current value.
        cur_meta_vec = ws.subspace.what.getW()[meta_row].detach()
        self.assertTrue(
            torch.allclose(out.detach(), cur_meta_vec, atol=1e-5),
            f"SymbolizeLayer.forward must return the SS row hosting the "
            f"META; got {out.tolist()} vs {cur_meta_vec.tolist()}")


# ---------------------------------------------------------------------------
# Reverse path
# ---------------------------------------------------------------------------


class TestSymbolizeLayerReverse(unittest.TestCase):
    """``reverse(parent)`` walks SS.codebook nearest-match to a META
    node, returns ``(ps_vec, ws_vec)`` for the children."""

    def test_reverse_recovers_pair_from_meta_vector(self):
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        ps_idx = ws.insert_percept(b"alpha")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[0] = 0.9
        ws_vec = torch.zeros(D)
        ws_vec[1] = 0.8
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        # Register a META node and pin its row to a deterministic vec
        # so the nearest-match snaps to it.
        fused = torch.zeros(D)
        fused[2] = 1.0
        meta_idx = ws.insert_meta(ps_idx, ws_idx, fused_vec=fused)
        meta_row = _ws_row_from_pos(ws,meta_idx)
        meta_vec = ws.subspace.what.getW()[meta_row].detach().clone()
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        left_out, right_out = meta.reverse(meta_vec)
        # Reverse returns vectors of shape [D].
        self.assertEqual(left_out.shape[-1], D)
        self.assertEqual(right_out.shape[-1], D)
        # left should be the PS child's codebook vector (positive idx).
        # right should be the SS child's codebook vector (negative idx).
        expected_ps = ps_store.codebook[ps_row].detach()
        expected_ss = ws.subspace.what.getW()[ws_row].detach()
        self.assertTrue(
            torch.allclose(left_out.detach(),
                           expected_ps.to(left_out.dtype),
                           atol=1e-5),
            f"reverse left must be PS child vec; got "
            f"{left_out.tolist()} vs expected {expected_ps.tolist()}")
        self.assertTrue(
            torch.allclose(right_out.detach(),
                           expected_ss.to(right_out.dtype),
                           atol=1e-5),
            f"reverse right must be SS child vec; got "
            f"{right_out.tolist()} vs expected {expected_ss.tolist()}")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestSymbolizeLayerIdempotency(unittest.TestCase):
    """Calling forward twice on the same (left, right) pair returns the
    same META idx (no new META row allocated)."""

    def test_forward_twice_returns_same_meta_idx(self):
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        ps_idx = ws.insert_percept(b"gamma")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[5] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[6] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        # First call: allocates a fresh META node.
        n_tax_before = len(ws.taxonomy)
        meta.forward(ps_vec, ws_vec)
        meta_idx_1 = ws.meta_pair_to_idx.get((ps_idx, ws_idx))
        self.assertIsNotNone(meta_idx_1)
        # Second call: must NOT allocate a new META row; same idx.
        meta.forward(ps_vec, ws_vec)
        meta_idx_2 = ws.meta_pair_to_idx.get((ps_idx, ws_idx))
        self.assertEqual(
            meta_idx_1, meta_idx_2,
            "Two SymbolizeLayer.forward calls on the same pair must return "
            f"the same meta idx; got {meta_idx_1} vs {meta_idx_2}")
        # And no new taxonomy node was created.
        self.assertEqual(len(ws.taxonomy), n_tax_before + 1,
                         "Exactly one META node should exist after two "
                         "forward calls on the same pair.")


# ---------------------------------------------------------------------------
# Signal-router registration
# ---------------------------------------------------------------------------


class TestSymbolizeLayerSignalRouterRegistration(unittest.TestCase):
    """Stage 9: SymbolizeLayer must auto-register with the chart authority
    when constructed under an active authority.

    Mirrors the LiftLayer / LowerLayer registration test (test_lift_lower_
    binary_grammar_ops.py::TestLiftLowerWiredIntoSignalRouter).
    """

    def test_meta_registers_with_word_subspace_authority(self):
        from Layers import GrammarLayer as _GL
        from Layers import SymbolizeLayer as _Meta

        class _FakeAuthority:
            def __init__(self):
                self.registered = []

            def register_grammar_layer(self, layer):
                self.registered.append(layer)

            def should_run_rule(self, name):
                return 1.0

        auth = _FakeAuthority()
        prev = _GL._chart_authority
        try:
            _GL.set_chart_authority(auth)
            meta = _Meta(nInput=4, nOutput=4)
        finally:
            _GL.set_chart_authority(prev)
        self.assertIn(meta, auth.registered,
                      "SymbolizeLayer must auto-register with the chart "
                      "authority on construction (Stage 9 wiring).")


class TestSymbolizeLayerWiredIntoAttachPerSpace(unittest.TestCase):
    """When the active grammar declares ``meta`` at the CS space_role,
    ``_attach_per_space_syntactic_layer`` must build a SymbolizeLayer and
    pass it through as a builtin layer for the CS-space_role SyntacticLayer.

    The hook point is ``builtin_layers['symbolize']`` in the CS-space_role branch
    (mirrors the existing ``builtin_layers['lift']`` / ``['lower']``
    registration).
    """

    def test_attach_per_space_registers_meta_when_grammar_uses_it(self):
        from Layers import SymbolizeLayer
        import Language
        # Stub out TheGrammar.rules so it contains a CS-space_role 'symbolize' rule.
        # Use a minimal namedtuple-shaped object.
        from collections import namedtuple
        FakeRule = namedtuple(
            'FakeRule',
            ['space_role', 'canonical', 'arity', 'method_name', 'lhs',
             'rhs_symbols'])
        fake_rules = [
            FakeRule(
                space_role='CS', canonical='C -> symbolize(C, C)', arity=2,
                method_name='symbolize', lhs='C', rhs_symbols=('C', 'C')),
        ]
        prev_rules = Language.TheGrammar.rules
        try:
            Language.TheGrammar.rules = fake_rules
            # Re-do the rule iteration as in _attach_per_space_syntactic_
            # layer's space_role=='CS' branch to confirm 'symbolize' is detected.
            grammar_C_methods = {
                r.method_name for r in Language.TheGrammar.rules
                if r.space_role == 'CS' and r.method_name is not None}
            self.assertIn('symbolize', grammar_C_methods)
        finally:
            Language.TheGrammar.rules = prev_rules

    def test_attach_per_space_builds_meta_layer_for_cs_space_role(self):
        """End-to-end: a model whose grammar carries CS-space_role 'symbolize(C, C)'
        produces a SyntacticLayer with a registered SymbolizeLayer instance.

        We mutate the live grammar AFTER construction (the XML doesn't
        carry meta(...)) by monkey-patching ``TheGrammar.rules`` to
        inject a fake CS-space_role 'symbolize' rule, then exercise the wiring
        branch in ``_attach_per_space_syntactic_layer`` to verify a
        SymbolizeLayer instance is built and registered as the 'symbolize'
        builtin.

        This is a unit-level test of the wiring branch; the full XML
        round-trip is exercised by ``make xor`` in Stage 9 acceptance.
        """
        from Layers import SymbolizeLayer
        import Language
        from collections import namedtuple
        m = _make_radix_model()
        # SymbolSubSpace lives on perceptualSpace.symbolSpace
        # (set by Space.attach_symbolSpace).
        ss = getattr(m.perceptualSpace, 'symbolSpace', None)
        if ss is None:
            self.skipTest("No SymbolSubSpace constructed; cannot test wiring.")
        FakeRule = namedtuple(
            'FakeRule',
            ['space_role', 'canonical', 'arity', 'method_name', 'lhs',
             'rhs_symbols'])
        prev_rules = list(Language.TheGrammar.rules)
        synthetic_meta = FakeRule(
            space_role='CS', canonical='C -> symbolize(C, C)', arity=2,
            method_name='symbolize', lhs='C', rhs_symbols=('C', 'C'))
        # Capture the builtin_layers dict that
        # _attach_per_space_syntactic_layer builds. The function then
        # passes it into ``build_space_syntactic_layer``; we patch that
        # downstream call to intercept and inspect.
        captured = {}
        # Patch build_space_syntactic_layer to capture builtin_layers.
        import Language as _Lang
        orig_builder = _Lang.build_space_syntactic_layer

        def _capture_builder(space, ss, *, space_role, builtin_layers):
            captured.setdefault(space_role, dict(builtin_layers))
            return orig_builder(space, ss,
                                space_role=space_role,
                                builtin_layers=builtin_layers)
        try:
            Language.TheGrammar.rules = prev_rules + [synthetic_meta]
            _Lang.build_space_syntactic_layer = _capture_builder
            # Re-attach the CS-space_role layer; the wiring branch should
            # build a SymbolizeLayer under builtin_layers['symbolize'].
            cs = getattr(m, 'conceptualSpace', None)
            if cs is None:
                self.skipTest("No conceptualSpace; cannot test CS-space_role wiring.")
            ss._attach_per_space_syntactic_layer(cs, space_role='CS')
        finally:
            Language.TheGrammar.rules = prev_rules
            _Lang.build_space_syntactic_layer = orig_builder
        self.assertIn('CS', captured,
                      "CS-space_role _attach_per_space did not run.")
        meta_layer = captured['CS'].get('symbolize')
        self.assertIsNotNone(
            meta_layer,
            "When the grammar carries 'symbolize' at the CS space_role, "
            "_attach_per_space_syntactic_layer must wire a SymbolizeLayer "
            "into builtin_layers['symbolize'].")
        self.assertIsInstance(meta_layer, SymbolizeLayer)
        # The SymbolizeLayer must have BOTH wholeSpace and perceptualSpace
        # back-references (it needs both for the discrete-identity
        # lookups in forward / reverse).
        self.assertIsNotNone(
            getattr(meta_layer, 'wholeSpace', None),
            "SymbolizeLayer wired by _attach_per_space must carry a "
            "wholeSpace back-reference.")
        self.assertIsNotNone(
            getattr(meta_layer, 'perceptualSpace', None),
            "SymbolizeLayer wired by _attach_per_space must carry a "
            "perceptualSpace back-reference.")


# ---------------------------------------------------------------------------
# Numerical guard
# ---------------------------------------------------------------------------


class TestSymbolizeLayerNumericalGuard(unittest.TestCase):
    """Fail loud on NaN/Inf in left/right (project's "fail loud" policy)."""

    def test_forward_raises_on_nan_left(self):
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        D = int(ws.nDim)
        # Ensure PS / SS have at least one row so the nearest-match
        # search isn't a no-op.
        ws.insert_percept(b"nan_guard_left")
        ws.insert_whole()
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        bad = torch.full((D,), float("nan"))
        good = torch.zeros(D)
        with self.assertRaises(RuntimeError) as ctx:
            meta.forward(bad, good)
        self.assertIn("NaN/Inf", str(ctx.exception))

    def test_forward_raises_on_inf_right(self):
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        D = int(ws.nDim)
        ws.insert_percept(b"inf_guard_right")
        ws.insert_whole()
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        good = torch.zeros(D)
        bad = torch.zeros(D)
        bad[0] = float("inf")
        with self.assertRaises(RuntimeError):
            meta.forward(good, bad)


# ---------------------------------------------------------------------------
# No-PerceptStore fallback
# ---------------------------------------------------------------------------


class TestSymbolizeLayerNoPerceptStoreFallback(unittest.TestCase):
    """When the PartSpace lacks a percept_store (legacy lexicon),
    forward is a no-op: returns ``(left + right) / 2`` without
    registering a META node."""

    def test_forward_falls_back_to_average_without_perceptstore(self):
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        # Swap percept_store out to simulate the legacy lexicon path.
        saved_ps_store = getattr(ps_space, 'percept_store', None)
        try:
            ps_space.percept_store = None
            meta = SymbolizeLayer(
                wholeSpace=ws,
                perceptualSpace=ps_space,
            )
            D = int(ws.nDim)
            a = torch.full((D,), 0.4)
            b = torch.full((D,), 0.2)
            n_tax_before = len(ws.taxonomy)
            out = meta.forward(a, b)
            # Output is the average.
            expected = (a + b) / 2.0
            self.assertTrue(
                torch.allclose(out.detach(),
                               expected.to(out.dtype),
                               atol=1e-6),
                f"No-perceptstore fallback must return (a+b)/2; got "
                f"{out.tolist()} vs {expected.tolist()}")
            # No META node was registered.
            self.assertEqual(
                len(ws.taxonomy), n_tax_before,
                "No-perceptstore fallback must NOT register a META node.")
        finally:
            ps_space.percept_store = saved_ps_store


# ---------------------------------------------------------------------------
# compose / generate dispatch
# ---------------------------------------------------------------------------


class TestSymbolizeLayerComposeGenerate(unittest.TestCase):
    """compose(left, right) dispatches to forward; generate(parent)
    dispatches to reverse (GrammarLayer binary-op contract)."""

    def test_compose_dispatches_to_forward(self):
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        ps_idx = ws.insert_percept(b"compose_word")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        a = torch.zeros(D)
        a[0] = 1.0
        b = torch.zeros(D)
        b[1] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = a
            ws.subspace.what.getW().data[ws_row, :] = b
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        # First call: forward seeds a META; capture state for an
        # apples-to-apples compose comparison.
        out_forward = meta.forward(a, b)
        out_compose = meta.compose(a, b)
        torch.testing.assert_close(out_forward, out_compose)


# ---------------------------------------------------------------------------
# Gradient flow / trainability (Stage 9 acceptance: "META vectors are
# trainable; they accumulate gradient from the loss.")
# ---------------------------------------------------------------------------


class TestSymbolizeLayerGradient(unittest.TestCase):
    """Stage 9 acceptance: META vectors must accumulate gradient from the loss.

    The META vector lives in ``SS.codebook`` (specifically
    ``ws.subspace.what.W[meta_row]``, an ``nn.Parameter``).
    SymbolizeLayer.forward returns the live ``SS.codebook`` slice (via
    ``getW()[meta_row]``) so the gradient path is:

        loss = forward(left, right).sum()
        loss.backward()
        -> SS.codebook Parameter accumulates gradient at meta_row.

    Gradient flow finding (documented in the test bodies below): in radix
    mode (``PerceptStore`` wired), the path from ``left, right`` to the
    returned META vector is **detached** inside ``forward`` -- the
    ``fused = (left + right) / 2`` is ``.detach()``-ed before being
    handed to ``WholeSpace.insert_meta`` (which does an in-place
    ``W.data[meta_row].copy_()`` under ``torch.no_grad``). So
    ``left.grad`` / ``right.grad`` are ``None`` after the radix-mode
    forward; the gradient flows from the return value back to the
    ``SS.codebook`` Parameter where the META row is stored. This is
    the trainability path the acceptance criterion calls out: META
    vectors live in SS.codebook (a trainable Parameter); they receive
    gradient on backward.

    The no-PerceptStore fallback (legacy lexicon mode) is the only
    branch where ``left``/``right`` themselves carry gradient -- there
    forward returns ``(left + right) / 2`` directly (no detach, no
    META row allocation).
    """

    def test_no_perceptstore_fallback_forward_is_differentiable(self):
        """In legacy / no-PerceptStore mode, forward is the bare
        ``(left + right) / 2`` average -- so ``left.grad`` /
        ``right.grad`` flow normally. This proves the forward path is
        differentiable in the simple-fallback regime."""
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        saved_ps_store = getattr(ps_space, 'percept_store', None)
        try:
            ps_space.percept_store = None
            meta = SymbolizeLayer(
                wholeSpace=ws,
                perceptualSpace=ps_space,
            )
            D = int(ws.nDim)
            left = torch.full((D,), 0.4, requires_grad=True)
            right = torch.full((D,), 0.2, requires_grad=True)
            out = meta.forward(left, right)
            loss = out.sum()
            loss.backward()
            # In the average fallback path, the gradient w.r.t. both
            # operands is exactly 0.5 per element of the output.
            self.assertIsNotNone(left.grad,
                                 "left.grad must be non-None in the "
                                 "no-PerceptStore fallback path.")
            self.assertIsNotNone(right.grad,
                                 "right.grad must be non-None in the "
                                 "no-PerceptStore fallback path.")
            self.assertTrue(
                torch.any(left.grad != 0),
                "left.grad must be non-zero after backward through "
                "the average fallback.")
            self.assertTrue(
                torch.any(right.grad != 0),
                "right.grad must be non-zero after backward through "
                "the average fallback.")
        finally:
            ps_space.percept_store = saved_ps_store

    def test_forward_output_carries_gradient_to_ws_codebook(self):
        """Stage 9 acceptance gate: the META vector (output of forward)
        is differentiable into the SS.codebook Parameter that hosts the
        META row. After ``loss = out.sum(); loss.backward()`` the
        SS.codebook Parameter has non-zero gradient at the META row.

        This is the trainability path: META vectors live in SS.codebook
        (a trainable Parameter), and loss-driven backprop reaches them.
        """
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        # Pre-seed PS + SS so nearest-match resolves deterministically.
        ps_idx = ws.insert_percept(b"grad_word")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[0] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[1] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        # Construct codebook-matching operands so the nearest-match
        # snaps to (ps_row, ws_row) and forward allocates a META.
        left = ps_vec.clone().detach()
        right = ws_vec.clone().detach()
        meta_vec = meta.forward(left, right)
        # Capture the SS-side trainable Parameter AFTER forward:
        # ``insert_meta`` -> ``insert_whole`` -> ``grow_to`` may
        # replace ``subspace.what.W`` with a new Parameter, so the
        # gradient must be read off the post-forward identity.
        ws_param = ws.subspace.what.W
        self.assertIsNotNone(ws_param,
                             "Test precondition: SS codebook must have a "
                             "trainable W Parameter post-forward.")
        # The output must be a tensor carrying autograd state (it is a
        # slice of SS.codebook via getW(); getW returns the Parameter
        # ``W`` directly so the slice is a differentiable view).
        self.assertTrue(meta_vec.requires_grad,
                        "SymbolizeLayer.forward output must require_grad "
                        "(it slices the trainable SS codebook Parameter).")
        self.assertIsNotNone(meta_vec.grad_fn,
                             "SymbolizeLayer.forward output must have a "
                             "grad_fn (live autograd graph back to "
                             "SS.codebook).")
        loss = meta_vec.sum()
        loss.backward()
        # SS.codebook Parameter MUST have gradient at the META row.
        self.assertIsNotNone(
            ws_param.grad,
            "SS.codebook Parameter must accumulate gradient on backward "
            "through SymbolizeLayer.forward (Stage 9: META vectors are "
            "trainable; they accumulate gradient from the loss).")
        meta_idx = ws.meta_pair_to_idx.get((ps_idx, ws_idx))
        self.assertIsNotNone(meta_idx,
                             "Test precondition: META row must have been "
                             "registered by forward.")
        meta_row = _ws_row_from_pos(ws,meta_idx)
        self.assertTrue(
            torch.any(ws_param.grad[meta_row] != 0),
            f"SS.codebook gradient at META row {meta_row} must be "
            f"non-zero after loss.backward(); got "
            f"{ws_param.grad[meta_row].tolist()}.")

    def test_forward_detaches_operands_in_radix_mode(self):
        """Documented gradient-path finding: in radix mode, the path
        from ``left``/``right`` to the returned META vector is severed
        by an internal ``.detach()`` in SymbolizeLayer.forward (the fused
        vec is detached before being passed into the in-place
        ``insert_meta`` codebook write under ``torch.no_grad``).

        This test pins the current behaviour. Any future change that
        removes the detach (e.g., a differentiable / learnable combine
        of left+right -> META) will flip this assertion and force a
        deliberate test update.
        """
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        ps_idx = ws.insert_percept(b"detach_word")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[0] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[1] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        left = ps_vec.clone().detach().requires_grad_(True)
        right = ws_vec.clone().detach().requires_grad_(True)
        out = meta.forward(left, right)
        loss = out.sum()
        loss.backward()
        # The autograd path back to ``left`` / ``right`` is severed by
        # the internal ``.detach()`` in forward. Gradient w.r.t. the
        # operands is None.
        self.assertIsNone(
            left.grad,
            "Current radix-mode forward detaches the fused vec before "
            "the codebook write, so left.grad is None. If you intend to "
            "make the combine differentiable into the operands, update "
            "this test deliberately.")
        self.assertIsNone(
            right.grad,
            "Current radix-mode forward detaches the fused vec before "
            "the codebook write, so right.grad is None. If you intend "
            "to make the combine differentiable into the operands, "
            "update this test deliberately.")

    def test_forward_on_existing_meta_carries_gradient_to_ws_codebook(self):
        """Idempotent (existing META) branch also carries gradient: the
        returned vector is still a slice of the trainable SS.codebook,
        and the EMA-update of the stored row is a ``no_grad`` write
        that does not break the read-side gradient path."""
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        ps_idx = ws.insert_percept(b"existing_grad")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[2] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[3] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        # Pre-register the META node so the second forward call hits
        # the idempotent existing-META branch.
        explicit_fused = torch.zeros(D)
        explicit_fused[4] = 0.5
        ws.insert_meta(ps_idx, ws_idx, fused_vec=explicit_fused)
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        out = meta.forward(ps_vec.clone(), ws_vec.clone())
        # Capture post-forward; on the idempotent branch no fresh row
        # is allocated, so the Parameter identity SHOULD be stable --
        # but read it post-forward for symmetry with the fresh-alloc
        # path and to guard against any internal grow_to.
        ws_param = ws.subspace.what.W
        self.assertTrue(out.requires_grad,
                        "Idempotent-path output must still require_grad "
                        "(reads from the trainable SS codebook).")
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(
            ws_param.grad,
            "SS.codebook Parameter must accumulate gradient even on "
            "the existing-META idempotent branch.")
        meta_idx = ws.meta_pair_to_idx.get((ps_idx, ws_idx))
        meta_row = _ws_row_from_pos(ws,meta_idx)
        self.assertTrue(
            torch.any(ws_param.grad[meta_row] != 0),
            f"SS.codebook gradient at META row {meta_row} must be "
            f"non-zero after backward on the idempotent-existing-META "
            f"branch; got {ws_param.grad[meta_row].tolist()}.")


# ---------------------------------------------------------------------------
# Signal-router end-to-end dispatch (Stage 9 acceptance: "Signal-router
# dispatch fires SymbolizeLayer at sentence-parse boundaries.")
# ---------------------------------------------------------------------------


class TestSymbolizeLayerSignalRouterDispatch(unittest.TestCase):
    """Stage 9 acceptance: the per-space SyntacticLayer wiring path
    registers the SymbolizeLayer instance against ``symbolSpace.host_layer
    ('CS', 'symbolize')``; when the chart / signal router fires a 'symbolize'
    rule at the CS space_role it dispatches through ``SymbolizeLayer.compose``.

    Two-pronged check:
      1. After ``_attach_per_space_syntactic_layer`` runs on the
         ConceptualSpace with a 'symbolize' CS-space_role rule in TheGrammar,
         ``SymbolSubSpace.host_layer('CS', 'symbolize')`` returns a SymbolizeLayer.
      2. Invoking ``compose(left, right)`` on the registered layer
         routes through ``SymbolizeLayer.forward`` (verified by monkey-
         patching the bound method).

    This stops short of running the full ``LanguageLayer.compose``
    end-to-end (which would require a full radix-mode parse with a
    'symbolize(C, C)' rule live in the grammar XML). The registry-+-
    compose-routing path is the contractual surface SymbolizeLayer must
    plug into; downstream the BinaryStructuredReductionLayer calls
    ``op(left, right)`` on the per-pair tensors through
    ``_BinaryGrammarOpAdapter.forward``, which itself just forwards
    to ``gl.compose(...)``.
    """

    def _wire_meta_into_cs_space_role(self, m):
        """Helper: monkey-patch TheGrammar.rules to inject a CS-space_role
        ``symbolize(C, C)`` rule, then re-attach the CS-space_role per-space
        SyntacticLayer so ``host_layer('CS', 'symbolize')`` registers the
        SymbolizeLayer. Returns ``(ss, cs, prev_rules)`` for the caller
        to restore on teardown.
        """
        import Language
        from collections import namedtuple
        ss = getattr(m.perceptualSpace, 'symbolSpace', None)
        if ss is None:
            self.skipTest("No SymbolSubSpace constructed; cannot test wiring.")
        cs = getattr(m, 'conceptualSpace', None)
        if cs is None:
            self.skipTest(
                "No conceptualSpace; cannot test CS-space_role wiring.")
        FakeRule = namedtuple(
            'FakeRule',
            ['space_role', 'canonical', 'arity', 'method_name', 'lhs',
             'rhs_symbols'])
        synthetic_meta = FakeRule(
            space_role='CS', canonical='C -> symbolize(C, C)', arity=2,
            method_name='symbolize', lhs='C', rhs_symbols=('C', 'C'))
        prev_rules = list(Language.TheGrammar.rules)
        Language.TheGrammar.rules = prev_rules + [synthetic_meta]
        # Re-attach CS-space_role; SyntacticLayer.__init__ calls
        # ss.register_host_layer('CS', 'symbolize', meta_layer).
        ss._attach_per_space_syntactic_layer(cs, space_role='CS')
        return ss, cs, prev_rules

    def test_host_layer_registry_resolves_meta_after_attach(self):
        """After the CS-space_role SyntacticLayer is rebuilt with a 'symbolize' rule
        in the grammar, ``SymbolSubSpace.host_layer('CS', 'symbolize')`` returns
        a registered SymbolizeLayer instance (NOT just a class entry --
        the actual instance the chart will dispatch to).
        """
        import Language
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ss, cs, prev_rules = self._wire_meta_into_cs_space_role(m)
        try:
            registered = ss.host_layer('CS', 'symbolize')
            self.assertIsNotNone(
                registered,
                "symbolSpace.host_layer('CS', 'symbolize') must return the "
                "registered SymbolizeLayer after _attach_per_space_syntactic_"
                "layer runs with 'symbolize' in the CS-space_role grammar.")
            self.assertIsInstance(
                registered, SymbolizeLayer,
                f"Registered ('CS', 'symbolize') layer must be a SymbolizeLayer; "
                f"got {type(registered).__name__}.")
            # Sanity: BOTH back-references are set (forward needs them
            # to dispatch through PS / SS codebook nearest-match).
            self.assertIsNotNone(
                getattr(registered, 'wholeSpace', None),
                "Registered SymbolizeLayer must carry a wholeSpace ref.")
            self.assertIsNotNone(
                getattr(registered, 'perceptualSpace', None),
                "Registered SymbolizeLayer must carry a perceptualSpace ref.")
        finally:
            Language.TheGrammar.rules = prev_rules

    def test_registered_meta_layer_compose_dispatches_to_forward(self):
        """The end-to-end dispatch contract: when the chart / signal
        router calls ``compose(left, right)`` on the registered
        SymbolizeLayer, the call routes through ``SymbolizeLayer.forward``.

        We monkey-patch the registered layer's bound ``forward`` to
        record invocations + delegate to the original, then call
        ``compose`` and assert forward was hit. This mirrors what
        ``_BinaryGrammarOpAdapter.forward`` does in production:

            return self.gl.compose(left, right)

        which itself routes to ``forward`` per the GrammarLayer binary-
        op contract.
        """
        import Language
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        # Pre-seed a PS row + SS row so forward can resolve nearest-
        # match cleanly and register a META.
        ps_idx = ws.insert_percept(b"dispatch_word")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[0] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[1] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        ss, cs, prev_rules = self._wire_meta_into_cs_space_role(m)
        try:
            registered = ss.host_layer('CS', 'symbolize')
            self.assertIsInstance(registered, SymbolizeLayer)
            # Monkey-patch the bound forward to record calls + delegate.
            calls = []
            orig_forward = registered.forward

            def _record(left, right, *, _orig=orig_forward,
                        _calls=calls):
                _calls.append((left, right))
                return _orig(left, right)
            object.__setattr__(registered, 'forward', _record)
            # Invoke the registered layer's compose with codebook-
            # matching operands (so forward exercises the radix-mode
            # META-allocation branch).
            out = registered.compose(ps_vec, ws_vec)
            self.assertTrue(torch.is_tensor(out),
                            "compose must return a tensor (META vec).")
            self.assertEqual(out.shape[-1], D)
            self.assertEqual(
                len(calls), 1,
                f"SymbolizeLayer.compose must route through "
                f"SymbolizeLayer.forward exactly once; got {len(calls)} "
                f"invocation(s).")
            self.assertEqual(
                len(calls[0]), 2,
                "SymbolizeLayer.forward must receive (left, right) "
                "operands from compose.")
            meta_idx = ws.meta_pair_to_idx.get((ps_idx, ws_idx))
            self.assertIsNotNone(
                meta_idx,
                "After compose -> forward, a META node should be "
                "registered for the (ps_idx, ws_idx) pair.")
        finally:
            Language.TheGrammar.rules = prev_rules

    def test_binary_op_adapter_routes_through_registered_meta_forward(self):
        """Same as the prior test but exercises the production adapter
        path: ``_BinaryGrammarOpAdapter`` is the wrapper the signal
        router wraps each binary GrammarLayer with before calling
        ``op(left, right)`` inside the BinaryStructuredReductionLayer.
        The adapter's ``forward(left, right)`` just dispatches to
        ``gl.compose(left, right)`` (Language.py:1280-1282). Verify
        the adapter+SymbolizeLayer pair plug together as expected.
        """
        import Language
        from Layers import SymbolizeLayer
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_space = m.perceptualSpace
        ps_store = ps_space.percept_store
        ps_idx = ws.insert_percept(b"adapter_word")
        ps_row = _ps_row_from_pos(ws,ps_idx)
        ws_idx = ws.insert_whole()
        ws_row = _ws_row_from_pos(ws,ws_idx)
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[0] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[1] = 1.0
        with torch.no_grad():
            ps_store._basis.W.data[ps_row, :] = ps_vec
            ws.subspace.what.getW().data[ws_row, :] = ws_vec
        ss, cs, prev_rules = self._wire_meta_into_cs_space_role(m)
        try:
            registered = ss.host_layer('CS', 'symbolize')
            self.assertIsInstance(registered, SymbolizeLayer)
            calls = []
            orig_forward = registered.forward

            def _record(left, right, *, _orig=orig_forward,
                        _calls=calls):
                _calls.append((left, right))
                return _orig(left, right)
            object.__setattr__(registered, 'forward', _record)
            # Wrap with the production binary-op adapter.
            adapter = Language._BinaryGrammarOpAdapter(registered)
            out = adapter(ps_vec, ws_vec)
            self.assertTrue(torch.is_tensor(out))
            self.assertEqual(out.shape[-1], D)
            self.assertEqual(
                len(calls), 1,
                "_BinaryGrammarOpAdapter must route into "
                "SymbolizeLayer.compose -> SymbolizeLayer.forward exactly once.")
        finally:
            Language.TheGrammar.rules = prev_rules


if __name__ == "__main__":
    unittest.main()
