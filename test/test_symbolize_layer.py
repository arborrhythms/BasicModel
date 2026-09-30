"""Pure SymbolizeLayer composition, tied reverse, dispatch and numerical guards.

Native concept admission is covered by the item-7 taxonomy tests.
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




# ---------------------------------------------------------------------------
# Reverse path
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------




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

        def _capture_builder(space, ss, *, space_role, builtin_layers,
                             owner_space=None):
            captured.setdefault(space_role, dict(builtin_layers))
            return orig_builder(space, ss,
                                space_role=space_role,
                                builtin_layers=builtin_layers,
                                owner_space=owner_space)
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
        # The pure numerical operator checks inputs before its arithmetic.
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
        D = int(ws.nDim)
        a = torch.zeros(D)
        a[0] = 1.0
        b = torch.zeros(D)
        b[1] = 1.0
        meta = SymbolizeLayer(
            wholeSpace=ws,
            perceptualSpace=ps_space,
        )
        # Compare direct numerical execution and the declared compose face.
        out_forward = meta.forward(a, b)
        out_compose = meta.compose(a, b)
        torch.testing.assert_close(out_forward, out_compose)


# ---------------------------------------------------------------------------
# Gradient flow / trainability (Stage 9 acceptance: "META vectors are
# trainable; they accumulate gradient from the loss.")
# ---------------------------------------------------------------------------




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
    plug into; downstream the OperationSelectionLayer calls
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
        # The registered operator consumes these two actual points.
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[0] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[1] = 1.0
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
            # Invoke the registered layer with the same two operand values.
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
            torch.testing.assert_close(out, (ps_vec + ws_vec) / 2)
            self.assertFalse(hasattr(ws, 'taxonomy'))
        finally:
            Language.TheGrammar.rules = prev_rules

    def test_binary_op_adapter_routes_through_registered_meta_forward(self):
        """Same as the prior test but exercises the production adapter
        path: ``_BinaryGrammarOpAdapter`` is the wrapper the signal
        router wraps each binary GrammarLayer with before calling
        ``op(left, right)`` inside the OperationSelectionLayer.
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
        D = int(ws.nDim)
        ps_vec = torch.zeros(D)
        ps_vec[0] = 1.0
        ws_vec = torch.zeros(D)
        ws_vec[1] = 1.0
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


class TestPureSymbolize(unittest.TestCase):
    def test_composition_retains_both_operand_gradients_without_a_memory_write(self):
        from Layers import SymbolizeLayer
        model = _make_radix_model()
        layer = SymbolizeLayer(wholeSpace=model.wholeSpace,
                               perceptualSpace=model.perceptualSpace)
        left, right = torch.randn(8, requires_grad=True), torch.randn(8, requires_grad=True)
        before = {k: v.clone() for k, v in model.wholeSpace.state_dict().items()}
        result = layer.compose(left, right)
        result.sum().backward()
        torch.testing.assert_close(left.grad, torch.full_like(left, .5))
        torch.testing.assert_close(right.grad, torch.full_like(right, .5))
        for key, value in before.items():
            torch.testing.assert_close(model.wholeSpace.state_dict()[key], value)
        assert not hasattr(model.wholeSpace, 'taxonomy')

    def test_reverse_is_pure_with_attached_stores(self):
        from Layers import SymbolizeLayer
        layer = SymbolizeLayer(nInput=8)
        point = torch.randn(8)
        left, right = layer.generate(point)
        torch.testing.assert_close(left + right, point)
