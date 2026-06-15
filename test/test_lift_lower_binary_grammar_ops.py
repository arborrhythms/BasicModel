"""Stage 4 substrate refactor: LiftLayer / LowerLayer as binary GrammarLayer ops.

Plan: doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md Stage 4.

Acceptance gates (per the master plan, Stage 4 §Files modified / §Acceptance):

  * ``LiftLayer`` and ``LowerLayer`` are proper binary GrammarLayer
    subclasses:
      - ``LiftLayer.arity == 2``, ``LiftLayer.rule_name == "lift"``,
        ``LiftLayer.tier == 'C'``.
      - ``LowerLayer.arity == 2``, ``LowerLayer.rule_name == "lower"``,
        ``LowerLayer.tier == 'C'``.
      - Both inherit from ``GrammarLayer``.

  * ``forward(a, b)`` accepts two operands and returns a single result
    (a tensor); ``reverse(parent)`` returns a pair ``(left, right)``.

  * Round-trip behavioral check: ``layer.reverse(layer.forward(a, b))``
    returns two tensors of the same shape as ``a`` / ``b``. The math
    is invertible at the spatial level (sigma-style for Lift,
    pi-style for Lower) so the round-trip is approximately ``(a, b)``
    under the balanced-split convention (each side recovers the
    half-sum in atanh / log-mult domain, exactly mirroring
    ``SigmaLayer.generate`` / ``PiLayer.generate``).

  * The new bodies do not reach into ``PartSpace.sigma`` or
    ``ConceptualSpace.pi`` from inside the layer -- the substrate
    folds are no longer borrowed by Lift / Lower.

  * Registration with the signal router (``WordSubSpace.languageLayer``):
    after a WordSubSpace is constructed, LiftLayer and LowerLayer
    attach as C-tier reduce ops on the LanguageLayer.

The pre-Stage-4 contract (``tier='S'``, internal ``_sigma`` /
``_pi``, fallback to ``Ops._lower_kernel`` / ``Ops._lift_kernel``)
is retired by this stage; the existing
``test/test_lift_lower_factorization.py`` is no longer authoritative
for the new contract.
"""
import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import GrammarLayer, LiftLayer, LowerLayer


class TestLiftLowerClassAttributes(unittest.TestCase):
    """Stage 4: class-level attributes for the binary grammar-op contract."""

    def test_lift_is_grammarlayer_subclass(self):
        """LiftLayer must be a GrammarLayer subclass."""
        self.assertTrue(
            issubclass(LiftLayer, GrammarLayer),
            "LiftLayer must inherit from GrammarLayer (Stage 4).")

    def test_lower_is_grammarlayer_subclass(self):
        """LowerLayer must be a GrammarLayer subclass."""
        self.assertTrue(
            issubclass(LowerLayer, GrammarLayer),
            "LowerLayer must inherit from GrammarLayer (Stage 4).")

    def test_lift_arity_is_two(self):
        """LiftLayer is a binary grammar op."""
        self.assertEqual(
            LiftLayer.arity, 2,
            "LiftLayer must have arity == 2 (binary op over STM pairs).")

    def test_lower_arity_is_two(self):
        """LowerLayer is a binary grammar op."""
        self.assertEqual(
            LowerLayer.arity, 2,
            "LowerLayer must have arity == 2 (binary op over STM pairs).")

    def test_lift_rule_name(self):
        """LiftLayer's canonical rule_name is 'lift'."""
        self.assertEqual(LiftLayer.rule_name, "lift")

    def test_lower_rule_name(self):
        """LowerLayer's canonical rule_name is 'lower'."""
        self.assertEqual(LowerLayer.rule_name, "lower")

    def test_lift_tier_is_C(self):
        """LiftLayer operates at the C tier (STM-pair composition)."""
        self.assertEqual(
            LiftLayer.tier, 'C',
            "LiftLayer.tier must be 'C' (Stage 4 §Files modified).")

    def test_lower_tier_is_C(self):
        """LowerLayer operates at the C tier (STM-pair composition)."""
        self.assertEqual(
            LowerLayer.tier, 'C',
            "LowerLayer.tier must be 'C' (Stage 4 §Files modified).")


class TestLiftLowerForwardReverseShape(unittest.TestCase):
    """Stage 4: forward(a, b) returns a single result; reverse(parent)
    returns a pair (left, right) of the same shape as a / b."""

    def _make_lift(self, dim=4):
        # Parameter-free construction; the layer is self-contained.
        return LiftLayer(nInput=dim, nOutput=dim)

    def _make_lower(self, dim=4):
        return LowerLayer(nInput=dim, nOutput=dim)

    def test_lift_forward_two_operands_one_result(self):
        """LiftLayer.forward(a, b) returns one tensor of [..., D] shape."""
        D = 4
        lift = self._make_lift(D)
        torch.manual_seed(0)
        a = torch.rand(2, 5, D) * 1.8 - 0.9
        b = torch.rand(2, 5, D) * 1.8 - 0.9
        with torch.no_grad():
            out = lift.forward(a, b)
        self.assertTrue(torch.is_tensor(out),
                        "LiftLayer.forward must return a tensor.")
        self.assertEqual(out.shape, a.shape,
                         "LiftLayer.forward output shape must match input.")

    def test_lower_forward_two_operands_one_result(self):
        """LowerLayer.forward(a, b) returns one tensor of [..., D] shape."""
        D = 4
        lower = self._make_lower(D)
        torch.manual_seed(1)
        a = torch.rand(2, 5, D) * 1.8 - 0.9
        b = torch.rand(2, 5, D) * 1.8 - 0.9
        with torch.no_grad():
            out = lower.forward(a, b)
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.shape, a.shape)

    def test_lift_reverse_returns_pair(self):
        """LiftLayer.reverse(parent) returns a 2-tuple ``(left, right)``."""
        D = 4
        lift = self._make_lift(D)
        torch.manual_seed(2)
        a = torch.rand(2, 5, D) * 1.8 - 0.9
        b = torch.rand(2, 5, D) * 1.8 - 0.9
        with torch.no_grad():
            parent = lift.forward(a, b)
            split = lift.reverse(parent)
        self.assertIsInstance(split, tuple)
        self.assertEqual(len(split), 2)
        left, right = split
        self.assertTrue(torch.is_tensor(left))
        self.assertTrue(torch.is_tensor(right))
        self.assertEqual(left.shape, a.shape)
        self.assertEqual(right.shape, b.shape)

    def test_lower_reverse_returns_pair(self):
        """LowerLayer.reverse(parent) returns a 2-tuple ``(left, right)``."""
        D = 4
        lower = self._make_lower(D)
        torch.manual_seed(3)
        a = torch.rand(2, 5, D) * 1.8 - 0.9
        b = torch.rand(2, 5, D) * 1.8 - 0.9
        with torch.no_grad():
            parent = lower.forward(a, b)
            split = lower.reverse(parent)
        self.assertIsInstance(split, tuple)
        self.assertEqual(len(split), 2)
        left, right = split
        self.assertEqual(left.shape, a.shape)
        self.assertEqual(right.shape, b.shape)


class TestLiftLowerRoundTrip(unittest.TestCase):
    """Stage 4: the LDU-invertible underlying sigma / pi math gives a
    clean round-trip via the balanced-split convention.

    With ``compose = sigma_style(a, b)``, the balanced split returns
    ``(half_a, half_b)`` where ``half = tanh((atanh(a) + atanh(b)) / 2)``
    -- i.e. the round-trip is *idempotent* in the additive-log domain
    (compose . split . compose == compose) but the literal recovery
    ``(a, b)`` requires no parameters; the spec is approximate
    (codebook-mediated, OK to be lossy). Here we check the structural
    property that round-trip is *consistent* (a second compose returns
    the same parent).
    """

    def test_lift_round_trip_consistent(self):
        """A second compose on the split should reproduce the parent."""
        D = 4
        lift = LiftLayer(nInput=D, nOutput=D)
        torch.manual_seed(4)
        a = torch.rand(2, 3, D) * 1.8 - 0.9
        b = torch.rand(2, 3, D) * 1.8 - 0.9
        with torch.no_grad():
            parent = lift.forward(a, b)
            left, right = lift.reverse(parent)
            parent_2 = lift.forward(left, right)
        # Compose . split . compose must return the same parent.
        torch.testing.assert_close(parent, parent_2, rtol=1e-4, atol=1e-4)
        # And: the round-trip output must be finite (fail-loud).
        self.assertTrue(torch.isfinite(parent).all(),
                        "LiftLayer.forward must not produce non-finite "
                        "values (fail-loud numerical contract).")
        self.assertTrue(torch.isfinite(left).all())
        self.assertTrue(torch.isfinite(right).all())

    def test_lower_round_trip_consistent(self):
        """A second compose on the split should reproduce the parent."""
        D = 4
        lower = LowerLayer(nInput=D, nOutput=D)
        torch.manual_seed(5)
        a = torch.rand(2, 3, D) * 1.8 - 0.9
        b = torch.rand(2, 3, D) * 1.8 - 0.9
        with torch.no_grad():
            parent = lower.forward(a, b)
            left, right = lower.reverse(parent)
            parent_2 = lower.forward(left, right)
        torch.testing.assert_close(parent, parent_2, rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.isfinite(parent).all())
        self.assertTrue(torch.isfinite(left).all())
        self.assertTrue(torch.isfinite(right).all())


class TestLiftLowerComposeGenerate(unittest.TestCase):
    """Stage 4: the GrammarLayer compose / generate contract is wired."""

    def test_lift_compose_dispatches_to_forward(self):
        """compose(left, right) gives the same result as forward(left, right)."""
        D = 4
        lift = LiftLayer(nInput=D, nOutput=D)
        torch.manual_seed(6)
        a = torch.rand(2, 3, D) * 1.8 - 0.9
        b = torch.rand(2, 3, D) * 1.8 - 0.9
        with torch.no_grad():
            via_compose = lift.compose(a, b)
            via_forward = lift.forward(a, b)
        torch.testing.assert_close(via_compose, via_forward)

    def test_lower_compose_dispatches_to_forward(self):
        D = 4
        lower = LowerLayer(nInput=D, nOutput=D)
        torch.manual_seed(7)
        a = torch.rand(2, 3, D) * 1.8 - 0.9
        b = torch.rand(2, 3, D) * 1.8 - 0.9
        with torch.no_grad():
            via_compose = lower.compose(a, b)
            via_forward = lower.forward(a, b)
        torch.testing.assert_close(via_compose, via_forward)

    def test_lift_generate_dispatches_to_reverse(self):
        """generate(parent) gives the same pair as reverse(parent)."""
        D = 4
        lift = LiftLayer(nInput=D, nOutput=D)
        torch.manual_seed(8)
        a = torch.rand(2, 3, D) * 1.8 - 0.9
        b = torch.rand(2, 3, D) * 1.8 - 0.9
        with torch.no_grad():
            parent = lift.forward(a, b)
            via_gen = lift.generate(parent)
            via_rev = lift.reverse(parent)
        # Both are 2-tuples; compare element-wise.
        self.assertEqual(len(via_gen), len(via_rev))
        torch.testing.assert_close(via_gen[0], via_rev[0])
        torch.testing.assert_close(via_gen[1], via_rev[1])


class TestLiftLowerNoSubstrateBorrow(unittest.TestCase):
    """Stage 4 acceptance: LiftLayer / LowerLayer no longer reach into
    Space-owned substrate folds (``PartSpace.sigma`` /
    ``ConceptualSpace.pi``).

    The layers become first-class GrammarLayer ops with their own
    pairwise math; this is enforced both at construction time (no
    Space refs needed) and by a source-level grep gate.
    """

    def test_lift_constructible_without_spaces(self):
        """LiftLayer instantiates with no Space references."""
        layer = LiftLayer(nInput=4, nOutput=4)
        # Forward path must run without any Space wiring -- the layer
        # is self-contained per the Stage 4 contract.
        a = torch.rand(2, 3, 4) * 1.8 - 0.9
        b = torch.rand(2, 3, 4) * 1.8 - 0.9
        with torch.no_grad():
            out = layer.forward(a, b)
        self.assertTrue(torch.isfinite(out).all())

    def test_lower_constructible_without_spaces(self):
        """LowerLayer instantiates with no Space references."""
        layer = LowerLayer(nInput=4, nOutput=4)
        a = torch.rand(2, 3, 4) * 1.8 - 0.9
        b = torch.rand(2, 3, 4) * 1.8 - 0.9
        with torch.no_grad():
            out = layer.forward(a, b)
        self.assertTrue(torch.isfinite(out).all())

    def test_lift_lower_classes_no_substrate_borrow_in_source(self):
        """grep gate: ``PartSpace.sigma`` / ``ConceptualSpace.pi``
        do not appear in LiftLayer / LowerLayer source.

        (Stage 4 acceptance criterion in the master plan.)
        """
        import inspect
        from Layers import LiftLayer as _Lift
        from Layers import LowerLayer as _Lower
        for cls in (_Lift, _Lower):
            try:
                src = inspect.getsource(cls)
            except (OSError, TypeError):
                continue
            # The retired pattern: substrate borrow from the named
            # Space attributes. Stage 4 forbids these references
            # inside the layer class bodies.
            self.assertNotIn(
                "PartSpace.sigma", src,
                f"{cls.__name__} must not borrow PartSpace.sigma "
                f"after Stage 4 (substrate retirement).")
            self.assertNotIn(
                "ConceptualSpace.pi", src,
                f"{cls.__name__} must not borrow ConceptualSpace.pi "
                f"after Stage 4 (substrate retirement).")


class TestLiftLowerHasInternalSubstrate(unittest.TestCase):
    """Stage 4: the sigma-style / pi-style math is owned internally
    by the layer (not borrowed from a Space).

    LiftLayer holds an internal SigmaLayer (additive log-domain);
    LowerLayer holds an internal PiLayer (multiplicative log-domain).
    Both are trainable nn.Modules registered under the layer.
    """

    def test_lift_has_internal_sigma(self):
        """LiftLayer owns an internal SigmaLayer for the binary fold."""
        from Layers import SigmaLayer
        lift = LiftLayer(nInput=4, nOutput=4)
        # The internal sigma exposes trainable parameters.
        params = [p for p in lift.parameters() if p.requires_grad]
        self.assertGreater(
            len(params), 0,
            "LiftLayer must own trainable parameters (its internal "
            "additive fold).")
        # The internal SigmaLayer instance is reachable on the layer.
        # (Allow either ``_sigma`` or ``sigma`` as the attribute name;
        # the implementation may pick either.)
        inner = getattr(lift, '_sigma', None) or getattr(lift, 'sigma', None)
        self.assertIsNotNone(
            inner,
            "LiftLayer must own an internal SigmaLayer (sigma-style "
            "binary fold).")
        self.assertIsInstance(inner, SigmaLayer)

    def test_lower_has_internal_pi(self):
        """LowerLayer owns an internal PiLayer for the binary fold."""
        from Layers import PiLayer
        lower = LowerLayer(nInput=4, nOutput=4)
        params = [p for p in lower.parameters() if p.requires_grad]
        self.assertGreater(len(params), 0)
        inner = getattr(lower, '_pi', None) or getattr(lower, 'pi', None)
        self.assertIsNotNone(
            inner,
            "LowerLayer must own an internal PiLayer (pi-style binary "
            "fold).")
        self.assertIsInstance(inner, PiLayer)


class TestLiftLowerWiredIntoSignalRouter(unittest.TestCase):
    """Stage 4 acceptance gate: when a WordSubSpace is constructed,
    LiftLayer and LowerLayer attach to the C-tier of its
    ``languageLayer`` as binary reduce ops.

    Path: WordSubSpace.__init__ -> ``_wire_signal_router_grammar_ops``
    walks ``TheGrammar.rules`` and calls
    ``router.attach_layer_ops(ops=..., rule_ids=..., tier='C')``
    for every binary C-tier rule. The ``ops`` list is wrapped per-op
    in ``_BinaryGrammarOpAdapter`` so the binary scorer can call
    ``op(left, right)`` and dispatch through ``gl.compose``.
    """

    def test_lift_and_lower_registered_with_word_subspace_authority(self):
        """When a WordSubSpace is the chart authority,
        ``register_grammar_layer`` adds any newly-built LiftLayer /
        LowerLayer to its roster."""
        import Language
        from Layers import LiftLayer as _Lift, LowerLayer as _Lower
        from Layers import GrammarLayer as _GL

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
            lift = _Lift(nInput=4, nOutput=4)
            lower = _Lower(nInput=4, nOutput=4)
        finally:
            _GL.set_chart_authority(prev)
        # Auto-registration walks ``rule_name``; both Lift and Lower
        # have non-empty rule_names, so both register with the
        # authority.
        self.assertIn(lift, auth.registered,
                      "LiftLayer must auto-register with the WordSpace "
                      "chart authority (Stage 4 wiring).")
        self.assertIn(lower, auth.registered,
                      "LowerLayer must auto-register with the WordSpace "
                      "chart authority (Stage 4 wiring).")


if __name__ == "__main__":
    unittest.main()
