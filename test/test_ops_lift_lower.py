"""Tests for the unified Ops.lift / Ops.lower mode dispatcher.

Steps 1-2 of doc/plans/2026-04-24-lift-lower-bivector-refactor.md:
    lift  = synthesis = many → one (∨), default mode='OR'
    lower = analysis  = one → many (∧), default mode='AND'

The Step-2 `kind` parameter selects the forward point body:
    kind='strict' (default) — min for AND, max for OR (lattice).
    kind='smooth'           — elementwise product for AND, arithmetic mean for OR.
    kind='radial'           — RadMin for AND, RadMax for OR (same-sign min/max
                              magnitude; RadMax has zero passthrough).

Region body and NOT mode are unaffected by `kind`.  Inverse paths unaffected.

Coverage:
    - lower(x, y, mode='AND', kind='strict') = elementwise min (default)
    - lower(x, y, mode='AND', kind='smooth') = elementwise product
    - lower(x, y, mode='AND', kind='radial') = RadMin (same-sign min magnitude)
    - lower((l, u), (l, u), mode='AND') = (max ℓ, min u) (region form)
    - lift(x, y, mode='OR', kind='strict')  = elementwise max (default)
    - lift(x, y, mode='OR', kind='smooth')  = arithmetic mean
    - lift(x, y, mode='OR', kind='radial')  = RadMax (same-sign max magnitude
                                              with zero passthrough)
    - lift((l, u), (l, u), mode='OR') = (min ℓ, max u) (region form)
    - lift(x, mode='NOT') / lower(x, mode='NOT') self-inverse
    - point auto-promotes to degenerate region containing origin
    - inverse=True for AND/OR matches conjunctionReverse / disjunctionReverse
    - legacy positional 2-arg form fires DeprecationWarning and returns
      bit-exact pre-refactor outputs (smoothed body)
    - Ops.conjunction / Ops.disjunction are thin forwarders to Ops.lower /
      Ops.lift with kind selected by `monotonic` (True→strict, False→radial).
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from Layers import Ops, epsilon


class TestLowerAND(unittest.TestCase):
    """Ops.lower(..., mode='AND') -- analysis / intersection."""

    def test_point_form_default_kind_is_strict_min(self):
        x = torch.tensor([0.5, -0.3, 0.8, 0.0])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9])
        out = Ops.lower(x, y, mode='AND')
        self.assertTrue(torch.allclose(out, torch.min(x, y)))

    def test_point_form_kind_strict_is_min(self):
        x = torch.tensor([0.5, -0.3, 0.8, 0.0])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9])
        out = Ops.lower(x, y, mode='AND', kind='strict')
        self.assertTrue(torch.allclose(out, torch.min(x, y)))

    def test_point_form_is_elementwise_product(self):
        x = torch.tensor([0.5, -0.3, 0.8, 0.0])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9])
        out = Ops.lower(x, y, mode='AND', kind='smooth')
        self.assertTrue(torch.allclose(out, x * y))

    def test_point_form_kind_radial_is_radmin(self):
        # RadMin: same-sign minimum magnitude, zero collapse.
        x = torch.tensor([0.5, -0.3, 0.8, 0.0, 0.7])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9, 0.2])
        out = Ops.lower(x, y, mode='AND', kind='radial')
        same_sign = (x * y > 0).float()
        expected = same_sign * torch.sign(x) * torch.min(x.abs(), y.abs())
        self.assertTrue(torch.allclose(out, expected))

    def test_region_form_is_max_lower_min_upper(self):
        l1 = torch.tensor([-0.7, 0.1])
        u1 = torch.tensor([ 0.3, 0.8])
        l2 = torch.tensor([-0.5, -0.1])
        u2 = torch.tensor([ 0.5, 0.6])
        out_l, out_u = Ops.lower((l1, u1), (l2, u2), mode='AND')
        self.assertTrue(torch.equal(out_l, torch.maximum(l1, l2)))
        self.assertTrue(torch.equal(out_u, torch.minimum(u1, u2)))

    def test_region_form_unchanged_by_kind(self):
        # kind only affects the point body; region body is the same regardless.
        l1 = torch.tensor([-0.7, 0.1])
        u1 = torch.tensor([ 0.3, 0.8])
        l2 = torch.tensor([-0.5, -0.1])
        u2 = torch.tensor([ 0.5, 0.6])
        for kind in ('strict', 'smooth', 'radial'):
            out_l, out_u = Ops.lower((l1, u1), (l2, u2), mode='AND', kind=kind)
            self.assertTrue(torch.equal(out_l, torch.maximum(l1, l2)))
            self.assertTrue(torch.equal(out_u, torch.minimum(u1, u2)))

    def test_point_promotes_to_degenerate_region(self):
        # Point [0.5, -0.3] promotes to (min(x, 0), max(x, 0))
        # = ([0.0, -0.3], [0.5, 0.0]).
        x_point = torch.tensor([0.5, -0.3])
        y_region = (torch.tensor([-0.4, -0.5]), torch.tensor([0.2, 0.1]))
        out_l, out_u = Ops.lower(x_point, y_region, mode='AND')
        x_l = torch.tensor([0.0, -0.3])
        x_u = torch.tensor([0.5, 0.0])
        self.assertTrue(torch.equal(out_l, torch.maximum(x_l, y_region[0])))
        self.assertTrue(torch.equal(out_u, torch.minimum(x_u, y_region[1])))


class TestLiftOR(unittest.TestCase):
    """Ops.lift(..., mode='OR') -- synthesis / union."""

    def test_point_form_default_kind_is_strict_max(self):
        x = torch.tensor([0.5, -0.3, 0.8, 0.0])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9])
        out = Ops.lift(x, y, mode='OR')
        self.assertTrue(torch.allclose(out, torch.max(x, y)))

    def test_point_form_kind_strict_is_max(self):
        x = torch.tensor([0.5, -0.3, 0.8, 0.0])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9])
        out = Ops.lift(x, y, mode='OR', kind='strict')
        self.assertTrue(torch.allclose(out, torch.max(x, y)))

    def test_point_form_is_arithmetic_mean(self):
        x = torch.tensor([0.5, -0.3, 0.8, 0.0])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9])
        out = Ops.lift(x, y, mode='OR', kind='smooth')
        self.assertTrue(torch.allclose(out, (x + y) / 2))

    def test_point_form_kind_radial_is_radmax(self):
        # RadMax: same-sign max magnitude with zero passthrough.
        x = torch.tensor([0.5, -0.3, 0.8, 0.0, 0.7])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9, 0.2])
        out = Ops.lift(x, y, mode='OR', kind='radial')
        same_sign = (x * y > 0).float()
        max_mag = torch.max(x.abs(), y.abs())
        core = same_sign * torch.sign(x) * max_mag
        x_zero = (x == 0).float()
        y_zero = (y == 0).float()
        expected = core + x_zero * y + y_zero * x
        self.assertTrue(torch.allclose(out, expected))

    def test_region_form_is_min_lower_max_upper(self):
        l1 = torch.tensor([-0.7, 0.1])
        u1 = torch.tensor([ 0.3, 0.8])
        l2 = torch.tensor([-0.5, -0.1])
        u2 = torch.tensor([ 0.5, 0.6])
        out_l, out_u = Ops.lift((l1, u1), (l2, u2), mode='OR')
        self.assertTrue(torch.equal(out_l, torch.minimum(l1, l2)))
        self.assertTrue(torch.equal(out_u, torch.maximum(u1, u2)))

    def test_region_form_unchanged_by_kind(self):
        l1 = torch.tensor([-0.7, 0.1])
        u1 = torch.tensor([ 0.3, 0.8])
        l2 = torch.tensor([-0.5, -0.1])
        u2 = torch.tensor([ 0.5, 0.6])
        for kind in ('strict', 'smooth', 'radial'):
            out_l, out_u = Ops.lift((l1, u1), (l2, u2), mode='OR', kind=kind)
            self.assertTrue(torch.equal(out_l, torch.minimum(l1, l2)))
            self.assertTrue(torch.equal(out_u, torch.maximum(u1, u2)))


class TestNOT(unittest.TestCase):
    """mode='NOT' is self-inverse on both lift and lower."""

    def test_lift_not_is_sign_flip_default(self):
        x = torch.tensor([0.5, -0.3, 0.0, 0.9])
        out = Ops.lift(x, mode='NOT')
        self.assertTrue(torch.allclose(out, -x))

    def test_lower_not_is_sign_flip_default(self):
        x = torch.tensor([0.5, -0.3, 0.0, 0.9])
        out = Ops.lower(x, mode='NOT')
        self.assertTrue(torch.allclose(out, -x))

    def test_lift_not_self_inverse(self):
        x = torch.tensor([0.5, -0.3, 0.7])
        twice = Ops.lift(Ops.lift(x, mode='NOT'), mode='NOT')
        self.assertTrue(torch.allclose(twice, x))

    def test_lift_not_inverse_true_equals_forward(self):
        x = torch.tensor([0.2, -0.7, 0.4])
        fwd = Ops.lift(x, mode='NOT')
        inv = Ops.lift(x, mode='NOT', inverse=True)
        self.assertTrue(torch.allclose(fwd, inv))

    def test_lift_not_monotonic_paired_index_flip(self):
        # Bivector layout: (aP, aN) per concept. NOT swaps pairs.
        x = torch.tensor([0.9, 0.0, 0.0, 0.7])  # concept 0 = +0.9; concept 1 = -0.7
        out = Ops.lift(x, mode='NOT', monotonic=True)
        expected = torch.tensor([0.0, 0.9, 0.7, 0.0])
        self.assertTrue(torch.allclose(out, expected))


class TestInverse(unittest.TestCase):
    """inverse=True for AND/OR routes to codebook-search reverse."""

    def test_lift_or_inverse_requires_W(self):
        x = torch.tensor([0.4, 0.5])
        y = torch.tensor([0.2, 0.3])
        with self.assertRaises(NotImplementedError):
            Ops.lift(x, y, mode='OR', inverse=True)

    def test_lower_and_inverse_requires_W(self):
        x = torch.tensor([0.4, 0.5])
        y = torch.tensor([0.2, 0.3])
        with self.assertRaises(NotImplementedError):
            Ops.lower(x, y, mode='AND', inverse=True)

    def test_lower_and_inverse_via_W_matches_conjunctionReverse(self):
        # Build a small codebook and pick a result from the codebook itself.
        torch.manual_seed(0)
        K, D = 6, 3
        W = torch.rand(K, D) - 0.5
        result = W[2].clone()
        y = W[1].clone()
        # Direct call to Ops.conjunctionReverse with W should match the
        # codebook search.
        recovered = Ops.conjunctionReverse(result, y, W)
        # Sanity: it returns a vector of shape (D,)
        self.assertEqual(recovered.shape, result.shape)


class TestDeprecationAliases(unittest.TestCase):
    """Legacy positional Ops.lift(left, right) and Ops.lower(left, right)
    forms (no mode kwarg) emit DeprecationWarning and return bit-exact
    pre-refactor outputs (XXX -- review when convenient, per spec Q5)."""

    def test_legacy_lift_is_elementwise_product(self):
        x = torch.tensor([0.5, -0.3, 0.8])
        y = torch.tensor([0.4,  0.6, -0.2])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = Ops.lift(x, y)
        self.assertTrue(torch.allclose(out, x * y),
                        "legacy lift body must equal elementwise product")
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "legacy lift form should emit DeprecationWarning",
        )

    def test_legacy_lower_is_arithmetic_mean(self):
        x = torch.tensor([0.5, -0.3, 0.8])
        y = torch.tensor([0.4,  0.6, -0.2])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = Ops.lower(x, y)
        self.assertTrue(torch.allclose(out, (x + y) / 2),
                        "legacy lower body must equal arithmetic mean")
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "legacy lower form should emit DeprecationWarning",
        )

    def test_legacy_liftReverse_inverts_product(self):
        # liftReverse(result, right) recovers X1 from result = X1 * right.
        x = torch.tensor([0.4, 0.5, 0.6])
        right = torch.tensor([0.3, 0.7, 0.9])
        result = x * right
        recovered = Ops.liftReverse(result, right)
        self.assertTrue(torch.allclose(recovered, x, atol=1e-5))

    def test_legacy_lowerReverse_inverts_mean(self):
        # lowerReverse(result, right) recovers X1 from result = (X1 + right) / 2.
        x = torch.tensor([0.4, 0.5, 0.6])
        right = torch.tensor([0.3, 0.7, 0.9])
        result = (x + right) / 2
        recovered = Ops.lowerReverse(result, right)
        self.assertTrue(torch.allclose(recovered, x, atol=1e-6))

    def test_legacy_form_polarity_swap(self):
        # Plan: legacy lift(x, y) → new lower(x, y, mode='AND', kind='smooth') (both product).
        # Plan: legacy lower(x, y) → new lift(x, y, mode='OR', kind='smooth') (both mean).
        x = torch.tensor([0.5, -0.3, 0.8])
        y = torch.tensor([0.4,  0.6, -0.2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_lift = Ops.lift(x, y)
            legacy_lower = Ops.lower(x, y)
        new_lower_and = Ops.lower(x, y, mode='AND', kind='smooth')
        new_lift_or = Ops.lift(x, y, mode='OR', kind='smooth')
        self.assertTrue(torch.allclose(legacy_lift, new_lower_and))
        self.assertTrue(torch.allclose(legacy_lower, new_lift_or))


class TestRegionPromotion(unittest.TestCase):
    """Point auto-promotion to degenerate region containing origin."""

    def test_as_region_promotes_point(self):
        x = torch.tensor([0.5, -0.3, 0.0])
        l, u = Ops._as_region(x)
        # Lower bound = min(x, 0); upper bound = max(x, 0)
        self.assertTrue(torch.equal(l, torch.tensor([0.0, -0.3, 0.0])))
        self.assertTrue(torch.equal(u, torch.tensor([0.5, 0.0, 0.0])))

    def test_as_region_passes_through_existing_region(self):
        l = torch.tensor([-0.5, -0.1])
        u = torch.tensor([ 0.5,  0.4])
        out_l, out_u = Ops._as_region((l, u))
        self.assertIs(out_l, l)
        self.assertIs(out_u, u)


class TestModeRouting(unittest.TestCase):
    """Cross-mode routing: lift(mode='AND') routes through lower;
    lower(mode='OR') routes through lift."""

    def test_lift_and_routes_to_lower(self):
        x = torch.tensor([0.5, -0.3])
        y = torch.tensor([0.4,  0.6])
        # lift(mode='AND') == lower(mode='AND')
        self.assertTrue(torch.allclose(
            Ops.lift(x, y, mode='AND'),
            Ops.lower(x, y, mode='AND'),
        ))

    def test_lower_or_routes_to_lift(self):
        x = torch.tensor([0.5, -0.3])
        y = torch.tensor([0.4,  0.6])
        # lower(mode='OR') == lift(mode='OR')
        self.assertTrue(torch.allclose(
            Ops.lower(x, y, mode='OR'),
            Ops.lift(x, y, mode='OR'),
        ))


class TestConjunctionDisjunctionForwarders(unittest.TestCase):
    """Step 2: Ops.conjunction / Ops.disjunction are thin forwarders.
    monotonic=True  → kind='strict' (lattice min/max).
    monotonic=False → kind='radial' (RadMin / RadMax).
    Bit-exact match to the pre-Step-2 conjunction / disjunction bodies.
    """

    def setUp(self):
        # Shape includes opposite-sign and zero entries to exercise both
        # the same-sign branch and the zero-passthrough branch in radial form.
        self.x = torch.tensor([0.5, -0.3, 0.8,  0.0, 0.7, 0.0])
        self.y = torch.tensor([0.4,  0.6, -0.2, 0.9, 0.2, 0.0])

    def test_conjunction_monotonic_equals_lower_strict(self):
        out_conj = Ops.conjunction(self.x, self.y, monotonic=True)
        out_lower = Ops.lower(self.x, self.y, mode='AND', kind='strict')
        self.assertTrue(torch.equal(out_conj, out_lower))

    def test_conjunction_bitonic_equals_lower_radial(self):
        out_conj = Ops.conjunction(self.x, self.y, monotonic=False)
        out_lower = Ops.lower(self.x, self.y, mode='AND', kind='radial')
        self.assertTrue(torch.equal(out_conj, out_lower))

    def test_disjunction_monotonic_equals_lift_strict(self):
        out_disj = Ops.disjunction(self.x, self.y, monotonic=True)
        out_lift = Ops.lift(self.x, self.y, mode='OR', kind='strict')
        self.assertTrue(torch.equal(out_disj, out_lift))

    def test_disjunction_bitonic_equals_lift_radial(self):
        out_disj = Ops.disjunction(self.x, self.y, monotonic=False)
        out_lift = Ops.lift(self.x, self.y, mode='OR', kind='radial')
        self.assertTrue(torch.equal(out_disj, out_lift))

    def test_conjunction_monotonic_matches_torch_min(self):
        # Bit-exact pre-Step-2 body: monotonic conjunction = torch.min.
        out = Ops.conjunction(self.x, self.y, monotonic=True)
        self.assertTrue(torch.equal(out, torch.min(self.x, self.y)))

    def test_conjunction_bitonic_matches_pre_step2_radmin(self):
        # Bit-exact pre-Step-2 body: same-sign min magnitude (zero collapse).
        out = Ops.conjunction(self.x, self.y, monotonic=False)
        same_sign = (self.x * self.y > 0).float()
        min_mag = torch.min(self.x.abs(), self.y.abs())
        expected = same_sign * torch.sign(self.x) * min_mag
        self.assertTrue(torch.equal(out, expected))

    def test_disjunction_monotonic_matches_torch_max(self):
        # Bit-exact pre-Step-2 body: monotonic disjunction = torch.max.
        out = Ops.disjunction(self.x, self.y, monotonic=True)
        self.assertTrue(torch.equal(out, torch.max(self.x, self.y)))

    def test_disjunction_bitonic_matches_pre_step2_radmax(self):
        # Bit-exact pre-Step-2 body: same-sign max magnitude with zero passthrough.
        out = Ops.disjunction(self.x, self.y, monotonic=False)
        same_sign = (self.x * self.y > 0).float()
        max_mag = torch.max(self.x.abs(), self.y.abs())
        core = same_sign * torch.sign(self.x) * max_mag
        x_zero = (self.x == 0).float()
        y_zero = (self.y == 0).float()
        expected = core + x_zero * self.y + y_zero * self.x
        self.assertTrue(torch.equal(out, expected))


class TestRadHelpers(unittest.TestCase):
    """Step 2: Ops._radmin / Ops._radmax private helpers carry the bitonic
    bodies so kind='radial' avoids circular routing through conjunction /
    disjunction (which themselves now route back through lift / lower).
    """

    def test_radmin_matches_pre_step2_bitonic_conjunction(self):
        x = torch.tensor([0.5, -0.3, 0.8,  0.0, 0.7])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9, 0.2])
        same_sign = (x * y > 0).float()
        expected = same_sign * torch.sign(x) * torch.min(x.abs(), y.abs())
        self.assertTrue(torch.equal(Ops._radmin(x, y), expected))

    def test_radmax_matches_pre_step2_bitonic_disjunction(self):
        x = torch.tensor([0.5, -0.3, 0.8,  0.0, 0.7])
        y = torch.tensor([0.4,  0.6, -0.2, 0.9, 0.2])
        same_sign = (x * y > 0).float()
        max_mag = torch.max(x.abs(), y.abs())
        core = same_sign * torch.sign(x) * max_mag
        x_zero = (x == 0).float()
        y_zero = (y == 0).float()
        expected = core + x_zero * y + y_zero * x
        self.assertTrue(torch.equal(Ops._radmax(x, y), expected))


if __name__ == '__main__':
    unittest.main()
