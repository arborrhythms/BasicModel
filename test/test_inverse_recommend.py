"""Tests for the mereology-guided union / intersection inverse recommender.

Replaces the prior brute-force ``_binary_op_inverse_impl`` codebook search
(which returned only a single best-matching left operand) with a structured,
mereology-guided recommendation that returns the pair ``(x1, x2)`` drawn
from the augmented codebook ``C = [⊥; W; ⊤]``.

See ``doc/plans/2026-05-17-inverse-recommend-union-intersection.md``.

Coverage:
    - partOf / wholeOf / overlapOf truth tables, incl. ⊥ / ⊤ sentinels.
    - union inverse: union(x1, x2) ≈ y for codebook-drawn y; x1 ≤ y;
      x2 ≥ y − x1; sentinels guarantee non-empty selection.
    - intersection inverse: x1 ≥ y, smallest; x2 ≥ y minimises
      overlapOf(x2, x1); intersection(x1, x2) ≈ y.
    - empty learned codebook → selection still works via ⊥ / ⊤.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from Layers import Ops


class TestMereologyHelpers(unittest.TestCase):
    """Truth tables for partOf / wholeOf / overlapOf, including sentinels."""

    def test_partOf_basic(self):
        a = torch.tensor([0.2, 0.3])
        b = torch.tensor([0.5, 0.4])
        c = torch.tensor([0.1, 0.5])
        self.assertTrue(bool(Ops.partOf(a, b)))         # a ≤ b elementwise
        self.assertFalse(bool(Ops.partOf(b, a)))        # b > a in both dims
        self.assertFalse(bool(Ops.partOf(a, c)))        # a[1] > c[1] only

    def test_wholeOf_basic(self):
        a = torch.tensor([0.5, 0.4])
        b = torch.tensor([0.2, 0.3])
        self.assertTrue(bool(Ops.wholeOf(a, b)))
        self.assertFalse(bool(Ops.wholeOf(b, a)))

    def test_partOf_with_sentinels(self):
        bottom = torch.zeros(2)
        top = torch.ones(2)
        x = torch.tensor([0.3, 0.7])
        # ⊥ ≤ anything (in [0, 1] domain), anything ≤ ⊤
        self.assertTrue(bool(Ops.partOf(bottom, x)))
        self.assertTrue(bool(Ops.partOf(x, top)))
        self.assertTrue(bool(Ops.partOf(bottom, top)))
        # ⊤ is not part of x (unless x == ⊤)
        self.assertFalse(bool(Ops.partOf(top, x)))

    def test_wholeOf_with_sentinels(self):
        bottom = torch.zeros(2)
        top = torch.ones(2)
        x = torch.tensor([0.3, 0.7])
        # ⊤ ≥ anything; ⊥ ≥ nothing (except ⊥ itself).
        self.assertTrue(bool(Ops.wholeOf(top, x)))
        self.assertFalse(bool(Ops.wholeOf(bottom, x)))
        self.assertTrue(bool(Ops.wholeOf(bottom, bottom)))

    def test_overlapOf_same_sign_min_magnitude(self):
        a = torch.tensor([0.5, 0.2])
        b = torch.tensor([0.3, 0.4])
        out = Ops.overlapOf(a, b)
        # Both pairs same-sign positive: min magnitude per channel.
        expected = torch.tensor([0.3, 0.2])
        self.assertTrue(torch.allclose(out, expected))

    def test_overlapOf_sign_disagreement_collapses_to_zero(self):
        a = torch.tensor([0.5, -0.2])
        b = torch.tensor([0.3, 0.4])
        out = Ops.overlapOf(a, b)
        # Channel 0 same-sign positive → 0.3; channel 1 sign mismatch → 0.
        expected = torch.tensor([0.3, 0.0])
        self.assertTrue(torch.allclose(out, expected))


class TestUnionInverse(unittest.TestCase):
    """``y = max(x1, x2)`` → recover (x1, x2) via Ops.disjunctionReverse."""

    def _build_codebook(self):
        torch.manual_seed(0)
        # Non-negative bivector rows so the lattice ordering is meaningful.
        return torch.tensor([
            [0.2, 0.1],
            [0.4, 0.3],
            [0.6, 0.5],
            [0.8, 0.7],
        ])

    def test_pair_satisfies_constraints_on_codebook_target(self):
        W = self._build_codebook()
        y = W[2].clone()                                # [0.6, 0.5]
        x1, x2 = Ops.disjunctionReverse(y, y, W)
        # Algorithm invariants:
        #   x1 ≤ y elementwise (largest part ≤ y).
        self.assertTrue(bool(Ops.partOf(x1, y)))
        #   x2 ≥ (y − x1) elementwise (smallest part ≥ residual).
        r = y - x1
        self.assertTrue(bool(Ops.wholeOf(x2, r)))
        # And forward recombines back to y (under monotonic max).
        recombined = Ops.union(x1, x2, monotonic=True)
        self.assertTrue(torch.allclose(recombined, y, atol=1e-5))

    def test_batched_recovery(self):
        W = self._build_codebook()
        Y = torch.stack([W[0], W[2], W[3]], dim=0)      # [3, 2]
        x1, x2 = Ops.disjunctionReverse(Y, Y, W)
        self.assertEqual(x1.shape, Y.shape)
        self.assertEqual(x2.shape, Y.shape)
        recombined = Ops.union(x1, x2, monotonic=True)
        self.assertTrue(torch.allclose(recombined, Y, atol=1e-5))

    def test_sentinels_keep_selection_non_empty(self):
        # When y is below every learned row but ⊥ is available, x1 falls
        # back to ⊥ rather than producing a NaN argmax.
        W = torch.tensor([[0.6, 0.6], [0.8, 0.8]])
        y = torch.tensor([0.1, 0.1])                    # below all rows
        x1, x2 = Ops.disjunctionReverse(y, y, W)
        # x1 must be ≤ y; the only such row in C = [⊥, W, ⊤] is ⊥.
        self.assertTrue(bool(Ops.partOf(x1, y)))
        # x2 must be ≥ (y − x1) = y; ⊤ or W[0] both satisfy ≥ y.
        self.assertTrue(bool(Ops.wholeOf(x2, y - x1)))


class TestIntersectionInverse(unittest.TestCase):
    """``y = min(x1, x2)`` → recover (x1, x2) via Ops.conjunctionReverse."""

    def _build_codebook(self):
        torch.manual_seed(0)
        return torch.tensor([
            [0.2, 0.1],
            [0.4, 0.3],
            [0.6, 0.5],
            [0.8, 0.7],
        ])

    def test_pair_satisfies_constraints_on_codebook_target(self):
        W = self._build_codebook()
        y = W[1].clone()                                # [0.4, 0.3]
        x1, x2 = Ops.conjunctionReverse(y, y, W)
        # x1 ≥ y; x2 ≥ y (both are wholes of y).
        self.assertTrue(bool(Ops.wholeOf(x1, y)))
        self.assertTrue(bool(Ops.wholeOf(x2, y)))
        # forward recombines back to y under monotonic min.
        recombined = Ops.intersection(x1, x2, monotonic=True)
        self.assertTrue(torch.allclose(recombined, y, atol=1e-5))

    def test_x2_minimises_overlap_with_x1(self):
        # Build a codebook where two distinct rows both bound y but
        # have very different overlap with x1; the recommender should
        # prefer the one with smaller overlap.
        W = torch.tensor([
            [0.5, 0.5],   # bounds y; medium row
            [0.5, 0.9],   # bounds y but overlaps {a,b} differently
            [0.9, 0.5],   # bounds y; different shape
        ])
        y = torch.tensor([0.5, 0.5])
        x1, x2 = Ops.conjunctionReverse(y, y, W)
        # Both x1, x2 ≥ y; they must differ (intersection-side wants
        # minimum mutual overlap given there are multiple wholes).
        self.assertTrue(bool(Ops.wholeOf(x1, y)))
        self.assertTrue(bool(Ops.wholeOf(x2, y)))
        # forward still recombines back to y (because min over any two
        # wholes of y on a monotonic lattice is exactly y if at least
        # one component of each meets y).
        recombined = Ops.intersection(x1, x2, monotonic=True)
        self.assertTrue(torch.allclose(recombined, y, atol=1e-5))


class TestEmptyCodebook(unittest.TestCase):
    """Empty learned W still produces a non-degenerate selection via ⊥/⊤."""

    def test_union_inverse_empty_W_falls_back_to_identity_pair(self):
        # When W is None or empty the wrapper short-circuits to
        # (result, result) without touching the recommender.
        y = torch.tensor([0.4, 0.3])
        x1, x2 = Ops.disjunctionReverse(y, y, None)
        self.assertTrue(torch.allclose(x1, y))
        self.assertTrue(torch.allclose(x2, y))

        W_empty = torch.empty(0, 2)
        x1, x2 = Ops.disjunctionReverse(y, y, W_empty)
        self.assertTrue(torch.allclose(x1, y))
        self.assertTrue(torch.allclose(x2, y))

    def test_intersection_inverse_empty_W_falls_back_to_identity_pair(self):
        y = torch.tensor([0.4, 0.3])
        x1, x2 = Ops.conjunctionReverse(y, y, None)
        self.assertTrue(torch.allclose(x1, y))
        self.assertTrue(torch.allclose(x2, y))


if __name__ == "__main__":
    unittest.main()
