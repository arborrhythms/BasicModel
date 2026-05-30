"""Forward-closure pinning for the union / intersection inverse
recommender, in the STM reverse-generation regime.

Task 9 (plan §6) of the STM serial/parallel modes plan. When the
reverse pass re-derives words from the held STM idea, the union /
intersection grammar ops are inverted by the mereology-guided
recommender ``Ops.disjunctionReverse`` / ``Ops.conjunctionReverse``,
which return a pair ``(x1, x2)`` drawn from the augmented codebook
``C = [bottom; W; top]``. This file pins the property the reverse pass
relies on:

    codebook-fixed, priming-OFF, candidate-recommendation -> forward
    closure: for a target ``y`` drawn from the (fixed) codebook, the
    recommended pair re-closes under the monotonic forward,
    ``union(x1, x2) ≈ y`` / ``intersection(x1, x2) ≈ y``.

It mirrors ``test/test_inverse_recommend.py`` (codebook construction,
mereology invariants, and the ``atol=1e-5`` reclosure tolerance) and
adds the STM-specific framings the reverse pass exercises:

  * priming-OFF is the explicit default (no ``left_priming`` /
    ``right_priming``); a priming buffer of all-ones (the unprimed
    identity) must give byte-identical recommendations.
  * candidate restriction via ``left_rows`` / ``right_rows`` (the typed
    hard-admissibility mask the reverse pass passes from the
    KnowledgeView) still re-closes -- the sentinels keep selection
    non-empty even when the restricted set excludes the natural pick.
  * batched targets (a whole STM row of ideas) re-close jointly.

TEST-ONLY: uses the REAL ``Ops`` recommender (no reimplementation).
``import Language`` binds the public ``Ops.union`` / ``Ops.intersection``
forward kernels used to check closure.
"""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

import Language  # noqa: F401  -- binds Ops.union / Ops.intersection
from Layers import Ops

ATOL_CLOSURE = 1e-5   # == test_inverse_recommend reclosure tolerance


def _codebook():
    """Non-negative, lattice-ordered bivector codebook -- same family as
    test_inverse_recommend.TestUnionInverse._build_codebook (the ordering
    is what makes partOf / wholeOf meaningful)."""
    return torch.tensor([
        [0.2, 0.1],
        [0.4, 0.3],
        [0.6, 0.5],
        [0.8, 0.7],
    ])


class TestUnionReverseClosure(unittest.TestCase):
    """y = max(x1, x2): disjunctionReverse pair re-closes under union."""

    def setUp(self):
        self.W = _codebook()

    def test_every_codebook_target_recloses(self):
        for i in range(self.W.shape[0]):
            y = self.W[i].clone()
            x1, x2 = Ops.disjunctionReverse(y, y, self.W)
            recl = Ops.union(x1, x2, monotonic=True)
            self.assertTrue(
                torch.allclose(recl, y, atol=ATOL_CLOSURE),
                f"union reclosure failed on codebook row {i}: "
                f"{recl.tolist()} != {y.tolist()}")

    def test_recommended_pair_obeys_mereology(self):
        y = self.W[2].clone()
        x1, x2 = Ops.disjunctionReverse(y, y, self.W)
        self.assertTrue(bool(Ops.partOf(x1, y)),
                        "x1 must be a part of y (x1 <= y).")
        self.assertTrue(bool(Ops.wholeOf(x2, y - x1)),
                        "x2 must enclose the residual (x2 >= y - x1).")

    def test_batched_targets_reclose(self):
        Y = torch.stack([self.W[0], self.W[2], self.W[3]], dim=0)   # [3,2]
        x1, x2 = Ops.disjunctionReverse(Y, Y, self.W)
        self.assertEqual(x1.shape, Y.shape)
        recl = Ops.union(x1, x2, monotonic=True)
        self.assertTrue(torch.allclose(recl, Y, atol=ATOL_CLOSURE))


class TestIntersectionReverseClosure(unittest.TestCase):
    """y = min(x1, x2): conjunctionReverse pair re-closes under
    intersection."""

    def setUp(self):
        self.W = _codebook()

    def test_every_codebook_target_recloses(self):
        for i in range(self.W.shape[0]):
            y = self.W[i].clone()
            x1, x2 = Ops.conjunctionReverse(y, y, self.W)
            recl = Ops.intersection(x1, x2, monotonic=True)
            self.assertTrue(
                torch.allclose(recl, y, atol=ATOL_CLOSURE),
                f"intersection reclosure failed on codebook row {i}: "
                f"{recl.tolist()} != {y.tolist()}")

    def test_recommended_pair_are_wholes(self):
        y = self.W[1].clone()
        x1, x2 = Ops.conjunctionReverse(y, y, self.W)
        self.assertTrue(bool(Ops.wholeOf(x1, y)))
        self.assertTrue(bool(Ops.wholeOf(x2, y)))

    def test_batched_targets_reclose(self):
        Y = torch.stack([self.W[0], self.W[1], self.W[3]], dim=0)
        x1, x2 = Ops.conjunctionReverse(Y, Y, self.W)
        recl = Ops.intersection(x1, x2, monotonic=True)
        self.assertTrue(torch.allclose(recl, Y, atol=ATOL_CLOSURE))


class TestPrimingOffIsIdentity(unittest.TestCase):
    """Priming-OFF (the reverse-pass default) must be byte-identical to an
    explicit all-ones (unity) priming buffer -- the multiplicative
    identity. This guards the reverse pass's no-priming fast path against
    drifting from the priming-aware path at unity."""

    def setUp(self):
        self.W = _codebook()

    def test_union_unprimed_equals_unity_priming(self):
        y = self.W[2].clone()
        K = self.W.shape[0]
        ones = torch.ones(K)
        x1a, x2a = Ops.disjunctionReverse(y, y, self.W)                  # priming OFF
        x1b, x2b = Ops.disjunctionReverse(
            y, y, self.W, left_priming=ones, right_priming=ones)        # unity
        self.assertTrue(torch.equal(x1a, x1b))
        self.assertTrue(torch.equal(x2a, x2b))

    def test_intersection_unprimed_equals_unity_priming(self):
        y = self.W[1].clone()
        K = self.W.shape[0]
        ones = torch.ones(K)
        x1a, x2a = Ops.conjunctionReverse(y, y, self.W)
        x1b, x2b = Ops.conjunctionReverse(
            y, y, self.W, left_priming=ones, right_priming=ones)
        self.assertTrue(torch.equal(x1a, x1b))
        self.assertTrue(torch.equal(x2a, x2b))


class TestCandidateRestrictionStillCloses(unittest.TestCase):
    """The typed hard-admissibility mask (``left_rows`` / ``right_rows``)
    the reverse pass forwards from the KnowledgeView restricts which
    learned rows are eligible, but the bottom/top sentinels keep the
    selection feasible so closure still holds."""

    def setUp(self):
        self.W = _codebook()

    def test_union_restricted_to_natural_rows_recloses(self):
        # Allow exactly the rows that can compose y = W[3] = [0.8,0.7]:
        # x1 <= y is satisfiable by any row, x2 >= residual by the top
        # rows. Restricting to {row3} for x1 and {row3} for x2 still lets
        # union(row3, anything>=0) reclose to row3 via the dominating part.
        y = self.W[3].clone()
        rows = torch.tensor([3], dtype=torch.long)
        x1, x2 = Ops.disjunctionReverse(
            y, y, self.W, left_rows=rows, right_rows=rows)
        recl = Ops.union(x1, x2, monotonic=True)
        self.assertTrue(torch.allclose(recl, y, atol=ATOL_CLOSURE))

    def test_union_overrestricted_falls_back_to_sentinels(self):
        # Restrict to a row that cannot serve as the dominating part of a
        # SMALL target: y = W[0] = [0.2,0.1] but only allow row3 (too big
        # to be <= y). The recommender must fall back to the bottom
        # sentinel for x1 so the pair stays feasible and x1 <= y holds.
        y = self.W[0].clone()
        rows = torch.tensor([3], dtype=torch.long)   # [0.8,0.7], not <= y
        x1, x2 = Ops.disjunctionReverse(
            y, y, self.W, left_rows=rows, right_rows=rows)
        # x1 must remain a part of y; only bottom (zeros) qualifies here.
        self.assertTrue(bool(Ops.partOf(x1, y)),
                        "over-restricted union must fall back to bottom for x1")

    def test_intersection_restricted_recloses(self):
        # y = W[1] = [0.4,0.3]. To re-close under min, the candidate set
        # must contain a row that *touches* y on every channel (here W[1]
        # itself) -- the typed mask the reverse pass builds includes the
        # admissible word, so we allow {row1, row2, row3}. intersection of
        # the recommended wholes then re-closes to y exactly.
        y = self.W[1].clone()
        rows = torch.tensor([1, 2, 3], dtype=torch.long)
        x1, x2 = Ops.conjunctionReverse(
            y, y, self.W, left_rows=rows, right_rows=rows)
        self.assertTrue(bool(Ops.wholeOf(x1, y)))
        self.assertTrue(bool(Ops.wholeOf(x2, y)))
        recl = Ops.intersection(x1, x2, monotonic=True)
        self.assertTrue(torch.allclose(recl, y, atol=ATOL_CLOSURE))

    def test_intersection_overrestricted_overshoots_above_y(self):
        # Honest counterpart: when the candidate set EXCLUDES every row
        # that touches y (only strictly-larger wholes {row2,row3} are
        # allowed, and the top sentinel = ones is also a whole), the
        # recommended pair are both strictly-greater wholes, so their min
        # OVERSHOOTS -- it stays a whole of y but does NOT equal y. This
        # documents that closure needs the admissible word in the mask;
        # it is a property of the lattice min, not a recommender bug.
        y = self.W[1].clone()                          # [0.4, 0.3]
        rows = torch.tensor([2, 3], dtype=torch.long)  # both strictly > y
        x1, x2 = Ops.conjunctionReverse(
            y, y, self.W, left_rows=rows, right_rows=rows)
        recl = Ops.intersection(x1, x2, monotonic=True)
        self.assertTrue(bool(Ops.wholeOf(recl, y)),
                        "overshoot still encloses y (recl >= y).")
        self.assertFalse(
            torch.allclose(recl, y, atol=ATOL_CLOSURE),
            "with y excluded from the mask, min of larger wholes overshoots.")

    def test_empty_restriction_still_feasible_via_sentinels(self):
        # An empty LongTensor restriction (category x order intersection
        # was empty) leaves ONLY the sentinels feasible; selection must
        # still return a pair (no NaN argmax).
        y = self.W[2].clone()
        empty = torch.empty(0, dtype=torch.long)
        x1, x2 = Ops.disjunctionReverse(
            y, y, self.W, left_rows=empty, right_rows=empty)
        self.assertEqual(x1.shape, y.shape)
        self.assertTrue(torch.isfinite(x1).all())
        self.assertTrue(torch.isfinite(x2).all())
        self.assertTrue(bool(Ops.partOf(x1, y)),
                        "sentinel-only union still yields x1 <= y (bottom).")


class TestEmptyCodebookFallback(unittest.TestCase):
    """No learned codebook -> the wrapper short-circuits to (y, y); the
    reverse pass before any words are learned must not crash."""

    def test_union_none_codebook_is_identity_pair(self):
        y = torch.tensor([0.4, 0.3])
        x1, x2 = Ops.disjunctionReverse(y, y, None)
        self.assertTrue(torch.allclose(x1, y))
        self.assertTrue(torch.allclose(x2, y))

    def test_intersection_empty_codebook_is_identity_pair(self):
        y = torch.tensor([0.4, 0.3])
        W_empty = torch.empty(0, 2)
        x1, x2 = Ops.conjunctionReverse(y, y, W_empty)
        self.assertTrue(torch.allclose(x1, y))
        self.assertTrue(torch.allclose(x2, y))


if __name__ == "__main__":
    unittest.main()
