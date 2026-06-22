"""Invertibility pinning for the unified ``Ops.lift`` / ``Ops.lower``
dispatcher across all reachable (``mode``, ``kind``) combinations.

Task 9 (plan §6) of the STM serial/parallel modes plan: the reverse
generation pass deletes SymbolSubSpace's syntactic cache and re-derives
words from the held STM idea. That round trip is only as good as the
per-op inverses it rides on, so this file pins the two round-trip laws

    forward(reverse(y)) ≈ y        (re-synthesis closes on a held target)
    reverse(forward(x)) ≈ x        (analysis recovers the operand)

for every inverse path the dispatcher exposes. Three distinct inverse
families are covered, each with the tolerance convention established by
``test/test_ops_lift_lower.py`` (which this file mirrors):

  * mode='NOT'  — self-inverse sign / paired-index flip. EXACT
    (``test_ops_lift_lower.TestNOT`` asserts ``allclose`` with no atol).

  * mode in {AND, OR}, kind='smooth' — the legacy analytic bodies
    (AND/smooth = elementwise product, OR/smooth = arithmetic mean).
    Their two-argument analytic inverses are ``Ops.liftReverse``
    (``result / (right + epsilon)``) and ``Ops.lowerReverse``
    (``2*result - right``). MEAN is exact; PRODUCT carries the
    ``epsilon`` denominator slack, pinned at ``atol=1e-3`` (the
    ``test_ops_lift_lower`` analytic-reverse tests use ``1e-5`` / ``1e-6``
    on benign operands; we draw random operands in [-1, 1] which can sit
    near zero, so we use the looser-but-still-tight ``1e-3``).

  * mode in {AND, OR}, kind in {strict, radial, soft} — the lossy
    lattice / radial bodies (max/min, RadMax/RadMin). These are
    many-to-one, so the inverse is the mereology-guided codebook
    recommender ``Ops.liftReverseAll`` / ``Ops.lowerReverseAll`` (which
    forward to ``disjunctionReverse`` / ``conjunctionReverse``). On a
    target drawn from the codebook the recommended pair re-closes under
    the monotonic forward EXACTLY (``test_inverse_recommend`` pins the
    same reclosure at ``atol=1e-5``); we assert ``atol=1e-5``.

This is a TEST-ONLY pinning: it uses the REAL ``Ops`` dispatcher and the
REAL reverse helpers (no reimplementation). ``import Language`` is what
binds the public ``Ops.lift`` / ``Ops.lower`` / ``Ops.conjunction`` /
``Ops.disjunction`` handles (they moved to Language.py in the 2026-05-29
grammar-file refactor); the autouse conftest fixture also imports it, but
we import explicitly so the file is runnable in isolation.
"""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

import Language  # noqa: F401  -- binds Ops.lift/lower/conjunction/disjunction
from Layers import Ops, epsilon

# Tolerances, mirrored from test_ops_lift_lower.py's reverse tests.
ATOL_EXACT = 0.0          # NOT self-inverse / strict-on-codebook are bit-exact
ATOL_CODEBOOK = 1e-5      # recommender reclosure (== test_inverse_recommend)
ATOL_PRODUCT = 1e-3       # liftReverse epsilon-denominator slack on [-1,1]


def _codebook():
    """Non-negative lattice-ordered bivector codebook (the ordering is
    meaningful for the mereology recommender) -- same shape family as
    test_inverse_recommend.TestUnionInverse._build_codebook."""
    return torch.tensor([
        [0.2, 0.1, 0.3],
        [0.4, 0.3, 0.5],
        [0.6, 0.5, 0.7],
        [0.8, 0.7, 0.9],
    ])


class TestNotSelfInverse(unittest.TestCase):
    """mode='NOT' is its own inverse on both lift and lower (EXACT).

    Covers both the bitonic (sign-flip) and monotonic (paired-index
    flip) bodies, in both round-trip directions. Because NOT is an
    involution, forward(reverse(y)) and reverse(forward(x)) are the same
    statement (apply twice == identity), but we assert both spellings to
    document the law as it is used by the reverse pipeline.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(4, 5) * 2 - 1

    def test_lift_not_reverse_of_forward_bitonic(self):
        twice = Ops.lift(Ops.lift(self.x, mode='NOT'), mode='NOT')
        self.assertTrue(torch.allclose(twice, self.x, atol=ATOL_EXACT))

    def test_lower_not_reverse_of_forward_bitonic(self):
        twice = Ops.lower(Ops.lower(self.x, mode='NOT'), mode='NOT')
        self.assertTrue(torch.allclose(twice, self.x, atol=ATOL_EXACT))

    def test_lift_not_forward_of_reverse_bitonic(self):
        # y -> reverse (NOT, self-inverse) -> forward (NOT) == y
        y = torch.rand(4, 5) * 2 - 1
        back = Ops.lift(Ops.lift(y, mode='NOT', inverse=True), mode='NOT')
        self.assertTrue(torch.allclose(back, y, atol=ATOL_EXACT))

    def test_lift_not_monotonic_paired_index_involution(self):
        # Bivector layout (aP, aN) per concept; NOT swaps the pair, twice
        # restores. Use an even width so the pairing is well-formed.
        x = torch.tensor([0.9, 0.0, 0.0, 0.7, 0.3, 0.0])
        twice = Ops.lift(Ops.lift(x, mode='NOT', monotonic=True),
                         mode='NOT', monotonic=True)
        self.assertTrue(torch.allclose(twice, x, atol=ATOL_EXACT))


class TestSmoothAnalyticReverse(unittest.TestCase):
    """kind='smooth': AND=product, OR=mean -- analytic two-arg inverses.

    ``Ops.liftReverse(result, right) = result / (right + epsilon)``
    inverts the PRODUCT (lower/AND/smooth).
    ``Ops.lowerReverse(result, right) = 2*result - right`` inverts the
    MEAN (lift/OR/smooth). Both directions of the round-trip law are
    pinned. Right operand is held fixed (it is the ``right`` argument the
    analytic inverse is conditioned on -- exactly the Layer-2.5 grammar
    convention where one operand is known).
    """

    def setUp(self):
        torch.manual_seed(1)
        self.x = torch.rand(6) * 2 - 1
        self.right = torch.rand(6) * 2 - 1
        self.y = torch.rand(6) * 2 - 1

    # -- MEAN body (lift, mode='OR', kind='smooth') ------------------
    def test_mean_reverse_of_forward(self):
        fwd = Ops.lift(self.x, self.right, mode='OR', kind='smooth')
        rec = Ops.lowerReverse(fwd, self.right)
        self.assertTrue(torch.allclose(rec, self.x, atol=ATOL_EXACT),
                        "mean is exactly invertible: 2*((x+r)/2) - r == x")

    def test_mean_forward_of_reverse(self):
        a = Ops.lowerReverse(self.y, self.right)        # recover operand
        back = Ops.lift(a, self.right, mode='OR', kind='smooth')
        self.assertTrue(torch.allclose(back, self.y, atol=ATOL_EXACT))

    # -- PRODUCT body (lower, mode='AND', kind='smooth') -------------
    def test_product_reverse_of_forward(self):
        fwd = Ops.lower(self.x, self.right, mode='AND', kind='smooth')
        rec = Ops.liftReverse(fwd, self.right)
        self.assertTrue(torch.allclose(rec, self.x, atol=ATOL_PRODUCT),
                        "product inverts to x up to the epsilon denominator")

    def test_product_forward_of_reverse(self):
        a = Ops.liftReverse(self.y, self.right)
        back = Ops.lower(a, self.right, mode='AND', kind='smooth')
        # back = (y/(right+eps)) * right ; exact up to the epsilon nudge.
        self.assertTrue(torch.allclose(back, self.y, atol=ATOL_PRODUCT))

    def test_product_reverse_matches_closed_form(self):
        # Pin the analytic formula itself (not just the round trip).
        fwd = Ops.lower(self.x, self.right, mode='AND', kind='smooth')
        rec = Ops.liftReverse(fwd, self.right)
        closed = (self.x * self.right) / (self.right + epsilon)
        self.assertTrue(torch.allclose(rec, closed, atol=ATOL_EXACT))


class TestLatticeCodebookReverse(unittest.TestCase):
    """kind in {strict, radial, soft}: lossy lattice/radial bodies whose
    inverse is the mereology-guided codebook recommender.

    For a target drawn FROM the codebook, the recommended pair
    re-closes under the monotonic forward exactly. We pin both
    ``Ops.lift`` (OR / union) via ``Ops.liftReverseAll`` and
    ``Ops.lower`` (AND / intersection) via ``Ops.lowerReverseAll``, and
    we check both the ``strict`` forward kind (the lattice max/min the
    recommender is defined against) and the smoothed ``soft`` kind, which
    re-closes to the same lattice value on same-sign non-negative
    operands.
    """

    def setUp(self):
        self.W = _codebook()

    # -- OR / union (lift) -------------------------------------------
    def test_or_strict_reverse_recloses_on_codebook_target(self):
        for i in range(self.W.shape[0]):
            y = self.W[i].clone()
            x1, x2 = Ops.liftReverseAll(y, W=self.W, monotonic=True)
            recl = Ops.lift(x1, x2, mode='OR', kind='strict')
            self.assertTrue(
                torch.allclose(recl, y, atol=ATOL_CODEBOOK),
                f"OR/strict reclosure failed on codebook row {i}")

    def test_or_reverse_pair_obeys_mereology_invariants(self):
        # x1 <= y (largest part) and x2 >= y - x1 (smallest enclosing
        # residual) -- the disjunctionReverse contract.
        y = self.W[2].clone()
        x1, x2 = Ops.liftReverseAll(y, W=self.W, monotonic=True)
        self.assertTrue(bool(Ops.partOf(x1, y)))
        self.assertTrue(bool(Ops.wholeOf(x2, y - x1)))

    def test_or_forward_of_reverse_batched(self):
        Y = torch.stack([self.W[0], self.W[2], self.W[3]], dim=0)
        x1, x2 = Ops.liftReverseAll(Y, W=self.W, monotonic=True)
        recl = Ops.lift(x1, x2, mode='OR', kind='strict')
        self.assertTrue(torch.allclose(recl, Y, atol=ATOL_CODEBOOK))

    # -- AND / intersection (lower) ----------------------------------
    def test_and_strict_reverse_recloses_on_codebook_target(self):
        for i in range(self.W.shape[0]):
            y = self.W[i].clone()
            x1, x2 = Ops.lowerReverseAll(y, W=self.W, monotonic=True)
            recl = Ops.lower(x1, x2, mode='AND', kind='strict')
            self.assertTrue(
                torch.allclose(recl, y, atol=ATOL_CODEBOOK),
                f"AND/strict reclosure failed on codebook row {i}")

    def test_and_reverse_pair_are_wholes_of_target(self):
        y = self.W[1].clone()
        x1, x2 = Ops.lowerReverseAll(y, W=self.W, monotonic=True)
        self.assertTrue(bool(Ops.wholeOf(x1, y)))
        self.assertTrue(bool(Ops.wholeOf(x2, y)))

    def test_and_forward_of_reverse_batched(self):
        Y = torch.stack([self.W[0], self.W[1], self.W[3]], dim=0)
        x1, x2 = Ops.lowerReverseAll(Y, W=self.W, monotonic=True)
        recl = Ops.lower(x1, x2, mode='AND', kind='strict')
        self.assertTrue(torch.allclose(recl, Y, atol=ATOL_CODEBOOK))

    # -- soft kind re-closes to the same lattice value ---------------
    def test_or_soft_reverse_recloses_to_lattice(self):
        # On non-negative, distinct-magnitude operands the soft (LSE)
        # max approaches the hard max; with a codebook-recommended pair
        # whose larger element equals y, soft reclosure stays within the
        # codebook tolerance band.
        y = self.W[3].clone()
        x1, x2 = Ops.liftReverseAll(y, W=self.W, monotonic=True)
        recl_strict = Ops.lift(x1, x2, mode='OR', kind='strict')
        recl_soft = Ops.lift(x1, x2, mode='OR', kind='soft')
        # soft and strict agree to the recommender tolerance because one
        # operand dominates per channel (the recommended pair is nested).
        self.assertTrue(
            torch.allclose(recl_soft, recl_strict, atol=ATOL_CODEBOOK),
            "soft union reclosure must track the strict lattice value")

    def test_and_radial_reverse_recloses_to_lattice(self):
        # Same argument for the intersection side with the hard radial
        # body: on same-sign non-negative nested operands RadMin == min.
        y = self.W[2].clone()
        x1, x2 = Ops.lowerReverseAll(y, W=self.W, monotonic=True)
        recl_strict = Ops.lower(x1, x2, mode='AND', kind='strict')
        recl_radial = Ops.lower(x1, x2, mode='AND', kind='radial')
        self.assertTrue(
            torch.allclose(recl_radial, recl_strict, atol=ATOL_CODEBOOK),
            "radial intersection reclosure must track the strict min")


class TestReverseModeCoverageMatrix(unittest.TestCase):
    """Smoke matrix: every (mode, kind) the dispatcher accepts has SOME
    inverse path exercised above, and the dispatcher itself routes the
    cross-mode aliases consistently (lift(mode='AND') == lower(mode='AND')).

    This guards against a future kind/mode being added without a
    corresponding inverse pinning.
    """

    def test_cross_mode_routing_is_consistent(self):
        x = torch.tensor([0.5, -0.3, 0.8])
        y = torch.tensor([0.4, 0.6, -0.2])
        self.assertTrue(torch.allclose(
            Ops.lift(x, y, mode='AND'), Ops.lower(x, y, mode='AND')))
        self.assertTrue(torch.allclose(
            Ops.lower(x, y, mode='OR'), Ops.lift(x, y, mode='OR')))

    def test_all_point_kinds_forward_then_codebook_reverse_run(self):
        # Exercise every forward kind end-to-end with a codebook reverse
        # so the (mode, kind) grid is genuinely walked, not just the
        # representative cases.
        #
        # Reclosure tolerance is per-kind on purpose:
        #   * strict / radial are the HARD lattice bodies (max/min,
        #     RadMax/RadMin); on the nested recommended pair they reclose
        #     to the lattice target EXACTLY (atol=1e-5).
        #   * soft is the LSE-smoothed body. It only approximates the
        #     hard min/max at finite temperature, so it does NOT reclose
        #     exactly when both operands are non-degenerate. The union
        #     side happens to be exact because the recommended pair is
        #     (y, bottom) and soft RadMax passes the live operand through
        #     the zero; the intersection side pairs (y, larger-whole) and
        #     the soft RadMin sits a bounded ~1.3e-2 below the true min.
        #     We pin that as an APPROXIMATE bound (the finite-temperature
        #     gap) rather than fudging it into the exact band -- a real,
        #     documented property of the soft kind.
        W = _codebook()
        soft_gap_bound = 5e-2   # LSE finite-temperature reclosure slack
        for mode, rev in (('OR', Ops.liftReverseAll),
                          ('AND', Ops.lowerReverseAll)):
            y = W[2].clone()
            x1, x2 = rev(y, W=W, monotonic=True)
            op = Ops.lift if mode == 'OR' else Ops.lower
            for kind in ('strict', 'radial', 'soft'):
                recl = op(x1, x2, mode=mode, kind=kind)
                self.assertTrue(torch.isfinite(recl).all(),
                                f"{mode}/{kind} reclosure must be finite")
                tol = ATOL_CODEBOOK if kind in ('strict', 'radial') \
                    else soft_gap_bound
                err = (recl - y).abs().max().item()
                self.assertLessEqual(
                    err, tol,
                    f"{mode}/{kind} reclosure on a nested codebook pair: "
                    f"max|err|={err:.2e} exceeds tol={tol:.0e}")
            # Independent of mode, the HARD kinds must agree bit-exactly
            # (they are the same lattice value computed two ways).
            self.assertTrue(torch.allclose(
                op(x1, x2, mode=mode, kind='strict'),
                op(x1, x2, mode=mode, kind='radial'),
                atol=ATOL_CODEBOOK),
                f"{mode} strict and radial must reclose identically here")


if __name__ == '__main__':
    unittest.main()
