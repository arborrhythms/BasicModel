"""Surviving inverse-operation priming contracts."""
import torch
from Language import Ops

class TestUnityPrimingByteIdentical:
    """Plan test 8: passing all-1.0 priming must produce the exact same
    (x1, x2) pair as passing no priming (None).

    Unity is the multiplicative identity for the priming boost, so the
    algorithms are provably equivalent.  This test exercises:
      - ``Ops.unionReverse``
      - ``Ops.intersectionReverse``
    at the Ops level (the Layer-level wrappers just forward kwargs, so
    Ops-level testing covers the substance).

    This validates the 'default-off' contract of Phase 3b:
    doc/plans/2026-06-06-symbolic-heat-retrieval.md §Phase 3b.
    """

    def _random_W_and_result(self, K=6, D=3, seed=0):
        torch.manual_seed(seed)
        W = torch.rand(K, D) * 0.8 + 0.1   # all positive, well-scaled
        result = torch.rand(1, 1, D) * 0.6 + 0.2   # shape [B=1, N=1, D]
        return W, result

    def test_disjunction_none_equals_unity(self):
        """unionReverse with no priming == all-1.0 priming."""
        W, result = self._random_W_and_result(seed=10)
        K = W.shape[0]
        ones = torch.ones(K)
        x1_none, x2_none = Ops.unionReverse(result, result, W)
        x1_ones, x2_ones = Ops.unionReverse(
            result, result, W,
            left_priming=ones, right_priming=ones)
        assert torch.equal(x1_none, x1_ones), (
            f"x1 differs: {x1_none} vs {x1_ones}")
        assert torch.equal(x2_none, x2_ones), (
            f"x2 differs: {x2_none} vs {x2_ones}")

    def test_conjunction_none_equals_unity(self):
        """intersectionReverse with no priming == all-1.0 priming."""
        W, result = self._random_W_and_result(seed=20)
        K = W.shape[0]
        ones = torch.ones(K)
        # For conjunction, result elements must be <= 1.0 and W must be >= result.
        # Scale W up to be feasible (>= result).
        W_feas = W.clamp(min=float(result.max().item()))
        x1_none, x2_none = Ops.intersectionReverse(result, result, W_feas)
        x1_ones, x2_ones = Ops.intersectionReverse(
            result, result, W_feas,
            left_priming=ones, right_priming=ones)
        assert torch.equal(x1_none, x1_ones), (
            f"x1 differs: {x1_none} vs {x1_ones}")
        assert torch.equal(x2_none, x2_ones), (
            f"x2 differs: {x2_none} vs {x2_ones}")

    def test_disjunction_none_rows_equals_unity_rows(self):
        """None priming with rows restriction == unity priming with same rows."""
        W, result = self._random_W_and_result(seed=30)
        K = W.shape[0]
        rows = torch.tensor([0, 2, 4], dtype=torch.long)
        ones = torch.ones(K)
        x1_none, x2_none = Ops.unionReverse(
            result, result, W, left_rows=rows, right_rows=rows)
        x1_ones, x2_ones = Ops.unionReverse(
            result, result, W,
            left_rows=rows, right_rows=rows,
            left_priming=ones, right_priming=ones)
        assert torch.equal(x1_none, x1_ones)
        assert torch.equal(x2_none, x2_ones)

class TestHeatContentPrefersExpectedCandidate:
    """Plan test 9: a boosted priming vector must make unionReverse
    select the primed row (for x1 / the 'argmax' direction), rather than
    the un-primed winner selected without priming.

    Construction mirrors ``test_priming_lifts_argmax_choice_union_x1`` in
    test_primed_reverse_combined.py:
      - W has two rows that are both feasible (both <= result element-wise).
      - Row 0 has a slightly smaller norm → loses the unprimed argmax.
      - Row 1 has a slightly larger norm → wins unprimed.
      - Boost row 0 by factor 2 so its primed score > row 1's un-primed score.
      - Assert x1_unprimed == W[1] and x1_primed == W[0].

    This validates the path
      retrieval_candidates_for_slot → priming vector → unionReverse
    returns the hot candidate (the simulated Phase-4 output of forward heat).

    doc/plans/2026-06-06-symbolic-heat-retrieval.md §Phase 3b test 9.
    """

    def test_primed_row_wins_over_unprimed_in_disjunction(self):
        """Hot-primed row 0 beats the otherwise-winning row 1 in unionReverse."""
        # Row 0: norm ≈ 0.50, slightly smaller than row 1 but still <= result.
        # Row 1: norm ≈ 0.515, slightly larger, unprimed winner.
        W = torch.tensor([
            [0.40, 0.30],   # norm ≈ 0.500
            [0.41, 0.31],   # norm ≈ 0.515
        ])
        result = torch.tensor([[[0.5, 0.4]]])

        # Without priming: row 1's larger norm wins the argmax-≤-result step.
        x1_un, _ = Ops.unionReverse(result, result, W)
        assert torch.allclose(x1_un[0, 0], W[1]), (
            "Sanity: unprimed pick should be row 1 (larger norm)")

        # Build priming that boosts row 0 to score 0.50 * 2.0 = 1.00,
        # which beats row 1's 0.515 * 1.0 = 0.515.
        left_priming = torch.tensor([2.0, 1.0])
        x1_pr, _ = Ops.unionReverse(
            result, result, W, left_priming=left_priming)
        assert torch.allclose(x1_pr[0, 0], W[0]), (
            f"Expected primed pick to be W[0]={W[0]}, got {x1_pr[0,0]}")

    def test_primed_row_wins_over_unprimed_in_conjunction(self):
        """Hot-primed row 0 beats otherwise-winning row 1 in intersectionReverse.

        For conjunction the inverse is argmin-≥-result (smallest feasible row);
        priming reduces effective norm by dividing, so a boosted row with
        a slightly larger raw norm can be selected over the un-primed winner.
        """
        W = torch.tensor([
            [0.80, 0.70],   # norm ≈ 1.063, slightly larger
            [0.75, 0.65],   # norm ≈ 0.991, smaller → wins unprimed
        ])
        result = torch.tensor([[[0.7, 0.6]]])

        # Without priming: row 1 (smaller norm) wins.
        x1_un, _ = Ops.intersectionReverse(result, result, W)
        assert torch.allclose(x1_un[0, 0], W[1]), (
            "Sanity: unprimed pick should be row 1 (smaller norm)")

        # Priming row 0 with factor 2: effective score = 1.063 / 2.0 = 0.5315
        # vs row 1's 0.991 / 1.0 = 0.991 → primed row 0 now has smaller
        # effective norm → argmin selects it.
        left_priming = torch.tensor([2.0, 1.0])
        x1_pr, _ = Ops.intersectionReverse(
            result, result, W, left_priming=left_priming)
        assert torch.allclose(x1_pr[0, 0], W[0]), (
            f"Expected primed pick to be W[0]={W[0]}, got {x1_pr[0,0]}")
