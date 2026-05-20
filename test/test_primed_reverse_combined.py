"""Tests for the priming-weighted inverse recommender (Phase B).

Plan: doc/plans/2026-05-20-primed-reverse-generation.md
§Application — A. selection time.

The priming weights are float boost-above-unity values per W-row:
  * 1.0 = multiplicative identity (default, no preference)
  * 2.0 = active ref (preferred by selection)
  * sentinels ⊥ / ⊤ are pinned to 1.0 regardless of input

Selection direction:
  * argmax steps (union x1 by norm) → multiply score by priming
  * argmin steps (union x2, intersection x1/x2) → divide by priming
Both directions preserve the invariant "higher priming → more likely
selected". With all-1.0 priming the result matches the un-primed
algorithm byte-for-byte.
"""
import sys
from pathlib import Path

import pytest
import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))

from Layers import Ops  # noqa: E402


# -- Identity priming preserves un-primed behavior --------------------------


def test_identity_priming_matches_unprimed_disjunction():
    """All-1.0 priming yields the same (x1, x2) as the un-primed call."""
    torch.manual_seed(0)
    W = torch.tensor([
        [0.2, 0.1],
        [0.6, 0.3],
        [0.4, 0.7],
    ])
    y = torch.tensor([[[0.5, 0.4]]])
    ones = torch.ones(W.shape[0])
    x1_a, x2_a = Ops.disjunctionReverse(y, y, W)
    x1_b, x2_b = Ops.disjunctionReverse(
        y, y, W, left_priming=ones, right_priming=ones)
    assert torch.equal(x1_a, x1_b)
    assert torch.equal(x2_a, x2_b)


def test_identity_priming_matches_unprimed_conjunction():
    """All-1.0 priming yields the same (x1, x2) for intersection too."""
    torch.manual_seed(0)
    W = torch.tensor([
        [0.7, 0.6],
        [0.8, 0.9],
        [0.95, 0.85],
    ])
    y = torch.tensor([[[0.6, 0.55]]])
    ones = torch.ones(W.shape[0])
    x1_a, x2_a = Ops.conjunctionReverse(y, y, W)
    x1_b, x2_b = Ops.conjunctionReverse(
        y, y, W, left_priming=ones, right_priming=ones)
    assert torch.equal(x1_a, x1_b)
    assert torch.equal(x2_a, x2_b)


def test_none_priming_matches_identity_priming():
    """None priming (default) is equivalent to passing all-1.0 weights."""
    W = torch.tensor([[0.3, 0.4], [0.5, 0.6]])
    y = torch.tensor([[[0.5, 0.55]]])
    x1_a, x2_a = Ops.disjunctionReverse(
        y, y, W, left_priming=None, right_priming=None)
    x1_b, x2_b = Ops.disjunctionReverse(
        y, y, W, left_priming=torch.ones(2), right_priming=torch.ones(2))
    assert torch.equal(x1_a, x1_b)
    assert torch.equal(x2_a, x2_b)


# -- Priming biases argmax selection (union x1) ----------------------------


def test_priming_lifts_argmax_choice_union_x1():
    """Among equally feasible candidates for union x1 (largest ≤ y), a
    primed one is preferred — even when an un-primed competitor has a
    slightly larger raw norm."""
    # Two W-rows with very close norms; both ≤ y elementwise.
    W = torch.tensor([
        [0.40, 0.30],   # norm ≈ 0.50
        [0.41, 0.31],   # norm ≈ 0.515 (slightly larger)
    ])
    y = torch.tensor([[[0.5, 0.4]]])
    # Without priming: row 1 is the largest ≤ y → x1 == W[1].
    x1_un, _ = Ops.disjunctionReverse(y, y, W)
    assert torch.allclose(x1_un[0, 0], W[1])
    # Prime row 0 with boost 1.0 → its score becomes 0.50 * 2.0 = 1.00,
    # which beats row 1's 0.515 * 1.0 = 0.515. Row 0 wins.
    priming = torch.tensor([2.0, 1.0])
    x1_pr, _ = Ops.disjunctionReverse(y, y, W, left_priming=priming)
    assert torch.allclose(x1_pr[0, 0], W[0])


# -- Priming biases argmin selection (intersection x1) ---------------------


def test_priming_lifts_argmin_choice_intersection_x1():
    """Among equally feasible candidates for intersection x1 (smallest
    ≥ y), priming makes an un-preferred candidate look smaller and
    thus more likely chosen."""
    # Two W-rows ≥ y; row 1 has slightly smaller norm and would win
    # un-primed; row 0 has a slightly larger norm but is primed enough
    # that its effective (norm / priming) drops below row 1's.
    W = torch.tensor([
        [0.80, 0.70],   # norm ≈ 1.063
        [0.75, 0.65],   # norm ≈ 0.991 (smaller, would win)
    ])
    y = torch.tensor([[[0.7, 0.6]]])
    x1_un, _ = Ops.conjunctionReverse(y, y, W)
    assert torch.allclose(x1_un[0, 0], W[1])
    # Prime row 0 with boost 1.0 → effective norm 1.063 / 2.0 = 0.5315,
    # which beats row 1's 0.991 / 1.0 = 0.991. Row 0 wins.
    priming = torch.tensor([2.0, 1.0])
    x1_pr, _ = Ops.conjunctionReverse(y, y, W, left_priming=priming)
    assert torch.allclose(x1_pr[0, 0], W[0])


# -- Hard mask trumps priming ----------------------------------------------


def test_hard_mask_blocks_priming_in_disjunction():
    """A row excluded by left_rows is never chosen, even with priming.

    Setup: row 0 is the only un-restricted W-row eligible by feasibility.
    Restricting left_rows=[1] makes row 1 the only admissible W-row;
    even if we crank priming on row 0 to ridiculous values, the result
    must be drawn from {⊥, W[1], ⊤}.
    """
    W = torch.tensor([
        [0.40, 0.30],
        [0.20, 0.10],
    ])
    y = torch.tensor([[[0.5, 0.4]]])
    left_rows = torch.tensor([1], dtype=torch.long)
    priming = torch.tensor([99.0, 1.0])
    x1, _ = Ops.disjunctionReverse(
        y, y, W, left_rows=left_rows, left_priming=priming)
    # x1 must be ⊥ (zeros), W[1], or ⊤ (ones). Row 0 is excluded.
    bottom = torch.zeros(2)
    top = torch.ones(2)
    chosen = x1.reshape(2)
    assert (torch.allclose(chosen, bottom)
            or torch.allclose(chosen, W[1])
            or torch.allclose(chosen, top))


def test_hard_mask_blocks_priming_in_intersection():
    """Same invariant for intersection: hard mask wins over priming."""
    W = torch.tensor([
        [0.95, 0.85],
        [0.80, 0.70],
    ])
    y = torch.tensor([[[0.7, 0.6]]])
    left_rows = torch.tensor([1], dtype=torch.long)
    priming = torch.tensor([99.0, 1.0])
    x1, _ = Ops.conjunctionReverse(
        y, y, W, left_rows=left_rows, left_priming=priming)
    bottom = torch.zeros(2)
    top = torch.ones(2)
    chosen = x1.reshape(2)
    assert (torch.allclose(chosen, bottom)
            or torch.allclose(chosen, W[1])
            or torch.allclose(chosen, top))


# -- Sentinel pinning ------------------------------------------------------


def test_priming_does_not_apply_to_sentinels():
    """⊥ / ⊤ sentinels are pinned to priming = 1.0; absurd W-row
    priming values do not change the sentinel scores.

    Construct a case where ⊤ is the un-primed winner for union x2
    (smallest ≥ residual). Even with huge priming on every W-row,
    ⊤'s effective score should match its un-primed score.
    """
    W = torch.tensor([
        [0.05, 0.05],
        [0.10, 0.10],
    ])
    # Choose y so no W-row satisfies (S ≥ r) — forces x2 = ⊤.
    y = torch.tensor([[[0.99, 0.99]]])
    huge_priming = torch.tensor([1e9, 1e9])
    _, x2 = Ops.disjunctionReverse(
        y, y, W, right_priming=huge_priming)
    # Un-primed result for the same setup:
    _, x2_un = Ops.disjunctionReverse(y, y, W)
    # Both should be the same sentinel (⊤) because priming can't
    # rescue an infeasible W-row.
    assert torch.equal(x2, x2_un)


# -- Empty / edge cases ----------------------------------------------------


def test_empty_priming_tensor_treated_as_identity():
    """An empty FloatTensor priming behaves like all-1.0."""
    W = torch.tensor([[0.4, 0.3], [0.5, 0.4]])
    y = torch.tensor([[[0.5, 0.4]]])
    empty = torch.empty(0)
    x1_a, x2_a = Ops.disjunctionReverse(y, y, W, left_priming=empty)
    x1_b, x2_b = Ops.disjunctionReverse(y, y, W)
    assert torch.equal(x1_a, x1_b)
    assert torch.equal(x2_a, x2_b)


def test_short_priming_tensor_pads_to_identity():
    """A priming tensor with fewer entries than K only weights the
    first ``len(priming)`` W-rows; the rest stay at 1.0."""
    W = torch.tensor([
        [0.40, 0.30],
        [0.41, 0.31],
    ])
    y = torch.tensor([[[0.5, 0.4]]])
    # Only the first row is supplied a priming weight.
    short = torch.tensor([2.0])
    x1_a, _ = Ops.disjunctionReverse(y, y, W, left_priming=short)
    x1_b, _ = Ops.disjunctionReverse(
        y, y, W, left_priming=torch.tensor([2.0, 1.0]))
    assert torch.equal(x1_a, x1_b)
