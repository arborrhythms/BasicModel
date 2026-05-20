"""Tests for restricted-candidate inverse recommender.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Lift/Lower Restricted-Candidate Inverse.

``Ops._binary_op_recommend`` gains ``left_rows`` / ``right_rows``
kwargs — ``LongTensor`` of W-row indices the recommender may pick from
for x1 / x2 respectively. The ⊥ / ⊤ sentinels remain feasible
regardless of restriction. When both are ``None`` behavior is unchanged
from before.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest


def _toy_codebook():
    """A 4-row bivector codebook used across tests.

      W[0] = [0.0, 0.0]    (≈ ⊥-like but learned)
      W[1] = [0.5, 0.0]
      W[2] = [0.5, 0.5]
      W[3] = [1.0, 1.0]    (≈ ⊤-like but learned)
    """
    import torch
    return torch.tensor([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])


def test_default_kwargs_match_unrestricted_baseline():
    """Passing ``left_rows=None, right_rows=None`` exactly matches the
    unrestricted call shape (regression: no behavior change for existing
    callers)."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    y = torch.tensor([[0.5, 0.5]])
    x1a, x2a = Ops._binary_op_recommend(y, W, 'union')
    x1b, x2b = Ops._binary_op_recommend(
        y, W, 'union', left_rows=None, right_rows=None)
    assert torch.allclose(x1a, x1b)
    assert torch.allclose(x2a, x2b)


def test_left_rows_restricts_x1_to_subset():
    """When ``left_rows = [1]`` (only allow W[1] for x1), the returned x1
    is either W[1] or a sentinel — never W[0]/W[2]/W[3]."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    y = torch.tensor([[0.5, 0.5]])
    # left_rows restricts: only allow W-row index 1 (or sentinels)
    x1, x2 = Ops._binary_op_recommend(
        y, W, 'union', left_rows=torch.tensor([1], dtype=torch.long))
    # x1 should be W[1] (= [0.5, 0.0]) or a sentinel ([0,0] or [1,1])
    valid_choices = [
        torch.tensor([0.0, 0.0]),  # ⊥
        torch.tensor([0.5, 0.0]),  # W[1]
        torch.tensor([1.0, 1.0]),  # ⊤
    ]
    matched = any(torch.allclose(x1[0], v) for v in valid_choices)
    assert matched, f"x1={x1[0]} not in restricted+sentinel set"


def test_right_rows_restricts_x2_to_subset():
    """``right_rows = [2]`` restricts x2 to W[2] (or sentinels)."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    y = torch.tensor([[0.5, 0.5]])
    x1, x2 = Ops._binary_op_recommend(
        y, W, 'union', right_rows=torch.tensor([2], dtype=torch.long))
    valid_choices = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([1.0, 1.0]),
    ]
    matched = any(torch.allclose(x2[0], v) for v in valid_choices)
    assert matched, f"x2={x2[0]} not in restricted+sentinel set"


def test_both_restricted_picks_from_respective_subsets():
    """``left_rows`` and ``right_rows`` set independently."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    y = torch.tensor([[0.5, 0.5]])
    x1, x2 = Ops._binary_op_recommend(
        y, W, 'intersection',
        left_rows=torch.tensor([3], dtype=torch.long),   # only W[3] for x1
        right_rows=torch.tensor([2], dtype=torch.long))  # only W[2] for x2
    # x1 must be W[3] or sentinel; x2 must be W[2] or sentinel
    valid_left = [torch.tensor([0.0, 0.0]),
                  torch.tensor([1.0, 1.0])]
    valid_right = [torch.tensor([0.0, 0.0]),
                   torch.tensor([0.5, 0.5]),
                   torch.tensor([1.0, 1.0])]
    assert any(torch.allclose(x1[0], v) for v in valid_left)
    assert any(torch.allclose(x2[0], v) for v in valid_right)


def test_empty_left_rows_falls_back_to_sentinels():
    """``left_rows = []`` (no learned candidates) → x1 must be a
    sentinel (⊥ or ⊤)."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    y = torch.tensor([[0.5, 0.5]])
    x1, _ = Ops._binary_op_recommend(
        y, W, 'union',
        left_rows=torch.empty(0, dtype=torch.long))
    sentinels = [torch.tensor([0.0, 0.0]),
                 torch.tensor([1.0, 1.0])]
    assert any(torch.allclose(x1[0], s) for s in sentinels)


def test_intersection_respects_restriction():
    """Same restriction semantics in the intersection branch as in union."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    y = torch.tensor([[0.25, 0.25]])
    x1, x2 = Ops._binary_op_recommend(
        y, W, 'intersection',
        left_rows=torch.tensor([1], dtype=torch.long))
    valid_left = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([1.0, 1.0]),
    ]
    assert any(torch.allclose(x1[0], v) for v in valid_left)


def test_sentinels_always_feasible():
    """Even with both restrictions to a single row, the recommender
    can still pick ⊥ or ⊤ as a fallback when nothing else matches."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    # Pick a y such that x1 = W[3] doesn't satisfy x1 ≤ y for union
    y = torch.tensor([[0.1, 0.1]])
    # Only W[3] = [1.0, 1.0] allowed, which is NOT ≤ y. ⊥ fallback.
    x1, _ = Ops._binary_op_recommend(
        y, W, 'union',
        left_rows=torch.tensor([3], dtype=torch.long))
    sentinels = [torch.tensor([0.0, 0.0]),
                 torch.tensor([1.0, 1.0])]
    matched = any(torch.allclose(x1[0], s) for s in sentinels)
    # Either bottom sentinel (0,0) which ≤ y, or W[3] couldn't be picked
    # so fell through to sentinel
    assert matched, f"x1={x1[0]} (expected sentinel since W[3] > y)"


def test_kwarg_forwarding_through_disjunctionReverse():
    """Public ``Ops.disjunctionReverse`` forwards left_rows/right_rows
    to the underlying implementation."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    y = torch.tensor([[0.5, 0.5]])
    # Without restriction
    x1a, x2a = Ops.disjunctionReverse(y, y, W)
    # With restriction
    x1b, x2b = Ops.disjunctionReverse(
        y, y, W, left_rows=torch.tensor([1], dtype=torch.long))
    # Restricted call's x1 should be in the restricted set ∪ sentinels
    valid = [torch.tensor([0.0, 0.0]),
             torch.tensor([0.5, 0.0]),
             torch.tensor([1.0, 1.0])]
    assert any(torch.allclose(x1b[0], v) for v in valid)


def test_kwarg_forwarding_through_conjunctionReverse():
    """``Ops.conjunctionReverse`` forwards the kwargs too."""
    from Layers import Ops
    import torch
    W = _toy_codebook()
    y = torch.tensor([[0.5, 0.5]])
    x1a, x2a = Ops.conjunctionReverse(
        y, y, W, left_rows=torch.tensor([2], dtype=torch.long))
    valid = [torch.tensor([0.0, 0.0]),
             torch.tensor([0.5, 0.5]),
             torch.tensor([1.0, 1.0])]
    assert any(torch.allclose(x1a[0], v) for v in valid)


# -- End-to-end: KnowledgeView drives the candidate masks --------------
# Plan: §Lift/Lower Restricted-Candidate Inverse — the call-site
# pattern. Build an artifact, load its view, intersect
# refs_by_category[cat] with refs_by_order[k] to produce per-operand
# masks, then call disjunctionReverse with those masks.


def _tiny_grammar():
    """Same grammar shape used in the phase2 e2e test."""
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", tier='S'),
        g._parse_rule("NP3", "lower(DET, NP4)", tier='S'),
        g._parse_rule("S3", "not(S3)", tier='S'),
    ]
    g._configured = True
    return g


def _intersect_long_tensors(a, b):
    """Set intersection of two LongTensor row-id lists."""
    import torch
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    sa, sb = set(a.tolist()), set(b.tolist())
    return torch.tensor(sorted(sa & sb), dtype=torch.long)


def test_typed_mask_from_knowledge_view_drives_recommender(tmp_path):
    """Full pipeline: artifact → load_knowledge_view → compute
    refs_by_category ∩ refs_by_order from the view → pass to
    disjunctionReverse → verify x1 obeys the restriction."""
    from embed import (save_artifact, build_knowledge_section,
                       load_knowledge_view)
    from Layers import Ops
    import torch
    path = str(tmp_path / "phase3.kv")
    save_artifact(path, knowledge=build_knowledge_section(_tiny_grammar()))
    view = load_knowledge_view(path)
    # The bootstrap has each class node at order 0; restrict by
    # category alone (intersect with refs_by_order(0)).
    np_refs = view.refs_by_category('NP')
    order_zero = view.refs_by_order(0)
    left_rows = _intersect_long_tensors(np_refs, order_zero)
    # The intersection should be exactly the NP class node (1 ref)
    assert left_rows.numel() == 1
    # Build a small bivector W indexed to match the artifact's ref_ids
    # (so left_rows points at real W rows).
    W = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [0.7, 0.7],
    ])[:view.n_refs_live]
    y = torch.tensor([[0.5, 0.5]])
    x1, x2 = Ops.disjunctionReverse(y, y, W, left_rows=left_rows)
    # x1 should be either W[left_rows[0]] or a sentinel ([0,0] / [1,1])
    np_ref_idx = int(left_rows[0].item())
    sentinels = [torch.tensor([0.0, 0.0]),
                 torch.tensor([1.0, 1.0])]
    valid_x1 = [W[np_ref_idx]] + sentinels
    assert any(torch.allclose(x1[0], v) for v in valid_x1)


def test_typed_mask_empty_intersection_falls_back_to_sentinels(tmp_path):
    """When refs_by_category ∩ refs_by_order is empty (no refs at
    that category & order in the bootstrap), the recommender falls
    back to ⊥ / ⊤ sentinels."""
    from embed import (save_artifact, build_knowledge_section,
                       load_knowledge_view)
    from Layers import Ops
    import torch
    path = str(tmp_path / "phase3_empty.kv")
    save_artifact(path, knowledge=build_knowledge_section(_tiny_grammar()))
    view = load_knowledge_view(path)
    # No NP refs at order 4 in the bootstrap.
    left_rows = _intersect_long_tensors(
        view.refs_by_category('NP'), view.refs_by_order(4))
    assert left_rows.numel() == 0
    W = torch.tensor([
        [0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [1.0, 1.0], [0.7, 0.7],
    ])[:view.n_refs_live]
    y = torch.tensor([[0.3, 0.3]])
    x1, _ = Ops.disjunctionReverse(y, y, W, left_rows=left_rows)
    sentinels = [torch.tensor([0.0, 0.0]),
                 torch.tensor([1.0, 1.0])]
    assert any(torch.allclose(x1[0], s) for s in sentinels)
