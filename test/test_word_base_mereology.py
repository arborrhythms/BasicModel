"""Parameter-free word-base aggregation before learned sigma recursion."""

from __future__ import annotations

import os
import sys

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import MeronymicFoldAdapter  # noqa: E402
from Spaces import ConceptualSpace, PartSpace  # noqa: E402


def _expected_union(codes, mask):
    complement = torch.where(
        mask.unsqueeze(-1), 1.0 - codes,
        torch.ones_like(codes))
    out = 1.0 - complement.prod(dim=-2)
    return torch.where(
        mask.any(dim=-1, keepdim=True), out, torch.zeros_like(out))


@pytest.mark.parametrize("butterfly", [False, True])
def test_base_set_aggregation_is_masked_de_morgan_and_permutation_invariant(
        butterfly):
    fold = MeronymicFoldAdapter(
        "sigma", 8, 8, stable=True, butterfly=butterfly, legacy_N=8)
    codes = torch.tensor([
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
        ],
        [
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        ],
    ], dtype=torch.float32)
    mask = torch.tensor([[True, False, True], [False, False, False]])

    got = fold.aggregate_over_set(codes, mask=mask)
    expected = _expected_union(codes, mask)
    assert torch.allclose(got, expected, atol=1e-7, rtol=0.0)

    permutation = torch.tensor([2, 0, 1])
    permuted = fold.aggregate_over_set(
        codes[:, permutation], mask=mask[:, permutation])
    assert torch.allclose(permuted, got, atol=1e-7, rtol=0.0)
    assert torch.equal(got[1], torch.zeros_like(got[1]))
    assert torch.all((got >= 0.0) & (got <= 1.0))


@pytest.mark.parametrize("butterfly", [False, True])
def test_base_set_aggregation_preserves_zero_union_identity(butterfly):
    fold = MeronymicFoldAdapter(
        "sigma", 8, 8, stable=True, butterfly=butterfly, legacy_N=8)
    codes = torch.zeros(2, 3, 8)
    mask = torch.tensor([[True, True, True], [True, False, True]])

    got = fold.aggregate_over_set(codes, mask=mask)

    assert torch.equal(got, torch.zeros_like(got))


@pytest.mark.parametrize("butterfly", [False, True])
def test_concept_evidence_matches_identified_codes_after_complete_part_union(butterfly):
    """Concept evidence is measured after, never before, word synthesis."""
    fold = MeronymicFoldAdapter(
        "sigma", 4, 4, stable=True, butterfly=butterfly, legacy_N=4)
    parts = torch.tensor([[[0.25, 0.00, 0.50, 0.00],
                           [0.00, 0.50, 0.50, 1.00],
                           [1.00, 0.00, 0.00, 0.00]]])
    mask = torch.tensor([[True, True, True]])

    union = fold.aggregate_over_set(parts, mask=mask)
    activation = ConceptualSpace.matched_code_evidence(
        union.unsqueeze(1), parts, n_what=4)
    expected_union = 1.0 - (1.0 - parts).prod(dim=-2)
    distance = (expected_union.unsqueeze(1) - parts).square().sum(-1)
    expected = (1 - distance / parts.square().sum(-1)).clamp(-1, 1).unsqueeze(1)

    torch.testing.assert_close(union, expected_union)
    torch.testing.assert_close(activation, expected)
    assert activation.shape == (1, 1, 3)  # three identified references, not one RMS
    assert bool((activation >= -1.0).all())
    assert bool((activation <= 1.0).all())


@pytest.mark.parametrize("butterfly", [False, True])
def test_base_set_aggregation_never_executes_learned_kernel(
        butterfly, monkeypatch):
    fold = MeronymicFoldAdapter(
        "sigma", 8, 8, stable=True, butterfly=butterfly, legacy_N=8)
    codes = torch.rand(2, 3, 8).requires_grad_()
    mask = torch.tensor([[True, True, False], [True, False, True]])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("base aggregation entered a learned fold")

    if butterfly:
        monkeypatch.setattr(fold, "_mem_cascade", forbidden)
    else:
        monkeypatch.setattr(fold.fold, "forward", forbidden)

    out = fold.aggregate_over_set(codes, mask=mask)
    out.square().sum().backward()
    assert codes.grad is not None
    assert bool((codes.grad[mask] != 0).any())
    assert all(parameter.grad is None for parameter in fold.parameters())



def test_partspace_word_code_is_max_and_anagrams_share_it():
    from types import SimpleNamespace
    events = torch.tensor([[[.1, .7, .25, -.5], [.8, .3, .75, .5]]])
    host = SimpleNamespace(nDim=2, _radix_part_events=lambda ids, offsets: events[:, ids[0]])
    mask = torch.tensor([[True, True]])
    forward = PartSpace.synthesize_word_parts(host, torch.tensor([[0, 1]]), mask)
    backward = PartSpace.synthesize_word_parts(host, torch.tensor([[1, 0]]), mask)
    torch.testing.assert_close(forward[..., :2], torch.tensor([[[.8, .7]]]))
    torch.testing.assert_close(forward[..., :2], backward[..., :2])
    torch.testing.assert_close(forward[..., 2:], events[:, :1, 2:])
    torch.testing.assert_close(backward[..., 2:], events[:, 1:, 2:])
