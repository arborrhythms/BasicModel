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

from Layers import MeronymicFoldAdapter, Ops  # noqa: E402
from Spaces import PartSpace  # noqa: E402


def _expected_union(codes, mask):
    membership = Ops.eval_chart(codes.clamp(-1.0, 1.0))
    complement = torch.where(
        mask.unsqueeze(-1), 1.0 - membership,
        torch.ones_like(membership))
    out = Ops.eval_chart_inv(1.0 - complement.prod(dim=-2))
    return torch.where(
        mask.any(dim=-1, keepdim=True), out, torch.zeros_like(out))


@pytest.mark.parametrize("butterfly", [False, True])
def test_base_set_aggregation_is_masked_de_morgan_and_permutation_invariant(
        butterfly):
    fold = MeronymicFoldAdapter(
        "sigma", 8, 8, stable=True, butterfly=butterfly, legacy_N=8)
    codes = torch.tensor([
        [
            [-0.8, -0.6, -0.4, -0.2, 0.1, 0.3, 0.5, 0.7],
            [0.6, 0.4, 0.2, 0.0, -0.1, -0.3, -0.5, -0.7],
            [0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6],
        ],
        [
            [0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5],
            [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
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


@pytest.mark.parametrize("butterfly", [False, True])
def test_base_set_aggregation_never_executes_learned_kernel(
        butterfly, monkeypatch):
    fold = MeronymicFoldAdapter(
        "sigma", 8, 8, stable=True, butterfly=butterfly, legacy_N=8)
    codes = torch.randn(2, 3, 8).tanh().requires_grad_()
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


def test_partspace_word_synthesis_selects_parameter_free_base_surface():
    class _Sigma:
        nInput = 4

        def __init__(self):
            self.calls = 0

        def aggregate_over_set(self, values, mask=None):
            self.calls += 1
            return values.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=-2)

        def synthesize_over_set(self, *_args, **_kwargs):
            raise AssertionError("word base used the learned set-fold surface")

    class _PartHost:
        def __init__(self, events, sigma):
            self.events = events
            self.sigma = sigma

        def _radix_part_events(self, _part_ids, _part_offsets):
            return self.events

        def _sigma_for_pass(self):
            return self.sigma

    sigma = _Sigma()
    events = torch.tensor([[[1., 2., 3., 4., 10., 11.],
                            [5., 6., 7., 8., 20., 21.],
                            [9., 9., 9., 9., 30., 31.]]])
    mask = torch.tensor([[True, False, True]])
    host = _PartHost(events, sigma)
    ids = torch.tensor([[0, 1, 2]])

    out = PartSpace.synthesize_word_parts(host, ids, mask)
    assert sigma.calls == 1
    assert torch.equal(out[..., :4], torch.tensor([[[10., 11., 12., 13.]]]))
    # The positional band comes from the first active constituent only.
    assert torch.equal(out[..., 4:], torch.tensor([[[10., 11.]]]))


def test_word_base_then_three_learned_sigma_rungs_execute_once_each():
    class _CountingSigma(torch.nn.Module):
        nInput = 4
        N = 0

        def __init__(self, scale):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
            self.aggregate_calls = 0
            self.forward_calls = 0

        def aggregate_over_set(self, values, mask=None):
            self.aggregate_calls += 1
            # Deliberately parameter-free: ``scale`` belongs only to forward.
            return values.masked_fill(
                ~mask.unsqueeze(-1), 0.0).sum(dim=-2)

        def forward(self, values):
            self.forward_calls += 1
            return values * self.scale

    class _PartHost:
        _synthesize_event = PartSpace._synthesize_event

        def __init__(self, events, sigmas):
            self.events = events
            self.sigmas = sigmas

        def _radix_part_events(self, _part_ids, _part_offsets):
            return self.events

        def _sigma_for_pass(self):
            return self.sigmas[0]

    sigmas = [_CountingSigma(1.1 + 0.1 * i) for i in range(4)]
    events = torch.tensor([[[1., 2., 3., 4., 10., 11.],
                            [5., 6., 7., 8., 20., 21.]]])
    mask = torch.tensor([[True, True]])
    host = _PartHost(events, sigmas)

    base = PartSpace.synthesize_word_parts(
        host, torch.tensor([[0, 1]]), mask)
    folds = PartSpace.fold_event_ladder(
        host, base, (0, 1, 2), strict=True)
    folds[-1].square().sum().backward()

    assert [fold.aggregate_calls for fold in sigmas] == [1, 0, 0, 0]
    assert [fold.forward_calls for fold in sigmas] == [1, 1, 1, 0]
    assert all(fold.scale.grad is not None
               and bool((fold.scale.grad != 0).all())
               for fold in sigmas[:3])
    # Rung 3 remains registered for checkpoint compatibility, but T=4's
    # aligned base-plus-three-fold path does not advertise or execute it.
    assert sigmas[3].scale.grad is None
