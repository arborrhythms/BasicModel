import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from SignalRouter import (
    copy_penalty, length_penalty, comparator_dp_kl,
)


def test_copy_penalty_zero_when_no_copies():
    routing = {"copy_marginal": torch.zeros(2, 5)}
    loss = copy_penalty([routing], lambda_copy=1.0)
    assert float(loss) == 0.0


def test_copy_penalty_grows_with_copy_mass():
    a = {"copy_marginal": torch.full((2, 5), 0.2)}
    b = {"copy_marginal": torch.full((2, 5), 0.9)}
    assert float(copy_penalty([b], 1.0)) > float(copy_penalty([a], 1.0))


def test_length_penalty_grows_with_lengths():
    a = {"lengths": torch.tensor([2, 3])}
    b = {"lengths": torch.tensor([5, 5])}
    assert float(length_penalty([b], 1.0)) > float(length_penalty([a], 1.0))


def test_comparator_dp_kl_zero_when_gates_match_marginals():
    B, N = 1, 4
    p_copy = torch.tensor([[0.4, 0.3, 0.5, 0.6]])
    p_reduce = torch.tensor([[0.3, 0.0, 0.0]])
    cum = torch.cumsum(p_reduce, dim=1)
    cumshift = torch.cat([torch.zeros(B, 1), cum], dim=1)        # [B, N]
    keep = p_copy * (1.0 - cumshift.clamp(0.0, 1.0))
    reduce_w = torch.cat([p_reduce, torch.zeros(B, 1)], dim=1)
    shift = cumshift.clamp(0.0, 1.0)
    pad = (1.0 - keep - reduce_w - shift).clamp(min=1e-8)
    target = torch.stack([keep, reduce_w, shift, pad], dim=-1)
    target = target / target.sum(-1, keepdim=True)
    routing = {
        "gates": target.clone(),
        "copy_marginal": p_copy,
        "reduce_marginal": p_reduce,
    }
    kl = comparator_dp_kl([routing], lambda_dp_prior=1.0)
    assert float(kl) < 1e-4


def test_losses_are_differentiable():
    p_copy = torch.full((1, 4), 0.5, requires_grad=True)
    p_reduce = torch.full((1, 3), 0.25, requires_grad=True)
    gates = torch.full((1, 4, 4), 0.25, requires_grad=True)
    routing = {
        "copy_marginal": p_copy,
        "reduce_marginal": p_reduce,
        "gates": gates,
        "lengths": torch.tensor([3]),
    }
    loss = (copy_penalty([routing], 1e-3)
            + length_penalty([routing], 1e-4)
            + comparator_dp_kl([routing], 1e-2))
    loss.backward()
    assert p_copy.grad is not None and p_copy.grad.abs().sum() > 0
    assert gates.grad is not None and gates.grad.abs().sum() > 0
