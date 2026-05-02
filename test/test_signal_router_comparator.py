import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from SignalRouter import ComparatorMixer


def _gather_branches(x, reduced):
    """Build [B, N, 4, D] where branches per j are
       (keep=x_j, reduce=r_j, shift=x_{j+1}, pad=0)."""
    B, N, D = x.shape
    pad = x.new_zeros(B, 1, D)
    x_shift_right = torch.cat([x[:, 1:, :], pad], dim=1)     # x_{j+1} with last=pad
    r_padded = torch.cat([reduced, pad], dim=1)              # r_j; last=pad
    return torch.stack([x, r_padded, x_shift_right, pad.expand_as(x)], dim=2)


def test_comparator_output_shape():
    B, N, D = 2, 5, 6
    cm = ComparatorMixer(d_model=D)
    x = torch.randn(B, N, D)
    reduced = torch.randn(B, N - 1, D)
    branches = _gather_branches(x, reduced)
    h = torch.randn(B, N, D)
    y, gates = cm(h=h, branches=branches)
    assert y.shape == (B, N, D)
    assert gates.shape == (B, N, 4)
    assert torch.allclose(gates.sum(-1), torch.ones(B, N), atol=1e-5)


def test_comparator_gradient_into_each_branch_and_into_h():
    B, N, D = 1, 4, 5
    cm = ComparatorMixer(d_model=D)
    x = torch.randn(B, N, D, requires_grad=True)
    reduced = torch.randn(B, N - 1, D, requires_grad=True)
    h = torch.randn(B, N, D, requires_grad=True)
    branches = _gather_branches(x, reduced)
    y, _ = cm(h=h, branches=branches)
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert reduced.grad is not None and reduced.grad.abs().sum() > 0
    assert h.grad is not None and h.grad.abs().sum() > 0


def test_comparator_temperature_sharpens_gates():
    B, N, D = 1, 4, 4
    torch.manual_seed(0)
    cm_hot = ComparatorMixer(d_model=D, temperature=10.0)
    cm_cold = ComparatorMixer(d_model=D, temperature=0.1)
    cm_cold.load_state_dict(cm_hot.state_dict())
    x = torch.randn(B, N, D)
    reduced = torch.randn(B, N - 1, D)
    h = torch.randn(B, N, D)
    branches = _gather_branches(x, reduced)
    _, g_hot = cm_hot(h=h, branches=branches)
    _, g_cold = cm_cold(h=h, branches=branches)
    # Cold (low T) ~ more peaked; max gate larger on average.
    assert g_cold.max(dim=-1).values.mean() > g_hot.max(dim=-1).values.mean()
