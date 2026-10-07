"""What spec Step 6: a tied invertible operator used in BOTH directions.

Section 11 "Loss and gradients": a tied invertible operator receives the
direct synthesis derivative (through ``reverse``) and the indirect derivative
through ``forward``; autograd must agree with finite differences on a small,
well-conditioned matrix, and the inverse must stay conditioned.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_BIN = Path(__file__).resolve().parent.parent / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Layers import InvertibleLinearLayer  # noqa: E402


def _tied_loss(layer, x, target_forward, target_reverse):
    # Analysis uses forward; synthesis uses the inverse of the SAME operator
    # on a state derived from the analysis (so both derivative paths exist).
    analysed = layer.forward(x)
    synthesized = layer.reverse(analysed * 1.5)
    return ((analysed - target_forward) ** 2).mean() + (
        (synthesized - target_reverse) ** 2).mean()


@pytest.mark.parametrize("n", [3, 5])
def test_autograd_matches_finite_differences_on_tied_operator(n):
    torch.manual_seed(0)
    layer = InvertibleLinearLayer(n, n, naive=False, ergodic=False).double()
    x = torch.randn(4, n, dtype=torch.float64)
    tf = torch.randn(4, n, dtype=torch.float64)
    tr = torch.randn(4, n, dtype=torch.float64)
    params = [p for p in layer.parameters() if p.requires_grad]
    assert params
    loss = _tied_loss(layer, x, tf, tr)
    grads = torch.autograd.grad(loss, params)
    eps = 1e-6
    for p, g in zip(params, grads):
        flat = p.detach().reshape(-1)
        # Sample a few coordinates per parameter.
        for idx in torch.linspace(0, flat.numel() - 1, steps=min(4, flat.numel())).long():
            with torch.no_grad():
                flat[idx] += eps
                up = float(_tied_loss(layer, x, tf, tr))
                flat[idx] -= 2 * eps
                down = float(_tied_loss(layer, x, tf, tr))
                flat[idx] += eps
            fd = (up - down) / (2 * eps)
            assert abs(fd - float(g.reshape(-1)[idx])) < 1e-5, (
                f"param {p.shape} idx {int(idx)}: fd={fd} autograd={float(g.reshape(-1)[idx])}")


def test_both_derivative_paths_are_live():
    torch.manual_seed(1)
    n = 4
    layer = InvertibleLinearLayer(n, n, naive=False, ergodic=False).double()
    x = torch.randn(2, n, dtype=torch.float64)
    analysed = layer.forward(x)
    synthesized = layer.reverse(analysed.detach() * 1.5)
    params = [p for p in layer.parameters() if p.requires_grad]
    direct = torch.autograd.grad(synthesized.sum(), params, allow_unused=True)
    indirect = torch.autograd.grad(analysed.sum(), params, allow_unused=True)
    assert any(g is not None and torch.count_nonzero(g) for g in direct)
    assert any(g is not None and torch.count_nonzero(g) for g in indirect)


def test_inverse_stays_conditioned_and_exact():
    torch.manual_seed(2)
    n = 6
    layer = InvertibleLinearLayer(n, n, naive=False, ergodic=False).double()
    x = torch.randn(3, n, dtype=torch.float64)
    y = layer.forward(x)
    back = layer.reverse(y)
    assert torch.allclose(back, x, atol=1e-8)
    W = layer.compute_W().detach()
    cond = torch.linalg.cond(W)
    assert torch.isfinite(cond) and float(cond) < 1e6
