"""The atanh / log-odds charts keep exact forward values and a bounded
backward slope (doc/Spaces.md, "Both charts are bounded in the backward";
doc/benchmarks/2026-09-11-fold-ladder-throughput.md, the packed-row NaN)."""
import math
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "bin") not in sys.path:
    sys.path.insert(0, str(_ROOT / "bin"))

import Layers  # noqa: E402


def _grad(fn, values):
    x = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    y = fn(x)
    y.sum().backward()
    return y.detach(), x.grad


def test_bounded_atanh_forward_is_exact_and_slope_is_capped():
    values = [-0.999, -0.5, 0.0, 0.5, 0.9, 0.99, 0.9999999, 1.0, 1.5]
    y, g = _grad(Layers.bounded_atanh, values)
    exact = torch.atanh(torch.tensor(values).clamp(-1 + Layers.epsilon, 1 - Layers.epsilon))
    assert torch.equal(y, exact)                                  # byte-identical forward
    x0 = Layers.ATANH_SLOPE_CAP_AT
    cap = 1.0 / (1.0 - x0 * x0)
    assert abs(float(g[2]) - 1.0) < 1e-6                          # d atanh / dx at 0 is 1
    assert abs(float(g[3]) - 1.0 / (1 - 0.25)) < 1e-5              # exact inside the cap
    assert all(abs(float(v) - cap) < 1e-4 for v in g[5:])         # capped beyond x0
    assert all(float(v) <= cap + 1e-4 for v in g)


def test_log_odds_chart_forward_is_exact_and_slope_is_capped():
    pi = Layers.PiLayer(nInput=4, nOutput=4, nonlinear=True)
    values = [-0.95, 0.0, 0.5, 0.9, 0.999, 1.0, 2.0]
    y, g = _grad(pi._log_mult, values)
    exact = torch.log(pi._to_mult(torch.tensor(values)))
    assert torch.equal(y, exact)
    x0 = pi._ODDS_SLOPE_CAP_AT
    cap = 2.0 / (1.0 - x0 * x0)
    assert abs(float(g[1]) - 2.0) < 1e-6                          # d 2atanh / dx at 0 is 2
    assert all(abs(float(v) - cap) < 1e-3 for v in g[4:])
    assert all(float(v) <= cap + 1e-3 for v in g)


def test_nested_lift_folds_over_saturated_operands_keep_gradients_bounded():
    """Thirty nested sigma composes whose operands sit at +-1 (a max-law
    unit code) used to compound the exact chart's 5e6 slope into
    overflow; with the cap the leaf gradient stays finite and modest."""
    torch.manual_seed(0)
    sigma = Layers.SigmaLayer(nInput=8, nOutput=8, nonlinear=True)
    leaf = torch.sign(torch.randn(2, 1, 8)).requires_grad_(True)  # exactly +-1
    acc = leaf
    for _ in range(30):
        unit = torch.sign(torch.randn(2, 1, 8))
        acc = sigma.compose(acc, unit)
    acc.square().mean().backward()
    assert bool(torch.isfinite(leaf.grad).all())
    assert float(leaf.grad.abs().max()) < 1e6
