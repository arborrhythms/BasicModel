"""Stage 1 of doc/plans/MeronomyPlan.md: ContractiveInvertibleLinearLayer.

Spec §10.10 (weight-law parametrization): on the new layer path,
``diag(W) >= 1`` (block-diagonals for binary folds), ``offdiag(W) >= 0``,
``b <= 0`` hold BY CONSTRUCTION at init and after arbitrary optimizer
steps; the ``stable`` clamp bounds d in ``[1, d_max]``. The exact LDU
inverse (triangular solves) round-trips.

The legacy regimes stay untouched: NonNegativeInvertibleLinearLayer's
``d = softplus(raw)`` (positive, NOT >= 1) and its ``stable`` clamp to
``(eps, 1.0]`` are the odds chart's law; the positive-bias path
(NonNegativeLinearLayer) and the unconstrained invertible bias path keep
their own signs. This file pins the NEW class only.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import (ContractiveInvertibleLinearLayer,
                    NonNegativeInvertibleLinearLayer, D_MAX_STABLE)

ATOL = 1e-6


def assert_weight_law(layer):
    """The §4 law on the materialized effective W, bias, and diagonal."""
    W = layer.compute_W()
    D = layer.nOutput
    assert (W >= -ATOL).all(), "offdiag(W) >= 0 violated"
    for b in range(layer.blocks):
        blk_diag = torch.diagonal(W[b * D:(b + 1) * D, :])
        assert (blk_diag >= 1.0 - ATOL).all(), (
            f"diag(W) >= 1 violated in operand block {b}: "
            f"min={blk_diag.min().item():.6f}")
    d = layer._d_effective()
    assert (d >= 1.0 - ATOL).all(), "d >= 1 violated"
    if layer.stable:
        assert (d <= layer.d_max + ATOL).all(), "stable d <= d_max violated"
    if layer.hasBias:
        bias = layer._effective_bias()
        assert (bias <= ATOL).all(), "b <= 0 violated"


# ---------------------------------------------------------------------------
# Constraints at init.
# ---------------------------------------------------------------------------

def test_constraints_hold_at_init_unary():
    layer = ContractiveInvertibleLinearLayer(6, 6)
    assert_weight_law(layer)
    # Near-identity init: d ~ 1 (just above), b ~ 0 (just below).
    d = layer._d_effective()
    assert (d - 1.0 <= 0.01).all(), "init d should be ~1"
    bias = layer._effective_bias()
    assert (bias.abs() <= 0.01).all(), "init b should be ~0"


def test_constraints_hold_at_init_binary_blocks():
    D = 5
    layer = ContractiveInvertibleLinearLayer(2 * D, D, blocks=2)
    assert_weight_law(layer)
    # Both operand-block diagonals sit just above 1 at init.
    W = layer.compute_W()
    assert (torch.diagonal(W[:D, :]) >= 1.0).all()
    assert (torch.diagonal(W[D:, :]) >= 1.0).all()
    assert (torch.diagonal(W[D:, :]) <= 1.1).all(), "near-identity init"


def test_blocks_shape_validation():
    with pytest.raises(ValueError):
        ContractiveInvertibleLinearLayer(7, 3, blocks=2)   # 7 != 2*3
    with pytest.raises(ValueError):
        ContractiveInvertibleLinearLayer(6, 3, blocks=0)
    # blocks=3 generalizes (nInput == 3 * nOutput).
    layer = ContractiveInvertibleLinearLayer(9, 3, blocks=3)
    assert_weight_law(layer)


def test_d_max_resolution():
    # Default comes from the Stage 0 constant (no config loaded here).
    layer = ContractiveInvertibleLinearLayer(4, 4, stable=True)
    assert layer.d_max == D_MAX_STABLE
    # Explicit argument wins.
    layer2 = ContractiveInvertibleLinearLayer(4, 4, stable=True, d_max=2.5)
    assert layer2.d_max == 2.5


def test_legacy_layer_untouched():
    # The parent keeps the odds-chart regime: d = softplus(raw) ~ 1 at
    # init but NOT bounded below by 1, and stable clamps to (eps, 1.0].
    legacy = NonNegativeInvertibleLinearLayer(4, 4, stable=True)
    with torch.no_grad():
        legacy.d.fill_(-5.0)  # softplus(-5) ~ 0.0067 -- legal for legacy
    d = legacy._d_effective()
    assert (d < 1.0).all(), "legacy d may sit below 1"
    assert not hasattr(legacy, 'blocks'), "blocks is new-class-only"


# ---------------------------------------------------------------------------
# Constraints after randomized optimizer steps (seed-pinned property
# test). The losses are chosen adversarially: one pushes every W entry
# down and the bias up (toward violating diag >= 1 / b <= 0), one pushes
# d up (toward violating the stable d_max bound). The constraints must
# hold by construction whatever the raw parameters do.
# ---------------------------------------------------------------------------

def _optimizer_steps(layer, loss_fn, steps=25, lr=0.5):
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(layer).backward()
        opt.step()


@pytest.mark.parametrize("blocks,nin,nout", [(1, 6, 6), (2, 8, 4)])
def test_constraints_hold_after_adversarial_descent(blocks, nin, nout):
    torch.manual_seed(1)
    layer = ContractiveInvertibleLinearLayer(nin, nout, stable=True,
                                             blocks=blocks)
    # Push W down / b up as hard as the raw parametrization allows.
    _optimizer_steps(layer, lambda l: l.compute_W().sum()
                     - l._effective_bias().sum())
    assert_weight_law(layer)


def test_stable_clamp_bounds_d_after_ascent():
    torch.manual_seed(1)
    layer = ContractiveInvertibleLinearLayer(5, 5, stable=True)
    # Push d (and everything else) UP; the clamp must cap d at d_max.
    _optimizer_steps(layer, lambda l: -l.compute_W().sum())
    assert_weight_law(layer)
    d = layer._d_effective()
    assert (d <= layer.d_max + ATOL).all()
    assert d.max() >= layer.d_max - 0.5, (
        "ascent should have driven d into the clamp")
    # Without stable, d is unbounded above (law only requires >= 1).
    torch.manual_seed(1)
    free = ContractiveInvertibleLinearLayer(5, 5, stable=False)
    _optimizer_steps(free, lambda l: -l.compute_W().sum())
    assert (free._d_effective() >= 1.0 - ATOL).all()
    assert free._d_effective().max() > D_MAX_STABLE, (
        "unclamped d should exceed the stable bound under ascent")


def test_constraints_hold_after_random_regression(seed=1):
    # A realistic objective: regress random targets through the layer.
    torch.manual_seed(seed)
    layer = ContractiveInvertibleLinearLayer(6, 3, blocks=2, stable=True)
    x = torch.rand(32, 6)
    target = torch.rand(32, 3)
    _optimizer_steps(layer, lambda l: ((l.forward(x) - target) ** 2).mean(),
                     steps=40, lr=0.1)
    assert_weight_law(layer)


# ---------------------------------------------------------------------------
# Exact LDU inverse round-trips (triangular solves, no approximation).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("naive", [False, True])
@pytest.mark.parametrize("hasBias", [True, False])
def test_square_roundtrip(naive, hasBias):
    torch.manual_seed(1)
    layer = ContractiveInvertibleLinearLayer(6, 6, naive=naive,
                                             hasBias=hasBias)
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(-5.0, 0.0)  # randomize within the law
    x = torch.randn(7, 6)
    y = layer.forward(x)
    x_rec = layer.reverse(y)
    err = torch.norm(x - x_rec) / torch.norm(x)
    assert err < 1e-4, f"roundtrip error {err:.2e}"


def test_square_W_Winverse_identity():
    torch.manual_seed(1)
    layer = ContractiveInvertibleLinearLayer(5, 5, hasBias=False)
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(-5.0, 0.0)
    W, W_inv = layer.compute_W(), layer.compute_Winverse()
    id_err = torch.norm(W @ W_inv - torch.eye(5))
    assert id_err < 1e-4, f"W @ W_inv != I: {id_err:.2e}"


def test_binary_right_inverse():
    # The 2D->D fold is lossy by rank; reverse() returns the canonical
    # preimage, and forward(reverse(z)) == z exactly (right inverse).
    torch.manual_seed(1)
    layer = ContractiveInvertibleLinearLayer(8, 4, blocks=2, hasBias=True)
    z = torch.randn(5, 4)
    z_rec = layer.forward(layer.reverse(z))
    err = torch.norm(z - z_rec) / torch.norm(z)
    assert err < 1e-4, f"binary right-inverse error {err:.2e}"


def test_roundtrip_survives_optimizer_steps():
    torch.manual_seed(1)
    layer = ContractiveInvertibleLinearLayer(6, 6, stable=True)
    x = torch.rand(16, 6)
    target = torch.rand(16, 6)
    _optimizer_steps(layer, lambda l: ((l.forward(x) - target) ** 2).mean(),
                     steps=30, lr=0.2)
    assert_weight_law(layer)
    y = layer.forward(x)
    x_rec = layer.reverse(y)
    err = torch.norm(x - x_rec) / torch.norm(x)
    assert err < 1e-4, f"post-training roundtrip error {err:.2e}"


# ---------------------------------------------------------------------------
# Ergodic mode: the law must hold under exploration noise too (the
# 1 + softplus floor is reapplied after the noise enters the raw domain).
# ---------------------------------------------------------------------------

def test_constraints_hold_under_ergodic_noise():
    torch.manual_seed(1)
    layer = ContractiveInvertibleLinearLayer(6, 3, blocks=2, ergodic=True,
                                             stable=True)
    with torch.no_grad():
        layer.var.fill_(0.5)
        layer.bias.fill_(0.8)
    for _ in range(5):
        layer.resample_noise()
        W = layer.compute_W_current()
        assert (W >= -ATOL).all()
        for b in range(layer.blocks):
            blk_diag = torch.diagonal(W[b * 3:(b + 1) * 3, :])
            assert (blk_diag >= 1.0 - ATOL).all()
        d = layer._d_eff()
        assert (d >= 1.0 - ATOL).all() and (d <= layer.d_max + ATOL).all()
        bias = layer._effective_bias()
        assert (bias <= ATOL).all()


def test_ergodic_d_matches_clean_law_at_zero_noise():
    # Review fix 2026-06-11: the ergodic diagonal must reproduce the
    # clean law exactly at bias=1, var=0 (raw-domain mixing, law
    # applied once). The earlier double-softplus inflated the init
    # diagonal to ~1.697 instead of the documented near-identity
    # ~1.0067.
    layer = ContractiveInvertibleLinearLayer(5, 5, ergodic=True,
                                             stable=True)
    with torch.no_grad():
        layer.var.zero_()
        layer.bias.fill_(1.0)
    assert torch.allclose(layer._d_eff(), layer._d_effective()), (
        "bias=1, var=0 ergodic diagonal == clean diagonal")
    d = layer._d_eff()
    assert (d - 1.0 <= 0.01).all(), (
        f"ergodic init diagonal must be near-identity, got {d.max():.4f}")


def test_ergodic_roundtrip():
    torch.manual_seed(1)
    layer = ContractiveInvertibleLinearLayer(5, 5, ergodic=True, stable=True)
    with torch.no_grad():
        layer.var.fill_(0.2)
        layer.bias.fill_(0.8)
    x = torch.randn(4, 5)
    y = layer.forward(x)          # resamples, then uses the new factors
    x_rec = layer.reverse(y)      # reuses the same factors, then resamples
    err = torch.norm(x - x_rec) / torch.norm(x)
    assert err < 1e-4, f"ergodic roundtrip error {err:.2e}"
