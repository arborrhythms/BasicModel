"""Soft DP correctness against brute-force enumeration."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))
sys.path.insert(0, os.path.dirname(__file__))

import math

import pytest
import torch

from Language import binary_tiling_soft_dp
from test_signal_router_brute_force import (
    enumerate_tilings, logsumexp_tilings, score_tiling,
)


def _rand_scores(B, N, rc, rr, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cs = torch.randn(B, N, rc, generator=g, device="cpu")
    rs = torch.randn(B, N - 1, rr, generator=g, device="cpu") if N >= 1 else \
        torch.empty(B, 0, rr, device="cpu")
    return cs, rs


def test_soft_dp_logZ_matches_brute_force_small_N():
    for N in (1, 2, 3, 4, 5):
        for rc, rr in ((1, 1), (2, 1), (1, 2), (3, 2)):
            cs, rs = _rand_scores(1, N, rc, rr, seed=N * 100 + rc * 10 + rr)
            out = binary_tiling_soft_dp(cs, rs)
            logZ_dp = out["logZ"][0].item()
            logZ_bf = logsumexp_tilings(
                cs[0].tolist(),
                rs[0].tolist() if N > 1 else [],
                N, rc, rr,
            )
            assert math.isclose(logZ_dp, logZ_bf, rel_tol=1e-5, abs_tol=1e-5), \
                f"N={N} rc={rc} rr={rr} dp={logZ_dp} bf={logZ_bf}"


def test_soft_dp_marginals_sum_consistency():
    # At every source position t in [0, N-2], exactly one of:
    #   - a copy fires at t (sum over ops)
    #   - a reduce fires at t (sum over ops, length-2 starting at t)
    #   - a reduce fires at t-1 (covers t)
    # the per-position "covered" mass = 1 in expectation.
    cs, rs = _rand_scores(2, 6, 3, 2, seed=42)
    out = binary_tiling_soft_dp(cs, rs)
    p_copy = out["copy_marginal"]      # [B, N]
    p_reduce = out["reduce_marginal"]  # [B, N-1]
    B, N = p_copy.shape
    for b in range(B):
        for t in range(N):
            covered = p_copy[b, t].item()
            if t < N - 1:
                covered += p_reduce[b, t].item()
            if t > 0:
                covered += p_reduce[b, t - 1].item()
            assert math.isclose(covered, 1.0, abs_tol=1e-4), \
                f"position t={t} covered_mass={covered}"


def test_soft_dp_per_op_marginals_sum_to_action_marginal():
    cs, rs = _rand_scores(1, 5, 3, 2, seed=7)
    out = binary_tiling_soft_dp(cs, rs)
    # copy_marginal_op: [B, N, R_copy]; copy_marginal: [B, N] = sum over ops.
    assert torch.allclose(
        out["copy_marginal_op"].sum(-1), out["copy_marginal"], atol=1e-5)
    assert torch.allclose(
        out["reduce_marginal_op"].sum(-1), out["reduce_marginal"], atol=1e-5)


def test_soft_dp_gradient_reaches_scores():
    cs = torch.randn(1, 4, 2, requires_grad=True, device="cpu")
    rs = torch.randn(1, 3, 2, requires_grad=True, device="cpu")
    out = binary_tiling_soft_dp(cs, rs)
    out["logZ"].sum().backward()
    assert cs.grad is not None and (cs.grad.abs().sum() > 0)
    assert rs.grad is not None and (rs.grad.abs().sum() > 0)


def test_soft_dp_extreme_finite_scores_have_finite_marginals():
    """Large anchor-dot scores must not overflow forward/backward marginals.

    The ranges reproduce the scale seen on the first N=64 FineWeb grammar
    pass before any optimiser step.  All inputs are finite; the old direct
    float32 subtraction produced +inf and NaN marginals here.
    """
    N = 64
    cs = torch.linspace(-110_104_816.0, 0.0, N).reshape(1, N, 1)
    rs = torch.linspace(
        -1_380_412_160.0,
        584_686_784.0,
        (N - 1) * 15,
    ).reshape(1, N - 1, 15)

    out = binary_tiling_soft_dp(cs, rs)

    for name, value in out.items():
        assert torch.isfinite(value).all(), name

    # Every source position is covered by exactly one expected COPY or by a
    # REDUCE beginning immediately before/at that position.
    covered = out["copy_marginal"].clone()
    covered[:, :-1] += out["reduce_marginal"]
    covered[:, 1:] += out["reduce_marginal"]
    assert torch.allclose(covered, torch.ones_like(covered), atol=2e-3)
    assert torch.allclose(
        out["copy_marginal_op"].sum(-1),
        out["copy_marginal"],
        atol=1e-5,
    )
    assert torch.allclose(
        out["reduce_marginal_op"].sum(-1),
        out["reduce_marginal"],
        atol=1e-5,
    )
