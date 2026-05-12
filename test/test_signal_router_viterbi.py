"""Hard Viterbi correctness against brute force, plus legality."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))
sys.path.insert(0, os.path.dirname(__file__))

import math

import torch

from Language import binary_tiling_viterbi
from test_signal_router_brute_force import best_tiling


def _rand_scores(B, N, rc, rr, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cs = torch.randn(B, N, rc, generator=g, device="cpu")
    rs = torch.randn(B, max(N - 1, 0), rr, generator=g, device="cpu")
    return cs, rs


def test_viterbi_score_matches_brute_force_small_N():
    for N in (1, 2, 3, 4, 5, 6):
        for rc, rr in ((1, 1), (2, 1), (1, 2), (2, 3)):
            cs, rs = _rand_scores(1, N, rc, rr, seed=N + rc * 17 + rr * 31)
            out = binary_tiling_viterbi(cs, rs)
            score_dp = out["score"][0].item()
            _, score_bf = best_tiling(
                cs[0].tolist(),
                rs[0].tolist() if N > 1 else [],
                N, rc, rr,
            )
            assert math.isclose(score_dp, score_bf, rel_tol=1e-5, abs_tol=1e-5)


def test_viterbi_route_is_legal_no_overlapping_reduces():
    cs, rs = _rand_scores(3, 6, 2, 2, seed=99)
    out = binary_tiling_viterbi(cs, rs)
    cm = out["copy_mask"]      # [B, N, R_copy] one-hot, summed over R_copy = {0,1}
    rm = out["reduce_mask"]    # [B, N-1, R_reduce]
    B, N, _ = cm.shape
    for b in range(B):
        # Per-position coverage: each t is covered by copy@t OR reduce@t OR
        # reduce@t-1, exactly one.
        for t in range(N):
            covered = int(cm[b, t].sum().item())
            if t < N - 1:
                covered += int(rm[b, t].sum().item())
            if t > 0:
                covered += int(rm[b, t - 1].sum().item())
            assert covered == 1, f"b={b} t={t} covered={covered}"


def test_viterbi_one_hot_per_active_action():
    cs, rs = _rand_scores(2, 5, 3, 2, seed=11)
    out = binary_tiling_viterbi(cs, rs)
    cm = out["copy_mask"]
    rm = out["reduce_mask"]
    # Where the action fires, exactly one op is selected; elsewhere all zero.
    cm_fired = cm.sum(-1)
    rm_fired = rm.sum(-1)
    assert torch.all((cm_fired == 0) | (cm_fired == 1))
    assert torch.all((rm_fired == 0) | (rm_fired == 1))
