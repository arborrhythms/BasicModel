"""Cross-product action distribution + route log-prob (NeuralToolUser).

doc/plans/NeuralToolUser.md, user-refined design: the chooser normalizes
its per-level scores JOINTLY over the (operation x location) cross-product
(one softmax across copy AND reduce choices), so probability is comparable
across rules and positions. The selected route stays a valid tiling
(binary_tiling_viterbi); the route's log-prob is read from this joint
distribution for the two-pass policy update. These tests pin the
primitive's correctness in isolation (it is not yet on any live path).
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import math
import torch

from Language import (
    cross_product_action_dist, cross_product_route_logprob,
    binary_tiling_viterbi,
)


def test_joint_distribution_normalizes_over_cross_product():
    B, N, R_copy, R_reduce = 2, 5, 1, 3
    copy_score = torch.randn(B, N, R_copy)
    reduce_score = torch.randn(B, N - 1, R_reduce)
    dist = cross_product_action_dist(copy_score, reduce_score)
    A = N * R_copy + (N - 1) * R_reduce
    assert dist["probs"].shape == (B, A)
    # JOINT normalization across copy AND reduce, not per-position.
    assert torch.allclose(dist["probs"].sum(-1), torch.ones(B), atol=1e-5)


def test_layout_action_counts():
    copy_score = torch.randn(2, 4, 2)
    reduce_score = torch.randn(2, 3, 5)
    dist = cross_product_action_dist(copy_score, reduce_score)
    lo = dist["layout"]
    assert lo["n_copy"] == 4 * 2
    assert lo["Nm1"] == 3 and lo["R_reduce"] == 5


def test_route_logprob_matches_manual_sum():
    torch.manual_seed(0)
    B, N, R_copy, R_reduce = 1, 4, 1, 2
    copy_score = torch.randn(B, N, R_copy)
    reduce_score = torch.randn(B, N - 1, R_reduce)
    dist = cross_product_action_dist(copy_score, reduce_score)
    route = binary_tiling_viterbi(copy_score, reduce_score)
    lp = cross_product_route_logprob(
        dist, route["copy_mask"], route["reduce_mask"])
    # Manual: sum log P over the actions the masks select.
    logp = torch.log(dist["probs"].clamp_min(1e-30))
    copy_flat = route["copy_mask"].reshape(B, N * R_copy)
    reduce_flat = route["reduce_mask"].reshape(B, (N - 1) * R_reduce)
    sel = torch.cat([copy_flat, reduce_flat], dim=1)
    manual = (sel * logp).sum(-1)
    assert torch.allclose(lp, manual, atol=1e-6)


def test_route_logprob_is_differentiable_into_scores():
    B, N, R_copy, R_reduce = 1, 4, 1, 2
    copy_score = torch.randn(B, N, R_copy, requires_grad=True)
    reduce_score = torch.randn(B, N - 1, R_reduce, requires_grad=True)
    dist = cross_product_action_dist(copy_score, reduce_score)
    route = binary_tiling_viterbi(copy_score, reduce_score)
    lp = cross_product_route_logprob(
        dist, route["copy_mask"], route["reduce_mask"])
    lp.sum().backward()
    assert copy_score.grad is not None
    assert torch.isfinite(copy_score.grad).all()


def test_entropy_within_bounds_and_temperature_flattens():
    B, N, R_copy, R_reduce = 1, 5, 1, 3
    copy_score = torch.randn(B, N, R_copy)
    reduce_score = torch.randn(B, N - 1, R_reduce)
    A = N * R_copy + (N - 1) * R_reduce
    cold = cross_product_action_dist(copy_score, reduce_score, temperature=0.5)
    hot = cross_product_action_dist(copy_score, reduce_score, temperature=4.0)
    assert (cold["entropy"] >= -1e-5).all()
    assert (hot["entropy"] <= math.log(A) + 1e-5).all()
    # Higher temperature -> flatter -> more entropy.
    assert (hot["entropy"] > cold["entropy"]).all()


def test_empty_and_degenerate_shapes():
    # N=0: no actions.
    d0 = cross_product_action_dist(
        torch.zeros(2, 0, 1), torch.zeros(2, 0, 0))
    assert d0["probs"].shape == (2, 0)
    assert torch.equal(d0["entropy"], torch.zeros(2))
    # R_copy=0 (grammars like MM_xor_loopback): reduce-only action space.
    d1 = cross_product_action_dist(
        torch.zeros(2, 4, 0), torch.randn(2, 3, 2))
    assert d1["probs"].shape == (2, 3 * 2)
    assert torch.allclose(d1["probs"].sum(-1), torch.ones(2), atol=1e-5)
