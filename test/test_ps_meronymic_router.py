"""Learned meronymic PS router (Phase R3).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§7.2 / §8 R3 / §10, updated for item 7.5. Signed-neighborhood evidence
selects one adjacent merge or perceptual STOP from a single softmax.
Tests cover one operation per round, depth penalty, and byte fallback
versus known words (coherent atoms merge, incoherent ones stay terminals).
"""

import itertools
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch


def test_route_once_selects_exactly_one_best_merge():
    from perceptual_analyzer import MeronymicRouter
    router = MeronymicRouter(keep_bias=0.0)
    copy = torch.zeros(1, 6, 1)
    scores = torch.tensor([[[3.], [-1.], [4.], [2.], [1.]]])
    assert router.route_once(copy, scores)['merges'] == [2]
    assert router.route_once(copy, -scores.abs())['merges'] == []


def test_route_once_returns_soft_marginals():
    """route_once exposes soft marginals alongside the one hard route; a
    confident merge has a near-1 reduce marginal."""
    from perceptual_analyzer import MeronymicRouter
    router = MeronymicRouter()
    copy_score = torch.zeros(1, 3, 1)
    reduce_score = torch.tensor([[[6.0], [-6.0]]])  # merge pair 0, not pair 1
    out = router.route_once(copy_score, reduce_score)
    assert out["merges"] == [0]
    marg = out["reduce_marginal"]
    assert marg[0] > 0.95 and marg[1] < 0.05


def test_depth_penalty_monotonically_reduces_merges():
    """Higher depth penalty -> non-increasing merge count (finer terminals).
    The penalty uniformly shifts the signed-neighborhood merge evidence, so
    one round can only lose positive pairs as it rises (provably
    monotonic; the iterated route mutates vectors and is not)."""
    from perceptual_analyzer import MeronymicRouter
    torch.manual_seed(1)
    atoms = torch.randn(9, 16)
    counts = []
    for pen in [-1.0, -0.25, 0.0, 0.25, 0.5, 0.9, 2.0]:
        router = MeronymicRouter(depth_penalty=pen)
        cs, rs = router.scores(atoms)
        counts.append(len(router.route_once(cs, rs)["merges"]))
    assert counts == sorted(counts, reverse=True), counts
    assert counts[-1] == 0, "a very high penalty leaves every atom a terminal"


def test_known_word_merges_unknown_stays_bytes():
    """Coherent atoms (a known word's bytes) merge into one chunk; an
    incoherent (unknown / byte-fallback) region stays as singleton
    terminals. This is the learned analogue of stop-vs-byte routing."""
    from perceptual_analyzer import MeronymicRouter
    D = 12
    w = torch.zeros(D); w[0] = 1.0            # 3 identical "known word" atoms
    u1 = torch.zeros(D); u1[5] = 1.0          # mutually orthogonal "unknown"
    u2 = torch.zeros(D); u2[9] = 1.0
    atoms = torch.stack([w, w, w, u1, u2])
    # Penalty between the unknown similarity (0) and the known similarity (1).
    router = MeronymicRouter(depth_penalty=0.5)
    segs = router.route(atoms)["segments"]
    assert (0, 3) in segs, segs              # the known word is one chunk
    assert (3, 4) in segs and (4, 5) in segs  # unknown bytes stay separate


def test_single_atom_and_empty_are_total():
    """A length-1 surface routes to one terminal; length-0 to none (the
    byte/atom cover is total, so the router always has a valid route)."""
    from perceptual_analyzer import MeronymicRouter
    router = MeronymicRouter()
    one = router.route(torch.randn(1, 8))
    assert one["segments"] == [(0, 1)] and one["n_merges"] == 0
    zero = router.route(torch.zeros(0, 8))
    assert zero["segments"] == [] and zero["n_merges"] == 0
