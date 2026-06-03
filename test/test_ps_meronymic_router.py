"""Learned meronymic PS router (Phase R3).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§7.2 / §8 R3 / §10. The analyzer beyond compatibility mode: a meronymic
Viterbi / soft-DP router with signed-neighborhood evidence that selects ONE
hard route plus soft marginals, reusing the SHARED inverse-routing primitive
(``binary_tiling_viterbi`` / ``binary_tiling_soft_dp``) that the symbolic
``BinaryStructuredReductionLayer`` uses. Tests: Viterbi-not-beam (exact DP
vs brute force), depth penalty (granularity control), byte-fallback vs
known-word (coherent atoms merge, incoherent stay byte terminals).
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


def _brute_force_best_merges(reduce_score):
    """Max-weight non-overlapping adjacent-pair matching on a path (the
    exact single-level binary tiling) by exhaustive enumeration. Returns
    (best_score, frozenset_of_merge_positions)."""
    m = len(reduce_score)
    best_score, best = 0.0, frozenset()
    positions = list(range(m))
    for r in range(m + 1):
        for combo in itertools.combinations(positions, r):
            # non-overlapping: no two chosen pairs share an index.
            if any(combo[i] + 1 == combo[i + 1] for i in range(len(combo) - 1)):
                continue
            s = float(sum(reduce_score[t] for t in combo))
            if s > best_score:
                best_score, best = s, frozenset(combo)
    return best_score, best


def test_route_once_is_exact_viterbi_not_beam():
    """The single-level route maximizes total reduce score exactly (matches
    brute force), proving an exact DP rather than a greedy / beam pick."""
    from perceptual_analyzer import MeronymicRouter
    router = MeronymicRouter(keep_bias=0.0)
    torch.manual_seed(0)
    for _ in range(20):
        N = 6
        # copy_score 0 so total route score == sum of chosen reduce scores.
        copy_score = torch.zeros(1, N, 1)
        reduce_score = torch.randn(1, N - 1, 1)
        out = router.route_once(copy_score, reduce_score)
        rs = reduce_score.view(-1).tolist()
        _, best = _brute_force_best_merges(rs)
        assert set(out["merges"]) == set(best), (
            f"router merges {sorted(out['merges'])} != brute force {sorted(best)}")


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
    one DP level can only lose positive pairs as it rises (provably
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
