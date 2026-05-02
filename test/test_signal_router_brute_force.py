"""Brute-force enumerator over legal COPY/REDUCE tilings.

Used as ground truth for the structured-DP and Viterbi tests.
N <= 8 is the practical ceiling; runtime is exponential.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Tuple

import pytest


# A tile is (kind, op_id) where kind in {"copy", "reduce"}.
Tile = Tuple[str, int]


def enumerate_tilings(
    n: int, r_copy: int, r_reduce: int
) -> Iterator[List[Tile]]:
    """Yield every legal tiling of length-n positions.

    A legal tiling covers positions 0..n-1 with non-overlapping tiles:
      - copy tile (length 1) at position t with op c in [0, r_copy)
      - reduce tile (length 2) at positions t, t+1 with op r in [0, r_reduce)
    """
    if n == 0:
        yield []
        return
    # Copy at position 0.
    for c in range(r_copy):
        for tail in enumerate_tilings(n - 1, r_copy, r_reduce):
            yield [("copy", c)] + tail
    # Reduce at positions 0,1.
    if n >= 2:
        for r in range(r_reduce):
            for tail in enumerate_tilings(n - 2, r_copy, r_reduce):
                yield [("reduce", r)] + tail


def score_tiling(
    tiling: List[Tile],
    copy_score,    # [N, R_copy] tensor or 2D list
    reduce_score,  # [N-1, R_reduce] tensor or 2D list
) -> float:
    """Sum the per-tile scalar scores along this tiling. Single-batch."""
    total = 0.0
    pos = 0
    for kind, op in tiling:
        if kind == "copy":
            total += float(copy_score[pos][op])
            pos += 1
        else:  # reduce
            total += float(reduce_score[pos][op])
            pos += 2
    return total


def best_tiling(
    copy_score, reduce_score, n: int, r_copy: int, r_reduce: int
) -> Tuple[List[Tile], float]:
    """Argmax legal tiling and its score."""
    best, best_score = None, -math.inf
    for t in enumerate_tilings(n, r_copy, r_reduce):
        s = score_tiling(t, copy_score, reduce_score)
        if s > best_score:
            best, best_score = t, s
    return best, best_score


def logsumexp_tilings(
    copy_score, reduce_score, n: int, r_copy: int, r_reduce: int
) -> float:
    """Brute-force partition function (log-sum-exp over all tilings)."""
    if n == 0:
        return 0.0
    scores = [
        score_tiling(t, copy_score, reduce_score)
        for t in enumerate_tilings(n, r_copy, r_reduce)
    ]
    m = max(scores)
    return m + math.log(sum(math.exp(s - m) for s in scores))


# ---- self-tests on the enumerator itself ----------------------------

def _count(n, rc, rr):
    return sum(1 for _ in enumerate_tilings(n, rc, rr))


def test_enumerate_count_n_zero():
    assert _count(0, 1, 1) == 1  # the empty tiling


def test_enumerate_count_n_one_no_reduce_possible():
    assert _count(1, 1, 1) == 1
    assert _count(1, 3, 5) == 3  # only copy, R_copy choices


def test_enumerate_count_n_two():
    # N=2: copy-copy (rc^2) + reduce (rr) = rc^2 + rr
    assert _count(2, 2, 3) == 2 * 2 + 3


def test_enumerate_count_n_three():
    # N=3: ccc (rc^3), Rc (rr*rc), cR (rc*rr)
    assert _count(3, 2, 3) == 2**3 + 3 * 2 + 2 * 3


def test_enumerate_no_overlapping_reduces():
    # No tiling at any N has two reduces touching the same position.
    for n in range(1, 6):
        for tiling in enumerate_tilings(n, 1, 1):
            covered = []
            pos = 0
            for kind, _ in tiling:
                if kind == "copy":
                    covered.append((pos, pos))
                    pos += 1
                else:
                    covered.append((pos, pos + 1))
                    pos += 2
            # flat coverage equals 0..n-1, no repeats
            flat = sorted(p for a, b in covered for p in range(a, b + 1))
            assert flat == list(range(n))
