"""Vectorized truth-layer staging tests (§6b).

The brick-vectorization handoff §6b drops the per-cell ``should_store``
gate entirely and replaces it with a pure batched insert that scales
each activation by its DegreeOfTruth. ``record_batch`` is the
production sync-free path: every cell is staged with its trust
score, and the post-tick ``compact()`` filters by trust.

These tests verify that ``record_batch`` produces results equivalent
to a reference Python loop over ``record(vec * dot)`` for high-trust
inputs, and that low-trust entries don't pollute the persistent
store after compact.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md §6b
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest

from Layers import TruthLayer


def test_should_store_function_deleted():
    """``TruthLayer.should_store`` was removed by §6b.

    Per-cell gating is no longer applied inside the brick; the codebook
    nearest-neighbor lookup at compact time naturally dedupes near-zero
    and near-duplicate vectors against the existing prototypes.
    """
    tl = TruthLayer(nDim=8, max_truths=16)
    assert not hasattr(tl, 'should_store'), (
        "should_store was deleted in §6b; the per-cell gate is gone "
        "and only record_batch / record / compact remain.")


def test_record_batch_matches_record_loop_high_trust():
    """``record_batch`` + ``compact`` matches a record-per-entry loop
    when every cell has high trust (above the compact threshold).
    """

    D = 8
    N = 6
    activations = torch.randn(N, D)
    trust = torch.full((N,), 0.8)  # all above the 0.5 default threshold
    degree = 1.0

    ref = TruthLayer(nDim=D, max_truths=16)
    for i in range(N):
        ref.record(activations[i], degree=degree)

    cand = TruthLayer(nDim=D, max_truths=16)
    cand.record_batch(activations, trust=trust, degree=degree)
    n_promoted = cand.compact(min_trust=0.5)

    assert n_promoted == N, (
        f"record_batch + compact should promote all {N} high-trust "
        f"entries; got {n_promoted}")
    assert int(cand.count.item()) == int(ref.count.item())
    # Stored vectors should match the reference (record stores vec*degree).
    cand_truths = cand.truths[:int(cand.count.item())]
    ref_truths = ref.truths[:int(ref.count.item())]
    assert torch.allclose(cand_truths, ref_truths, atol=1e-6), (
        "record_batch + compact persistent store diverged from record-loop")


def test_record_batch_low_trust_dropped_at_compact():
    """Low-trust entries don't reach the persistent store.

    The plan's intent: the codebook's near-zero prototype matches
    low-DoT (=low-magnitude after DoT scaling) entries so they don't
    grow the store. Here we test the simpler trust-threshold fence at
    ``compact()``: low-trust entries are filtered out.
    """

    D = 8
    N_high = 3
    N_low = 5
    activations = torch.randn(N_high + N_low, D)
    trust = torch.cat([
        torch.full((N_high,), 0.9),
        torch.full((N_low,), 0.1),
    ])

    tl = TruthLayer(nDim=D, max_truths=16)
    tl.record_batch(activations, trust=trust, degree=1.0)
    promoted = tl.compact(min_trust=0.5)
    assert promoted == N_high, (
        f"compact should promote only the {N_high} high-trust entries; "
        f"got {promoted}")
    assert int(tl.count.item()) == N_high


def test_record_batch_dot_scaling_negative_degree_bivector():
    """For ``degree<0`` with a monotonic basis, ``record_batch`` uses
    the bivector-aware path (paired-index flip), matching ``record``.
    """

    K = 3
    D = 2 * K  # even, monotonic-ready

    class _MonotonicBasis:
        monotonic = True
        def negation(self, vec, monotonic=False):
            # Swap pairs (2k, 2k+1) along the last dim.
            v = vec.clone()
            v[..., 0::2], v[..., 1::2] = vec[..., 1::2].clone(), vec[..., 0::2].clone()
            return v

    basis = _MonotonicBasis()
    activations = torch.randn(2, D)
    trust = torch.full((2,), 0.9)

    ref = TruthLayer(nDim=D, max_truths=8)
    for i in range(2):
        ref.record(activations[i], degree=-0.6, basis=basis)

    cand = TruthLayer(nDim=D, max_truths=8)
    cand.record_batch(activations, trust=trust, degree=-0.6, basis=basis)
    cand.compact(min_trust=0.5)

    assert torch.allclose(
        cand.truths[:int(cand.count.item())],
        ref.truths[:int(ref.count.item())],
        atol=1e-6,
    ), "bivector-mode record_batch diverged from per-entry record"


def test_record_batch_no_host_sync_in_brick():
    """``record_batch`` runs without a host sync inside the brick.

    The implementation uses Python int counters for the pending
    cursor (no tensor reduction, no ``.item()`` until ``compact()``
    runs outside the brick). This test exercises the path with the
    stage-only call and checks the persistent ``count`` did NOT
    advance — promotion happens at ``compact``, not at staging.
    """
    D = 4
    tl = TruthLayer(nDim=D, max_truths=8)
    activations = torch.randn(3, D)
    trust = torch.full((3,), 0.9)
    tl.record_batch(activations, trust=trust, degree=1.0)
    # The persistent buffer is untouched; only the pending buffer holds the data.
    assert int(tl.count.item()) == 0
    assert tl._pending_count == 3
    promoted = tl.compact(min_trust=0.5)
    assert promoted == 3
    assert int(tl.count.item()) == 3
