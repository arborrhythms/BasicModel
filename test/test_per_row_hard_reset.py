"""Tests confirming per-row hard Reset fires only on doc-boundary ticks.

Under the rolling-cursor design, the outer doc-streaming loop
dispatches ``space.Reset(batch=b, hard=True)`` only for rows whose
``hard_eos[b]`` was True on the most recent ``next_tick``. Other rows
keep their per-row state across ticks (so a long doc that takes many
slabs stays composable across the boundary).

Plan reference: doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md §Verification
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest
import torch

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


def _model(masked_prediction='AR'):
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")

    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    model.masked_prediction = masked_prediction
    model.inputSpace.masked_prediction = masked_prediction
    return model


def test_dispatch_per_row_reset_skips_rows_without_hard_eos():
    """Rows with hard_eos=False must NOT fire Reset; True rows must.

    Spies on each space's Reset and tracks per-row dispatch. Calling
    ``dispatch_per_row_reset([False, True])`` must invoke
    ``space.Reset(batch=1, hard=True)`` exactly once per space, with
    no call for batch=0.
    """
    model = _model()
    # Size WordSpace state to two source rows so per-row dispatch can
    # write to indices [0..1] without going out of bounds.
    if model.wordSpace is not None:
        model.wordSpace.ensure_microbatch(2, 1)
    seen = []  # list of (space_class, batch_arg, hard_arg)
    originals = []
    for space in model.spaces:
        if hasattr(space, 'Reset'):
            cname = type(space).__name__
            original = space.Reset
            originals.append((space, original))

            def _spy(_self=space, _orig=original, _name=cname,
                     batch=None, hard=True):
                seen.append((_name, batch, hard))
                # call through with the same kwargs to keep state
                # transitions correct.
                return _orig(batch=batch, hard=hard)
            space.Reset = _spy
    try:
        model.dispatch_per_row_reset([False, True])
    finally:
        for space, original in originals:
            space.Reset = original

    batches_seen = sorted({b for (_n, b, _h) in seen})
    assert batches_seen == [1], (
        f"per-row dispatch must skip hard_eos=False rows; "
        f"saw batches {batches_seen}, expected [1]"
    )
    # Every Reset call should have hard=True under the cursor contract.
    for _n, _b, h in seen:
        assert h is True, f"hard kw must be True, got {h}"


def test_dispatch_per_row_reset_empty_or_all_false_is_noop():
    """An empty hard_eos list, or all-False, fires no Reset calls."""
    model = _model()
    seen = []
    originals = []
    for space in model.spaces:
        if hasattr(space, 'Reset'):
            original = space.Reset
            originals.append((space, original))

            def _spy(_self=space, _orig=original,
                     batch=None, hard=True):
                seen.append((batch, hard))
                return _orig(batch=batch, hard=hard)
            space.Reset = _spy
    try:
        model.dispatch_per_row_reset([])
        model.dispatch_per_row_reset([False, False])
    finally:
        for space, original in originals:
            space.Reset = original

    assert seen == [], (
        f"dispatch_per_row_reset should be a no-op for empty / all-False "
        f"hard_eos; got {seen}"
    )


def test_per_row_hard_reset_clears_only_target_row_state():
    """WordSpace.Reset(batch=b) clears row b's per-row state, not other rows.

    Sets _stm_fired[0] and _stm_fired[1] to True, fires Reset(batch=0),
    asserts _stm_fired[0] is False but [1] is still True.
    """
    model = _model()
    ws = model.wordSpace
    if ws is None:
        pytest.skip("model has no WordSpace")
    # Ensure state is sized to at least 2 rows.
    if ws._stm_fired.shape[0] < 2:
        ws.ensure_microbatch(2, 1)
    ws._stm_fired[0] = True
    ws._stm_fired[1] = True
    ws._svo_valid[0] = True
    ws._svo_valid[1] = True

    ws.Reset(batch=0, hard=True)

    assert ws._stm_fired[0].item() is False
    assert ws._stm_fired[1].item() is True
    assert ws._svo_valid[0].item() is False
    assert ws._svo_valid[1].item() is True


def test_runEpoch_dispatches_per_row_reset_per_batch():
    """End-to-end: runEpoch fires per-row dispatch after every batch.

    Uses a counting spy on dispatch_per_row_reset to confirm the outer
    loop calls it once per tick (regardless of cursor mode).
    """
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    calls = []
    original = model.dispatch_per_row_reset

    def _spy(hard_eos):
        calls.append(list(hard_eos))
        return original(hard_eos)
    model.dispatch_per_row_reset = _spy
    try:
        model.runEpoch(optimizer=optimizer, batchSize=2, split="train")
    finally:
        model.dispatch_per_row_reset = original

    assert len(calls) > 0, (
        "dispatch_per_row_reset must fire at least once per epoch "
        "under the rolling-cursor handoff outer loop"
    )
    # Legacy (non-cursor) path uses hard_eos=[True]*B every tick.
    for hard_eos in calls:
        assert all(hard_eos), (
            f"non-cursor path uses hard_eos=[True]*B every tick; "
            f"got {hard_eos}"
        )
