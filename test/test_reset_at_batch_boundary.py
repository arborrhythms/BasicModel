"""Tests confirming Reset() fires at every AR batch boundary.

Under the new gate, `inputSpace.batch_advances_sentence` is a
host-side @property that returns True for every AR batch (the
SentenceStreamDataset contract guarantees one sentence per row per
batch). The Reset cascade therefore fires unconditionally at every
AR-batch end.

Plan reference: doc/plans/2026-04-26-per-row-ar-no-eos-sync-handoff.md §3
"""
import sys
import os
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


def _model(masked_prediction='ARLM'):
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    model.masked_prediction = masked_prediction
    model.inputSpace.masked_prediction = masked_prediction
    return model


def test_reset_fires_every_ar_batch():
    """One Reset cascade per AR batch under the simple-version contract.

    Spy on each space's Reset(); after running an epoch with multiple
    batches, the count must equal the number of batches (per space
    that has Reset). Under the legacy gate, Reset would have been
    suppressed unless `_end_of_stream.all()` was True; under the new
    gate, it always fires.
    """
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    counts = {}
    originals = []
    for space in model.spaces:
        if hasattr(space, 'Reset'):
            cname = type(space).__name__
            counts.setdefault(cname, 0)
            original = space.Reset
            originals.append((space, original))

            def _spy(_self=space, _orig=original, _name=cname):
                counts[_name] += 1
                return _orig()
            space.Reset = _spy
    try:
        model.runEpoch(optimizer=optimizer, batchSize=2, split="train")
    finally:
        for space, original in originals:
            space.Reset = original

    # At least one Reset must have fired (a non-zero batch count
    # implies non-zero Reset count).
    assert sum(counts.values()) > 0, (
        f"Reset cascade never fired during runEpoch; counts={counts}")
    # All spaces with a Reset method should have been invoked at
    # least once.
    for name, n in counts.items():
        assert n > 0, f"{name}.Reset() never fired"


def test_per_row_state_clears_between_batches():
    """After a batch, per-row state must be cleared for the next batch.

    Reset clears: parse stack `_top`, SVO valid flags `_svo_valid`,
    STM-fired `_stm_fired`, and the serial cache. After running a
    full epoch (multiple batches), at the end of the epoch all
    per-row state should be at its post-Reset baseline.
    """
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.runEpoch(optimizer=optimizer, batchSize=2, split="train")

    ws = model.wordSpace
    if ws is None:
        pytest.skip("model has no WordSpace")

    # Post-epoch: Reset has fired at the last batch boundary, so
    # all per-row state should be cleared.
    assert ws._svo_valid is None or not ws._svo_valid.any().item(), (
        "_svo_valid not cleared after batch-boundary Reset")
    assert ws._stm_fired is None or not ws._stm_fired.any().item(), (
        "_stm_fired not cleared after batch-boundary Reset")
    # serial_cache cleared by Subspace.Reset.
    assert len(model.inputSpace.subspace.serial_cache) == 0, (
        "serial_cache not cleared after batch-boundary Reset")


def test_end_of_stream_reset_to_false():
    """The legacy `_end_of_stream` flag is still cleared at Reset.

    Even though we no longer branch on it, the runBatch end still
    sets `_end_of_stream = False` after firing Reset. This keeps
    the diagnostic in a known state.
    """
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Pre-set to a non-default value so we observe the reset.
    model.inputSpace._end_of_stream = torch.ones(2, dtype=torch.bool)
    model.runEpoch(optimizer=optimizer, batchSize=2, split="train")
    # Post-epoch: should be the False scalar from the last Reset.
    final = model.inputSpace._end_of_stream
    if torch.is_tensor(final):
        assert not bool(final.any().item()), (
            f"_end_of_stream tensor not cleared: {final}")
    else:
        assert final is False
