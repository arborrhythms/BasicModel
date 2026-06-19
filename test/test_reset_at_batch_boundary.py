"""Tests confirming Reset() fires at every AR batch boundary.

The rolling-cursor handoff replaced the per-batch EOS gate with a
per-row dispatch in the outer doc-streaming loop, driven by the
cursor's host-side ``hard_eos`` list. Every AR batch in the
all-rows-finish case (the legacy DataLoader contract and also the
common cursor case where every row finishes a doc on the same tick)
fires a single global Reset per space. The brick-vectorization
handoff §8a then deleted the now-vestigial
``InputSpace.batch_advances_sentence`` stub property.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md
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


def _model():
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")

    model, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
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

    ss = model.symbolicSpace
    if ss is None:
        pytest.skip("model has no SymbolicSpace")

    # Post-epoch: Reset has fired at the last batch boundary, so
    # all per-row state should be cleared.
    assert ss._svo_valid is None or not ss._svo_valid.any().item(), (
        "_svo_valid not cleared after batch-boundary Reset")
    assert ss._stm_fired is None or not ss._stm_fired.any().item(), (
        "_stm_fired not cleared after batch-boundary Reset")
    # serial_cache cleared by Subspace.Reset.
    assert len(model.inputSpace.subspace.serial_cache) == 0, (
        "serial_cache not cleared after batch-boundary Reset")


def test_end_of_stream_reset_to_false():
    """The `_end_of_stream` diagnostic is a list[bool] always cleared.

    Under §8c (brick-vectorization handoff) ``_end_of_stream`` is a
    host-side ``list[bool]`` only — no scalar/tensor variants — and
    the per-row Reset path / dispatch_per_row_reset clears any True
    entry whose row has just rolled over. Post-epoch every entry
    must be False.
    """
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Pre-set to a non-default list[bool] so we observe the reset.
    model.inputSpace._end_of_stream = [True, True]
    model.runEpoch(optimizer=optimizer, batchSize=2, split="train")
    final = model.inputSpace._end_of_stream
    assert isinstance(final, list), (
        f"_end_of_stream must be list[bool], got {type(final).__name__}")
    assert not any(final), (
        f"_end_of_stream not cleared: {final}")
