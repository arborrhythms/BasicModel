"""Tests confirming the per-batch EOS sync at runBatch end is gone.

The legacy gate read `inputSpace._end_of_stream` (a [B] bool tensor)
and called `.all().item()` to decide whether to fire the per-sentence
Reset cascade. That `.item()` was a per-batch GPU->CPU sync that
prevented CUDA-graph capture and idled the SMs every step.

The rolling-cursor handoff dispatched Reset per-row from
``next_tick``'s ``hard_eos`` signal in the outer loop, removing the
in-brick gate entirely. The brick-vectorization handoff then deleted
the ``batch_advances_sentence`` stub property (it had no remaining
readers); these tests assert that absence.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md
"""
import sys
import os
import inspect
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


def test_batch_advances_sentence_property_removed():
    """The legacy stub property has been deleted.

    The predecessor (per-row-AR-no-eos-sync) handoff introduced
    ``batch_advances_sentence`` as a host-side gate replacing the
    tensor-typed ``_end_of_stream.all().item()``. The rolling-cursor
    handoff then moved Reset to the outer loop and stopped consulting
    the property. With no remaining readers, the brick-vectorization
    handoff deletes it outright. This test guards against accidental
    resurrection.
    """
    model = _model()
    assert not hasattr(model.inputSpace, 'batch_advances_sentence'), (
        "batch_advances_sentence property should be deleted; the per-row "
        "Reset cascade in the outer doc-streaming loop is the canonical "
        "gate now (see plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md)"
    )


def test_runBatch_source_does_not_consume_eos_for_control_flow():
    """Under the rolling-cursor handoff, the EOS gate is gone entirely.

    runBatch is now a pure compute brick: no Reset cascade, no
    truth_layer.compact, no host sync on _end_of_stream. The Reset
    decision lives in the outer doc-streaming loop (runEpoch), driven
    by ``next_tick``'s host-side ``hard_eos`` per-row signal.

    Plan reference: doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md
    """
    from Models import BasicModel
    src = inspect.getsource(BasicModel.runBatch)
    # Reset cascade must be out of runBatch's source.
    assert "space.Reset()" not in src, (
        "runBatch must not call space.Reset() inline -- the rolling-cursor "
        "handoff relocated the Reset cascade to the outer doc-streaming loop"
    )
    # Any remaining .item() on _end_of_stream-related values must be
    # behind a MODEL_DEBUG guard (the only acceptable place).
    if "_end_of_stream" in src and ".item()" in src:
        lines = src.split("\n")
        for i, line in enumerate(lines):
            if ".item()" in line and "_end_of_stream" not in line:
                continue
            if ".item()" in line:
                guarded = any("MODEL_DEBUG" in lines[j]
                              for j in range(max(0, i - 10), i))
                assert guarded, (
                    f"Found unguarded .item() on _end_of_stream-related "
                    f"value at line {i}: {line!r}; must be inside a "
                    f"MODEL_DEBUG branch")


def test_reset_fires_even_when_end_of_stream_is_all_false():
    """Pre-set `_end_of_stream` to all-False; Reset must still fire.

    Under the legacy gate `eos.all().item() == False` would have
    blocked the Reset cascade. Under the brick-vectorization handoff
    (§8c), ``_end_of_stream`` is a host-side ``list[bool]`` diagnostic
    that is no longer consulted for control flow — the cursor's
    ``hard_eos`` list is the canonical signal. Reset fires regardless
    of the diagnostic's value.
    """
    import util as _util
    prior = _util.MODEL_DEBUG
    _util.init_model_debug(False)
    try:
        model = _model()
        reset_count = {"n": 0}
        originals = []
        for space in model.spaces:
            if hasattr(space, 'Reset'):
                original = space.Reset
                originals.append((space, original))

                def _spy(_self=space, _orig=original):
                    reset_count["n"] += 1
                    return _orig()
                space.Reset = _spy
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            # Pre-set _end_of_stream to a list[bool] of all-False —
            # the legacy gate's "no row reached EOS" signal.
            model.inputSpace._end_of_stream = [False, False]
            model.runEpoch(optimizer=optimizer, batchSize=2, split="train")
        finally:
            for space, original in originals:
                space.Reset = original

        assert reset_count["n"] > 0, (
            "Reset cascade did not fire under all-False _end_of_stream; "
            "the new gate should fire it on every AR batch regardless")
    finally:
        _util.init_model_debug(prior)
