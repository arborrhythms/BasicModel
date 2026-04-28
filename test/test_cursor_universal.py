"""Cursor-universal end-to-end tests (§8e).

The brick-vectorization handoff §8e unified the data path so non-AR /
numeric data flows through ``SentenceStreamDataset.next_tick`` in
trial mode (each tick = one batch of trials, ``hard_eos = [True] * B``).
The runEpoch outer loop drives ``ds.next_tick()`` directly for both
the byte-cursor (AR text byte) and trial-cursor (non-AR / numeric)
paths; there is no longer a separate DataLoader-iteration branch.

These tests exercise the trial-cursor end-to-end through ``runEpoch``
to catch regressions in the unified path.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md §8e
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest


def test_xor_runs_through_trial_cursor():
    """XOR (non-AR text, label classifier) trains via the trial cursor.

    XOR doesn't use AR prediction (single text frame per trial, label
    output). Under §8e the runEpoch outer loop drives the cursor for
    every dataset; XOR ends up in trial mode (slab_bytes=None, one
    tick per trial). Sanity: one full epoch runs without crashing,
    Reset fires for every row each tick.
    """
    from data import TheData
    from Models import BaseModel

    config = str(_project / "data" / "MM_xor.xml")
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = BaseModel.from_config(config, data=TheData)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    reset_count = {"n": 0}
    originals = []
    for space in model.spaces:
        if hasattr(space, 'Reset'):
            original = space.Reset
            originals.append((space, original))

            def _spy(*args, _orig=original, **kwargs):
                reset_count["n"] += 1
                return _orig(*args, **kwargs)
            space.Reset = _spy
    try:
        model.runEpoch(optimizer=optimizer, batchSize=2, split="train")
    finally:
        for space, original in originals:
            space.Reset = original

    # Trial-cursor mode dispatches one Reset per row per tick. Total
    # invocations across the epoch must be > 0 (i.e. the trial cursor
    # actually drove ticks through the unified runEpoch path).
    assert reset_count["n"] > 0, (
        "Reset cascade never fired during runEpoch via trial cursor; "
        "the unified §8e dispatch is broken")


def test_trial_cursor_no_dataloader_iteration():
    """The unified runEpoch must not iterate the DataLoader.

    Under §8e the outer loop drives ``ds.next_tick()`` directly for
    both modes; the surrounding DataLoader is built only because
    existing tests access ``loader.dataset``. This source-inspection
    test guards against accidental reintroduction of the
    ``for inp_items, out_items in loader:`` pattern.
    """
    import inspect
    from Models import BasicModel

    src = inspect.getsource(BasicModel.runEpoch)
    # The unified path uses ``while not ds.all_done(): ... ds.next_tick()``.
    # The legacy non-cursor path used ``for ... in loader:`` -- a
    # regression here would resurrect the dual-branch dispatch.
    code_lines = [
        line.split("#", 1)[0]
        for line in src.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert "for inp_items, out_items in loader" not in code_only, (
        "runEpoch reintroduced DataLoader iteration; the §8e unified "
        "cursor must drive ``ds.next_tick()`` directly.")
    assert "ds.next_tick()" in code_only, (
        "runEpoch lost the next_tick dispatch; the cursor-universal "
        "path is missing")


def test_trial_cursor_three_tuple_signature():
    """``next_tick`` returns a 3-tuple ``(input, output, hard_eos)``
    in both byte-cursor and trial-cursor modes.

    The unified dispatch contract (§8e) requires this signature so
    runEpoch can handle both modes via one ``ds.next_tick()`` call.
    """
    from data import SentenceStreamDataset

    # Byte cursor
    docs = ["hello", "world"]
    ds_byte = SentenceStreamDataset(docs, num_streams=2, slab_bytes=64)
    inp_b, out_b, hard_b = ds_byte.next_tick()
    assert hard_b == [True, True]
    assert out_b is None  # AR is self-supervised; no separate target
    assert inp_b.shape == (2, 64)

    # Trial cursor
    ds_trial = SentenceStreamDataset(
        ["a", "b", "c", "d"], num_streams=2)  # no slab_bytes
    inp_t, out_t, hard_t = ds_trial.next_tick()
    assert hard_t == [True, True]
    # outputs is None when no labels were passed -> output is None
    assert out_t is None
    # Trial mode yields the raw input items (one per row)
    assert len(inp_t) == 2
