"""Tests confirming the rolling-cursor loader loses no input bytes.

Under the rolling-cursor design, ``SentenceStreamDataset.next_tick()``
walks each row's documents one ``slab_bytes``-wide tick at a time.
Concatenating the populated prefixes of consecutive ticks for one row
must reproduce the original document bytes exactly — no truncation,
no byte loss, no spurious padding contribution.

Plan reference: doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md §Verification
"""
import os
import random
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch

from data import SentenceStreamDataset


def _gen_doc(rng, min_bytes=10, max_bytes=8000):
    """Build a UTF-8 doc whose byte length is somewhere in the range.

    Uses a mix of ASCII and multi-byte characters so the byte-vs-char
    distinction is exercised (encode('utf-8') ≠ len()).
    """
    n_chars = rng.randint(min_bytes, max_bytes)
    chunks = []
    while len(''.join(chunks).encode('utf-8')) < min_bytes:
        cp = rng.choice([
            rng.randint(0x20, 0x7e),                # ASCII printable
            rng.randint(0xa1, 0xff),                # Latin-1 supplement
            rng.choice([0x2014, 0x2026, 0x2122]),   # punctuation / symbols
        ])
        chunks.append(chr(cp))
        if len(''.join(chunks).encode('utf-8')) >= n_chars:
            break
    return ''.join(chunks)


def _drain_row(ds, row, max_ticks=10000):
    """Walk next_tick until row crosses its first hard_eos.

    Returns the concatenated bytes of the populated prefixes for that row.
    The ``ds`` cursor is consumed in place.
    """
    out = bytearray()
    for _ in range(max_ticks):
        slab, _, hard_eos = ds.next_tick()
        # Find the populated prefix length: bytes after the doc's tail
        # are zero-padded NULL. For docs containing 0x00 we'd need a
        # smarter strategy, but the docs we generate don't include NULL
        # so the trailing-NULL count is the padding.
        row_bytes = bytes(slab[row].tolist())
        # Active prefix = up to the first run of trailing NULLs.
        if hard_eos[row]:
            # On the doc-completion tick, the populated portion is
            # remaining = len(doc) - prior_offset. We can recover by
            # rstripping zeros (docs we generate never contain NULL).
            populated = row_bytes.rstrip(b'\x00')
            out.extend(populated)
            return bytes(out)
        out.extend(row_bytes)
    raise AssertionError(
        f"row {row}: did not hit hard_eos within {max_ticks} ticks")


def test_concat_per_tick_slabs_equals_original_doc_bytes():
    """For 100 random docs, per-row slab concatenation equals the doc bytes."""
    rng = random.Random(20260426)
    docs = [_gen_doc(rng) for _ in range(100)]
    # Pick a small num_streams so each stream owns several docs.
    num_streams = 2
    ds = SentenceStreamDataset(
        docs, num_streams=num_streams, slab_bytes=1024)

    # For each stream, walk the assigned doc window doc-by-doc and
    # confirm the accumulated bytes match.
    L = ds.stream_length
    for b in range(num_streams):
        # Fresh dataset per row so each row's first-doc trace is
        # uncontaminated by other rows' state.
        ds_b = SentenceStreamDataset(
            docs, num_streams=num_streams, slab_bytes=1024)
        for d in range(L):
            doc_idx = b * L + d
            expected = docs[doc_idx].encode('utf-8')
            got = _drain_row(ds_b, b)
            assert got == expected, (
                f"row {b} doc {doc_idx}: byte mismatch.\n"
                f"  expected ({len(expected)} bytes): {expected[:40]!r}...\n"
                f"  got      ({len(got)} bytes): {got[:40]!r}..."
            )


def test_short_doc_one_tick_eos():
    """A doc shorter than slab_bytes finishes in a single tick with hard_eos."""
    docs = ["hi", "world!"]
    ds = SentenceStreamDataset(docs, num_streams=2, slab_bytes=1024)
    slab, _, hard_eos = ds.next_tick()
    assert hard_eos == [True, True], (
        f"both rows should hit hard_eos in one tick, got {hard_eos}")
    assert bytes(slab[0].tolist()).rstrip(b'\x00') == b"hi"
    assert bytes(slab[1].tolist()).rstrip(b'\x00') == b"world!"


def test_long_doc_multi_tick_only_last_is_eos():
    """A doc of 2500 bytes with slab=1024 takes 3 ticks; only tick 3 has eos."""
    long_doc = "X" * 2500   # 2500 bytes (ASCII)
    short_doc = "Y" * 100
    # Two streams; row 0 owns long_doc, row 1 owns short_doc.
    docs = [long_doc, short_doc]
    ds = SentenceStreamDataset(docs, num_streams=2, slab_bytes=1024)

    # Tick 1: row 0 has 1024 bytes of long_doc, no eos. Row 1 finishes
    # short_doc in one tick (hard_eos=True).
    slab1, _, hard_eos1 = ds.next_tick()
    assert hard_eos1 == [False, True], (
        f"tick 1 hard_eos: expected [False, True], got {hard_eos1}")
    assert (slab1[0] == ord('X')).sum().item() == 1024

    # Tick 2: row 0 has another 1024 bytes (still 452 left), no eos.
    # Row 1 is exhausted (no more docs in its window).
    slab2, _, hard_eos2 = ds.next_tick()
    assert hard_eos2[0] is False
    assert (slab2[0] == ord('X')).sum().item() == 1024

    # Tick 3: row 0 has the final 452 bytes, hard_eos.
    slab3, _, hard_eos3 = ds.next_tick()
    assert hard_eos3[0] is True
    assert (slab3[0] == ord('X')).sum().item() == 452
    # Trailing positions should be NULL.
    assert (slab3[0, 452:] == 0).all().item() is True


def test_all_done_after_window_exhausted():
    """all_done() returns True iff every row has consumed its window."""
    docs = ["a", "b"]
    ds = SentenceStreamDataset(docs, num_streams=2, slab_bytes=64)
    assert not ds.all_done(), "fresh cursor must not report done"
    ds.next_tick()
    assert ds.all_done(), (
        "after one tick consuming both single-byte docs, all rows "
        "should be exhausted")


def test_reset_cursor_restarts_from_beginning():
    """reset_cursor() rewinds every row to the start of its window."""
    docs = ["alpha", "beta"]
    ds = SentenceStreamDataset(docs, num_streams=2, slab_bytes=64)
    ds.next_tick()
    assert ds.all_done()
    ds.reset_cursor()
    assert not ds.all_done(), "after reset_cursor we should iterate again"
    slab, _, hard_eos = ds.next_tick()
    assert hard_eos == [True, True]
    assert bytes(slab[0].tolist()).rstrip(b'\x00') == b"alpha"
    assert bytes(slab[1].tolist()).rstrip(b'\x00') == b"beta"


def test_trial_cursor_yields_one_trial_per_tick():
    """Trial-mode cursor (slab_bytes=None) yields one batch of trials
    per tick with hard_eos=[True]*B every tick.

    Brick-vectorization handoff §8e: the cursor is universal -- non-AR
    data flows through this trial mode rather than the legacy
    DataLoader iteration. Each trial completes immediately so every
    row's hard_eos fires every tick (the outer loop runs Reset on
    every row each tick, matching the pre-handoff DataLoader contract).
    """
    docs = ["a", "b", "c", "d"]  # 4 docs, num_streams=2 -> stream_length=2
    ds = SentenceStreamDataset(docs, num_streams=2)  # no slab_bytes
    assert not ds.all_done(), "fresh trial cursor must not report done"

    inp1, out1, hard_eos1 = ds.next_tick()
    assert hard_eos1 == [True, True]
    assert inp1 == ["a", "c"], f"step 0: row 0='a', row 1='c', got {inp1}"
    assert out1 is None, "outputs is None -> no output yielded"

    assert not ds.all_done(), "after 1 tick of 2-step stream, not done"
    inp2, out2, hard_eos2 = ds.next_tick()
    assert hard_eos2 == [True, True]
    assert inp2 == ["b", "d"], f"step 1: row 0='b', row 1='d', got {inp2}"

    assert ds.all_done(), "after 2 ticks of 2-step stream, must be done"


def test_trial_cursor_reset_rewinds_step():
    """reset_cursor() in trial mode returns to step 0."""
    docs = ["x", "y"]
    ds = SentenceStreamDataset(docs, num_streams=2)
    inp1, _, _ = ds.next_tick()
    assert ds.all_done()
    ds.reset_cursor()
    assert not ds.all_done(), "after reset_cursor, trial mode iterates again"
    inp2, _, _ = ds.next_tick()
    assert inp1 == inp2, f"reset cursor should yield same first batch"
