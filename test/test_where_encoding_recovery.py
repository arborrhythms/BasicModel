"""Stage 2: ``WhereEncoding.recover`` + ``WholeSpace.allocate_position``.

Covers the integer-recovery primitive on ``WhereEncoding`` and the
monotonic position counter on ``WholeSpace`` introduced by
[`doc/plans/2026-05-28-where-keyed-taxonomy.md`](../doc/plans/2026-05-28-where-keyed-taxonomy.md).

``recover(vec) -> int`` is the inverse view onto a sinusoidal encoding:
it delegates to ``QuadratureEncoding.decode`` (atan2-based), snaps to
the nearest integer, and validates the result is in ``[0, maxP)``.
``WhereEncoding(maxP, nWhere=0)`` (the disabled / no-op encoding form,
which is the current default everywhere) must raise — recovery has no
meaning when the encoding is zero-width.

``WholeSpace.allocate_position()`` returns monotonically increasing
positive ints starting at 1. Position 0 is reserved as the
frozen-zeros anchor (never allocated to a content row). The counter
survives ``vocab_extras`` save/load.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_radix_model():
    """Build the MM_xor radix-chunking model for fixture-backed tests."""
    import warnings
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    m.eval()
    return m


# ---------------------------------------------------------------------------
# ``WhereEncoding.recover`` RETIRED (2026-06-04, modality re-architecture
# Phase 4): .where no longer keys the codebook (identity is the row index),
# so the .where -> int identity inverse was removed and its
# TestWhereEncodingRecover suite is dropped with it. The position counter
# below (WholeSpace.allocate_position) is row/position-keyed, not
# .where-quadrature-based, and is kept.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ``WholeSpace.allocate_position`` — monotonic positive-int allocator.
# ---------------------------------------------------------------------------


class TestSymbolSpaceAllocatePosition(unittest.TestCase):
    """Monotonic position counter on WholeSpace; position 0 reserved."""

    def test_allocate_starts_at_one(self):
        """First call returns 1 (position 0 is the reserved anchor)."""
        m = _make_radix_model()
        ws = m.wholeSpace
        self.assertEqual(ws.allocate_position(), 1)

    def test_allocate_monotonic(self):
        """Subsequent calls return strictly increasing ints."""
        m = _make_radix_model()
        ws = m.wholeSpace
        positions = [ws.allocate_position() for _ in range(8)]
        self.assertEqual(positions, list(range(1, 9)),
                         "allocate_position must produce 1, 2, 3, …")

    def test_allocate_position_persists_via_vocab_extras(self):
        """``next_position`` survives a ``vocab_extras`` roundtrip."""
        m = _make_radix_model()
        ws = m.wholeSpace
        # Burn a few positions on the live SS.
        for _ in range(5):
            ws.allocate_position()
        # The next allocation would yield 6.
        self.assertEqual(ws.allocate_position(), 6)
        blob = ws.vocab_extras()
        self.assertIn("next_position", blob,
                      "vocab_extras must persist the position counter")
        self.assertEqual(int(blob["next_position"]), 7,
                         "next_position should point at the NEXT unused "
                         "position (7 after 1..6 were taken)")

        # Build a fresh model + apply the blob; new SS should resume the
        # counter rather than restart at 1.
        m2 = _make_radix_model()
        ss2 = m2.wholeSpace
        ss2.load_vocab_extras(blob)
        self.assertEqual(ss2.allocate_position(), 7,
                         "after load, the next allocation must be 7, not 1")

    def test_load_vocab_extras_without_counter_defaults_to_one(self):
        """Older blobs without ``next_position`` start the counter at 1."""
        m = _make_radix_model()
        ws = m.wholeSpace
        # Construct a minimal extras blob with no ``next_position`` key.
        legacy_blob = {
            "well_known_atoms": {},
            "paired_orth_to_sem": {},
            "paired_next_row": -1,
            "taxonomy": {},
            "taxonomy_parent": {},
            "meta_pair_to_idx": {},
        }
        ws.load_vocab_extras(legacy_blob)
        self.assertEqual(ws.allocate_position(), 1,
                         "legacy blob without next_position must default "
                         "the counter to 1")


if __name__ == "__main__":
    unittest.main()
