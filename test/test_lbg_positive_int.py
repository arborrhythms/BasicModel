"""LBG codebook splitting under the positive-int taxonomy
(doc/plans/2026-05-28-where-keyed-taxonomy.md Stage 4).

``record_lbg_pull(pos, vec)`` accumulates per-position displacement
statistics for SS-side rows; ``maybe_split_lbg(pos)`` triggers a
codebook-row split when the variance + count thresholds are met. The
split allocates a fresh position via :meth:`WholeSpace.allocate_position`
and -- when the original was a META child -- registers a new META edge
for the split-off row so both halves remain reachable via the reverse
decode walk.

PS positions are ignored by both methods (LBG only splits SS-side
rows).
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


class TestRecordLbgPullPositions(unittest.TestCase):
    """``record_lbg_pull`` accepts an SS-kind position and accumulates."""

    def test_record_pull_on_ws_position_accumulates(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        ws_pos = ws.insert_symbol()
        D = int(ws.nDim)
        vec = torch.zeros(D)
        vec[0] = 1.0
        # First pull seeds the accumulators.
        ws.record_lbg_pull(ws_pos, vec)
        self.assertIn(ws_pos, ws._lbg_count,
                      "first record_lbg_pull must seed _lbg_count")
        self.assertEqual(ws._lbg_count[ws_pos], 1)
        # Second pull increments count.
        ws.record_lbg_pull(ws_pos, vec)
        self.assertEqual(ws._lbg_count[ws_pos], 2)

    def test_record_pull_on_ps_position_is_a_noop(self):
        """PS positions don't get split; LBG should ignore them silently."""
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_pos = ws.insert_percept(b"lbg_ps")
        D = int(ws.nDim)
        vec = torch.zeros(D)
        vec[0] = 1.0
        ws.record_lbg_pull(ps_pos, vec)
        self.assertNotIn(ps_pos, ws._lbg_count,
                         "PS positions must NOT seed LBG accumulators")

    def test_record_pull_raises_on_nan(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        ws_pos = ws.insert_symbol()
        D = int(ws.nDim)
        bad = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            ws.record_lbg_pull(ws_pos, bad)
        self.assertIn("NaN/Inf", str(ctx.exception))


class TestMaybeSplitLbgPositions(unittest.TestCase):
    """``maybe_split_lbg`` allocates a fresh position on a successful split."""

    def test_no_split_below_threshold(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        ws_pos = ws.insert_symbol()
        # No data, no split.
        result = ws.maybe_split_lbg(ws_pos)
        self.assertIsNone(result,
                          "maybe_split_lbg with empty counts must return None")

    def test_split_allocates_fresh_position(self):
        """Forcing the variance state past the threshold triggers a split
        and returns a new positive-int position."""
        m = _make_radix_model()
        ws = m.wholeSpace
        ws_pos = ws.insert_symbol()
        # Force enough pulls + high variance to trip the split.
        D = int(ws.nDim)
        # Alternate +1 and -1 to drive variance up while keeping the
        # mean direction non-zero.
        plus = torch.zeros(D)
        plus[0] = 5.0
        minus = torch.zeros(D)
        minus[0] = -5.0
        # Need at least _lbg_min_count pulls.
        n = max(int(ws._lbg_min_count) + 2, 12)
        for i in range(n):
            ws.record_lbg_pull(ws_pos, plus if i % 2 == 0 else minus)
        # Sanity: the accumulators are seeded.
        self.assertGreaterEqual(ws._lbg_count[ws_pos], int(ws._lbg_min_count))
        new_pos = ws.maybe_split_lbg(ws_pos)
        self.assertIsNotNone(new_pos,
                             "maybe_split_lbg should fire with high-variance "
                             "alternating pulls past min_count")
        self.assertIsInstance(new_pos, int)
        self.assertGreater(new_pos, 0,
                           f"split returns a positive position; got {new_pos}")
        self.assertNotEqual(new_pos, ws_pos,
                            "the new position must differ from the original")
        # The new position is bound to an SS-side row and tagged "ws".
        self.assertEqual(ws._pos_kind.get(new_pos), "ws",
                         "split-off position must be tagged 'ws'")
        self.assertIn(new_pos, ws._ws_pos_to_row,
                      "split-off position must have an _ws_pos_to_row entry")
        # Accumulators for the original position are cleared.
        self.assertNotIn(ws_pos, ws._lbg_count)

    def test_split_inherits_meta_binding_to_new_position(self):
        """If the original SS row was bound under a META, the split-off
        gets its own META edge so reverse decode can still reach it."""
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_pos = ws.insert_percept(b"lbg_split_meta")
        ws_pos = ws.insert_symbol()
        meta_pos = ws.insert_meta(ps_pos, ws_pos)
        # Drive a split on the SS child.
        D = int(ws.nDim)
        plus = torch.zeros(D)
        plus[0] = 5.0
        minus = torch.zeros(D)
        minus[0] = -5.0
        n = max(int(ws._lbg_min_count) + 2, 12)
        for i in range(n):
            ws.record_lbg_pull(ws_pos, plus if i % 2 == 0 else minus)
        new_pos = ws.maybe_split_lbg(ws_pos)
        self.assertIsNotNone(new_pos)
        # The new position must have a META binding too -- look up its
        # parent and verify it's tagged "meta" and includes the original
        # PS child in its children list.
        new_parent = ws.taxonomy_parent(new_pos)
        self.assertIsNotNone(new_parent,
                             "split-off must inherit a META binding")
        self.assertTrue(ws.is_meta(new_parent),
                        "the parent must be a META node")
        new_children = ws.taxonomy_children(new_parent)
        self.assertIn(ps_pos, new_children,
                      "the inherited META must keep the original PS child")
        self.assertIn(new_pos, new_children,
                      "the inherited META must list the split-off position")


if __name__ == "__main__":
    unittest.main()
