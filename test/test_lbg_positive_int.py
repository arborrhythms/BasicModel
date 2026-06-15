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

    def test_record_pull_on_ss_position_accumulates(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        ss_pos = ss.insert_symbol()
        D = int(ss.nDim)
        vec = torch.zeros(D)
        vec[0] = 1.0
        # First pull seeds the accumulators.
        ss.record_lbg_pull(ss_pos, vec)
        self.assertIn(ss_pos, ss._lbg_count,
                      "first record_lbg_pull must seed _lbg_count")
        self.assertEqual(ss._lbg_count[ss_pos], 1)
        # Second pull increments count.
        ss.record_lbg_pull(ss_pos, vec)
        self.assertEqual(ss._lbg_count[ss_pos], 2)

    def test_record_pull_on_ps_position_is_a_noop(self):
        """PS positions don't get split; LBG should ignore them silently."""
        m = _make_radix_model()
        ss = m.symbolicSpace
        ps_pos = ss.insert_percept(b"lbg_ps")
        D = int(ss.nDim)
        vec = torch.zeros(D)
        vec[0] = 1.0
        ss.record_lbg_pull(ps_pos, vec)
        self.assertNotIn(ps_pos, ss._lbg_count,
                         "PS positions must NOT seed LBG accumulators")

    def test_record_pull_raises_on_nan(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        ss_pos = ss.insert_symbol()
        D = int(ss.nDim)
        bad = torch.full((D,), float("nan"))
        with self.assertRaises(RuntimeError) as ctx:
            ss.record_lbg_pull(ss_pos, bad)
        self.assertIn("NaN/Inf", str(ctx.exception))


class TestMaybeSplitLbgPositions(unittest.TestCase):
    """``maybe_split_lbg`` allocates a fresh position on a successful split."""

    def test_no_split_below_threshold(self):
        m = _make_radix_model()
        ss = m.symbolicSpace
        ss_pos = ss.insert_symbol()
        # No data, no split.
        result = ss.maybe_split_lbg(ss_pos)
        self.assertIsNone(result,
                          "maybe_split_lbg with empty counts must return None")

    def test_split_allocates_fresh_position(self):
        """Forcing the variance state past the threshold triggers a split
        and returns a new positive-int position."""
        m = _make_radix_model()
        ss = m.symbolicSpace
        ss_pos = ss.insert_symbol()
        # Force enough pulls + high variance to trip the split.
        D = int(ss.nDim)
        # Alternate +1 and -1 to drive variance up while keeping the
        # mean direction non-zero.
        plus = torch.zeros(D)
        plus[0] = 5.0
        minus = torch.zeros(D)
        minus[0] = -5.0
        # Need at least _lbg_min_count pulls.
        n = max(int(ss._lbg_min_count) + 2, 12)
        for i in range(n):
            ss.record_lbg_pull(ss_pos, plus if i % 2 == 0 else minus)
        # Sanity: the accumulators are seeded.
        self.assertGreaterEqual(ss._lbg_count[ss_pos], int(ss._lbg_min_count))
        new_pos = ss.maybe_split_lbg(ss_pos)
        self.assertIsNotNone(new_pos,
                             "maybe_split_lbg should fire with high-variance "
                             "alternating pulls past min_count")
        self.assertIsInstance(new_pos, int)
        self.assertGreater(new_pos, 0,
                           f"split returns a positive position; got {new_pos}")
        self.assertNotEqual(new_pos, ss_pos,
                            "the new position must differ from the original")
        # The new position is bound to an SS-side row and tagged "ss".
        self.assertEqual(ss._pos_kind.get(new_pos), "ss",
                         "split-off position must be tagged 'ss'")
        self.assertIn(new_pos, ss._ss_pos_to_row,
                      "split-off position must have an _ss_pos_to_row entry")
        # Accumulators for the original position are cleared.
        self.assertNotIn(ss_pos, ss._lbg_count)

    def test_split_inherits_meta_binding_to_new_position(self):
        """If the original SS row was bound under a META, the split-off
        gets its own META edge so reverse decode can still reach it."""
        m = _make_radix_model()
        ss = m.symbolicSpace
        ps_pos = ss.insert_percept(b"lbg_split_meta")
        ss_pos = ss.insert_symbol()
        meta_pos = ss.insert_meta(ps_pos, ss_pos)
        # Drive a split on the SS child.
        D = int(ss.nDim)
        plus = torch.zeros(D)
        plus[0] = 5.0
        minus = torch.zeros(D)
        minus[0] = -5.0
        n = max(int(ss._lbg_min_count) + 2, 12)
        for i in range(n):
            ss.record_lbg_pull(ss_pos, plus if i % 2 == 0 else minus)
        new_pos = ss.maybe_split_lbg(ss_pos)
        self.assertIsNotNone(new_pos)
        # The new position must have a META binding too -- look up its
        # parent and verify it's tagged "meta" and includes the original
        # PS child in its children list.
        new_parent = ss.taxonomy_parent(new_pos)
        self.assertIsNotNone(new_parent,
                             "split-off must inherit a META binding")
        self.assertTrue(ss.is_meta(new_parent),
                        "the parent must be a META node")
        new_children = ss.taxonomy_children(new_parent)
        self.assertIn(ps_pos, new_children,
                      "the inherited META must keep the original PS child")
        self.assertIn(new_pos, new_children,
                      "the inherited META must list the split-off position")


if __name__ == "__main__":
    unittest.main()
