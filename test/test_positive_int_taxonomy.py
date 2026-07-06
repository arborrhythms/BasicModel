"""Stage 3+4: positive-integer taxonomy + ``.where``-keyed lookup tables.

End-to-end test for the new taxonomy keying introduced by
[`doc/plans/2026-05-28-where-keyed-taxonomy.md`](../doc/plans/2026-05-28-where-keyed-taxonomy.md):

  * Every ``insert_*`` allocates a position via
    :meth:`WholeSpace.allocate_position`. The return is the position
    (a positive int), not a signed-int row reference.
  * ``meta_pair_to_idx[(ps_pos, ws_pos)] -> meta_pos`` keys positive
    ints to positive ints. Idempotent on the pair.
  * ``taxonomy_children`` / ``taxonomy_parent`` consume positive ints.
  * ``_pos_kind[pos]`` tags each position as ``"ps"``, ``"ws"``, or
    ``"meta"``. PS/SS/META each live in the unified position namespace
    but resolve to different codebook rows via the
    ``_ps_pos_to_row`` / ``_ws_pos_to_row`` indirection.
  * ``RadixLayer.position_for(row)`` / ``row_for_position(pos)`` mirror
    the SS-side lookup tables.
  * ``vocab_extras`` persists every table; load is backwards-compatible
    with legacy signed-int blobs (rekey on load).

The sign-convention helpers (``_ps_signed`` / ``_ws_signed`` /
``_ps_row_of`` / ``_ws_row_of``) are retired in this stage; tests that
hand-constructed signed refs migrate to positive-int positions.
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
    """Build the MM_xor radix-chunking model for end-to-end tests."""
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
# Positive-int returns from insert_*.
# ---------------------------------------------------------------------------


class TestInsertReturnsPositiveInt(unittest.TestCase):
    """Each ``insert_*`` allocates a position and returns it as a positive int."""

    def test_insert_percept_returns_positive_position(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        pos = ws.insert_percept(b"alpha")
        self.assertIsInstance(pos, int)
        self.assertGreater(pos, 0,
                           "insert_percept must return a positive position "
                           f"(allocated via allocate_position); got {pos}")
        # Repeat insert of the same bytes returns the SAME position.
        again = ws.insert_percept(b"alpha")
        self.assertEqual(again, pos,
                         "idempotent insert: same bytes -> same position")
        # _pos_kind tags this as a PS slot.
        self.assertEqual(ws._pos_kind.get(pos), "ps",
                         "PS-side insert should tag _pos_kind[pos]='ps'")

    def test_insert_whole_returns_positive_position(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        init = torch.zeros(int(ws.nDim))
        init[0] = 0.5
        pos = ws.insert_whole(init_vec=init)
        self.assertIsInstance(pos, int)
        self.assertGreater(pos, 0,
                           f"insert_whole must return a positive position; got {pos}")
        self.assertEqual(ws._pos_kind.get(pos), "ws",
                         "SS-side insert should tag _pos_kind[pos]='ws'")
        # The position resolves to a real SS row.
        ws_row = ws._ws_pos_to_row[pos]
        self.assertIsInstance(ws_row, int)
        self.assertGreaterEqual(ws_row, 0)
        # Inverse direction works too.
        self.assertEqual(ws._ws_row_to_pos[ws_row], pos)

    def test_insert_meta_returns_positive_position_and_keys_pair(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_pos = ws.insert_percept(b"beta")
        ws_pos = ws.insert_whole()
        meta_pos = ws.insert_meta(ps_pos, ws_pos)
        self.assertIsInstance(meta_pos, int)
        self.assertGreater(meta_pos, 0,
                           f"insert_meta must return a positive position; got {meta_pos}")
        self.assertEqual(ws._pos_kind.get(meta_pos), "meta",
                         "META insert should tag _pos_kind[pos]='meta'")
        # Idempotent on the pair.
        again = ws.insert_meta(ps_pos, ws_pos)
        self.assertEqual(again, meta_pos)
        # meta_pair_to_idx keyed by positive (ps_pos, ws_pos).
        self.assertEqual(ws.meta_pair_to_idx[(ps_pos, ws_pos)], meta_pos)

    def test_insert_meta_rejects_zero_or_negative(self):
        """Position 0 is the anchor; negative inputs are not legal under
        the new convention."""
        m = _make_radix_model()
        ws = m.wholeSpace
        with self.assertRaises((ValueError, AssertionError)):
            ws.insert_meta(0, 0)
        with self.assertRaises((ValueError, AssertionError)):
            ws.insert_meta(-1, -2)


# ---------------------------------------------------------------------------
# Taxonomy storage uses positive ints throughout.
# ---------------------------------------------------------------------------


class TestTaxonomyStoragePositiveInt(unittest.TestCase):
    """Taxonomy / parent / meta_pair maps key on positive ints."""

    def test_taxonomy_children_and_parent_use_positions(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_pos = ws.insert_percept(b"gamma")
        ws_pos = ws.insert_whole()
        meta_pos = ws.insert_meta(ps_pos, ws_pos)

        children = ws.taxonomy_children(meta_pos)
        self.assertCountEqual(children, [ps_pos, ws_pos],
                              "taxonomy_children(meta_pos) must list the "
                              "positive PS + SS child positions")
        # All children are positive.
        for c in children:
            self.assertGreater(c, 0,
                               f"every child position must be positive; got {c}")

        self.assertEqual(ws.taxonomy_parent(ps_pos), meta_pos)
        self.assertEqual(ws.taxonomy_parent(ws_pos), meta_pos)
        self.assertTrue(ws.is_meta(meta_pos))
        self.assertFalse(ws.is_meta(ps_pos))
        self.assertFalse(ws.is_meta(ws_pos))


# ---------------------------------------------------------------------------
# vocab_extras roundtrip with positive-int keys + legacy migration.
# ---------------------------------------------------------------------------


class TestVocabExtrasPositiveInt(unittest.TestCase):
    """``vocab_extras`` persists positive-int taxonomy + lookup tables."""

    def test_roundtrip_preserves_positive_int_keys(self):
        m = _make_radix_model()
        ws = m.wholeSpace
        ps_pos = ws.insert_percept(b"delta")
        ws_pos = ws.insert_whole()
        meta_pos = ws.insert_meta(ps_pos, ws_pos)
        blob = ws.vocab_extras()

        # Every key in the taxonomy persist blob is a positive int.
        for k in blob.get("taxonomy", {}).keys():
            self.assertGreater(int(k), 0,
                               f"taxonomy keys must be positive; got {k}")
        for k in blob.get("taxonomy_parent", {}).keys():
            self.assertGreater(int(k), 0,
                               f"taxonomy_parent keys must be positive; got {k}")
        # meta_pair_to_idx's stringified pair is "ps_pos,ws_pos" — both positive.
        for raw_key, meta_i in blob.get("meta_pair_to_idx", {}).items():
            a, b = raw_key.split(",")
            self.assertGreater(int(a), 0)
            self.assertGreater(int(b), 0)
            self.assertGreater(int(meta_i), 0)

        # Load into a fresh model and verify state restored.
        m2 = _make_radix_model()
        ss2 = m2.wholeSpace
        ss2.load_vocab_extras(blob)
        self.assertEqual(ss2.taxonomy_children(meta_pos), [ps_pos, ws_pos])
        self.assertEqual(ss2.taxonomy_parent(ps_pos), meta_pos)
        self.assertEqual(ss2.meta_pair_to_idx[(ps_pos, ws_pos)], meta_pos)

    def test_legacy_signed_int_blob_migrates_on_load(self):
        """A pre-Stage-3 ``vocab_extras`` blob with signed-int keys
        rekeys to positive ints on load."""
        m = _make_radix_model()
        ws = m.wholeSpace
        # Hand-construct a legacy blob: positive PS keys, negative SS / META keys.
        legacy_blob = {
            "well_known_atoms": {},
            "paired_orth_to_sem": {},
            "paired_next_row": -1,
            # taxonomy: meta_signed (negative) -> [ps_signed (positive),
            #                                       ws_signed (negative)]
            "taxonomy": {
                -3: [0, -1],  # META at signed -3 binds PS row 0 + SS row 0
            },
            "taxonomy_parent": {
                0: -3,    # PS row 0's parent is META -3
                -1: -3,   # SS row 0's parent is META -3
            },
            "meta_pair_to_idx": {
                "0,-1": -3,   # (ps_row=0, ws_signed=-1) -> meta_signed=-3
            },
        }
        ws.load_vocab_extras(legacy_blob)
        # After load every taxonomy key is positive.
        for k in ws.taxonomy.keys():
            self.assertGreater(k, 0,
                               f"after legacy migration, taxonomy keys "
                               f"must be positive; got {k}")
        for k in ws.taxonomy_parent_map.keys():
            self.assertGreater(k, 0)
        for (a, b) in ws.meta_pair_to_idx.keys():
            self.assertGreater(a, 0)
            self.assertGreater(b, 0)


if __name__ == "__main__":
    unittest.main()
