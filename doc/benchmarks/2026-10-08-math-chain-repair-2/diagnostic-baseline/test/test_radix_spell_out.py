"""RadixLayer.spell_out: longest-percept emission with byte spell-out.

2026-06-04 radix model: "send the next-largest percept, up to the size of
the word; that percept may be the percept of a single byte." Unfamiliar
words are spelled out as a run of byte/prefix percepts; as a byte-sequence
recurs it is concatenated (promoted) into one larger percept. Crucially,
EVERY emitted id indexes a single ``.what`` codebook row -- which is what
lets the SubSpace store a ``.active`` selection and materialize percepts
without copying their vectors.
"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = str(Path(__file__).resolve().parent.parent / "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import RadixLayer


class TestSpellOut(unittest.TestCase):
    def _rl(self):
        return RadixLayer(4, initial_cap=8)

    def test_cold_word_spells_out_to_bytes(self):
        rl = self._rl()
        pids = rl.spell_out(b"hello")
        # No multi-byte percept known yet -> one percept per byte.
        self.assertEqual(len(pids), len(b"hello"))
        # Repeated byte reuses the same percept row.
        self.assertEqual(pids[2], pids[3], "both 'l' bytes share one percept")
        # Round-trips byte-for-byte.
        self.assertEqual(b"".join(rl.bytes_for(p) for p in pids), b"hello")

    def test_longest_match_after_promotion(self):
        rl = self._rl()
        rl.spell_out(b"hello")            # seed the byte-percepts
        rl.insert(b"hel")                 # concatenate a prefix percept
        pids = rl.spell_out(b"hello")
        self.assertEqual(len(pids), 3, "hello -> [hel, l, o]")
        self.assertEqual(rl.bytes_for(pids[0]), b"hel")
        self.assertEqual(b"".join(rl.bytes_for(p) for p in pids), b"hello")

    def test_full_word_promotion_is_single_percept(self):
        rl = self._rl()
        rl.spell_out(b"hello")
        rl.insert(b"hello")
        pids = rl.spell_out(b"hello")
        self.assertEqual(pids, [rl.get_id(b"hello")])
        self.assertEqual(len(pids), 1)

    def test_every_pid_is_a_valid_codebook_row(self):
        rl = self._rl()
        pids = rl.spell_out(b"hello world")
        V = int(rl.codebook.shape[0])
        for p in pids:
            self.assertTrue(0 <= p < V,
                            f"pid {p} must index a valid .what row [0,{V})")

    def test_empty_chunk_emits_nothing(self):
        rl = self._rl()
        self.assertEqual(rl.spell_out(b""), [])


if __name__ == "__main__":
    unittest.main()
