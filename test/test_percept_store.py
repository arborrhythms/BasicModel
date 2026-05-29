"""PerceptStore unit tests (Stage 7 of the PerceptStore plan).

Covers the 6 acceptance items in
``doc/plans/2026-05-27-perceptstore-meta-taxonomy-reentrancy.md``
Stage 7 §Tests:

1. Radix trie insertion + longest-match.
2. Hash map cache consistency.
3. Inverse table exact roundtrip: ``percept_id -> bytes -> percept_id``.
4. Byte fallback encoding produces a vector + increments counter.
5. Promotion triggers after ``promotion_threshold`` hits and
   ``>= min_length``.
6. Codebook grows on insert; existing rows preserved.

Plus a few persistence smoke tests so the ``vocab_extras`` round trip
is exercised end-to-end.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


class TestRadixTrie(unittest.TestCase):
    """Item 1: radix trie insertion + longest-match."""

    def test_empty_trie_returns_none(self):
        from Layers import RadixTrie
        trie = RadixTrie()
        self.assertEqual(len(trie), 0)
        self.assertIsNone(trie.get(b"hello"))
        match_id, match_len = trie.longest_match(b"hello")
        self.assertIsNone(match_id)
        self.assertEqual(match_len, 0)

    def test_single_insert_and_exact_lookup(self):
        from Layers import RadixTrie
        trie = RadixTrie()
        self.assertTrue(trie.insert(b"hello", 0))
        self.assertEqual(len(trie), 1)
        self.assertEqual(trie.get(b"hello"), 0)
        self.assertIsNone(trie.get(b"hel"))

    def test_insert_returns_false_on_duplicate(self):
        from Layers import RadixTrie
        trie = RadixTrie()
        self.assertTrue(trie.insert(b"hi", 7))
        self.assertFalse(trie.insert(b"hi", 99),
                         "Re-inserting an existing key must return False")
        self.assertEqual(trie.get(b"hi"), 7,
                         "Existing percept_id must be preserved")

    def test_shared_prefix_forks_correctly(self):
        from Layers import RadixTrie
        trie = RadixTrie()
        trie.insert(b"cat", 1)
        trie.insert(b"car", 2)
        self.assertEqual(trie.get(b"cat"), 1)
        self.assertEqual(trie.get(b"car"), 2)
        self.assertEqual(len(trie), 2)

    def test_longest_match_returns_deepest_terminal(self):
        from Layers import RadixTrie
        trie = RadixTrie()
        trie.insert(b"cat", 1)
        trie.insert(b"category", 2)
        # "categoryweight" matches "category" (longer than "cat").
        match_id, match_len = trie.longest_match(b"categoryweight")
        self.assertEqual(match_id, 2)
        self.assertEqual(match_len, len(b"category"))

    def test_longest_match_returns_shorter_when_deep_terminal_missing(self):
        from Layers import RadixTrie
        trie = RadixTrie()
        trie.insert(b"cat", 1)
        # No "category" entry -- the input "categoryweight" only
        # matches the "cat" terminal.
        match_id, match_len = trie.longest_match(b"categoryweight")
        self.assertEqual(match_id, 1)
        self.assertEqual(match_len, 3)

    def test_insert_splits_existing_edge_at_partial_overlap(self):
        from Layers import RadixTrie
        trie = RadixTrie()
        trie.insert(b"hello", 1)
        # Forking on the shared prefix "he"; "help" forks off the old edge.
        trie.insert(b"help", 2)
        self.assertEqual(trie.get(b"hello"), 1)
        self.assertEqual(trie.get(b"help"), 2)
        # Lookup of a non-terminal prefix must still return None.
        self.assertIsNone(trie.get(b"hel"))


class TestHashMapCache(unittest.TestCase):
    """Item 2: hash map cache consistency."""

    def test_hash_map_mirrors_trie_after_insert(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=8, initial_cap=4)
        pid = ps.insert(b"hello")
        self.assertEqual(ps.hash_map[b"hello"], pid)
        self.assertEqual(ps.radix_trie.get(b"hello"), pid)
        # Hash-map presence + lookup agree.
        self.assertIn(b"hello", ps)
        self.assertEqual(ps.get_id(b"hello"), pid)

    def test_hash_map_unknown_keys_return_none(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=8)
        self.assertIsNone(ps.get_id(b"never_seen"))
        self.assertNotIn(b"never_seen", ps)


class TestInverseTableRoundtrip(unittest.TestCase):
    """Item 3: inverse table exact roundtrip."""

    def test_roundtrip_percept_id_to_bytes_to_percept_id(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=8, initial_cap=8)
        for word in (b"cat", b"dog", b"category", b"do"):
            ps.insert(word)
        for pid in range(len(ps)):
            recovered = ps.bytes_for(pid)
            self.assertEqual(ps.get_id(recovered), pid,
                             f"roundtrip failed for percept_id {pid}: "
                             f"bytes={recovered!r}")

    def test_inverse_table_indexing_is_exact(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, initial_cap=4)
        words = [b"alpha", b"beta", b"gamma"]
        ids = [ps.insert(w) for w in words]
        self.assertEqual(ids, [0, 1, 2])
        for w, pid in zip(words, ids):
            self.assertEqual(ps.bytes_for(pid), w)

    def test_invalid_percept_id_raises(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4)
        ps.insert(b"x")
        with self.assertRaises(IndexError):
            ps.bytes_for(7)
        with self.assertRaises(IndexError):
            ps.bytes_for(-1)


class TestByteFallback(unittest.TestCase):
    """Item 4: byte fallback encoding produces a vector + increments
    counter."""

    def test_byte_fallback_returns_a_vector(self):
        from Layers import BytesFallbackEncoder
        enc = BytesFallbackEncoder(dim=8)
        v = enc.encode(b"abc")
        self.assertEqual(v.shape, (8,))
        self.assertTrue(torch.isfinite(v).all().item())

    def test_byte_fallback_increments_hit_counter(self):
        from Layers import BytesFallbackEncoder
        enc = BytesFallbackEncoder(dim=4)
        self.assertEqual(enc.hits(b"foo"), 0)
        enc.encode(b"foo")
        self.assertEqual(enc.hits(b"foo"), 1)
        enc.encode(b"foo")
        enc.encode(b"foo")
        self.assertEqual(enc.hits(b"foo"), 3)
        # A different chunk has its own independent counter.
        self.assertEqual(enc.hits(b"bar"), 0)
        enc.encode(b"bar")
        self.assertEqual(enc.hits(b"bar"), 1)

    def test_lookup_via_percept_store_increments_hits(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, promotion_threshold=1000)
        v = ps.lookup(b"novel")
        self.assertEqual(v.shape, (4,))
        # The full chunk's hit counter must have been bumped.
        self.assertGreaterEqual(ps.byte_fallback.hits(b"novel"), 1)


class TestPromotion(unittest.TestCase):
    """Item 5: promotion triggers after threshold hits and >= min_length."""

    def test_promotion_after_threshold_hits(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, promotion_threshold=3,
                          promotion_min_length=2)
        self.assertNotIn(b"abc", ps)
        # Hit count must reach the threshold before promotion fires.
        for i in range(2):
            ps.lookup(b"abc")
            self.assertNotIn(b"abc", ps,
                             f"abc promoted prematurely on hit {i + 1}")
        # The 3rd hit should trigger promotion (hit count reaches 3).
        ps.lookup(b"abc")
        self.assertIn(b"abc", ps,
                      "abc should have been promoted after 3 hits")
        # Now it's a hash-map hit, no further hit counter mutation.
        prev_hits = ps.byte_fallback.hits(b"abc")
        ps.lookup(b"abc")
        self.assertEqual(ps.byte_fallback.hits(b"abc"), prev_hits,
                         "promoted chunks should not bump the fallback "
                         "counter")

    def test_promotion_respects_min_length(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, promotion_threshold=2,
                          promotion_min_length=3)
        # "ab" has length 2, below the min_length=3 threshold.
        for _ in range(5):
            ps.lookup(b"ab")
        self.assertNotIn(b"ab", ps,
                         "promotion should be blocked for chunks shorter "
                         "than promotion_min_length")
        # "abc" has length 3, at the threshold.
        ps.lookup(b"abc")
        ps.lookup(b"abc")
        self.assertIn(b"abc", ps,
                      "abc should have promoted after 2 hits (>= "
                      "promotion_threshold=2)")


class TestCodebookGrowth(unittest.TestCase):
    """Item 6: codebook grows on insert; existing rows preserved."""

    def test_codebook_grows_when_capacity_exceeded(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, initial_cap=2)
        self.assertEqual(ps.capacity, 2)
        ids = []
        for w in (b"a", b"b", b"c", b"d", b"e"):
            ids.append(ps.insert(w))
        self.assertEqual(ids, [0, 1, 2, 3, 4])
        self.assertGreaterEqual(ps.capacity, 5,
                                "Capacity must have grown to fit 5 inserts")

    def test_existing_codebook_rows_preserved_through_growth(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, initial_cap=2)
        ps.insert(b"a")
        ps.insert(b"b")
        # Snapshot rows before growth.
        row_a = ps.codebook[0].detach().clone()
        row_b = ps.codebook[1].detach().clone()
        # Force growth.
        ps.insert(b"c")
        self.assertGreaterEqual(ps.capacity, 3)
        # Rows 0 and 1 must remain bit-identical after the grow.
        self.assertTrue(torch.equal(ps.codebook[0].detach(), row_a),
                        "Row 0 changed after codebook growth")
        self.assertTrue(torch.equal(ps.codebook[1].detach(), row_b),
                        "Row 1 changed after codebook growth")

    def test_capacity_doubles_on_overflow(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, initial_cap=2)
        ps.insert(b"a")
        ps.insert(b"b")
        self.assertEqual(ps.capacity, 2)
        ps.insert(b"c")
        self.assertEqual(ps.capacity, 4,
                         "PerceptStore should double its codebook on "
                         "overflow")


class TestForwardLookupPath(unittest.TestCase):
    """Integrated forward-lookup behaviour: hash-map hit returns the
    permanent row; full unknown returns a fallback vector; partial match
    composes prefix + fallback residual."""

    def test_hashmap_hit_returns_codebook_row(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, initial_cap=4)
        pid = ps.insert(b"hello")
        # Force a known row value so we can check identity.
        with torch.no_grad():
            ps.codebook[pid] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        v = ps.lookup(b"hello")
        self.assertTrue(torch.equal(
            v.detach(),
            torch.tensor([1.0, 2.0, 3.0, 4.0])))

    def test_partial_match_composes_prefix_plus_residual(self):
        import math
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, initial_cap=4,
                          promotion_threshold=1000)
        pid = ps.insert(b"hel")
        with torch.no_grad():
            ps.codebook[pid] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # Snapshot the codebook + byte_fallback BEFORE the lookup so
        # we can recompute the exact expected vector independent of
        # post-call mutations (the byte_fallback.byte_codebook is a
        # learned Parameter and only bumps hit-counts on encode(),
        # so its row values are stable -- but we clone anyway to make
        # the test resilient to future changes).
        prefix_row = ps.codebook.data[pid].clone()
        byte_cb_snapshot = ps.byte_fallback.byte_codebook.data.clone()
        # The lookup should add the prefix vector to the byte-fallback
        # vector for the residual "lo".
        v = ps.lookup(b"hello")
        # Compute the expected vector using the snapshotted state:
        # prefix codebook row + byte_fallback("lo") computed as
        # sum_b(byte_codebook[b]) / sqrt(len) -- mirroring the
        # contract in BytesFallbackEncoder.encode.
        residual = b"lo"
        residual_fb_expected = (
            byte_cb_snapshot[residual[0]]
            + byte_cb_snapshot[residual[1]]
        ) / math.sqrt(len(residual))
        expected = prefix_row + residual_fb_expected
        # The lookup's vector must match the expected to tight
        # tolerance. A regression that drops the residual (returning
        # just prefix_row) or mis-sums the byte_fallback contribution
        # would fail this assertion.
        self.assertTrue(
            torch.allclose(v.detach(), expected, atol=1e-6),
            f"lookup vector {v.detach()!r} != expected {expected!r}")


class TestPersistence(unittest.TestCase):
    """``vocab_extras`` + ``state_dict`` round trip."""

    def test_vocab_extras_dump_and_load_preserves_state(self):
        from Layers import RadixLayer
        ps = RadixLayer(dim=4, initial_cap=4,
                          promotion_threshold=2,
                          promotion_min_length=2)
        ps.insert(b"cat")
        ps.insert(b"dog")
        ps.insert(b"category")
        extras = ps.vocab_extras()
        state = ps.state_dict()
        # Build a fresh PerceptStore and load.
        ps2 = RadixLayer(dim=4, initial_cap=4,
                           promotion_threshold=2,
                           promotion_min_length=2)
        # The replay codebook needs to be at the target capacity
        # before ``load_state_dict`` (Parameter shape must match).
        ps2._grow_to(extras["capacity"])
        ps2.load_state_dict(state)
        ps2.load_vocab_extras(extras)
        # Same lookups should now succeed.
        for w in (b"cat", b"dog", b"category"):
            pid = ps.get_id(w)
            self.assertEqual(ps2.get_id(w), pid,
                             f"reloaded percept_id mismatch for {w!r}")
            self.assertEqual(ps2.bytes_for(pid), w)
        self.assertEqual(len(ps2), len(ps))


if __name__ == "__main__":
    unittest.main()
