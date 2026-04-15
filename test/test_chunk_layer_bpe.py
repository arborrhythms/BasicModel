"""BPE ChunkLayer tests.

Verifies the new BPE path in ``ChunkLayer``:

  1. Cold-start invariants — empty merge table, 256 single-byte ids.
  2. ``train_step`` grows ``merges`` and ``vocab`` monotonically on a
     small repeating text corpus; frequent byte pairs end up in the
     vocab.
  3. Greedy longest-match ``forward`` is deterministic across batches —
     identical input yields identical chunk ids.
  4. ``hard_merge_spans`` + ``compact`` + ``uncompact`` still roundtrips
     exactly in BPE mode (span-stored originals drive reconstruction).
  5. Loading ``MM_bpe.xml`` propagates ``chunkBPE`` / ``chunkTargetVocabSize``
     / ``chunkMinPairFrequency`` into a ``ChunkLayer`` instance.
"""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def _byte_tensor(texts, width=None):
    import torch
    rows = [list(t.encode("utf-8")) for t in texts]
    if width is None:
        width = max(len(r) for r in rows)
    padded = [r + [0] * (width - len(r)) for r in rows]
    return torch.tensor(padded, dtype=torch.long)


class TestChunkLayerBPE(unittest.TestCase):

    def test_cold_start_vocab_has_bytes(self):
        from Layers import ChunkLayer
        layer = ChunkLayer(nDim=8, bpe=True,
                           target_vocab_size=1024, min_pair_frequency=2)
        self.assertEqual(len(layer.vocab), 256)
        self.assertEqual(len(layer.merges), 0)
        self.assertEqual(layer._next_id, 256)
        self.assertEqual(layer._max_merge_len, 1)
        for i in range(256):
            self.assertIn((i,), layer.vocab)
            self.assertEqual(layer.vocab[(i,)], i)

    def test_merges_grow_on_repeating_corpus(self):
        import torch
        from Layers import ChunkLayer
        layer = ChunkLayer(nDim=8, bpe=True,
                           target_vocab_size=1024, min_pair_frequency=2)
        layer.train()
        corpus = ["hello hello world world the the",
                  "the hello world the world hello"]
        batch = _byte_tensor(corpus)
        sizes = [len(layer.vocab)]
        for _ in range(12):
            layer.train_step(batch, k_merges=1)
            sizes.append(len(layer.vocab))

        self.assertGreater(len(layer.merges), 0,
                           "train_step should add merges on a repeating corpus")
        for a, b in zip(sizes, sizes[1:]):
            self.assertGreaterEqual(b, a,
                                    "vocab size must grow monotonically")

        # The most-frequent byte pair in the corpus is ' t' / 'th' / 'he'
        # etc.  At least one of the common English digraphs should land
        # in the learned vocab after 12 training steps.
        common_digraphs = [b"th", b"he", b"lo", b"or", b"wo", b"ld"]
        learned = [d for d in common_digraphs
                   if tuple(d) in layer.vocab]
        self.assertGreater(
            len(learned), 0,
            f"expected at least one common digraph in vocab, got merges={layer.merges[:10]}")

    def test_forward_is_deterministic_across_batches(self):
        import torch
        from Layers import ChunkLayer
        layer = ChunkLayer(nDim=8, bpe=True,
                           target_vocab_size=1024, min_pair_frequency=2)
        layer.train()
        train_batch = _byte_tensor(["abcabcabcabc"])
        for _ in range(5):
            layer.train_step(train_batch, k_merges=1)

        layer.eval()
        batch1 = _byte_tensor(["abcabc"])
        batch2 = _byte_tensor(["abcabc"])
        chunks1, spans1 = layer.forward(batch1)
        chunks2, spans2 = layer.forward(batch2)
        self.assertEqual(chunks1, chunks2)
        self.assertEqual(spans1, spans2)

    def test_forward_legacy_mode_is_identity(self):
        from Layers import ChunkLayer
        layer = ChunkLayer(nDim=8, bpe=False)
        batch = _byte_tensor(["hello"])
        out = layer.forward(batch)
        self.assertTrue((out == batch).all().item())

    def test_hard_merge_spans_bpe_roundtrip(self):
        import torch
        from Layers import ChunkLayer
        layer = ChunkLayer(nDim=4, bpe=True,
                           target_vocab_size=1024, min_pair_frequency=2)
        layer.train()
        train_batch = _byte_tensor(["hello hello"])
        for _ in range(3):
            layer.train_step(train_batch, k_merges=1)

        batch = _byte_tensor(["hello"], width=8)
        B, N = batch.shape
        torch.manual_seed(0)
        data = torch.randn(B, N, 4)
        original_before_mutation = data.clone()

        merged, span_meta = layer.hard_merge_spans(data, batch)
        self.assertEqual(len(span_meta), B)
        self.assertGreater(len(span_meta[0]), 0)

        nWordSlots = 8
        dense, compact_map = layer.compact(merged, nWordSlots, span_meta,
                                           where_encoding=None)
        restored = layer.uncompact(dense, compact_map, nByteSlots=N)

        # Only the active (non-padding) positions are reconstructed.
        active = batch[0] != 0
        rec = restored[0][active]
        orig = original_before_mutation[0][active]
        diff = (rec - orig).abs().max().item()
        self.assertLess(
            diff, 1e-6,
            f"BPE roundtrip must reconstruct byte-vectors exactly, max diff={diff}")

    def test_target_vocab_size_bounds_growth(self):
        from Layers import ChunkLayer
        layer = ChunkLayer(nDim=4, bpe=True,
                           target_vocab_size=260, min_pair_frequency=1)
        layer.train()
        batch = _byte_tensor(["ababababababababababab"])
        for _ in range(50):
            layer.train_step(batch, k_merges=2)
        self.assertLessEqual(len(layer.vocab), 260)

    def test_mm_bpe_config_drives_chunk_layer_flags(self):
        """MM_bpe.xml config should produce a BPE-mode ChunkLayer."""
        from util import init_config
        import Spaces
        from Layers import ChunkLayer

        cfg_path = os.path.join(_PROJECT, "data", "MM_bpe.xml")
        init_config(
            path=cfg_path,
            defaults_path=os.path.join(_PROJECT, "data", "model.xml"),
        )
        cfg = Spaces.TheXMLConfig
        self.assertTrue(bool(cfg.space("PerceptualSpace", "chunkBPE")),
                        "MM_bpe.xml must set chunkBPE=true")
        target = int(cfg.space("PerceptualSpace", "chunkTargetVocabSize"))
        min_freq = int(cfg.space("PerceptualSpace", "chunkMinPairFrequency"))
        layer = ChunkLayer(
            nDim=8,
            bpe=bool(cfg.space("PerceptualSpace", "chunkBPE")),
            target_vocab_size=target,
            min_pair_frequency=min_freq,
        )
        self.assertTrue(layer.bpe)
        self.assertEqual(layer.target_vocab_size, target)
        self.assertEqual(layer.min_pair_frequency, min_freq)


if __name__ == "__main__":
    unittest.main()
