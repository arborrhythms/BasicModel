import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from embed import (
    iter_documents, CorpusBuilder, EmbeddingTrainer, build_embeddings,
    WordVectors,
)


# ---------------------------------------------------------------------------
# FineWeb streaming
# ---------------------------------------------------------------------------

class TestFineWebIterator(unittest.TestCase):

    @patch('embed.pq')
    def test_yields_strings(self, mock_pq):
        """iter_documents yields string documents."""
        mock_file = MagicMock()
        mock_file.num_row_groups = 1
        mock_rg = MagicMock()
        mock_rg.column.return_value.to_pylist.return_value = [
            "The dog barks. The cat meows.",
            "Water is wet."
        ]
        mock_file.read_row_group.return_value = mock_rg
        mock_pq.ParquetFile.return_value = mock_file

        docs = list(iter_documents(shard_paths=["/fake/shard.parquet"], max_docs=10))
        self.assertEqual(len(docs), 2)
        self.assertIsInstance(docs[0], str)

    @patch('embed.pq')
    def test_max_docs_limit(self, mock_pq):
        """iter_documents stops after max_docs."""
        mock_file = MagicMock()
        mock_file.num_row_groups = 1
        mock_rg = MagicMock()
        mock_rg.column.return_value.to_pylist.return_value = [
            f"Sentence {i}." for i in range(100)
        ]
        mock_file.read_row_group.return_value = mock_rg
        mock_pq.ParquetFile.return_value = mock_file

        docs = list(iter_documents(shard_paths=["/fake/shard.parquet"], max_docs=5))
        self.assertEqual(len(docs), 5)


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

class TestCorpusBuilder(unittest.TestCase):

    def test_process_document(self):
        """Processing a document updates vocabulary and examples."""
        cb = CorpusBuilder()
        cb.process_document("The dog barks. The cat sits.")
        self.assertGreater(len(cb.vocab), 0)
        self.assertIn("dog", cb.vocab)
        self.assertGreater(len(cb.examples), 0)

    def test_vocab_counts(self):
        """Vocabulary tracks word frequency."""
        cb = CorpusBuilder()
        cb.process_document("The dog barks. The cat barks.")
        self.assertGreaterEqual(cb.vocab_counts["barks"], 2)

    def test_example_structure(self):
        """Each example has target word and context words."""
        cb = CorpusBuilder()
        cb.process_document("The big dog barks.")
        ex = cb.examples[0]
        self.assertIn('target', ex)
        self.assertIn('context', ex)
        self.assertIsInstance(ex['target'], str)
        self.assertIsInstance(ex['context'], list)

    def test_context_is_sentence_minus_target(self):
        """Context is the other words in the same sentence."""
        cb = CorpusBuilder()
        cb.process_document("The big dog barks.")
        for ex in cb.examples:
            self.assertIsInstance(ex['context'], list)
            self.assertGreater(len(ex['context']), 0)

    def test_multiple_documents(self):
        """Processing multiple documents accumulates results."""
        cb = CorpusBuilder()
        cb.process_document("The dog barks.")
        n1 = len(cb.examples)
        cb.process_document("The cat meows.")
        n2 = len(cb.examples)
        self.assertGreater(n2, n1)

    def test_get_vocab_list(self):
        """get_vocab_list returns words sorted by frequency."""
        cb = CorpusBuilder()
        cb.process_document("The dog barks. The cat barks. The bird sings.")
        vocab = cb.get_vocab_list(min_count=1)
        self.assertIsInstance(vocab, list)
        self.assertGreater(len(vocab), 0)

    def test_min_count_filter(self):
        """min_count filters out rare words."""
        cb = CorpusBuilder()
        cb.process_document("The dog barks. The cat barks.")
        vocab_all = cb.get_vocab_list(min_count=1)
        vocab_freq = cb.get_vocab_list(min_count=2)
        self.assertLessEqual(len(vocab_freq), len(vocab_all))

    def test_sentences_from_parse(self):
        """Words are grouped into sentences by parse_buffer."""
        cb = CorpusBuilder()
        cb.process_document("Big dog. Small cat.")
        # Should have 2 sentences worth of examples
        # "Big dog" -> 2 examples, "Small cat" -> 2 examples
        self.assertEqual(len(cb.examples), 4)


# ---------------------------------------------------------------------------
# Embedding trainer
# ---------------------------------------------------------------------------

class MockCorpusBuilder:
    """Minimal corpus builder for testing embedding trainer."""
    def __init__(self):
        self.examples = []
        self.vocab_counts = {}

    def get_vocab_list(self, min_count=1):
        counts = {}
        for ex in self.examples:
            w = ex['target']
            counts[w] = counts.get(w, 0) + 1
            for c in ex['context']:
                counts[c] = counts.get(c, 0) + 1
        return [w for w, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= min_count]

def _make_corpus():
    cb = MockCorpusBuilder()
    words_sets = [
        ["The", "big", "dog", "barks", "loudly"],
        ["The", "small", "cat", "meows", "softly"],
        ["Water", "is", "wet"],
        ["Fire", "is", "hot"],
        ["Ice", "is", "cold"],
    ]
    for _ in range(10):
        for words in words_sets:
            for i, w in enumerate(words):
                context = [words[j] for j in range(len(words)) if j != i]
                cb.examples.append({'target': w, 'context': context})
    return cb

class TestEmbeddingTrainer(unittest.TestCase):

    def test_train_produces_vectors(self):
        cb = _make_corpus()
        trainer = EmbeddingTrainer(vector_size=20)
        wv = trainer.train(cb, epochs=5)
        self.assertIsInstance(wv, WordVectors)
        self.assertGreater(len(wv), 0)

    def test_vector_dimensions(self):
        cb = _make_corpus()
        trainer = EmbeddingTrainer(vector_size=32)
        wv = trainer.train(cb, epochs=3)
        self.assertEqual(wv._vectors.shape[1], 32)

    def test_vocab_coverage(self):
        cb = _make_corpus()
        vocab = cb.get_vocab_list(min_count=2)
        trainer = EmbeddingTrainer(vector_size=20, min_count=2)
        wv = trainer.train(cb, epochs=3)
        for word in vocab:
            self.assertIn(word, wv, f"{word} not in trained vectors")

    def test_save_and_load(self):
        cb = _make_corpus()
        trainer = EmbeddingTrainer(vector_size=20)
        wv = trainer.train(cb, epochs=3)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            wv.save(path)
            loaded = WordVectors.load(path)
            self.assertEqual(len(loaded), len(wv))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestBuildEmbeddings(unittest.TestCase):

    @patch('embed.iter_documents')
    def test_end_to_end(self, mock_iter):
        mock_iter.return_value = iter([
            "The big dog barks loudly. The small cat meows softly.",
            "Water is wet. Fire is hot. Ice is cold.",
        ] * 20)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "embeddings.pt")
            build_embeddings(
                shard_paths=["/fake"],
                output_path=output_path,
                max_docs=40,
                vector_size=20,
                epochs=3,
                min_count=2,
            )
            self.assertTrue(os.path.exists(output_path))
            wv = WordVectors.load(output_path)
            self.assertGreater(len(wv), 0)
            self.assertEqual(wv._vectors.shape[1], 20)


if __name__ == '__main__':
    unittest.main()
