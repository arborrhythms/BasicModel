import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from embed import (
    iter_documents, StreamingSBOWTrainer, build_embeddings,
    WordVectors,
)


# ---------------------------------------------------------------------------
# FineWeb streaming
# ---------------------------------------------------------------------------

class TestFineWebIterator(unittest.TestCase):

    @patch('data.pq')
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

    @patch('data.pq')
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
# Streaming CBOW trainer
# ---------------------------------------------------------------------------

class TestStreamingSBOWTrainer(unittest.TestCase):

    def _make_trainer(self, docs=None, **kwargs):
        """Create a trainer with vocabulary already built from *docs*.

        The two-pass API requires scan_document + build_vocab before
        process_document can train.
        """
        defaults = dict(vector_size=20, min_count=1)
        defaults.update(kwargs)
        t = StreamingSBOWTrainer(**defaults)
        if docs:
            for d in docs:
                t.scan_document(d)
            t.build_vocab()
        return t

    def test_scan_grows_vocab(self):
        """Scanning documents and building vocab adds words."""
        t = self._make_trainer(docs=["The dog barks. The cat sits."])
        self.assertGreater(t.vocab_size, 0)
        self.assertIn("dog", t.word_to_idx)

    def test_vocab_counts(self):
        """Word counts are tracked correctly after scanning."""
        t = StreamingSBOWTrainer(vector_size=20, min_count=1)
        t.scan_document("The dog barks. The cat barks.")
        self.assertGreaterEqual(t.word_counts["barks"], 2)

    def test_min_count_filters(self):
        """Words below min_count are not promoted to trainable vocab."""
        t = StreamingSBOWTrainer(vector_size=20, min_count=3)
        t.scan_document("The dog barks. The cat barks.")
        t.build_vocab()
        # "The" appears twice, "barks" appears twice — neither hits 3
        for word, idx in t.word_to_idx.items():
            self.assertGreaterEqual(t.word_counts[word], 3,
                                    f"{word} promoted with count {t.word_counts[word]}")

    @patch('embed.util.compile')
    def test_build_vocab_compiles_model(self, mock_compile):
        """Vocabulary build routes the SBOW model through util.compile()."""
        mock_compile.side_effect = lambda model: model
        t = StreamingSBOWTrainer(vector_size=20, min_count=1)
        t.scan_document("The dog barks.")

        t.build_vocab()

        mock_compile.assert_called_once()
        compiled_model = mock_compile.call_args.args[0]
        self.assertIs(t.model, compiled_model)

    def test_trains_and_produces_vectors(self):
        """Training produces WordVectors with correct dimensions."""
        docs = [
            "The big dog barks loudly. The small cat meows softly.",
            "Water is wet. Fire is hot. Ice is cold.",
        ]
        t = self._make_trainer(docs=docs, vector_size=16)
        for _ in range(10):
            for d in docs:
                t.process_document(d)
        wv = t.finish()
        self.assertIsInstance(wv, WordVectors)
        self.assertGreater(len(wv), 0)
        self.assertEqual(wv.vector_size, 16)

    def test_multiple_documents(self):
        """Processing multiple documents increases trained examples."""
        docs = ["The dog barks.", "The cat meows."]
        t = self._make_trainer(docs=docs)
        t.process_document("The dog barks.")
        n1 = t.n_examples
        t.process_document("The cat meows.")
        n2 = t.n_examples
        self.assertGreaterEqual(n2, n1)

    def test_save_and_load(self):
        """Trained vectors can be saved and loaded."""
        docs = ["The big dog barks. The small cat meows."]
        t = self._make_trainer(docs=docs)
        for _ in range(10):
            t.process_document(docs[0])
        wv = t.finish()
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
        docs = [
            "The big dog barks loudly. The small cat meows softly.",
            "Water is wet. Fire is hot. Ice is cold.",
        ] * 20

        # iter_documents is called once per epoch; return fresh iterator each time
        mock_iter.side_effect = lambda *a, **kw: iter(docs)

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
            self.assertEqual(wv.vector_size, 20)


if __name__ == '__main__':
    unittest.main()
