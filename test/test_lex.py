import unittest
import torch
from Lex import Lex


class TestLexWordLevel(unittest.TestCase):
    """Lex produces span tables over a source buffer."""

    def _make_source(self, text):
        """Convert string to uint8 tensor (source buffer)."""
        return torch.tensor(list(text.encode('utf-8')), dtype=torch.uint8)

    def test_build_vocab(self):
        source = self._make_source("the cat sat on the mat")
        lex = Lex(granularity="word")
        lex.build_vocab(source)
        self.assertIn("the", lex.vocab)
        self.assertIn("cat", lex.vocab)
        self.assertEqual(len(lex.vocab), 5)  # the, cat, sat, on, mat

    def test_encode_produces_span_table(self):
        source = self._make_source("the cat")
        lex = Lex(granularity="word")
        lex.build_vocab(source)
        spans = lex.encode(source)
        self.assertEqual(spans.shape[1], 3)  # (start, end, type)
        self.assertEqual(spans.shape[0], 2)  # two words

    def test_span_offsets_match_source(self):
        text = "the cat"
        source = self._make_source(text)
        lex = Lex(granularity="word")
        lex.build_vocab(source)
        spans = lex.encode(source)
        # "the" is bytes 0-3, "cat" is bytes 4-7
        self.assertEqual(spans[0, 0].item(), 0)   # start of "the"
        self.assertEqual(spans[0, 1].item(), 3)   # end of "the"
        self.assertEqual(spans[1, 0].item(), 4)   # start of "cat"
        self.assertEqual(spans[1, 1].item(), 7)   # end of "cat"

    def test_span_type_is_token_id(self):
        source = self._make_source("the cat the")
        lex = Lex(granularity="word")
        lex.build_vocab(source)
        spans = lex.encode(source)
        # Both "the" spans should have the same type (token ID)
        self.assertEqual(spans[0, 2].item(), spans[2, 2].item())
        # "cat" should have a different token ID
        self.assertNotEqual(spans[0, 2].item(), spans[1, 2].item())

    def test_decode_reconstructs_words(self):
        text = "the cat sat"
        source = self._make_source(text)
        lex = Lex(granularity="word")
        lex.build_vocab(source)
        spans = lex.encode(source)
        words = lex.decode(source, spans)
        self.assertEqual(words, ["the", "cat", "sat"])

    def test_vocab_to_id_and_id_to_word(self):
        source = self._make_source("hello world")
        lex = Lex(granularity="word")
        lex.build_vocab(source)
        # Round-trip: word -> id -> word
        hello_id = lex.vocab["hello"]
        self.assertEqual(lex.id_to_word[hello_id], "hello")


class TestLexMultipleExamples(unittest.TestCase):
    """Lex handles source buffers with multiple examples."""

    def _make_source(self, text):
        return torch.tensor(list(text.encode('utf-8')), dtype=torch.uint8)

    def test_encode_with_example_offsets(self):
        """Given example offsets, encode only spans within that range."""
        text = "the cat sat on the mat"
        source = self._make_source(text)
        lex = Lex(granularity="word")
        lex.build_vocab(source)
        # Full span table
        all_spans = lex.encode(source)
        self.assertEqual(all_spans.shape[0], 6)  # six words total
        # Slice for first 7 bytes ("the cat")
        offsets = torch.tensor([[0, 7]])
        subset = lex.encode(source, example_offsets=offsets)
        self.assertEqual(subset.shape[0], 2)  # "the", "cat"
