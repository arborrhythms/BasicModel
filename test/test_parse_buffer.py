# basicmodel/test/test_parse_buffer.py
"""Tests for util.parse(data, lex=...) -- sentence / word / byte modes."""
import unittest
from util import parse


class TestParseSentences(unittest.TestCase):
    def test_single_sentence(self):
        spans = parse("The dog barks.", lex='sentences')
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][0], "The dog barks.")

    def test_two_sentences(self):
        spans = parse("The dog barks. The cat sits.", lex='sentences')
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0][0], "The dog barks.")
        self.assertEqual(spans[1][0], "The cat sits.")

    def test_trailing_fragment_kept(self):
        spans = parse("The dog barks. The cat", lex='sentences')
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[1][0], "The cat")

    def test_byte_offsets_point_to_first_nonspace(self):
        buf = "The dog barks. The cat sits."
        spans = parse(buf, lex='sentences')
        for text, start in spans:
            self.assertEqual(buf[start:start + len(text)], text)

    def test_empty_buffer(self):
        self.assertEqual(parse("", lex='sentences'), [])

    def test_punctuation_variants(self):
        spans = parse("Hello! How are you? Fine.", lex='sentences')
        self.assertEqual(len(spans), 3)


class TestParseWords(unittest.TestCase):
    def test_word_split(self):
        spans = parse("The dog barks.", lex='words')
        words = [t for t, _ in spans if not t.isspace()]
        self.assertIn("dog", words)
        self.assertIn(".", words)

    def test_byte_offsets_contiguous(self):
        buf = "abc 123!"
        spans = parse(buf, lex='words')
        for tok, start in spans:
            self.assertEqual(buf[start:start + len(tok)], tok)


class TestParseBytes(unittest.TestCase):
    def test_byte_count(self):
        spans = parse("AB", lex='bytes')
        self.assertEqual(spans, [('A', 0), ('B', 1)])

    def test_unknown_lex_raises(self):
        with self.assertRaises(ValueError):
            parse("x", lex='nope')


if __name__ == '__main__':
    unittest.main()
