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

    def test_abbreviation_and_parenthetical_punctuation_stay_together(self):
        text = (
            "Material remains (i.e., osteological, architectural, etc.). "
            "Related categories.")
        spans = parse(text, lex='sentences')
        self.assertEqual(
            [surface for surface, _ in spans],
            [
                "Material remains (i.e., osteological, architectural, etc.).",
                "Related categories.",
            ])

    def test_titles_dotted_acronyms_and_decimals_do_not_split(self):
        text = "Dr. Smith joined the U.S. team at 3.14 p.m. He left."
        spans = parse(text, lex='sentences')
        self.assertEqual(
            [surface for surface, _ in spans],
            ["Dr. Smith joined the U.S. team at 3.14 p.m.", "He left."])

    def test_punctuation_only_fragments_are_not_sentences(self):
        spans = parse("One sentence. ). ... Two sentences.", lex='sentences')
        self.assertEqual(
            [surface for surface, _ in spans],
            ["One sentence.", "Two sentences."])

    def test_newlines_separate_headings_and_list_rows(self):
        text = "HOW TO SUBMIT DATA\nThere are two ways.\nfirst item\nsecond item"
        spans = parse(text, lex='sentences')
        self.assertEqual(
            [surface for surface, _ in spans],
            [
                "HOW TO SUBMIT DATA",
                "There are two ways.",
                "first item",
                "second item",
            ])


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
