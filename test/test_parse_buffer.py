# basicmodel/test/test_parse_buffer.py
import unittest
from parse import parse, parse_buffer, parse_deep

class TestParseBuffer(unittest.TestCase):
    """Tests for sentence.cfg-based sentence grouping."""

    def test_single_sentence(self):
        result, next_pos = parse_buffer("The dog barks.")
        self.assertEqual(len(result['sentences']), 1)
        self.assertGreater(len(result['sentences'][0]['tokens']), 0)

    def test_two_sentences(self):
        result, next_pos = parse_buffer("The dog barks. The cat sits.")
        self.assertEqual(len(result['sentences']), 2)

    def test_sentence_has_word_tokens(self):
        result, _ = parse_buffer("The dog barks.")
        words = result['sentences'][0]['tokens']
        word_texts = [t['text'] for t in words]
        self.assertIn("dog", word_texts)
        # Separators should not be in the token list
        self.assertNotIn(".", word_texts)

    def test_trailing_fragment_unconsumed(self):
        result, next_pos = parse_buffer("The dog barks. The cat")
        self.assertEqual(len(result['sentences']), 1)
        self.assertLess(next_pos, len("The dog barks. The cat"))
        # next_pos should point to start of unconsumed fragment
        remaining = "The dog barks. The cat"[next_pos:]
        self.assertTrue(remaining.strip().startswith("The cat"))

    def test_start_offset(self):
        buf = "The dog barks. The cat sits."
        result1, pos1 = parse_buffer(buf, 0)
        result2, pos2 = parse_buffer(buf, pos1)
        self.assertEqual(len(result1['sentences']), 2)
        self.assertEqual(len(result2['sentences']), 0)

    def test_sentence_span_offsets(self):
        buf = "The dog barks. The cat sits."
        result, _ = parse_buffer(buf)
        for sent in result['sentences']:
            self.assertIn('start', sent)
            self.assertIn('end', sent)
            self.assertLessEqual(sent['start'], sent['end'])

    def test_empty_buffer(self):
        result, next_pos = parse_buffer("")
        self.assertEqual(len(result['sentences']), 0)
        self.assertEqual(next_pos, 0)

    def test_existing_parse_unchanged(self):
        xml = parse("The dog barks.")
        self.assertIn("dog", xml)
        self.assertIn("<", xml)


class TestParseDeep(unittest.TestCase):
    """Tests for grammar.cfg-based deep parsing with positions."""

    def test_single_sentence(self):
        result, next_pos = parse_deep("The dog barks.")
        self.assertEqual(len(result['sentences']), 1)
        self.assertIn('xml', result['sentences'][0])
        self.assertIn("dog", result['sentences'][0]['xml'])

    def test_two_sentences(self):
        result, next_pos = parse_deep("The dog barks. The cat sits.")
        self.assertEqual(len(result['sentences']), 2)
        for sent in result['sentences']:
            self.assertIn('xml', sent)

    def test_trailing_fragment_unconsumed(self):
        result, next_pos = parse_deep("The dog barks. The cat")
        self.assertEqual(len(result['sentences']), 1)
        self.assertLess(next_pos, len("The dog barks. The cat"))
        remaining = "The dog barks. The cat"[next_pos:]
        self.assertTrue(remaining.strip().startswith("The cat"))

    def test_sentence_has_tokens_and_xml(self):
        result, _ = parse_deep("The dog barks.")
        sent = result['sentences'][0]
        self.assertIn('tokens', sent)
        self.assertIn('xml', sent)
        self.assertIn('start', sent)
        self.assertIn('end', sent)

    def test_empty_buffer(self):
        result, next_pos = parse_deep("")
        self.assertEqual(len(result['sentences']), 0)
        self.assertEqual(next_pos, 0)

    def test_xml_matches_parse(self):
        """parse_deep XML should match what parse() produces for the same sentence."""
        result, _ = parse_deep("The dog barks.")
        deep_xml = result['sentences'][0]['xml']
        direct_xml = parse("The dog barks.")
        self.assertEqual(deep_xml, direct_xml)


if __name__ == '__main__':
    unittest.main()
