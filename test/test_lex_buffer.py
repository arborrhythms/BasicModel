import unittest
from lex import Lex

class TestLexBuffer(unittest.TestCase):

    def test_basic_tokenization(self):
        """lex_buffer returns token spans with WORD/SEPARATOR categories."""
        lex = Lex()
        buf = "The dog barks."
        tokens = lex.lex_buffer(buf, 0)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(tokens[-1]['category'], 'SEPARATOR')
        word_tokens = [t for t in tokens if t['category'] == 'WORD']
        self.assertEqual(len(word_tokens), 3)

    def test_token_span_fields(self):
        """Each token span has start, end, text, category."""
        lex = Lex()
        buf = "The dog barks."
        tokens = lex.lex_buffer(buf, 0)
        for tok in tokens:
            self.assertIn('start', tok)
            self.assertIn('end', tok)
            self.assertIn('text', tok)
            self.assertIn('category', tok)
            self.assertEqual(buf[tok['start']:tok['end']], tok['text'])

    def test_start_offset(self):
        """Tokenizing from a non-zero offset skips earlier text."""
        lex = Lex()
        buf = "The dog barks. The cat sits."
        tokens = lex.lex_buffer(buf, 15)
        word_texts = [t['text'] for t in tokens if t['category'] == 'WORD']
        self.assertNotIn("dog", word_texts)
        self.assertIn("cat", word_texts)

    def test_empty_buffer(self):
        """Empty buffer returns no tokens."""
        lex = Lex()
        tokens = lex.lex_buffer("", 0)
        self.assertEqual(len(tokens), 0)

    def test_multiple_separators(self):
        """Different separator types are recognized."""
        lex = Lex()
        for sep in ['.', '!', '?']:
            buf = f"The dog barks{sep}"
            tokens = lex.lex_buffer(buf, 0)
            seps = [t for t in tokens if t['category'] == 'SEPARATOR']
            self.assertEqual(len(seps), 1, f"Separator '{sep}' not detected")

    def test_vocab_updated(self):
        """lex_buffer updates vocabulary with words encountered."""
        lex = Lex()
        buf = "The dog barks."
        lex.lex_buffer(buf, 0)
        self.assertIn("The", lex.vocab)
        self.assertIn("dog", lex.vocab)
        self.assertIn("barks", lex.vocab)

    def test_absolute_offsets(self):
        """Token offsets are absolute positions in the buffer."""
        lex = Lex()
        buf = "Hello world."
        tokens = lex.lex_buffer(buf, 0)
        for tok in tokens:
            self.assertEqual(buf[tok['start']:tok['end']], tok['text'])

    def test_existing_api_unchanged(self):
        """The existing Lex.encode() API still works."""
        import torch
        lex = Lex()
        source = torch.tensor(list("hello world".encode('utf-8')), dtype=torch.uint8)
        lex.build_vocab(source)
        spans = lex.encode(source)
        self.assertEqual(spans.shape[1], 3)

    def test_bracket_splits_from_word(self):
        """[MASK] from source text splits into 3 tokens."""
        lex = Lex()
        tokens = lex.lex_buffer("[MASK]", 0)
        texts = [t['text'] for t in tokens]
        self.assertEqual(texts, ['[', 'MASK', ']'])
        self.assertEqual(tokens[0]['category'], 'PUNCT')
        self.assertEqual(tokens[1]['category'], 'WORD')
        self.assertEqual(tokens[2]['category'], 'PUNCT')

    def test_brackets_in_sentence(self):
        lex = Lex()
        tokens = lex.lex_buffer("the [quick] fox", 0)
        words = [t['text'] for t in tokens if t['category'] == 'WORD']
        self.assertEqual(words, ['the', 'quick', 'fox'])

    def test_numbers_separate(self):
        lex = Lex()
        tokens = lex.lex_buffer("room 101", 0)
        self.assertEqual(tokens[0]['text'], 'room')
        self.assertEqual(tokens[0]['category'], 'WORD')
        self.assertEqual(tokens[1]['text'], '101')
        self.assertEqual(tokens[1]['category'], 'NUMBER')

    def test_semicolon_is_punct_not_separator(self):
        lex = Lex()
        tokens = lex.lex_buffer("hello; world", 0)
        semi = [t for t in tokens if t['text'] == ';']
        self.assertEqual(semi[0]['category'], 'PUNCT')

    def test_bracket_offsets_correct(self):
        lex = Lex()
        buf = "[MASK]"
        tokens = lex.lex_buffer(buf, 0)
        for tok in tokens:
            self.assertEqual(buf[tok['start']:tok['end']], tok['text'])

if __name__ == '__main__':
    unittest.main()
