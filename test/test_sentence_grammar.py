import unittest
from nltk.grammar import CFG
from nltk.parse.earleychart import EarleyChartParser
from pathlib import Path

class TestSentenceGrammar(unittest.TestCase):

    def setUp(self):
        cfg_path = Path(__file__).resolve().parent.parent / "data" / "sentence.cfg"
        self.grammar_text = cfg_path.read_text()
        self.cfg = CFG.fromstring(self.grammar_text)

    def test_grammar_loads(self):
        """sentence.cfg loads as a valid NLTK CFG."""
        self.assertIsNotNone(self.cfg)

    def test_single_sentence(self):
        """A sentence ending with period parses."""
        parser = EarleyChartParser(self.cfg)
        tokens = ["WORD", "WORD", "WORD", "SEPARATOR"]
        trees = list(parser.parse(tokens))
        self.assertGreater(len(trees), 0)

    def test_two_sentences(self):
        """Two sentences separated by period parse as S -> S SENT."""
        parser = EarleyChartParser(self.cfg)
        tokens = ["WORD", "WORD", "SEPARATOR", "WORD", "WORD", "SEPARATOR"]
        trees = list(parser.parse(tokens))
        self.assertGreater(len(trees), 0)

    def test_no_separator_no_parse(self):
        """Text without a separator does not parse as a complete sentence."""
        parser = EarleyChartParser(self.cfg)
        tokens = ["WORD", "WORD", "WORD"]
        trees = list(parser.parse(tokens))
        self.assertEqual(len(trees), 0)

if __name__ == '__main__':
    unittest.main()
