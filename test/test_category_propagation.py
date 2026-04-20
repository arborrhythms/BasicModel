"""Category tensor propagates through chart compose; rhs_symbols gate rules."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
from util import TheXMLConfig
import Language
from Language import Grammar, SyntacticLayer, TheGrammar


class TestCategoryPropagation(unittest.TestCase):
    def setUp(self):
        try:
            self._prior_chart = TheXMLConfig.get('WordSpace.chartCompose')
        except KeyError:
            self._prior_chart = None
        TheXMLConfig.set('WordSpace.chartCompose', True)
        self._saved_grammar_state = TheGrammar._configured
        TheGrammar._configured = False

    def tearDown(self):
        if self._prior_chart is None:
            TheXMLConfig.set('WordSpace.chartCompose', False)
        else:
            TheXMLConfig.set('WordSpace.chartCompose', self._prior_chart)
        TheGrammar._configured = self._saved_grammar_state

    def _layer(self, rules_dict, N=3, D=3):
        TheGrammar._configured = False
        TheGrammar.configure(rules_dict)
        layer = SyntacticLayer(nInput=N, nOutput=N,
                               rules=list(range(len(TheGrammar.rules))),
                               max_depth=N - 1, hidden_dim=16,
                               grammar=TheGrammar)
        from Spaces import WordSubSpace
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=1)
        return layer, sub, TheGrammar

    def test_merging_V_and_O_yields_VO_slot(self):
        layer, sub, g = self._layer(
            {'upward': {'S': ['S VO'], 'VO': ['V O']}}, N=3)
        layer._ensure_category_table(g)
        cats = torch.tensor([[layer._category_index['S'],
                              layer._category_index['V'],
                              layer._category_index['O']]])
        layer._seed_category(cats)
        data = torch.randn(1, 3, 3)
        layer.compose(data, sub, g)
        # Final trace entry should have merged_category == S.
        last = layer._derivation_trace[0][-1]
        self.assertEqual(last[4], layer._category_index['S'])

    def test_incompatible_categories_block_the_rule(self):
        layer, sub, g = self._layer(
            {'upward': {'S': ['S VO'], 'VO': ['V O']}}, N=3)
        layer._ensure_category_table(g)
        # Seed all leaves as S: there is no rule (S, S) -> anything typed,
        # and the pair-scorer + compat mask should zero-out every rule.
        cats = torch.tensor([[layer._category_index['S']] * 3])
        layer._seed_category(cats)
        data = torch.randn(1, 3, 3)
        layer.compose(data, sub, g)
        # No legal typed merges -> trace is empty.
        self.assertEqual(layer._derivation_trace[0], [])


class TestNLTKPOSSeeding(unittest.TestCase):
    def test_basic_map(self):
        from Language import map_nltk_tags_to_categories
        cats = map_nltk_tags_to_categories(
            ['the', 'teacher', 'helped', 'the', 'student'])
        self.assertEqual(cats, ['DET', 'N', 'V', 'DET', 'N'])


if __name__ == '__main__':
    unittest.main()
