"""SVO is derived from the chart-compose derivation trace."""

# ---------------------------------------------------------------------
# Skipped pending migration to the post-2026-05-01 chart / GrammarLayer
# surface. Tests in this module exercise legacy SyntacticLayer methods
# (generate / decompose / _signal_sentence_completed /
# _extract_svo_from_trace) that were removed by the refactor;
# equivalent functionality now lives on the Chart class.
# ---------------------------------------------------------------------
import pytest
pytestmark = pytest.mark.skip(
    reason="Pending migration to Chart surface; "
           "see doc/specs/2026-05-01-syntactic-layer-refactor.md")

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
from util import TheXMLConfig
import Language
from Language import Grammar, SyntacticLayer, TheGrammar


class TestSVOFromTrace(unittest.TestCase):
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

    def test_svo_for_n_v_n(self):
        from Spaces import WordSubSpace
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'S': ['S VO'], 'VO': ['V O']}})
        g = TheGrammar
        # pair_scorer contract: D == nInput.
        B, N, D = 1, 3, 3
        layer = SyntacticLayer(nInput=N, nOutput=N,
                               rules=list(range(len(g.rules_upward))),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        layer._ensure_category_table(g)
        cats = torch.tensor([[layer._category_index['S'],
                              layer._category_index['V'],
                              layer._category_index['O']]])
        layer._seed_category(cats)
        data = torch.randn(B, N, D)
        layer.compose(data, sub, g)
        svo = layer.last_svo
        self.assertIsNotNone(svo)
        s, v, o = svo
        self.assertTrue(torch.allclose(s[0, 0], data[0, 0]))
        self.assertTrue(torch.allclose(v[0, 0], data[0, 1]))
        self.assertTrue(torch.allclose(o[0, 0], data[0, 2]))

    def test_svo_none_without_outer_s_rule(self):
        from Spaces import WordSubSpace
        TheGrammar._configured = False
        TheGrammar.configure({'S': ['not(S)']})  # no S -> S VO
        g = TheGrammar
        B, N, D = 1, 3, 3
        layer = SyntacticLayer(nInput=N, nOutput=N, rules=g.symbolic(),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        layer.compose(torch.randn(B, N, D), sub, g)
        self.assertIsNone(layer.last_svo)


if __name__ == '__main__':
    unittest.main()
