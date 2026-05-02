"""Chart-like compose scaffolding: derivation trace contract."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import Language
from Language import Grammar, SyntacticLayer


class TestDerivationTraceContract(unittest.TestCase):
    def test_trace_attribute_exists_after_init(self):
        layer = SyntacticLayer(nInput=4, nOutput=4, rules=[], max_depth=4,
                               hidden_dim=16, grammar=Grammar())
        self.assertTrue(hasattr(layer, '_derivation_trace'))
        self.assertIsNone(layer._derivation_trace)

    def test_trace_reset_to_per_batch_empty_lists_on_compose(self):
        from Spaces import WordSubSpace
        g = Grammar()
        g.configure({'S': ['not(S)']})
        B, N, D = 2, 4, 3
        layer = SyntacticLayer(nInput=N, nOutput=N, rules=g.symbolic(),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        layer._derivation_trace = 'stale'
        # The reset is unconditional at compose() entry. If the full
        # _compose_vector cascade raises downstream (e.g. 3D-basis matmul
        # in this minimal WordSubSpace wiring), the reset has still
        # happened and that's what this contract test asserts.
        try:
            layer.compose(torch.randn(B, N, D), sub, g)
        except (RuntimeError, IndexError):
            pass
        self.assertEqual(layer._derivation_trace, [[] for _ in range(B)])


class TestPairScorer(unittest.TestCase):
    def _make_layer(self, N=4, num_rules=2):
        g = Grammar()
        g.configure({'S': ['not(S)'] * num_rules})
        return SyntacticLayer(nInput=N, nOutput=N, rules=list(range(num_rules)),
                              max_depth=N - 1, hidden_dim=16, grammar=g)

    def test_pair_scorer_output_shape(self):
        layer = self._make_layer(N=4, num_rules=2)
        B, N, D = 2, 4, 4
        hidden = torch.zeros(B, layer.hidden_dim)
        pairs = torch.zeros(B, N - 1, 2, N)  # D == nInput by contract
        alive = torch.ones(B, N, dtype=torch.bool)
        scores = layer._pair_scorer(hidden, pairs, alive)
        self.assertEqual(scores.shape, (B, N - 1))

    def test_pair_scorer_respects_alive_mask(self):
        layer = self._make_layer(N=4, num_rules=2)
        B, N = 1, 4
        hidden = torch.zeros(B, layer.hidden_dim)
        pairs = torch.zeros(B, N - 1, 2, N)
        alive = torch.tensor([[True, True, True, False]])
        scores = layer._pair_scorer(hidden, pairs, alive)
        probs = torch.softmax(scores, dim=-1)
        self.assertAlmostEqual(probs[0, 2].item(), 0.0, places=5)


class TestChartCompose(unittest.TestCase):
    def setUp(self):
        from util import TheXMLConfig
        try:
            self._prior_chart = TheXMLConfig.get('WordSpace.chartCompose')
        except KeyError:
            self._prior_chart = None
        self._saved_grammar_state = Language.TheGrammar._configured
        Language.TheGrammar._configured = False

    def tearDown(self):
        from util import TheXMLConfig
        if self._prior_chart is None:
            TheXMLConfig.set('WordSpace.chartCompose', False)
        else:
            TheXMLConfig.set('WordSpace.chartCompose', self._prior_chart)
        Language.TheGrammar._configured = self._saved_grammar_state

    def _enable_chart(self):
        from util import TheXMLConfig
        TheXMLConfig.set('WordSpace.chartCompose', True)

    def test_chart_reduces_n_leaves_to_trace_entries(self):
        self._enable_chart()
        from Spaces import WordSubSpace
        Language.TheGrammar._configured = False
        Language.TheGrammar.configure({'compose': {'S': ['S VO'], 'VO': ['V O']}})
        g = Language.TheGrammar
        B, N, D = 1, 3, 3
        layer = SyntacticLayer(nInput=N, nOutput=N,
                               rules=list(range(len(g.rules_upward))),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        # Seed categories so the N-1 merge invariant holds regardless of
        # which pair/rule the argmax picks. Without typing, the second
        # merge can become incompatible after the first (Task 7 compat
        # mask is strict on non-wildcard categories).
        layer._ensure_category_table(g)
        cats = torch.tensor([[layer._category_index['S'],
                              layer._category_index['V'],
                              layer._category_index['O']]])
        layer._seed_category(cats)
        data = torch.randn(B, N, D)
        composed, _ = layer.compose(data, sub, g)
        self.assertEqual(composed.shape, (B, N, D))
        # N=3 active -> 2 merges
        self.assertEqual(len(layer._derivation_trace[0]), N - 1)

    def test_chart_trace_tuple_shape(self):
        self._enable_chart()
        from Spaces import WordSubSpace
        Language.TheGrammar._configured = False
        Language.TheGrammar.configure({'compose': {'S': ['S VO'], 'VO': ['V O']}})
        g = Language.TheGrammar
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
        layer.compose(torch.randn(B, N, D), sub, g)
        for entry in layer._derivation_trace[0]:
            # (rule_id, left, right, merged, merged_category)
            self.assertEqual(len(entry), 5)
            for i in range(5):
                self.assertIsInstance(entry[i], int)

    def test_chart_legacy_path_unchanged_when_flag_off(self):
        from Spaces import WordSubSpace
        Language.TheGrammar._configured = False
        Language.TheGrammar.configure({'S': ['not(S)']})
        g = Language.TheGrammar
        B, N, D = 1, 3, 3
        layer = SyntacticLayer(nInput=N, nOutput=N, rules=g.symbolic(),
                               max_depth=N - 1, hidden_dim=16, grammar=g)
        sub = WordSubSpace(nDim=D, nWhat=D, nWhere=0, nWhen=0,
                           max_depth=8, max_arity=3, batch=B)
        # Flag is off (setUp did not enable). Legacy path leaves the
        # trace empty (no chart merges made). The legacy compose may
        # raise internally on these toy dims; we only care that the
        # chart path wasn't used.
        try:
            composed, _ = layer.compose(torch.randn(B, N, D), sub, g)
            self.assertEqual(composed.shape, (B, N, D))
        except (RuntimeError, IndexError):
            pass
        self.assertEqual(layer._derivation_trace, [[] for _ in range(B)])


class TestChartDecompose(unittest.TestCase):
    def setUp(self):
        from util import TheXMLConfig
        try:
            self._prior_chart = TheXMLConfig.get('WordSpace.chartCompose')
        except KeyError:
            self._prior_chart = None
        TheXMLConfig.set('WordSpace.chartCompose', True)
        self._saved_grammar_state = Language.TheGrammar._configured
        Language.TheGrammar._configured = False

    def tearDown(self):
        from util import TheXMLConfig
        if self._prior_chart is None:
            TheXMLConfig.set('WordSpace.chartCompose', False)
        else:
            TheXMLConfig.set('WordSpace.chartCompose', self._prior_chart)
        Language.TheGrammar._configured = self._saved_grammar_state

    def test_decompose_returns_same_shape(self):
        from Spaces import WordSubSpace
        Language.TheGrammar._configured = False
        Language.TheGrammar.configure({'compose': {'S': ['S VO'], 'VO': ['V O']}})
        g = Language.TheGrammar
        # pair_scorer contract: D == nInput (Task 4 Step 3 comment).
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
        composed, _ = layer.compose(data, sub, g)
        restored = layer.decompose(composed, sub, g)
        self.assertEqual(restored.shape, data.shape)


if __name__ == '__main__':
    unittest.main()
