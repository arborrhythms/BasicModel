"""Tests for MentalModel with syntax=true.

Verifies that MentalModel can:
1. Create from MentalModel.xml and run forward+reverse (passing)
2. Learn the toy grammar through the full pipeline (xfail — needs projectConcepts implementations)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import warnings
import torch
import matplotlib
import Models
import Spaces
matplotlib.use('Agg')

from util import init_config, ProjectPaths, TheXMLConfig


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    """Reload defaults + MentalModel.xml to avoid test-ordering state leakage."""
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Spaces.TheGrammar._configured = False
    Spaces.TheGrammar._configured = False


class TestMentalModelForwardReverse(unittest.TestCase):
    """MentalModel with syntax=true runs forward+reverse without error."""

    def setUp(self):
        _reload_config()

    def test_forward_reverse_runs(self):
        _reload_config()
        model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        sentences = ['the cat sat on the mat', 'a dog chased the ball']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        # Untrained model — suppress range checks (shape-only test)
        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:2])

            model.eval()
            model.set_sigma(0)
            with torch.no_grad():
                input_state, concepts, symbols = model.forward(x)

            self.assertEqual(symbols.ndim, 3)
            self.assertEqual(symbols.shape[0], 2)  # batch=2

            if model.reversible:
                with torch.no_grad():
                    try:
                        inputData, inputLatent = model.reverse(symbols, model.outputs.materialize())
                    except ValueError:
                        self.skipTest("Untrained model range violation (expected)")
                self.assertEqual(inputData.ndim, 3)
                self.assertEqual(inputData.shape[0], 2)

    def test_grammar_has_syntactic_layers(self):
        """TheGrammar is initialized and Spaces own SyntacticLayers after init_layers."""
        _reload_config()
        model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        # Grammar should be initialized
        self.assertTrue(Spaces.TheGrammar._configured)
        # SyntacticLayers are now on the Spaces, not Grammar
        self.assertIsNotNone(model.wordSpace.conceptualSyntacticLayer)
        self.assertIsNotNone(model.wordSpace.symbolicSyntacticLayer)
        # C-tier methods (equals/part are on S-tier)
        self.assertNotIn('equals', Spaces.TheGrammar.c_methods)
        self.assertNotIn('part', Spaces.TheGrammar.c_methods)
        self.assertIn('union', Spaces.TheGrammar.c_methods)
        self.assertIn('not', Spaces.TheGrammar.c_methods)
        # S-tier methods
        self.assertIn('equals', Spaces.TheGrammar.s_methods)
        self.assertIn('part', Spaces.TheGrammar.s_methods)

    def test_subspace_words_clearable(self):
        """SubSpace word lists can be cleared on all tiers."""
        _reload_config()
        model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        # Should not raise
        for space in (model.symbolicSpace, model.conceptualSpace, model.perceptualSpace):
            space.subspace.set_words([])


class TestMentalModelGrammarConfiguration(unittest.TestCase):
    """MentalModel should expose the grammar configured in MentalModel.xml."""

    # Expected canonical rule strings from MentalModel.xml after the
    # trinity / coordination / demux / query extensions. The S-tier
    # ordering follows the literal XML order: trinity → coordination →
    # demux selectors → query → swap/equals/part → S→C transition.
    EXPECTED_RULES = [
        "START → S",
        "S → true(S)",
        "S → false(S)",
        "S → non(S)",
        "S → conjunction(S, S)",
        "S → disjunction(S, S)",
        "S → what(S)",
        "S → where(S)",
        "S → when(S)",
        "S → query(S, S)",
        "S → swap(S, S)",
        "S → equals(S, S)",
        "S → part(S, S)",
        "S → C",
        "C → not(C)",
        "C → intersection(C, C)",
        "C → union(C, C)",
        "C → lower(C, C)",
        "C → lift(C, C)",
        "C → lift(C, C, C)",
        "C → P",
        "P → chunk(I, P)",
        "P → I",
    ]

    def setUp(self):
        _reload_config()

    def test_configured_grammar_matches_xml(self):
        _reload_config()
        model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))

        canonicals = [rule.canonical for rule in Spaces.TheGrammar.rules]
        self.assertEqual(canonicals, self.EXPECTED_RULES)
        # 1 START + 13 S-tier + 7 C-tier + 2 P-tier = 23 total rules.
        self.assertEqual(len(Spaces.TheGrammar.rules), 23)
        self.assertEqual(Spaces.TheGrammar.interpretation, 0.5)

        # S-tier indices 1..13 (12 method rules + transition at index 13)
        self.assertEqual(model.wordSpace.symbolicSyntacticLayer.all_rules,
                         list(range(1, 14)))
        self.assertEqual(model.wordSpace.symbolicSyntacticLayer.transition_rule, 13)

        # C-tier indices 14..20 (6 method rules + transition at index 20)
        self.assertEqual(model.wordSpace.conceptualSyntacticLayer.all_rules,
                         list(range(14, 21)))
        self.assertEqual(model.wordSpace.conceptualSyntacticLayer.transition_rule, 20)

        # P-tier indices 21..22 (chunk, terminal I)
        self.assertEqual(model.wordSpace.perceptualSyntacticLayer.all_rules, [21, 22])
        self.assertIsNone(model.wordSpace.perceptualSyntacticLayer.transition_rule)


if __name__ == '__main__':
    unittest.main()
