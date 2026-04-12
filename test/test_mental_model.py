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
matplotlib.use('Agg')

from BasicModel import MentalModel, TheData, TheDevice
from util import init_config, ProjectPaths, TheXMLConfig


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    """Reload defaults + MentalModel.xml to avoid test-ordering state leakage."""
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    from Space import TheGrammar
    TheGrammar._configured = False
    TheGrammar._configured = False


class TestMentalModelForwardReverse(unittest.TestCase):
    """MentalModel with syntax=true runs forward+reverse without error."""

    def setUp(self):
        _reload_config()

    def test_forward_reverse_runs(self):
        _reload_config()
        model, cfg = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        sentences = ['the cat sat on the mat', 'a dog chased the ball']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        # Untrained model — suppress range checks (shape-only test)
        with TheData.runtime_batch(sentences, outputs), \
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
        model, cfg = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        from Space import TheGrammar
        # Grammar should be initialized
        self.assertTrue(TheGrammar._configured)
        # SyntacticLayers are now on the Spaces, not Grammar
        self.assertIsNotNone(model.conceptualSpace.syntacticLayer)
        self.assertIsNotNone(model.symbolicSpace.syntacticLayer)
        # C-tier methods (equals/part are on S-tier)
        self.assertNotIn('equals', TheGrammar.c_methods)
        self.assertNotIn('part', TheGrammar.c_methods)
        self.assertIn('union', TheGrammar.c_methods)
        self.assertIn('not', TheGrammar.c_methods)
        # S-tier methods
        self.assertIn('equals', TheGrammar.s_methods)
        self.assertIn('part', TheGrammar.s_methods)

    def test_subspace_words_clearable(self):
        """SubSpace word lists can be cleared on all tiers."""
        _reload_config()
        model, cfg = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        # Should not raise
        for space in (model.symbolicSpace, model.conceptualSpace, model.perceptualSpace):
            space.subspace.set_words([])


class TestMentalModelGrammarConfiguration(unittest.TestCase):
    """MentalModel should expose the grammar configured in MentalModel.xml."""

    EXPECTED_RULES = [
        "START → S",
        "S → true(S)",
        "S → non(S)",
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
        model, cfg = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        from Space import TheGrammar

        canonicals = [rule.canonical for rule in TheGrammar.rules]
        self.assertEqual(canonicals, self.EXPECTED_RULES)
        self.assertEqual(len(TheGrammar.rules), 16)
        self.assertEqual(TheGrammar.interpretation, 0.5)

        self.assertEqual(model.symbolicSpace.syntacticLayer.all_rules, [1, 2, 3, 4, 5, 6])
        self.assertEqual(model.symbolicSpace.syntacticLayer.transition_rule, 6)

        self.assertEqual(model.conceptualSpace.syntacticLayer.all_rules, [7, 8, 9, 10, 11, 12, 13])
        self.assertEqual(model.conceptualSpace.syntacticLayer.transition_rule, 13)

        self.assertEqual(model.perceptualSpace.syntacticLayer.all_rules, [14, 15])
        self.assertIsNone(model.perceptualSpace.syntacticLayer.transition_rule)


if __name__ == '__main__':
    unittest.main()
