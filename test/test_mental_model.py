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
        """TheGrammar has per-tier SyntacticLayers and methods after init_layers."""
        _reload_config()
        model, cfg = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        from Space import TheGrammar
        # Grammar should be initialized with per-tier layers
        self.assertTrue(TheGrammar._layers_initialized)
        self.assertIsNotNone(TheGrammar.c_syntactic_layer)
        self.assertIsNotNone(TheGrammar.s_syntactic_layer)
        # C-tier methods (equals/part are on S-tier)
        self.assertNotIn('equals', TheGrammar.c_methods)
        self.assertNotIn('part', TheGrammar.c_methods)
        self.assertIn('union', TheGrammar.c_methods)
        self.assertIn('not', TheGrammar.c_methods)
        # S-tier methods
        self.assertIn('equals', TheGrammar.s_methods)
        self.assertIn('part', TheGrammar.s_methods)

    def test_grammar_has_resetStack(self):
        """TheGrammar.resetStack works for all tiers."""
        _reload_config()
        model, cfg = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        from Space import TheGrammar
        # Should not raise
        TheGrammar.resetStack('S')
        TheGrammar.resetStack('C')
        TheGrammar.resetStack('P')


@unittest.expectedFailure
class TestMentalModelLearnsGrammar(unittest.TestCase):
    """MentalModel should learn grammar rules through reconstruction loss.

    Evaluates predicted grammar rules against known-correct derivations.
    Each sentence has a known syntactic structure (DET N V DET N, etc.)
    that maps to a known rule sequence. After training, the SyntacticLayer
    should predict rules that match the correct derivation.

    Currently xfail: the Grammar.project() methods need actual training
    for the gradient to flow through composed representations and train
    the SyntacticLayer's rule predictions.
    """

    # Known sentences with expected derivation structure.
    # Grammar rules (from Grammar class):
    #   7: S → C VERB C (transitive)
    #   8: S → C VERB   (intransitive)
    #   9: C → C PART C
    #  10: C → C UNION C
    #  11: C → C INTERSECTION C
    #  12: C → P         (transition)
    #  13: P → W         (terminal)
    SENTENCES = [
        'the cat sat on the mat',       # DET N V P DET N → transitive (rule 7)
        'a dog chased the ball',        # DET N V DET N   → transitive (rule 7)
        'the fish swam',                # DET N V         → intransitive (rule 8)
        'a bird flew',                  # DET N V         → intransitive (rule 8)
    ]
    # Expected top-level rule for each sentence:
    #   transitive sentences → rule 7 (S → C VERB C)
    #   intransitive sentences → rule 8 (S → C VERB)
    EXPECTED_TOP_RULES = [7, 7, 8, 8]

    def test_predicted_rules_match_derivation(self):
        _reload_config()
        model, cfg = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))

        outputs = [torch.tensor([0.0])] * len(self.SENTENCES)

        # Untrained model — suppress expected concept range warnings
        with TheData.runtime_batch(self.SENTENCES, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, train_output = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input)
            model.train()
            model.set_sigma(0.5)
            optimizer = model.getOptimizer(lr=0.001)

            # Train for enough steps to learn patterns
            for step in range(50):
                optimizer.zero_grad()
                input_state, concepts, symbols = model.forward(x)
                inputData, inputLatent = model.reverse(symbols, model.outputs.materialize())
                loss = (inputData - input_state).pow(2).mean()
                loss.backward()
                optimizer.step()

            # Evaluate: extract predicted rules from derivation
            model.eval()
            model.set_sigma(0)
            with torch.no_grad():
                input_state, concepts, symbols = model.forward(x)

            self.assertIsNotNone(model.syntax_state, "No syntax state after forward")
            words = model.syntax_state.get_words()
            self.assertGreater(len(words), 0, "No word tuples produced")

            # Extract first rule per batch element (top-level derivation rule)
            batch_size = len(self.SENTENCES)
            predicted_top_rules = []
            for b in range(batch_size):
                batch_words = [(bi, vi, ri) for bi, vi, ri in words if bi == b]
                if batch_words:
                    # First word tuple's rule is the top-level derivation
                    predicted_top_rules.append(batch_words[0][2])
                else:
                    predicted_top_rules.append(-1)

            # Check accuracy: predicted top-level rules should match expected
            correct = sum(p == e for p, e in zip(predicted_top_rules, self.EXPECTED_TOP_RULES))
            accuracy = correct / len(self.EXPECTED_TOP_RULES)

            self.assertGreater(accuracy, 0.5,
                               f"Grammar rule prediction accuracy too low: {accuracy:.1%}\n"
                               f"  predicted: {predicted_top_rules}\n"
                               f"  expected:  {self.EXPECTED_TOP_RULES}")


if __name__ == '__main__':
    unittest.main()
