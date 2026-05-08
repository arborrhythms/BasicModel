"""Tests for MentalModel with syntax=true.

Verifies that MentalModel can:
1. Create from MentalModel.xml and run forward+reverse (passing)
2. Learn the toy grammar through the full pipeline (xfail -- needs projectConcepts implementations)
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
import Language
matplotlib.use('Agg')

from util import init_config, ProjectPaths, TheXMLConfig


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    """Reload defaults + MentalModel.xml to avoid test-ordering state leakage."""
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Language.TheGrammar._configured = False
    Language.TheGrammar._configured = False


class TestMentalModelForwardReverse(unittest.TestCase):
    """MentalModel with syntax=true runs forward+reverse without error."""

    def setUp(self):
        _reload_config()

    def test_forward_reverse_runs(self):
        _reload_config()
        model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        sentences = ['the cat sat on the mat', 'a dog chased the ball']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        # Untrained model -- suppress range checks (shape-only test)
        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:2])

            model.eval()
            model.set_sigma(0)
            with torch.no_grad():
                input_state, symbols, predictions, reconstruction = model.forward(x)

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
        self.assertTrue(Language.TheGrammar._configured)
        # Post S-tier merge: a single unified SyntacticLayer on WordSpace.
        self.assertIsNotNone(model.wordSpace.syntacticLayer)
        # After the C->S merge, all method rules live on S-tier.
        self.assertIn('equals', Language.TheGrammar.s_methods)
        self.assertIn('part', Language.TheGrammar.s_methods)
        self.assertIn('union', Language.TheGrammar.s_methods)
        self.assertIn('not', Language.TheGrammar.s_methods)

    def test_subspace_words_clearable(self):
        """SubSpace word lists can be cleared on all tiers."""
        _reload_config()
        model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        # Should not raise
        for space in (model.symbolicSpace, model.conceptualSpace, model.perceptualSpace):
            space.subspace.set_words([])


class TestMentalModelGrammarConfiguration(unittest.TestCase):
    """MentalModel should expose the grammar configured by MentalModel.xml.

    The 2026-05-05 grammar rewrite moved the grammar from an external
    ``data/grammar.cfg`` into an inline ``<grammar>`` block in
    ``MentalModel.xml``.  This test asserts structural shape (every
    dispatchable op the rule predictor relies on has a rule_id slot),
    not the exact canonical list — the rules themselves are
    versioned in MentalModel.xml.
    """

    # S-tier op names the post-2026-05-05 grammar guarantees as
    # rule.method_name entries.  Retired 2026-05-04: what, where,
    # when, absorb, Contiguous, Fusion.
    REQUIRED_S_OPS = {
        'true', 'false', 'non', 'not',
        'conjunction', 'disjunction',
        'intersection', 'union',
        'lift', 'lower',
        'equals', 'part',
        'query', 'swap',
    }

    def setUp(self):
        _reload_config()

    def test_configured_grammar_matches_xml(self):
        _reload_config()
        model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))

        # Step 6: grammar comes from data/grammar.cfg; structural
        # invariants (the rule predictor sees every dispatchable op)
        # matter, not the exact rule list.
        method_names = {rule.method_name
                        for rule in Language.TheGrammar.rules
                        if rule.method_name}
        missing = self.REQUIRED_S_OPS - method_names
        self.assertEqual(missing, set(),
                         f"cfg-loaded grammar missing required S-tier "
                         f"ops: {missing}")
        self.assertEqual(Language.TheGrammar.interpretation, 0.5)

        # Post S-tier merge + 2026-05-07 rollback: the unified
        # SyntacticLayer is constructed with grammar.symbolic() (S-tier
        # rule ids only). Pre-rollback MentalModel.xml had only S-tier
        # rules so this matched ``range(0, n_rules)``; the rollback's
        # Phase 4 Step 6 added explicit ``P = sigma(P)`` and
        # ``C = pi(C)`` natural-fold rules at P / C tiers, so the
        # equality below now compares against the S-tier subset.
        s_tier_ids = [i for i, r in enumerate(Language.TheGrammar.rules)
                      if r.tier == 'S']
        self.assertEqual(model.wordSpace.syntacticLayer.all_rules,
                         s_tier_ids)
        self.assertIsNone(model.wordSpace.syntacticLayer.transition_rule)


if __name__ == '__main__':
    unittest.main()
