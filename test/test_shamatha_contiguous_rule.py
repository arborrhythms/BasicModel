"""Tests for the Contiguous grammar rule registration and gating.

The ``Contiguous`` rule is registered in
``SyntacticLayer._RULE_METHODS`` next to ``not`` / ``union`` /
``intersection``. Its rule_probability is pinned to 1.0 in
``thought_free`` mode, 0.0 otherwise (a per-request runtime gate set
by serve.py via ``TheGrammar.thought_free``).
"""

import os
import sys
import unittest

import torch

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Layers
import Language


class TestContiguousRuleRegistration(unittest.TestCase):
    def test_contiguous_in_rule_methods(self):
        self.assertIn("Contiguous", Language.SyntacticLayer._RULE_METHODS)

    def test_contiguous_dispatches_to_contiguous_layer(self):
        """ContiguousForward delegates to ContiguousLayer.forward.

        Stateless dispatch path: no SymbolicSpace context, so
        ContiguousForward constructs a transient layer of matching
        width. Output shape and values must match the layer's hull.
        """
        layer = Language.SyntacticLayer(
            nInput=3, nOutput=3, rules=[], grammar=Language.Grammar()
        )
        x = torch.tensor([
            [[0.5, 0.0, 0.2],
             [0.1, 0.9, 0.4]],
        ])
        out = layer.ContiguousForward(x, subspace=None)
        # Reference: layer-level hull on the same operand.
        ref_layer = Layers.ContiguousLayer(3, 3)
        ref = ref_layer(x)
        self.assertTrue(torch.allclose(out, ref))

    def test_contiguous_reverse_passes_through(self):
        layer = Language.SyntacticLayer(
            nInput=3, nOutput=3, rules=[], grammar=Language.Grammar()
        )
        y = torch.tensor([[[0.5, 0.9, 0.4], [0.5, 0.9, 0.4]]])
        rev = layer.ContiguousReverse(y, subspace=None)
        self.assertTrue(torch.allclose(rev, y))


class TestContiguousRuleProbability(unittest.TestCase):
    def setUp(self):
        # Reset state so each test sees a clean grammar.
        Language.TheGrammar.thought_free = False
        Language.TheGrammar._fired_bodies = None
        Language.TheGrammar._learned_rule_probs = None

    def tearDown(self):
        Language.TheGrammar.thought_free = False

    def test_rule_probability_high_when_thought_free(self):
        Language.TheGrammar.thought_free = True
        self.assertEqual(
            Language.TheGrammar.rule_probability("Contiguous(S)"),
            1.0,
        )

    def test_rule_probability_low_when_not_thought_free(self):
        Language.TheGrammar.thought_free = False
        self.assertEqual(
            Language.TheGrammar.rule_probability("Contiguous(S)"),
            0.0,
        )

    def test_other_rules_unchanged_by_thought_free(self):
        """Setting thought_free must not change the dormant defaults of
        other rules (intersection, union -> 1.0; not, non -> 0.0)."""
        Language.TheGrammar.thought_free = True
        self.assertEqual(
            Language.TheGrammar.rule_probability("intersection(C, C)"),
            1.0,
        )
        self.assertEqual(
            Language.TheGrammar.rule_probability("union(C, C)"),
            1.0,
        )
        self.assertEqual(
            Language.TheGrammar.rule_probability("not(S)"),
            0.0,
        )
        self.assertEqual(
            Language.TheGrammar.rule_probability("non(S)"),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
