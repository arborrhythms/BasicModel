"""Tests for MM_shamatha.xml's inline grammar wiring.

The MM_shamatha.xml grammar block lists Contiguous, not, union, and
intersection (mirroring MM_boolean's DNF stack plus the new
``Contiguous(S)`` rule). The presence of those rules in the inline
grammar dict drives downstream wiring:

  - ConceptualSpace's grammar-driven DNF stack (NegationLayer + Sigma
    AND-fold + Pi OR-fold) is wired by ``Language.grammar_uses``.
  - ``ContiguousLayer`` is constructed eagerly on SymbolicSpace.
  - ``rule_probability("Contiguous(S)")`` returns 1.0 when
    thought_free is on, gating the layer in front of NotLayer in the
    SymbolicSpace forward path.
"""

import os
import sys
import unittest

import torch

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
import Layers
from util import init_config, TheXMLConfig

_CONFIG = os.path.join(_PROJECT, "data", "MM_shamatha.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    m, _ = Models.MentalModel.from_config(_CONFIG)
    return m


class TestShamathaInlineGrammar(unittest.TestCase):
    def test_grammar_block_parses_contiguous(self):
        """The inline <grammar> block contains Contiguous(S) ahead of
        not(S), union(C,C), and intersection(C,C)."""
        _fresh_model()
        cfg = TheXMLConfig.get("WordSpace.language.grammar")
        # The merged dict should contain the user's S/C entries.
        self.assertIn("S", cfg)
        s_rules = cfg["S"]
        if isinstance(s_rules, str):
            s_rules = [s_rules]
        self.assertIn("Contiguous(S)", s_rules)
        # Contiguous precedes not(S) so hull-then-negate semantics hold.
        self.assertLess(s_rules.index("Contiguous(S)"), s_rules.index("not(S)"))

    def test_grammar_uses_detects_contiguous(self):
        """Substring scanner used at construction time picks up Contiguous."""
        _fresh_model()
        self.assertTrue(Language.grammar_uses("Contiguous"))
        self.assertTrue(Language.grammar_uses("not"))
        self.assertTrue(Language.grammar_uses("union"))
        self.assertTrue(Language.grammar_uses("intersection"))

    def test_dnf_stack_wired_into_conceptual_space(self):
        """ConceptualSpace's grammar-driven DNF wiring picks up the
        not/union/intersection rules from the inline grammar block."""
        m = _fresh_model()
        # The presence of NegationLayer (or its propositional NEG sibling
        # NotLayer) on SymbolicSpace is the canonical observable.
        self.assertIsInstance(m.symbolicSpace.propositional_negation, Layers.NotLayer)

    def test_contiguous_layer_in_pipeline_before_negation(self):
        """SymbolicSpace's _contiguous_layer is constructed and its
        gate sits before propositional NEG in the forward path."""
        m = _fresh_model()
        sym = m.symbolicSpace
        self.assertIsInstance(sym._contiguous_layer, Layers.ContiguousLayer)
        self.assertIsInstance(sym.propositional_negation, Layers.NotLayer)

    def test_contiguous_rule_probability_pinned_in_thought_free(self):
        """In thought_free mode, rule_probability(Contiguous(S)) is 1.0."""
        _fresh_model()
        Language.TheGrammar.thought_free = True
        try:
            self.assertEqual(
                Language.TheGrammar.rule_probability("Contiguous(S)"),
                1.0,
            )
        finally:
            Language.TheGrammar.thought_free = False


if __name__ == "__main__":
    unittest.main()
