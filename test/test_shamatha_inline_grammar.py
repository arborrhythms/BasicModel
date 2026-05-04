"""Tests for MM_shamatha.xml's inline grammar wiring.

The MM_shamatha.xml grammar block lists ``disjunction(S, S)``,
``not(S)``, ``union(C, C)``, and ``intersection(C, C)``
(mirroring MM_boolean's DNF stack plus the post-codebook scalar-
max fold that replaced the retired ``Contiguous(S)`` 2026-05-04).
The presence of those rules drives downstream wiring:

  - ConceptualSpace's grammar-driven DNF stack (NotLayer + Sigma
    AND-fold + Pi OR-fold) is wired by ``Language.grammar_uses``.
  - ``DisjunctionLayer`` is constructed lazily in WordSpace's
    per-space SyntacticLayer when the grammar references
    ``disjunction(S, S)``.
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
    def test_grammar_block_parses_disjunction(self):
        """The inline <grammar> block contains disjunction(S, S)
        ahead of not(S), union(C, C), and intersection(C, C)."""
        _fresh_model()
        cfg = TheXMLConfig.get("WordSpace.language.grammar")
        self.assertIn("S", cfg)
        s_rules = cfg["S"]
        if isinstance(s_rules, str):
            s_rules = [s_rules]
        self.assertIn("disjunction(S, S)", s_rules)
        # Disjunction precedes not(S) so hull-then-negate semantics
        # hold (post-codebook scalar max is the new contiguity fold).
        self.assertLess(
            s_rules.index("disjunction(S, S)"), s_rules.index("not(S)"))

    def test_dnf_stack_wired_into_conceptual_space(self):
        """ConceptualSpace's grammar-driven DNF wiring picks up the
        not/union/intersection rules from the inline grammar block."""
        m = _fresh_model()
        self.assertIsInstance(
            m.symbolicSpace.propositional_negation, Layers.NotLayer)


if __name__ == "__main__":
    unittest.main()
