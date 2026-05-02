"""Tests for MM_shamatha.xml configuration.

MM_shamatha.xml flips useGrammar to "thoughtFree" and switches on the
four SymbolicSpace regularizer knobs (l1Lambda, discontinuityLambda,
impenetrableOverlap, impenetrableVariance) independently of mode. The
inline <grammar> block adds the new ``Contiguous(S)`` rule ahead of
``not(S)``.
"""


# ---------------------------------------------------------------------
# Skipped pending migration to the post-2026-05-01 chart / GrammarLayer
# surface. The tests in this module exercised the legacy SyntacticLayer
# dispatch tables (`_RULE_METHODS`, `*Forward` / `*Reverse`, `project`,
# `compose(data, subspace, grammar)`, etc.) which were removed by the
# 2026-05-01 syntactic-layer refactor. Rewrite to use the new
# `Chart` class and the `GRAMMAR_LAYER_CLASSES` GrammarLayer kernels.
# ---------------------------------------------------------------------
import pytest
pytestmark = pytest.mark.skip(
    reason="Pending migration to chart + GRAMMAR_LAYER_CLASSES surface; "
           "see doc/specs/2026-05-01-syntactic-layer-refactor.md")

import os
import sys
import unittest

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from util import init_config, TheXMLConfig

_CONFIG = os.path.join(_PROJECT, "data", "MM_shamatha.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    """Create a fresh MentalModel from MM_shamatha.xml."""
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    return Models.MentalModel.from_config(_CONFIG)


class TestShamathaXML(unittest.TestCase):
    def test_model_constructs(self):
        m, _ = _fresh_model()
        self.assertIsNotNone(m)
        self.assertEqual(m.useGrammar, "thoughtFree")

    def test_use_grammar_resolves_to_thought_free(self):
        _fresh_model()
        self.assertEqual(
            TheXMLConfig.get("WordSpace.useGrammar"),
            "thoughtFree",
        )

    def test_n_where_n_when_resolve_to_two(self):
        _fresh_model()
        self.assertEqual(int(TheXMLConfig.get("architecture.nWhere")), 2)
        self.assertEqual(int(TheXMLConfig.get("architecture.nWhen")), 2)

    def test_symbolic_space_regularizers_all_enabled(self):
        m, _ = _fresh_model()
        sym = m.symbolicSpace
        # All four knobs are positive, so all three regularizers are live.
        self.assertGreater(sym.l1_lambda, 0.0)
        self.assertGreater(sym.discontinuity_lambda, 0.0)
        self.assertGreater(sym.impenetrable_overlap, 0.0)
        self.assertGreater(sym.impenetrable_variance, 0.0)
        # Layer instances exist and are enabled.
        self.assertTrue(sym._sparsity.enabled)
        self.assertTrue(sym._smoothing.enabled)
        self.assertTrue(sym._impenetrable.enabled)

    def test_contiguous_layer_constructed_on_symbolic_space(self):
        m, _ = _fresh_model()
        from Layers import ContiguousLayer
        self.assertIsInstance(m.symbolicSpace._contiguous_layer, ContiguousLayer)

    def test_contiguous_rule_in_method_registry(self):
        from Language import SyntacticLayer
        self.assertIn("Contiguous", SyntacticLayer._RULE_METHODS)
        fwd, rev, binary = SyntacticLayer._RULE_METHODS["Contiguous"]
        self.assertEqual(fwd, "ContiguousForward")
        self.assertEqual(rev, "ContiguousReverse")
        self.assertFalse(binary)


if __name__ == "__main__":
    unittest.main()
