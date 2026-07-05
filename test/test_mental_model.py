"""Tests for BasicModel with syntax=true.

Verifies that BasicModel can:
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


class TestBasicModelForwardReverse(unittest.TestCase):
    """BasicModel with syntax=true runs forward+reverse without error."""

    def setUp(self):
        _reload_config()

    # test_forward_reverse_runs retired 2026-05-14 (reverse pipeline / <maskedPrediction> retired in IR-only refactor).

    def test_grammar_has_syntactic_layers(self):
        """TheGrammar is initialized and Spaces own SyntacticLayers after init_layers."""
        _reload_config()
        model, cfg = Models.BasicModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        # Grammar should be initialized
        self.assertTrue(Language.TheGrammar._configured)
        # Post 2026-05-08 SyntacticLayer rename: per-space dispatchers
        # live on the home spaces (CS / SS each own a SyntacticLayer).
        # Bivector retirement (2026-05-20) made the PartSpace-
        # space_role SyntacticLayer optional — not all configs wire a
        # ``subsymbolic`` SyntacticLayer, so accept None there.
        for space in (model.conceptualSpace, model.wholeSpace):
            self.assertIsNotNone(getattr(space, 'syntacticLayer', None),
                                 f"{space.name} missing per-space "
                                 f"SyntacticLayer")
        # Post-2026-05-29 grammar-file refactor: space_role comes from the
        # layer class (per ``_reassign_space_roles_from_layer_classes``), so
        # ``s_methods`` returns only methods whose layer declares
        # ``space_role='SS'``. Assertive ``conjunction`` /
        # ``disjunction`` / ``exist`` are SS-space_role; query and logical
        # answer operations live on CS-space_role per their GrammarLayer
        # subclasses. ADAPTED 2026-07-05: isEqual RELOCATED to <Queries>
        # (Alec: equality is a query, not parse structure) -- it must NOT
        # appear as a parse method any more.
        self.assertNotIn('isEqual', Language.TheGrammar.s_methods)
        self.assertIn('conjunction', Language.TheGrammar.s_methods)
        self.assertIn('disjunction', Language.TheGrammar.s_methods)
        self.assertIn('exist', Language.TheGrammar.s_methods)

    def test_subspace_words_clearable(self):
        """SubSpace word lists can be cleared on all space_roles."""
        _reload_config()
        model, cfg = Models.BasicModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
        # Should not raise
        for space in (model.wholeSpace, model.conceptualSpace, model.perceptualSpace):
            space.subspace.set_words([])


class TestBasicModelGrammarConfiguration(unittest.TestCase):
    """BasicModel should expose the grammar configured by MentalModel.xml.

    The 2026-05-05 grammar rewrite moved the grammar from an external
    ``data/grammar.cfg`` into ``data/complete.grammar`` (referenced by
    ``MentalModel.xml``).  This test asserts structural shape (every
    dispatchable op the rule predictor relies on has a rule_id slot),
    not the exact canonical list — the rules themselves are
    versioned in MentalModel.xml.
    """

    # Dispatchable op names the current complete grammar guarantees as
    # rule.method_name entries. Retired 2026-05-04: what, where, when,
    # absorb, Contiguous, Fusion. The relation rewrite split old
    # query/part into explicit queryPart/assertPart forms, while
    # equality stays ``isEqual`` with rule.query metadata.
    # Retired 2026-05-30 (subsymbolic-analyzer-terminal-emitter, decision
    # #4): copy / swap are removed from the symbolic grammar. The MARKER
    # idiom they implemented is replaced by markers that are learned and
    # owned by each operator (absorb/emit on the T1-T5 SurfaceSchema), so
    # the symbolic rule predictor no longer dispatches copy/swap.
    # Role-collapse (default since 2026-06-03, D1 gate met) unifies the part
    # operator: the grammar declares the single relative op ``isPart`` and the
    # query/assert split is recovered by dispatch (``_dispatch_method_name_
    # for_rule`` maps ``isPart`` -> ``queryPart`` in a query context), so the
    # rule method_names carry ``isPart`` where the transitional
    # ``complete.grammar`` carried ``assertPart`` + ``queryPart``.
    # ADAPTED 2026-07-05: the relation family (isEqual, isPart) relocated
    # to <Queries>; lattice max renamed union -> join (the additive pair
    # owns 'union').
    REQUIRED_OPS = {
        'non', 'not',
        'conjunction', 'disjunction',
        'intersection', 'join',
        'lift', 'lower',
        'exist',
    }

    def setUp(self):
        _reload_config()

    def test_configured_grammar_matches_xml(self):
        _reload_config()
        model, cfg = Models.BasicModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))

        # grammar comes from MentalModel.xml (complete.grammar, the
        # default since 2026-06-03); structural invariants (the rule predictor
        # sees every dispatchable op) matter, not the exact rule list.
        method_names = {rule.method_name
                        for rule in Language.TheGrammar.rules
                        if rule.method_name}
        missing = self.REQUIRED_OPS - method_names
        self.assertEqual(missing, set(),
                         f"cfg-loaded grammar missing required "
                         f"ops: {missing}")
        self.assertEqual(Language.TheGrammar.interpretation, 0.5)

        # Stage 3 (2026-05-27): the chart retired; the signal router
        # (``SymbolSubSpace.languageLayer``) carries the grammar reference
        # for diagnostics and gating.
        self.assertIs(model.symbolSpace.languageLayer.grammar,
                      Language.TheGrammar)
        # Post-2026-05-29 grammar-file refactor: ``load_from_grammar_file``
        # injects an ``S = S`` identity rule (method_name=None, arity=1,
        # space_role='SS') so the per-word cursor has a no-op transition to
        # pad against. ``symbolic_transition()`` now returns that
        # rule's id rather than None.
        self.assertIsNotNone(Language.TheGrammar.symbolic_transition())


if __name__ == '__main__':
    unittest.main()
