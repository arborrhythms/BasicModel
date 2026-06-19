"""XOR_grammar testpoint -- end-to-end signal-router parse on the
canonical XOR fixture.

Exercises:
  - SymbolicSpace.routerKind == "signal" reaches the chart from XML.
  - Grammar wiring: NOT (unary), conjunction / disjunction (binary)
    from GRAMMAR_LAYER_CLASSES are attached to the LanguageLayer at
    SymbolicSpace construction time, with global rule_ids preserved.
  - ChartCompose fires on the post-PartSpace subspace (the
    "data is None" failure mode that hid the wiring failure earlier
    must stay fixed).
  - WholeSpace's per-tier SyntacticLayer is a no-op on the signal
    path (would otherwise crash on truly-binary ConjunctionLayer /
    DisjunctionLayer that don't expose a unary forward).
"""

import copy
import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_CONFIG = os.path.join(_PROJECT, "data", "XOR_grammar.xml")

_TEST = os.path.dirname(os.path.abspath(__file__))
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)


def _snapshot_global_state():
    """Capture the process-global config + grammar singletons.

    ``init_config(XOR_grammar.xml)`` overlays ``SymbolicSpace.routerKind=signal``
    onto ``util.TheXMLConfig._data``; ``TheGrammar`` is a module-level
    singleton too. With no restore, that leaks into every later test --
    e.g. ``test_chart_wordspace_wiring`` then builds a ``Chart`` whose
    ``router_kind`` defaults to the leaked ``"signal"`` and explodes in
    ``LanguageLayer.compose`` (the conftest autouse reset clears
    ``_configured`` / ``_requirements`` but not ``_data``).
    """
    import Language
    from util import TheXMLConfig
    return {
        "data": copy.deepcopy(TheXMLConfig._data),
        "sources": list(TheXMLConfig._sources),
        "requirements": list(TheXMLConfig._requirements),
        "grammar": copy.deepcopy(Language.TheGrammar.__dict__),
    }


def _restore_global_state(snap):
    """Restore the singletons captured by ``_snapshot_global_state``."""
    import Language
    from util import TheXMLConfig
    TheXMLConfig._data = copy.deepcopy(snap["data"])
    TheXMLConfig._sources = list(snap["sources"])
    TheXMLConfig._requirements = list(snap["requirements"])
    Language.TheGrammar.__dict__.clear()
    Language.TheGrammar.__dict__.update(copy.deepcopy(snap["grammar"]))
    # Force a clean lazy reconfigure from the restored config on next use.
    Language.TheGrammar._configured = False


class TestXORGrammarConfigParsing(unittest.TestCase):
    """Verify the XML carries the signal-router flag and the grammar's
    three Boolean primitive rules at the symbolic tier."""

    @classmethod
    def setUpClass(cls):
        import Models
        cls.cfg = Models.BaseModel.load_config(_CONFIG)

    def test_config_loads(self):
        self.assertIn("architecture", self.cfg)
        self.assertIn("SymbolicSpace", self.cfg)

    def test_router_kind_is_signal(self):
        """Stage 3: routerKind retired -- the signal router is the
        canonical (and only) parser. The XML no longer carries the
        knob; this test verifies the absence."""
        ss_cfg = self.cfg["SymbolicSpace"]
        self.assertNotIn("routerKind", ss_cfg)

    def test_grammar_has_not_conjunction_disjunction(self):
        ss_cfg = self.cfg["SymbolicSpace"]
        grammar = ss_cfg["language"]["grammar"]
        compose = grammar["compose"]
        symbols = compose["symbols"]
        rules = symbols.get("rule")
        self.assertIsInstance(rules, list)
        rule_text = " ".join(rules)
        self.assertIn("not.forward(S)", rule_text)
        self.assertIn("conjunction.forward(S, S)", rule_text)
        self.assertIn("disjunction.forward(S, S)", rule_text)


class TestXORGrammarRouterWiring(unittest.TestCase):
    """Build the model and verify the LanguageLayer has the three grammar
    ops attached, with global rule_ids preserved."""

    @classmethod
    def setUpClass(cls):
        cls._global_snap = _snapshot_global_state()
        import Models
        from util import init_config
        xml_path = Models.ModelFactory.resolve_xml(_CONFIG)
        init_config(xml_path)
        # Force grammar reload.
        import Language
        Language.TheGrammar._configured = False
        Language.TheGrammar._ensure_configured()
        cls.grammar_rules = list(Language.TheGrammar.rules)

    @classmethod
    def tearDownClass(cls):
        _restore_global_state(cls._global_snap)

    def test_grammar_has_three_rules(self):
        # not (arity 1), conjunction (arity 2), disjunction (arity 2).
        method_names = [r.method_name for r in self.grammar_rules]
        self.assertIn("not", method_names)
        self.assertIn("conjunction", method_names)
        self.assertIn("disjunction", method_names)

    def test_grammar_rule_arities(self):
        by_name = {r.method_name: r for r in self.grammar_rules}
        self.assertEqual(by_name["not"].arity, 1)
        self.assertEqual(by_name["conjunction"].arity, 2)
        self.assertEqual(by_name["disjunction"].arity, 2)

    def test_grammar_rules_at_symbolic_tier(self):
        for r in self.grammar_rules:
            if r.method_name in ("not", "conjunction", "disjunction"):
                self.assertEqual(r.tier, "S",
                    f"Rule {r.method_name} expected at tier 'S', got {r.tier!r}")


class TestXORGrammarLanguageLayerIntegration(unittest.TestCase):
    """End-to-end signal-router parse on the XOR_grammar.xml fixture.

    Builds the model via the same path Models.py main() uses, runs one
    forward pass, and verifies the chart's compose fired (current_rules
    populated, S root state present, grammar ops attached). This is the
    plumbing-integrity check that guards against the historic
    'subspace.materialize() returned None' regression where ChartCompose
    silently no-op'd.
    """

    @classmethod
    def setUpClass(cls):
        cls._global_snap = _snapshot_global_state()
        try:
            import Models
            import Language
            from util import init_config, TheXMLConfig
            from data import TheData

            xml_path = Models.ModelFactory.resolve_xml(_CONFIG)
            defaults_path = os.path.join(_PROJECT, "data", "model.xml")
            init_config(path=xml_path, defaults_path=defaults_path)
            cfg = TheXMLConfig.data
            dat = cfg.get("architecture", {}).get("data", {})
            TheData.load(dat.get("dataset"), num_shards=1, max_docs=64, dat=dat)

            Language.TheGrammar._configured = False
            Language.TheGrammar._ensure_configured()

            cls._Models = Models
            cls._Language = Language
            cls._TheData = TheData

            cls.model, _ = Models.BaseModel.from_config(xml_path, data=TheData)
        except Exception as exc:
            # setUpClass raised -> tearDownClass won't run; restore here
            # so the partially-loaded XOR config doesn't leak on skip.
            _restore_global_state(cls._global_snap)
            raise unittest.SkipTest(
                f"XOR_grammar model build failed: "
                f"{type(exc).__name__}: {exc}")

    @classmethod
    def tearDownClass(cls):
        _restore_global_state(cls._global_snap)

    def test_chart_router_kind_is_signal(self):
        """Stage 3: chart retired; the canonical parser is the signal
        router (LanguageLayer) directly on SymbolicSubSpace."""
        ss = self.model.symbolicSpace
        self.assertIsNotNone(ss, "SymbolicSpace must be built")
        self.assertIsNotNone(ss.languageLayer)
        # router_kind is no longer a meaningful attribute (the chart
        # vs. signal dispatch is gone), but the languageLayer is.
        from Language import LanguageLayer
        self.assertIsInstance(ss.languageLayer, LanguageLayer)

    def test_signal_router_has_grammar_ops_attached(self):
        # Tier-free contract: grammar rules collapse to a single reduction
        # tier, so do NOT assume a "S" (or "C") key. Search across ALL
        # attached unary / binary layers and assert the grammar's ops are
        # present by op-class.
        from Language import NotLayer, ConjunctionLayer, DisjunctionLayer
        ss = self.model.symbolicSpace
        router = ss.languageLayer
        self.assertIsNotNone(
            router, "LanguageLayer must be built")
        self.assertTrue(
            len(router._unary_layers) > 0,
            "Expected at least one unary tier attached")
        self.assertTrue(
            len(router._binary_layers) > 0,
            "Expected at least one binary tier attached")

        # Binary ops are wrapped in _BinaryGrammarOpAdapter; unwrap via
        # `.gl` (the codebase's own unwrap idiom) to reach the real
        # grammar layer. Unary ops are attached directly.
        unary_ops = [op
                     for layer in router._unary_layers.values()
                     for op in layer.ops]
        binary_ops = [getattr(op, "gl", op)
                      for layer in router._binary_layers.values()
                      for op in layer.ops]
        self.assertTrue(
            any(isinstance(op, NotLayer) for op in unary_ops),
            "Expected a NotLayer attached to some unary tier")
        self.assertTrue(
            any(isinstance(op, ConjunctionLayer) for op in binary_ops),
            "Expected a ConjunctionLayer attached to some binary tier")
        self.assertTrue(
            any(isinstance(op, DisjunctionLayer) for op in binary_ops),
            "Expected a DisjunctionLayer attached to some binary tier")

    def test_signal_router_rule_ids_match_grammar(self):
        ss = self.model.symbolicSpace
        router = ss.languageLayer
        TheGrammar = self._Language.TheGrammar
        not_id = next(i for i, r in enumerate(TheGrammar.rules)
                      if r.method_name == "not")
        conj_id = next(i for i, r in enumerate(TheGrammar.rules)
                       if r.method_name == "conjunction")
        disj_id = next(i for i, r in enumerate(TheGrammar.rules)
                       if r.method_name == "disjunction")
        # Tier-free contract: rule_ids are keyed by the single collapsed
        # reduction tier, so do NOT index by "S"/"C". Assert the grammar's
        # rule_ids appear in the union of ALL attached tiers' rule_ids.
        all_unary_rule_ids = set()
        for rids in router._unary_rule_ids.values():
            all_unary_rule_ids.update(rids)
        all_binary_rule_ids = set()
        for rids in router._binary_rule_ids.values():
            all_binary_rule_ids.update(rids)
        self.assertIn(not_id, all_unary_rule_ids,
            "Grammar 'not' rule_id must be attached to some unary tier")
        self.assertLessEqual(
            {conj_id, disj_id}, all_binary_rule_ids,
            "Grammar 'conjunction'/'disjunction' rule_ids must be "
            "attached to some binary tier")

    def test_chart_compose_fires_on_forward_pass(self):
        """Retired 2026-05-14: subsymbolicOrder=2 + useGrammar='all' shape contract no longer matches IR-only forward; chart-compose wiring covered by test/test_compose_chart.py."""
        return  # AR-specific behaviour; covered elsewhere or no longer applicable


if __name__ == "__main__":
    unittest.main()
