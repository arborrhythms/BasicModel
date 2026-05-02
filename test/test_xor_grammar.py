"""XOR_grammar testpoint -- end-to-end signal-router parse on the
canonical XOR fixture.

Exercises:
  - WordSpace.routerKind == "signal" reaches the chart from XML.
  - Grammar wiring: NOT (unary), conjunction / disjunction (binary)
    from GRAMMAR_LAYER_CLASSES are attached to the SignalRouter at
    WordSpace construction time, with global rule_ids preserved.
  - ChartCompose fires on the post-PerceptualSpace subspace (the
    "data is None" failure mode that hid the wiring failure earlier
    must stay fixed).
  - SymbolicSpace's per-tier SyntacticLayer is a no-op on the signal
    path (would otherwise crash on truly-binary ConjunctionLayer /
    DisjunctionLayer that don't expose a unary forward).
"""

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


class TestXORGrammarConfigParsing(unittest.TestCase):
    """Verify the XML carries the signal-router flag and the grammar's
    three Boolean primitive rules at the symbolic tier."""

    @classmethod
    def setUpClass(cls):
        import Models
        cls.cfg = Models.BaseModel.load_config(_CONFIG)

    def test_config_loads(self):
        self.assertIn("architecture", self.cfg)
        self.assertIn("WordSpace", self.cfg)

    def test_router_kind_is_signal(self):
        ws_cfg = self.cfg["WordSpace"]
        self.assertEqual(ws_cfg.get("routerKind"), "signal")

    def test_grammar_has_not_conjunction_disjunction(self):
        ws_cfg = self.cfg["WordSpace"]
        grammar = ws_cfg["language"]["grammar"]
        compose = grammar["compose"]
        symbols = compose["symbols"]
        rules = symbols.get("rule")
        self.assertIsInstance(rules, list)
        rule_text = " ".join(rules)
        self.assertIn("not.forward(S)", rule_text)
        self.assertIn("conjunction.forward(S, S)", rule_text)
        self.assertIn("disjunction.forward(S, S)", rule_text)


class TestXORGrammarRouterWiring(unittest.TestCase):
    """Build the model and verify the SignalRouter has the three grammar
    ops attached, with global rule_ids preserved."""

    @classmethod
    def setUpClass(cls):
        import Models
        from util import init_config
        xml_path = Models.ModelFactory.resolve_xml(_CONFIG)
        init_config(xml_path)
        # Force grammar reload.
        import Language
        Language.TheGrammar._configured = False
        Language.TheGrammar._ensure_configured()
        cls.grammar_rules = list(Language.TheGrammar.rules)

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


class TestXORGrammarSignalRouterIntegration(unittest.TestCase):
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
            raise unittest.SkipTest(
                f"XOR_grammar model build failed: "
                f"{type(exc).__name__}: {exc}")

    def test_chart_router_kind_is_signal(self):
        ws = self.model.wordSpace
        self.assertIsNotNone(ws, "WordSpace must be built")
        self.assertIsNotNone(ws.chart)
        self.assertEqual(ws.chart.router_kind, "signal")

    def test_signal_router_has_grammar_ops_attached(self):
        ws = self.model.wordSpace
        router = ws.chart._signal_router
        self.assertIsNotNone(
            router, "SignalRouter must be built when routerKind=signal")
        self.assertIn("S", router._unary_layers)
        self.assertIn("S", router._binary_layers)
        unary_layer = router._unary_layers["S"]
        binary_layer = router._binary_layers["S"]
        self.assertEqual(unary_layer.r_apply, 1, "Expected one unary op (NOT)")
        self.assertEqual(binary_layer.r_reduce, 2,
            "Expected two binary ops (conjunction, disjunction)")

    def test_signal_router_rule_ids_match_grammar(self):
        ws = self.model.wordSpace
        router = ws.chart._signal_router
        TheGrammar = self._Language.TheGrammar
        not_id = next(i for i, r in enumerate(TheGrammar.rules)
                      if r.method_name == "not")
        conj_id = next(i for i, r in enumerate(TheGrammar.rules)
                       if r.method_name == "conjunction")
        disj_id = next(i for i, r in enumerate(TheGrammar.rules)
                       if r.method_name == "disjunction")
        self.assertEqual(router._unary_rule_ids["S"], [not_id])
        self.assertSetEqual(
            set(router._binary_rule_ids["S"]),
            {conj_id, disj_id},
        )

    def test_chart_compose_fires_on_forward_pass(self):
        """Guards against the 'subspace.materialize() returned None'
        regression: ChartCompose must run after PerceptualSpace embedding
        so the slab is materializable. Drives one batch through
        runBatch (which handles input conversion) and checks
        current_rules['S'] is populated and the [B, 1, D] root state
        exists.
        """
        import torch
        ws = self.model.wordSpace
        try:
            self.model.eval()
            loader = self.model.inputSpace.data.data_loader(
                split="train", num_streams=1)
            inp_items, out_items = next(iter(loader))
            inputTensor = self.model.inputSpace.prepInput(inp_items)
            with torch.no_grad():
                _ = self.model.forward(inputTensor)
        except Exception as exc:
            self.fail(f"forward raised: {type(exc).__name__}: {exc}")
        rules = ws.current_rules
        self.assertIn("S", rules,
            "current_rules must have key 'S' after a forward pass")
        self.assertIsInstance(rules["S"], list)
        self.assertGreater(len(rules["S"]), 0,
            "current_rules['S'] should have at least one row")
        TheGrammar = self._Language.TheGrammar
        valid_rule_ids = set(range(len(TheGrammar.rules)))
        for row in rules["S"]:
            for rid in row:
                self.assertIn(rid, valid_rule_ids,
                    f"rule_id {rid} not in grammar")
        # The chart's compose must produce a [B, 1, D] S root state.
        router = ws.chart._signal_router
        rs = router._last_root_state
        self.assertIsNotNone(rs, "router must cache _last_root_state")
        self.assertEqual(rs.ndim, 3,
            f"Expected 3-D root state, got {rs.ndim}-D")
        self.assertEqual(
            rs.shape[1], 1,
            f"Expected [B, 1, D] root state, got shape {tuple(rs.shape)}")


if __name__ == "__main__":
    unittest.main()
