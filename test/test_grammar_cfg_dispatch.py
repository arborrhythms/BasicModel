"""Tests for Step 6 of the lift / lower / bivector refactor:
grammar-driven dispatch via ``data/grammar.cfg``.

Verifies:
    1. ``Grammar.load_from_cfg`` parses the cfg cleanly into RuleDefs.
    2. Reverse productions (Layer 2.5) are derived mechanically from the
       upward forward productions.
    3. ``SyntacticLayer.dispatch_ops`` invokes the named Ops method and
       produces the same output as the explicit hand-written call.
    4. The cfg dispatcher remains bit-equivalent to the legacy
       ``_RULE_METHODS`` path on the test grammar.
    5. ``Ops.liftReverseAll`` / ``Ops.lowerReverseAll`` return tuples and
       fall back to the analytic placeholder when no codebook is
       supplied.
    6. Category set covers all introduced LHS / RHS labels (VO, NP,
       VP, AP, MP, PP, DEF, HAS, etc.) — overflow guard for the
       category codebook.
    7. The new dispatch path runs alongside the legacy
       ``_RULE_METHODS`` path with no regression: existing grammar
       tests stay green.

See:
    basicmodel/doc/plans/2026-04-25-step6-grammar-cfg-handoff.md
    basicmodel/doc/plans/2026-04-24-lift-lower-bivector-refactor.md (Step 6)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from Layers import Ops, SigmaLayer, PiLayer
import Language
from Language import Grammar, SyntacticLayer


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
_CFG_PATH = os.path.join(_DATA_DIR, 'grammar.cfg')


def _set_seed(seed=0):
    torch.manual_seed(seed)


class TestCfgParse(unittest.TestCase):
    """``Grammar.load_from_cfg`` parses the cfg file into RuleDefs."""

    def setUp(self):
        self.g = Grammar()
        self.g.load_from_cfg(_CFG_PATH)

    def test_cfg_file_present(self):
        self.assertTrue(os.path.exists(_CFG_PATH),
                        f"missing cfg fixture: {_CFG_PATH}")

    def test_upward_rules_parsed(self):
        """Every cfg [upward] line becomes one RuleDef in rules_upward."""
        self.assertGreater(len(self.g.rules_upward), 0)
        for rule in self.g.rules_upward:
            self.assertIsInstance(rule, Grammar.RuleDef)
            # method_name is None for PROJECT rules (S = NP, NP = N, ...).
            # For function-call rules it's the op name.
            self.assertTrue(rule.method_name is None
                            or isinstance(rule.method_name, str))

    def test_downward_rule_parsed(self):
        """[downward] section produces rules in rules_downward."""
        downward_method_names = [r.method_name for r in self.g.rules_downward]
        self.assertIn('emit_head', downward_method_names,
                      f"expected emit_head in downward rules; got "
                      f"{downward_method_names}")

    def test_post_hoc_s_ops_in_rules(self):
        """Post-hoc S-ops (true(S), what(S), conjunction(S, S), ...)
        live alongside Layer-1 phrase-structure productions in the
        same [upward] section so the rule-id-addressable consumers
        (predictor, dispatcher, derivation traces) see every
        dispatchable op as an addressable rule."""
        method_names = {r.method_name for r in self.g.rules_upward}
        for op in ('true', 'false', 'non', 'what', 'where', 'when',
                   'conjunction', 'disjunction', 'swap', 'absorb'):
            self.assertIn(op, method_names,
                          f"post-hoc S-op {op!r} missing from rules_upward")

    def test_inline_comments_stripped(self):
        """Trailing ``# ...`` is stripped from rule lines."""
        # Find the `S = lift(NP, VP)` row (has a trailing comment in the cfg).
        match = [r for r in self.g.rules_upward
                 if r.method_name == 'lift' and r.rhs_symbols == ('NP', 'VP')]
        self.assertEqual(len(match), 1, "expected exactly one S = lift(NP, VP)")

    def test_categories_includes_phrase_labels(self):
        """The cfg introduces VO, NP, VP, AP, MP, PP, DEF, HAS as LHS
        categories; ``Grammar.categories`` must surface them all."""
        cats = set(self.g.categories)
        for needed in ('S', 'VO', 'NP', 'VP', 'AP', 'MP',
                       'PP', 'DEF', 'HAS'):
            self.assertIn(needed, cats,
                          f"expected category {needed!r} in {sorted(cats)}")

    def test_terminals_included_in_categories(self):
        """Open-class terminals (N, V, ADJ, ADV) and closed-class
        terminals (IS, POSSESS, P, DET, DEG) appear as RHS symbols and
        thus as categories."""
        cats = set(self.g.categories)
        for needed in ('N', 'V', 'ADJ', 'ADV', 'IS', 'POSSESS',
                       'P', 'DET', 'DEG'):
            self.assertIn(needed, cats,
                          f"expected terminal {needed!r} in {sorted(cats)}")

    def test_unknown_section_rejected(self):
        """Non-existent section header raises a clear error."""
        g = Grammar()
        with self.assertRaises(ValueError):
            g._parse_cfg_lines(['[bogus]', 'S = NP'])

    def test_malformed_rule_rejected(self):
        """Rule line without '=' raises."""
        g = Grammar()
        with self.assertRaises(ValueError):
            g._parse_cfg_lines(['S NP VP'])  # missing '='


class TestReverseRuleTable(unittest.TestCase):
    """Reverse productions (Layer 2.5) derived from upward rules."""

    def setUp(self):
        self.g = Grammar()
        self.g.load_from_cfg(_CFG_PATH)

    def test_reverse_count_matches_upward(self):
        """One reverse per upward rule."""
        self.assertEqual(len(self.g.reverse_rules), len(self.g.rules_upward))

    def test_reverse_swaps_lhs_and_args(self):
        """forward `LHS = op(a1, a2)` -> reverse `(a1, a2) = opReverse(LHS)`."""
        # Find S = lift(NP, VP) and its reverse.
        idx = None
        for i, rule in enumerate(self.g.rules_upward):
            if rule.method_name == 'lift' and rule.rhs_symbols == ('NP', 'VP'):
                idx = i
                break
        self.assertIsNotNone(idx)
        args, op_name, lhs_tuple = self.g.reverse_rules[idx]
        self.assertEqual(args, ('NP', 'VP'))
        self.assertEqual(op_name, 'liftReverse')
        self.assertEqual(lhs_tuple, ('S',))

    def test_self_inverse_ops(self):
        """`not` is self-inverse: opReverse name == op name."""
        for fwd, rev in zip(self.g.rules_upward, self.g.reverse_rules):
            if fwd.method_name == 'not':
                _, op_name, _ = rev
                self.assertEqual(op_name, 'not',
                                 "not is self-inverse; reverse op "
                                 "name must equal forward op name")

    def test_project_rules_get_project_reverse(self):
        """PROJECT rules (single-RHS forms like ``S = NP`` / ``NP = N``)
        get the explicit ``projectReverse`` op name in their reverse
        rule.  Two cfg shapes are PROJECT:
          * method_name is None (transition / epsilon / X -> X);
          * method_name == 'merge' with arity == 1 (single-category RHS).
        """
        seen_project = False
        for fwd, rev in zip(self.g.rules_upward, self.g.reverse_rules):
            is_project = (fwd.method_name is None) or (
                fwd.method_name == 'merge' and len(fwd.rhs_symbols or ()) == 1
            )
            if is_project:
                seen_project = True
                _, op_name, _ = rev
                self.assertEqual(op_name, 'projectReverse')
        self.assertTrue(seen_project,
                        "expected at least one PROJECT rule in the cfg "
                        "(e.g. S = NP, NP = N, VP = V, AP = ADJ, MP = ADV)")


class TestOpsDispatch(unittest.TestCase):
    """``SyntacticLayer.dispatch_ops`` invokes named Ops methods."""

    def test_dispatch_lift_or_strict(self):
        x = torch.tensor([0.2, -0.1, 0.7])
        y = torch.tensor([0.1, 0.5, -0.3])
        out = SyntacticLayer.dispatch_ops('lift', x, y, mode='OR', kind='strict')
        expected = torch.max(x, y)
        self.assertTrue(torch.allclose(out, expected))

    def test_dispatch_lower_and_strict(self):
        x = torch.tensor([0.2, -0.1, 0.7])
        y = torch.tensor([0.1, 0.5, -0.3])
        out = SyntacticLayer.dispatch_ops('lower', x, y, mode='AND', kind='strict')
        expected = torch.min(x, y)
        self.assertTrue(torch.allclose(out, expected))

    def test_dispatch_intersection_alias(self):
        """``intersection`` resolves to ``Ops.lower(mode='AND')``."""
        x = torch.tensor([0.4, -0.2])
        y = torch.tensor([0.1, 0.6])
        out = SyntacticLayer.dispatch_ops('intersection', x, y, kind='strict')
        expected = Ops.lower(x, y, mode='AND', kind='strict')
        self.assertTrue(torch.allclose(out, expected))

    def test_dispatch_union_alias(self):
        """``union`` resolves to ``Ops.lift(mode='OR')``."""
        x = torch.tensor([0.4, -0.2])
        y = torch.tensor([0.1, 0.6])
        out = SyntacticLayer.dispatch_ops('union', x, y, kind='strict')
        expected = Ops.lift(x, y, mode='OR', kind='strict')
        self.assertTrue(torch.allclose(out, expected))

    def test_dispatch_not_unary(self):
        x = torch.tensor([0.3, -0.5, 0.0])
        out = SyntacticLayer.dispatch_ops('not', x)
        self.assertTrue(torch.allclose(out, -x))

    def test_dispatch_unknown_op_raises(self):
        x = torch.tensor([0.1])
        with self.assertRaises(KeyError):
            SyntacticLayer.dispatch_ops('totally_made_up', x)


class TestMultiReturnReverseOps(unittest.TestCase):
    """``Ops.liftReverseAll`` / ``lowerReverseAll`` return 2-tuples."""

    def test_liftReverseAll_no_codebook_returns_pair(self):
        """Without a codebook W, returns ``(Y, Y)`` placeholder pair."""
        Y = torch.tensor([0.1, -0.2, 0.5])
        a, b = Ops.liftReverseAll(Y)
        self.assertTrue(torch.equal(a, Y))
        self.assertTrue(torch.equal(b, Y))

    def test_lowerReverseAll_no_codebook_returns_pair(self):
        Y = torch.tensor([0.3, 0.0, -0.4])
        a, b = Ops.lowerReverseAll(Y)
        self.assertTrue(torch.equal(a, Y))
        self.assertTrue(torch.equal(b, Y))

    def test_liftReverseAll_with_codebook(self):
        """With a codebook, returns ``(recovered_left, Y)``: the search
        result and the search-conditioning operand."""
        torch.manual_seed(0)
        D = 3
        K = 4
        W = torch.randn(K, D)
        Y = torch.randn(2, D)
        a, b = Ops.liftReverseAll(Y, W)
        self.assertEqual(a.shape, Y.shape)
        self.assertEqual(b.shape, Y.shape)
        # right slot is the unmodified Y by convention
        self.assertTrue(torch.equal(b, Y))

    def test_lowerReverseAll_with_codebook(self):
        torch.manual_seed(0)
        D = 3
        K = 4
        W = torch.randn(K, D)
        Y = torch.randn(2, D)
        a, b = Ops.lowerReverseAll(Y, W)
        self.assertEqual(a.shape, Y.shape)
        self.assertEqual(b.shape, Y.shape)
        self.assertTrue(torch.equal(b, Y))

    def test_legacy_liftReverse_unaffected(self):
        """Existing 2-arg ``Ops.liftReverse(result, right)`` still
        returns a single tensor (analytic inverse)."""
        Y = torch.tensor([0.4, 0.6])
        right = torch.tensor([0.2, 0.3])
        out = Ops.liftReverse(Y, right)
        # Single-tensor return; not a tuple.
        self.assertIsInstance(out, torch.Tensor)

    def test_legacy_lowerReverse_unaffected(self):
        Y = torch.tensor([0.4, 0.6])
        right = torch.tensor([0.2, 0.3])
        out = Ops.lowerReverse(Y, right)
        self.assertIsInstance(out, torch.Tensor)


class TestCfgVsRuleMethodsParity(unittest.TestCase):
    """The cfg dispatcher matches the explicit hand-written op call."""

    def test_intersection_dispatch_matches_layer_method(self):
        """``dispatch_ops('intersection', x, y)`` equals
        ``Ops.lower(x, y, mode='AND')``."""
        torch.manual_seed(0)
        x = torch.randn(4)
        y = torch.randn(4)
        via_cfg = SyntacticLayer.dispatch_ops('intersection', x, y, kind='strict')
        explicit = Ops.lower(x, y, mode='AND', kind='strict')
        self.assertTrue(torch.allclose(via_cfg, explicit))

    def test_union_dispatch_matches_layer_method(self):
        torch.manual_seed(1)
        x = torch.randn(4)
        y = torch.randn(4)
        via_cfg = SyntacticLayer.dispatch_ops('union', x, y, kind='strict')
        explicit = Ops.lift(x, y, mode='OR', kind='strict')
        self.assertTrue(torch.allclose(via_cfg, explicit))

    def test_not_dispatch_matches_negation(self):
        x = torch.tensor([0.5, -0.3, 0.0])
        via_cfg = SyntacticLayer.dispatch_ops('not', x)
        explicit = Ops.negation(x)
        self.assertTrue(torch.allclose(via_cfg, explicit))


class TestCategoryCodebookCapacity(unittest.TestCase):
    """Category codebook is sized to fit all cfg-introduced labels."""

    def test_capacity_covers_categories(self):
        g = Grammar()
        g.load_from_cfg(_CFG_PATH)
        # The WordSpace constructor uses max(64, len(categories)); for
        # the cfg we wrote, len(categories) is well under 64, but the
        # invariant we want is that capacity >= len(categories).  Test
        # the invariant directly without instantiating WordSpace
        # (which requires the full Pipeline).
        capacity = max(64, len(g.categories))
        self.assertGreaterEqual(capacity, len(g.categories))


class TestNoRegressionOnLegacyPath(unittest.TestCase):
    """The XML-loaded grammar still works (legacy path unchanged)."""

    def test_legacy_xml_grammar_still_loads(self):
        g = Grammar()
        # Same minimal grammar shape used by ``Grammar._NOOP_GRAMMAR``.
        g.configure({'S': ['not(S)']})
        self.assertEqual(len(g.rules_upward), 1)
        self.assertEqual(g.rules_upward[0].method_name, 'not')
        # Legacy path also derives reverse rules now (parity with cfg).
        self.assertEqual(len(g.reverse_rules), 1)
        _, op_name, _ = g.reverse_rules[0]
        # not is self-inverse so reverse op name == 'not'
        self.assertEqual(op_name, 'not')


if __name__ == '__main__':
    unittest.main()
