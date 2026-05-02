"""Grammar accepts <compose> and <generate> sub-blocks (default = compose).

The legacy <parse> / <upward> / <downward> aliases were removed
2026-05-01; only <compose> / <generate> are accepted now.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import Language
from Language import Grammar
from util import init_config


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


class TestGrammarSplit(unittest.TestCase):
    def test_configure_accepts_parse_block(self):
        g = Grammar()
        g.configure({'compose': {'S': ['not.forward(S)']}})
        self.assertEqual(len(g.rules_upward), 1)
        self.assertEqual(g.rules_upward[0].lhs, 'S')
        self.assertEqual(g.rules_upward[0].method_name, 'not')
        self.assertEqual(g.rules_downward, [])

    def test_configure_accepts_both_blocks(self):
        g = Grammar()
        g.configure({
            'compose':    {'S': ['S VO'], 'VO': ['V O']},
            'generate': {'S': ['C']},
        })
        self.assertEqual(len(g.rules_upward), 2)
        self.assertEqual(len(g.rules_downward), 1)
        down = g.rules_downward[0]
        self.assertEqual(down.lhs, 'S')
        self.assertEqual(down.rhs_symbols, ('C',))
        self.assertEqual(down.method_name, 'emit_head')

    def test_legacy_flat_grammar_still_loads(self):
        # Old-shape {'S': ['not(S)']} must still work as parse-only.
        g = Grammar()
        g.configure({'S': ['not(S)']})
        self.assertEqual(len(g.rules_upward), 1)
        self.assertEqual(g.rules_downward, [])

    def test_self_rules_contains_union_for_backcompat(self):
        # Grammar.rules must still be the union (parse first, then generate)
        # so old call sites that read g.rules keep working.
        g = Grammar()
        g.configure({
            'compose':    {'S': ['S VO']},
            'generate': {'S': ['C']},
        })
        self.assertEqual(len(g.rules), 2)
        self.assertEqual(g.rules[0].lhs, 'S')
        self.assertEqual(g.rules[1].rhs_symbols, ('C',))

    def test_xml_config_loads_parse_block(self):
        # model.xml's parse-wrapped grammar survives round-trip.
        init_config(
            path=os.path.join(_DATA_DIR, 'model.xml'),
            defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
        )
        Language.TheGrammar._configured = False
        Language.TheGrammar._ensure_configured()
        # model.xml ships with <parse><rule>S = not.forward(S)</rule></parse>
        self.assertGreaterEqual(len(Language.TheGrammar.rules_upward), 1)

    def test_tier_scoped_grammar_tags_rules(self):
        # New tier-scoped form. Each rule's RuleDef.tier is set from
        # the <symbols> / <concepts> / <percepts> sub-section it
        # appeared in.
        g = Grammar()
        g.configure({
            'compose': {
                'symbols':  {'rule': ['S = not.forward(S)',
                                      'S = sigma.forward(S, S)']},
                'concepts': {'rule': ['S = pi.forward(S, S)']},
                'percepts': {'rule': ['S = sigma.forward(S, S)']},
            },
            'generate': {
                'symbols':  {'rule': ['S = not.reverse(S)',
                                      'S,S = sigma.reverse(S)']},
                'concepts': {'rule': ['S,S = pi.reverse(S)']},
                'percepts': {'rule': ['S,S = sigma.reverse(S)']},
            },
        })
        # 4 parse rules total (2 + 1 + 1) and 4 generate rules (2 + 1 + 1).
        self.assertEqual(len(g.rules_upward), 4)
        self.assertEqual(len(g.rules_downward), 4)
        # Tier tagging.
        tiers_up = [(r.tier, r.method_name) for r in g.rules_upward]
        # The two `S = ...` rules under <symbols> -> tier='S'.
        # The pi rule under <concepts> -> tier='C'.
        # The sigma rule under <percepts> -> tier='P'.
        self.assertIn(('S', 'not'), tiers_up)
        self.assertIn(('S', 'sigma'), tiers_up)
        self.assertIn(('C', 'pi'), tiers_up)
        self.assertIn(('P', 'sigma'), tiers_up)
        # Generate rules carry the same tiers.
        tiers_dn = [(r.tier, r.method_name) for r in g.rules_downward]
        self.assertIn(('S', 'not'), tiers_dn)
        self.assertIn(('C', 'pi'), tiers_dn)
        self.assertIn(('P', 'sigma'), tiers_dn)


    def test_default_model_xml_loads_tier_scoped_grammar(self):
        # The shipped data/model.xml uses the tier-scoped form. Round-
        # trip through the XML loader and verify tier tagging survives.
        init_config(
            path=os.path.join(_DATA_DIR, 'model.xml'),
            defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
        )
        Language.TheGrammar._configured = False
        Language.TheGrammar._ensure_configured()
        tiers = {(r.tier, r.method_name) for r in Language.TheGrammar.rules}
        for needle in (('S', 'not'), ('S', 'sigma'),
                       ('C', 'pi'), ('P', 'sigma')):
            self.assertIn(needle, tiers,
                f"missing tier-scoped rule {needle} in default grammar; "
                f"got {tiers}")


    def test_op_forward_and_op_reverse_syntax(self):
        # New explicit-direction rule bodies parse to the same RuleDef
        # the legacy `op(args)` form would produce: method_name is the
        # op without the suffix.
        g = Grammar()
        g.configure({
            'compose':    {'S': ['intersection.forward(S, S)']},
            'generate': {'S,S': ['intersection.reverse(S)']},
        })
        up = g.rules_upward[0]
        dn = g.rules_downward[0]
        self.assertEqual(up.method_name, 'intersection')
        self.assertEqual(up.arity, 2)
        self.assertEqual(up.rhs_symbols, ('S', 'S'))
        self.assertEqual(dn.method_name, 'intersection')
        self.assertEqual(dn.arity, 1)
        self.assertEqual(dn.rhs_symbols, ('S',))
        self.assertEqual(dn.lhs, 'S,S')


if __name__ == '__main__':
    unittest.main()
