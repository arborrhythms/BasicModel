"""Grammar accepts <upward> and <downward> sub-blocks; default = upward."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import Language
from Language import Grammar
from util import init_config


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


class TestGrammarSplit(unittest.TestCase):
    def test_configure_accepts_upward_block(self):
        g = Grammar()
        g.configure({'upward': {'S': ['not(S)']}})
        self.assertEqual(len(g.rules_upward), 1)
        self.assertEqual(g.rules_upward[0].lhs, 'S')
        self.assertEqual(g.rules_downward, [])

    def test_configure_accepts_both_blocks(self):
        g = Grammar()
        g.configure({
            'upward': {'S': ['S VO'], 'VO': ['V O']},
            'downward': {'S': ['C']},
        })
        self.assertEqual(len(g.rules_upward), 2)
        self.assertEqual(len(g.rules_downward), 1)
        down = g.rules_downward[0]
        self.assertEqual(down.lhs, 'S')
        self.assertEqual(down.rhs_symbols, ('C',))
        self.assertEqual(down.method_name, 'emit_head')

    def test_legacy_flat_grammar_still_loads(self):
        # Old-shape {'S': ['not(S)']} must still work as upward-only.
        g = Grammar()
        g.configure({'S': ['not(S)']})
        self.assertEqual(len(g.rules_upward), 1)
        self.assertEqual(g.rules_downward, [])

    def test_self_rules_contains_union_for_backcompat(self):
        # Grammar.rules must still be the union (upward first, then downward)
        # so old call sites that read g.rules keep working.
        g = Grammar()
        g.configure({
            'upward': {'S': ['S VO']},
            'downward': {'S': ['C']},
        })
        self.assertEqual(len(g.rules), 2)
        self.assertEqual(g.rules[0].lhs, 'S')
        self.assertEqual(g.rules[1].rhs_symbols, ('C',))

    def test_xml_config_loads_upward_block(self):
        # model.xml's upward-wrapped grammar survives round-trip.
        init_config(
            path=os.path.join(_DATA_DIR, 'model.xml'),
            defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
        )
        Language.TheGrammar._configured = False
        Language.TheGrammar._ensure_configured()
        # model.xml ships with <upward><S>not(S)</S></upward>
        self.assertGreaterEqual(len(Language.TheGrammar.rules_upward), 1)


if __name__ == '__main__':
    unittest.main()
