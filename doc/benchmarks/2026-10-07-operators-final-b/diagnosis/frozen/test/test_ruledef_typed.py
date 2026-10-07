"""RuleDef carries lhs/rhs_symbols; Grammar.configure accepts multi-LHS."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import Language
from Language import Grammar


class TestRuleDefTyped(unittest.TestCase):
    def test_ruledef_has_lhs_and_rhs_symbols(self):
        g = Grammar()
        g.configure({'S': ['not(S)']})
        rule = g.rules[0]
        self.assertEqual(rule.lhs, 'S')
        self.assertEqual(rule.rhs_symbols, ('S',))
        self.assertEqual(rule.method_name, 'not')
        self.assertEqual(rule.arity, 1)

    def test_bare_symbol_form_parses_to_merge(self):
        g = Grammar()
        g.configure({'S': ['S VO'], 'VO': ['V O']})
        self.assertEqual(len(g.rules), 2)
        r0, r1 = g.rules[0], g.rules[1]
        self.assertEqual(r0.lhs, 'S')
        self.assertEqual(r0.rhs_symbols, ('S', 'VO'))
        self.assertEqual(r0.method_name, 'merge')
        self.assertEqual(r0.arity, 2)
        self.assertEqual(r1.lhs, 'VO')
        self.assertEqual(r1.rhs_symbols, ('V', 'O'))

    def test_s_first_ordering_is_stable(self):
        g = Grammar()
        g.configure({'VO': ['V O'], 'S': ['S VO']})
        self.assertEqual(g.rules[0].lhs, 'S')
        self.assertEqual(g.rules[1].lhs, 'VO')

    def test_function_call_rhs_populates_rhs_symbols(self):
        g = Grammar()
        g.configure({'S': ['lift(S, S)']})
        rule = g.rules[0]
        self.assertEqual(rule.rhs_symbols, ('S', 'S'))
        self.assertEqual(rule.method_name, 'lift')
        self.assertEqual(rule.arity, 2)

    def test_epsilon_rule_has_empty_rhs(self):
        g = Grammar()
        g.configure({'S': ['epsilon']})
        rule = g.rules[0]
        self.assertEqual(rule.rhs_symbols, ())
        self.assertEqual(rule.arity, 0)
        self.assertIsNone(rule.method_name)


if __name__ == '__main__':
    unittest.main()
