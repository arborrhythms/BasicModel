"""Clause structure belongs to grammar roles, never to English token tests."""
import torch

from Language import Grammar


def test_complete_grammar_declares_sentence_predicate_generic_and_implication():
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    rules = {rule.method_name: rule for rule in grammar.rules_upward}
    assert rules['lift'].clause_form == 'S'
    assert rules['verb'].clause_form == 'VP'
    assert 'exist' not in rules
    assert grammar.ws_absolute_starts == frozenset({'S'})
    assert rules['generic'].reference_kinds == (('I1', 'generic'),)
    assert rules['lower'].head_role == 2
    assert rules['implies'].clause_form == 'implies'
    assert 'operator_O1' in grammar.ws_relative_starts
    assert 'implies_O1' in grammar.ws_relative_starts


def test_clause_facets_preserve_operator_input_roles():
    grammar = Grammar()
    grammar.configure({'compose': {'rule': [
        {'_': 'lift_O1 = lift.forward(lift_I1, lift_I2)', 'clause': 'S'},
        {'_': 'verb_O1 = verb.forward(verb_I1, verb_I2)', 'clause': 'VP'},
        {'_': 'lower_O1 = lower.forward(lower_I1, lower_I2)', 'head': 'I2',
         'reference': 'I2:particular'}]}})
    rules = {rule.method_name: rule for rule in grammar.rules_upward}
    assert rules['lift'].arity == rules['verb'].arity == 2
    assert rules['lift'].clause_form == 'S' and rules['verb'].clause_form == 'VP'
    assert rules['lower'].head_role == 2
