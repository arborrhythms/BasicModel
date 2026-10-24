"""Clause structure belongs to grammar roles, never to lexical token tests."""
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
    assert rules['generic'].reference_modes == (('argin1', 'generic'),)
    assert rules['lower'].head_role == 2
    assert rules['implies'].clause_form == 'implies'
    assert 'operator_argout1' in grammar.ws_relative_starts
    assert 'implies_argout1' in grammar.ws_relative_starts


def test_clause_facets_preserve_operator_input_roles():
    grammar = Grammar()
    grammar.configure({'compose': {'rule': [
        {'_': 'lift_argout1 = lift.forward(lift_argin1, lift_argin2)', 'clause': 'S'},
        {'_': 'verb_argout1 = verb.forward(verb_argin1, verb_argin2)', 'clause': 'VP'},
        {'_': 'lower_argout1 = lower.forward(lower_argin1, lower_argin2)', 'head': 'argin2',
         'reference': 'argin2'}]}})
    rules = {rule.method_name: rule for rule in grammar.rules_upward}
    assert rules['lift'].arity == rules['verb'].arity == 2
    assert rules['lift'].clause_form == 'S' and rules['verb'].clause_form == 'VP'
    assert rules['lower'].head_role == 2
