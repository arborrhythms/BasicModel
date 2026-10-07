"""Thought-capable structural faces compose without geometrically answering."""
from types import SimpleNamespace

import pytest
import torch

import Language
from test_basicmodel import _populate_test_config
from test_ispart_query_dispatch import _part_grammar, _load


@pytest.mark.parametrize("name", ["part", "equal"])
def test_thought_capable_rule_dispatches_the_same_pure_compose_face(
        name, monkeypatch, tmp_path):
    grammar = _load(
        _part_grammar().replace("part", name), monkeypatch, tmp_path)
    rule = next(rule for rule in grammar.rules_upward if rule.method_name == name)
    assert Language._dispatch_method_name_for_rule(rule) == name
    monkeypatch.setattr(
        Language, "_parthood_geometric",
        lambda *args: pytest.fail("answered during composition"))
    left, right = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    selected = Language.GRAMMAR_LAYER_CLASSES[
        Language._dispatch_method_name_for_rule(rule)]()
    pure = Language.GRAMMAR_LAYER_CLASSES[name]()
    torch.testing.assert_close(selected.compose(left, right), pure.compose(left, right))


def test_language_owns_rule_meaning_for_each_local_compose_action(monkeypatch):
    _populate_test_config(
        inputDim=8, perceptDim=8, conceptDim=8, symbolDim=8, wordDim=8,
        outputDim=8, nWhere=0, nWhen=0)
    grammar = Language.Grammar()
    grammar.load_from_grammar_file("complete.grammar")
    binary = tuple(
        index for index, rule in enumerate(grammar.rules_upward)
        if rule.space_role == "CS" and rule.arity == 2)
    unary = tuple(
        index for index, rule in enumerate(grammar.rules_upward)
        if rule.space_role == "CS" and rule.arity == 1)
    assert binary and unary
    monkeypatch.setattr(Language, "TheGrammar", grammar)
    layer = SimpleNamespace(
        _binary_rule_ids={"CS": binary}, _unary_rule_ids={"CS": unary})
    owner = Language.LanguageSpace(SimpleNamespace(
        subspace=SimpleNamespace(languageLayer=layer, muxedSize=0)))
    captured = owner._compose_binary_rules
    assert isinstance(captured, tuple)
    assert captured == tuple(grammar.rules_upward[index] for index in binary)
    assert owner._compose_unary_rules == tuple(
        grammar.rules_upward[index] for index in unary)
    grammar.configure({"compose": {"S": ["sum(S,S)"]}})
    assert owner._compose_binary_rules is captured
    assert captured[0].method_name == "part"
