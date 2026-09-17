"""Structured query grounding, mode, live payload and durable native identity."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Language import Grammar
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Queries import BUILTIN_QUERIES, GrammaticalQueryRegistry, QueryContext
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


def _world():
    cs = _cs()
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    registry = GrammaticalQueryRegistry.install(cs, grammar)
    a, b = (('sym', cs.new_concept()) for _ in range(2))
    for ref in (a, b):
        cs._csw_concept_row(0, ref[1])
    cs.add_whole(a[1], b)
    context = QueryContext(TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs)))
    return cs, grammar, registry, a, b, context


def test_foreign_width_query_fails_before_the_taxonomy_executor(monkeypatch):
    _, _, registry, a, b, context = _world()
    question = registry.form('isPart', a, b)
    monkeypatch.setattr(context.reasoner, 'taxonomy_evidence', lambda *a, **k: pytest.fail('executed'))
    with pytest.raises(ValueError, match='width'):
        registry.execute(replace(question, roles=question.roles[:, :4]), context)


def test_what_cannot_turn_assertive_content_into_an_executed_subgoal():
    calls = []
    content = ConceptualMeaning.from_description(torch.ones(8))
    context = QueryContext(TruthGroundedReasoner(), schedule_subgoal=lambda value: calls.append(value) or value)
    with pytest.raises(ValueError, match='interrogative|question'):
        BUILTIN_QUERIES['what'].invoke(context, content)
    question = replace(content, mode='interrogative')
    assert BUILTIN_QUERIES['what'].invoke(context, question)['value'] is question
    assert calls == [question]


def test_registry_installation_reuses_native_bindings_and_checkpoint(tmp_path):
    from Models import BaseModel
    cs, grammar, registry, a, b, _ = _world()
    question = registry.form('isPart', a, b)
    before = dict(cs._concept_allocator.placement)
    GrammaticalQueryRegistry.install(cs, grammar)
    assert cs._concept_allocator.placement == before
    def model(space):
        value = BaseModel()
        value.name = 'GrammaticalVPCheckpoint'
        value.spaces = [space]
        value.conceptualSpaces = torch.nn.ModuleList([space])
        object.__setattr__(value, 'conceptualSpace', space)
        value.wholeSpaces = []
        return value
    source = model(cs)
    path = tmp_path / 'query-vps.ckpt'
    source.save_weights(path)
    restored = _cs()
    target = model(restored)
    assert target.load_weights(path)
    bindings = dict(restored._frozen_named)
    loaded = GrammaticalQueryRegistry.install(restored, grammar)
    restored_question = loaded.form('isWhole', b, a)
    assert restored._frozen_named == bindings
    assert restored_question.role_refs == question.role_refs
    torch.testing.assert_close(restored_question.roles, question.roles)
    context = QueryContext(TruthGroundedReasoner(model=target))
    assert loaded.execute(restored_question, context)['support_true'] == 1


def test_occurrence_namespace_and_bound_are_never_rebound_to_matching_row():
    cs, _, registry, _, _, _ = _world()
    store = TernaryTruthStore(8)
    description = ConceptualMeaning.from_description(torch.eye(8)[:3])
    first = store.append_meaning(description)
    second = store.append_meaning(description)
    context = QueryContext(TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs), store=store), max_records=1)
    with pytest.raises(ValueError, match='limit|unavailable'):
        registry.form('isTrue', store.occurrence_of(second), context=context)
    foreign = ('ltm', 'foreign-namespace', store.occurrence_of(first)[2])
    with pytest.raises(ValueError, match='namespace'):
        registry.form('isTrue', foreign, context=context)
    assert len(store) == 2


def test_compose_without_a_query_declaration_cannot_execute():
    cs, _, _, a, b, context = _world()
    grammar = Grammar()
    grammar.configure({'Symbolic': {'compose': {'rule': ['S = part.forward(S,S)']}}})
    registry = GrammaticalQueryRegistry.install(cs, grammar)
    question = registry.form('part', a, b)
    with pytest.raises(ValueError, match='interface'):
        registry.execute(question, context)


def test_converse_only_interface_still_dispatches_canonical_roles():
    cs, _, _, a, b, context = _world()
    grammar = Grammar()
    grammar.configure({'Queries': {'query': 'isWhole(X,Y)'}})
    registry = GrammaticalQueryRegistry.install(cs, grammar)
    question = registry.form('isWhole', b, a)
    assert question.role_refs[0] == a and question.role_refs[2] == b
    assert registry.execute(question, context)['support_true'] == 1


def test_query_proposition_preserves_live_operand_gradients_without_new_parameters():
    cs, _, registry, _, _, _ = _world()
    before = tuple(id(p) for p in cs.parameters())
    a, b = (torch.randn(8, requires_grad=True) for _ in range(2))
    question = registry.form('isEqual', a, b)
    loss = question.roles[0].square().sum() + 2 * question.roles[2].square().sum()
    loss.backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert tuple(id(p) for p in cs.parameters()) == before


def test_unknown_negated_query_remains_unknown():
    _, _, registry, a, b, context = _world()
    result = registry.execute(registry.form('isPart', b, a, polarity=False), context)
    assert result['support_true'] == result['support_false'] == 0
