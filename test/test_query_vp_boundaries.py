"""Structural thought grounding, mode, live payload and native identity."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Language import Grammar
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Queries import (
    GrammaticalThoughtRegistry, THOUGHT_EXECUTORS, ThoughtConceptualCapability,
    ThoughtGrammarContext, ThoughtLTMCapability, ThoughtSignature,
    ThoughtTaxonomyCapability,
)
from QueryWork import QueryWorkBudget
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


def _context(cs, *, store=None, model=None, continuation=None, row=0,
             max_nodes=256, max_records=1024, max_steps=8,
             max_expansions=1024, work=None, memory=None, discourse=None,
             boundary=None):
    """Build the public thought context without passing a reasoner through it."""
    if model is None:
        model = SimpleNamespace(conceptualSpace=cs)
    symbol_space = getattr(model, 'symbolSpace', None)
    if memory is None:
        memory = getattr(symbol_space, 'what_memory', None)
    if discourse is None:
        discourse = getattr(symbol_space, 'discourse', None)
    reasoner = TruthGroundedReasoner(model=model, store=store)
    return ThoughtGrammarContext(
        word_stream=(),
        conceptual_space=ThoughtConceptualCapability(
            cs, TruthGroundedReasoner.equal),
        primed_symbols=(),
        ltm=ThoughtLTMCapability(
            existence_evidence=reasoner.existence_evidence,
            store=reasoner.reasoning_store,
            equal=TruthGroundedReasoner.equal,
            tau_id=reasoner.tau_id,
            memory=memory,
            discourse=discourse,
        ),
        taxonomy=ThoughtTaxonomyCapability(cs),
        work=work or QueryWorkBudget(4096),
        continuation=continuation,
        boundary=boundary or (lambda _row: None),
        row=row,
        max_nodes=max_nodes,
        max_records=max_records,
        max_steps=max_steps,
        max_expansions=max_expansions,
    )


def _signature(semantic_id, *roles):
    """Exercise an executor's public thought signature without an alias."""
    return ThoughtSignature(
        SimpleNamespace(semantic_id=semantic_id, operand_roles=tuple(roles)),
        THOUGHT_EXECUTORS[semantic_id], tuple(roles))


def _world():
    cs = _cs()
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    a, b = (('sym', cs.new_concept()) for _ in range(2))
    for ref in (a, b):
        cs._csw_concept_row(0, ref[1])
    cs.add_whole(a[1], b)
    context = _context(cs)
    return cs, grammar, registry, a, b, context


def test_foreign_width_query_fails_before_the_taxonomy_executor(monkeypatch):
    _, _, registry, a, b, context = _world()
    question = registry.form('part', a, b)
    monkeypatch.setattr(
        ThoughtTaxonomyCapability, 'evidence',
        lambda *a, **k: pytest.fail('executed'))
    with pytest.raises(ValueError, match='width'):
        registry.execute(replace(question, roles=question.roles[:, :4]), context)


def test_what_cannot_turn_assertive_content_into_an_executed_subgoal():
    calls = []
    content = ConceptualMeaning.from_description(torch.ones(8))
    cs = _cs()
    context = _context(cs, continuation=lambda value: calls.append(value) or value)
    with pytest.raises(ValueError, match='interrogative|question'):
        _signature('what', 'I1').invoke(context, content)
    question = replace(content, mode='interrogative')
    result = _signature('what', 'I1').invoke(context, question)
    assert result['value'].mode == 'interrogative'
    assert len(calls) == 1
    torch.testing.assert_close(calls[0].roles, question.roles)


def test_registry_installation_reuses_native_bindings_and_checkpoint(tmp_path):
    from Models import BaseModel
    cs, grammar, registry, a, b, _ = _world()
    question = registry.form('part', a, b)
    before = dict(cs._concept_allocator.placement)
    GrammaticalThoughtRegistry.install(cs, grammar)
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
    loaded = GrammaticalThoughtRegistry.install(restored, grammar)
    restored_question = loaded.form('whole', b, a)
    assert restored._frozen_named == bindings
    assert restored_question.role_refs == question.role_refs
    torch.testing.assert_close(restored_question.roles, question.roles)
    context = _context(restored, model=target)
    assert loaded.execute(restored_question, context).support_true == 1


def test_occurrence_namespace_and_bound_are_never_rebound_to_matching_row():
    cs, _, registry, _, _, _ = _world()
    store = TernaryTruthStore(8)
    description = ConceptualMeaning.from_description(torch.eye(8)[:3])
    first = store.append_meaning(description)
    second = store.append_meaning(description)
    context = _context(cs, store=store, max_records=1)
    with pytest.raises(ValueError, match='limit|unavailable'):
        registry.form('exist', store.occurrence_of(second), context=context)
    foreign = ('ltm', 'foreign-namespace', store.occurrence_of(first)[2])
    with pytest.raises(ValueError, match='namespace'):
        registry.form('exist', foreign, context=context)
    assert len(store) == 2


def test_structure_only_compose_face_cannot_execute():
    cs, _, _, a, b, context = _world()
    grammar = Grammar()
    grammar.configure({'Symbolic': {'compose': {
        'rule': ['union_O1 = union.forward(union_I1, union_I2)']}}})
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    with pytest.raises(ValueError, match='no executor|structural'):
        registry.form('union', a, b)


def test_converse_only_interface_still_dispatches_canonical_roles():
    cs, _, _, a, b, context = _world()
    grammar = Grammar()
    # A structural spelling supplies the grammar-owned converse metadata;
    # there is no separately declared query interface.
    grammar.configure({'Symbolic': {'compose': {'rule': [
        {'_': 'whole_O1 = whole.forward(whole_I1, whole_I2)',
         'family': 'part', 'permutation': 'I2,I1'}]}}})
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    question = registry.form('whole', b, a)
    assert question.role_refs[0] == a and question.role_refs[2] == b
    assert registry.execute(question, context).support_true == 1


def test_query_proposition_preserves_live_operand_gradients_without_new_parameters():
    cs, _, registry, _, _, _ = _world()
    before = tuple(id(p) for p in cs.parameters())
    a, b = (torch.randn(8, requires_grad=True) for _ in range(2))
    question = registry.form('equal', a, b)
    loss = question.roles[0].square().sum() + 2 * question.roles[2].square().sum()
    loss.backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert tuple(id(p) for p in cs.parameters()) == before


def test_unknown_negated_query_remains_unknown():
    _, _, registry, a, b, context = _world()
    result = registry.execute(registry.form('part', b, a, polarity=False), context)
    assert result.support_true == result.support_false == 0
