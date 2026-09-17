"""Linguistic operator faces and internal queries need the same middle VP."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Language import Grammar
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Queries import QueryContext
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


def _world():
    from Queries import GrammaticalQueryRegistry
    cs = _cs()
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    registry = GrammaticalQueryRegistry.install(cs, grammar)
    a = ('sym', cs.new_concept())
    b = ('sym', cs.new_concept())
    cs._csw_concept_row(0, a[1])
    cs._csw_concept_row(0, b[1])
    cs.add_whole(a[1], b)
    model = SimpleNamespace(conceptualSpace=cs)
    return cs, registry, a, b, QueryContext(TruthGroundedReasoner(model=model))


def test_compose_query_and_converse_faces_share_the_same_canonical_vp():
    cs, registry, a, b, context = _world()
    assertion = registry.form('part', a, b, mode='assertive')
    question = registry.form('isPart', a, b)
    converse = registry.form('isWhole', b, a)
    assert isinstance(question, ConceptualMeaning)
    assert question.mode == 'interrogative' and assertion.mode == 'assertive'
    assert question.role_mask.tolist() == [True, True, True]
    assert question.role_refs[0] == a and question.role_refs[2] == b
    assert question.role_refs == assertion.role_refs == converse.role_refs
    torch.testing.assert_close(question.roles, assertion.roles)
    torch.testing.assert_close(question.roles, converse.roles)
    vp = question.role_refs[1]
    assert vp[0] == 'sym'
    torch.testing.assert_close(question.roles[1], cs.similarity_codebook.getW()[cs._csw_row_of(vp[1])])


def test_execution_is_derived_from_middle_vp_and_retains_the_proposition():
    cs, registry, a, b, context = _world()
    question = registry.form('isWhole', b, a, scope={'place': 'workshop'})
    result = registry.execute(question, context)
    assert result['support_true'] == 1
    assert result['meaning'] is question
    assert result['meaning'].scope == (('place', 'workshop'),)
    assert registry.execute(registry.form('isPart', b, a), context)['support_true'] == 0
    negated = registry.execute(replace(question, polarity=False), context)
    assert negated['support_false'] == 1 and negated['support_true'] == 0


def test_assertive_or_unrecognized_vp_does_not_execute_a_query(monkeypatch):
    cs, registry, a, b, context = _world()
    monkeypatch.setattr(context.reasoner, 'taxonomy_evidence', lambda *a, **k: pytest.fail('executed'))
    with pytest.raises(ValueError, match='interrogative|question'):
        registry.execute(registry.form('part', a, b, mode='assertive'), context)
    question = registry.form('isPart', a, b)
    with pytest.raises(ValueError, match='VP|relation|registered'):
        registry.execute(replace(question, role_refs=(a, a, b)), context)


def test_open_taxonomy_roles_use_the_same_vp_without_fabricating_an_operand():
    cs, registry, a, b, context = _world()
    bound = registry.form('isPart', a, b)
    parts = registry.form('parts', b)
    wholes = registry.form('wholes', a)
    assert parts.role_mask.tolist() == [False, True, True]
    assert wholes.role_mask.tolist() == [True, True, False]
    assert parts.role_refs[1] == wholes.role_refs[1] == bound.role_refs[1]
    assert parts.role_refs[0] is None and wholes.role_refs[2] is None
    assert registry.execute(parts, context)['value'][0]['reference'] == a
    assert registry.execute(wholes, context)['value'][0]['reference'] == b


def test_forming_candidates_does_not_mint_or_write_memory_after_setup():
    cs, registry, a, b, context = _world()
    placement = dict(cs._concept_allocator.placement)
    named = dict(cs._frozen_named)
    records = tuple(cs._concept_allocator._layers[0].constituents(a[1]))
    for name, args in [('part', (a, b)), ('isWhole', (b, a)), ('parts', (b,))]:
        registry.form(name, *args)
    assert cs._concept_allocator.placement == placement
    assert cs._frozen_named == named
    assert tuple(cs._concept_allocator._layers[0].constituents(a[1])) == records


def test_unary_existence_keeps_full_description_through_an_existing_occurrence():
    cs, registry, a, b, _ = _world()
    store = TernaryTruthStore(8)
    description = ConceptualMeaning(torch.eye(8)[:3], torch.ones(3, dtype=torch.bool),
                                   scope={'place': 'workshop'})
    index = store.append_meaning(description, trust=0.75)
    context = QueryContext(TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs), store=store))
    reference = store.occurrence_of(index)
    question = registry.form('isTrue', reference, context=context)
    assert question.role_refs[0] == reference
    assert question.role_mask.tolist() == [True, True, False]
    result = registry.execute(question, context)
    assert result['support_true'] == pytest.approx(0.75)
    torch.testing.assert_close(result['evidence']['meaning'].roles, description.roles)
    assert result['evidence']['meaning'].scope == description.scope
    assert len(store) == 1
    # A descriptor without an owned occurrence may not be silently flattened
    # into NP1 or committed by candidate construction.
    with pytest.raises((TypeError, ValueError), match='occurrence|reference'):
        registry.form('isTrue', description, context=context)


def test_missing_or_retired_vp_binding_fails_without_lazy_reinstallation():
    cs, registry, a, b, context = _world()
    question = registry.form('isPart', a, b)
    # Named VPs do not retire through the normal frozen-concept lifecycle.
    # Simulate a missing native record in an incomplete structural restore.
    cs._concept_allocator.drop(question.role_refs[1][1])
    placement = dict(cs._concept_allocator.placement)
    with pytest.raises(ValueError, match='unavailable|retired|VP'):
        registry.form('isPart', a, b)
    assert cs._concept_allocator.placement == placement
