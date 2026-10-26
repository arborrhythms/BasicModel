"""Preserved evidence contracts from the retired 61 frame-kernel tests."""
from dataclasses import replace

import pytest
import torch

from Layers import LTM0
from test_cs_symbol_table import _cs
from test_thought_model_fixture import model_for


def _world():
    cs = _cs()
    refs = tuple(('sym', cs.synthesize_higher_order([('sym', cs.new_concept())]))
                 for _ in range(4))
    for ref in refs:
        cs._csw_concept_row(0, ref[1])
    store = LTM0(8, capacity=32)
    model = model_for(cs, store)
    return model, model.grammatical_thoughts, store, refs


@pytest.mark.parametrize('hops', [1, 2, 3])
def test_direct_and_depth_one_two_chains_keep_native_premises(hops):
    model, registry, store, refs = _world()
    for a, b in zip(refs[:hops], refs[1:hops + 1]):
        model.conceptualSpace.add_whole(a[1], b)
    query = registry.form('isPart', refs[0], refs[hops])
    from reasoning import QuerySpec, TruthGroundedReasoner
    proof = TruthGroundedReasoner(model).evaluate(
        QuerySpec.from_surface('isPart', refs[0], refs[hops]))
    assert len(proof['path']) == hops
    result = model.reason_about(query)
    assert result.posture == 'TRUE' and result.support_true == 1
    # Direct membership can already be the retained conclusion. A proved
    # chain adds that same native Part edge, not a second LTM1 relation.
    edge = store.parts.find(refs[0][1], refs[hops][1])
    assert edge is not None and store.parts.truth_plus[edge] == 1
    assert len(store) == 0
    if hops > 1:
        model.conceptualSpace.retire_concept(edge)
    model.conceptualSpace.retire_concept(refs[1][1])
    if hops > 1:
        assert model.reason_about(query).posture == 'UNKNOWN'


@pytest.mark.parametrize('cycle', [False, True])
def test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm(cycle):
    model, registry, store, (a, b, c, _d) = _world()
    model.conceptualSpace.add_whole(a[1], b)
    if cycle:
        model.conceptualSpace.add_whole(b[1], a)
    result = model.reason_about(registry.form('isPart', a, c))
    assert result.posture == 'UNKNOWN' and result.support_true == 0
    assert result.work.spent <= model.attention_budget
    assert result.records[-1].kind == 'finish' and len(store) == 0
    edge = store.parts.find(a[1], c[1])
    assert store.row(('sym', edge))['kind'] == 'unverified'


def test_budget_exhaustion_is_bounded_unknown_with_diagnostic():
    model, registry, store, (a, b, _c, _d) = _world()
    model.conceptualSpace.add_whole(a[1], b)
    model.attention_budget = 1
    result = model.reason_about(registry.form('isPart', a, b))
    assert result.posture == 'UNKNOWN'
    assert 'work_budget' in result.evidence['incomplete']
    assert result.work.spent == 1 and len(store) == 0
    assert store.row(('sym', store.parts.find(a[1], b[1])))['evidence'] == (1., 0.)


def test_concluding_without_evidence_is_illegal_while_work_remains(monkeypatch):
    model,registry,store,(a,b,_,_)=_world()
    monkeypatch.setattr(model,'_choose_selected_thought_action',lambda *_a,**_k:None)
    with pytest.raises(ValueError,match='cannot conclude'):
        model.reason_about(registry.form('isPart', a, open_roles=('argin2',)))
    assert len(store)==0


@pytest.mark.parametrize('trusts', [(-.8,), (.8, -.7)])
def test_world_refutation_or_conflict_never_closes_taxonomy_false(trusts):
    model, registry, store, (a, b, _c, _d) = _world()
    for trust in trusts:
        from Meaning import ConceptualMeaning
        store.append_meaning(ConceptualMeaning.from_description(
            (registry._payload(a) + registry._payload(b)) / 2), kind='fact', trust=trust)
    result = model.reason_about(registry.form('isPart', a, b))
    assert result.posture == 'UNKNOWN'
    assert result.support_true == result.support_false == 0


@pytest.mark.parametrize('positive,negative,posture', [(.8, .7, 'BOTH'), (.8, .1, 'TRUE'), (0, 0, 'UNKNOWN')])
def test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees(positive,negative,posture):
    from index_fixtures import terminal_model_index
    model,registry,store,(a,b,_,_)=_world()
    from Meaning import ConceptualMeaning
    meaning=ConceptualMeaning.from_description(registry._payload(a))
    order=terminal_model_index(model,meaning.role_refs)
    index=store.append_meaning(meaning,kind='fact',order=order,evidence=(positive,negative))
    from QueryWork import QueryWorkBudget
    with model._query_boundary_scope((0,)):
        context=model._thought_grammar_context(meaning,row=0,work=QueryWorkBudget(32),continuation=None)
        query=registry.form('ask',store.occurrence_of(index),context=context)
    result=model.reason_about(query)
    assert result.posture==posture
    assert result.support_true==pytest.approx(positive) and result.support_false==pytest.approx(negative)
    assert store.row(len(store)-1)['kind']==('unverified' if posture=='UNKNOWN' else 'inference')


def test_supplied_mechanism_walk_uses_one_controller_at_every_level(monkeypatch):
    model,registry,store,(a,b,_,_)=_world()
    model.conceptualSpace.add_whole(a[1],b)
    from test_thought_answer_adapters import _open_goal
    from meaning_fixtures import replay_thought_plan
    query=_open_goal(model, registry, a, b)
    replay_thought_plan(model, monkeypatch, ['ask', 'query', None, None])
    result=model.reason_about(registry.form('ask',query))
    assert [(r.level,r.operation) for r in result.records if r.kind=='thought']==[
        (0,'ask'),(1,'query'),(1,'conclude'),(0,'conclude')]
    assert not hasattr(model,'selected_thought_choosers')
    assert model._selected_thought_chooser(query) is model.shared_grammar


def test_public_disabled_budget_never_opens_an_episode():
    model, registry, store, (a, b, _c, _d) = _world()
    model.attention_budget = 0
    query = registry.form('isPart', a, b)
    assert model.reason_about(query) is model.think_about(query) is None
    assert not model._what_memory().thought_history()
    assert len(store) == 0
