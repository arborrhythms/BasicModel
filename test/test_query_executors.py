"""A selected checked thought preserves its declared object and effects."""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from Meaning import ConceptualMeaning
from Layers import LTM0
from test_cs_symbol_table import _cs
from order_fixtures import force_order
from Language import Grammar
from Queries import GrammaticalThoughtRegistry, THOUGHT_EXECUTORS, ThoughtSignature, _existing_row
from test_query_vp_boundaries import _context, _signature


def _meaning():
    return replace(ConceptualMeaning.from_description(torch.eye(8)[0]),
                   scope={"place": "workshop"})


def test_content_query_keeps_all_roles_and_conflicting_fact_sources():
    cs = _cs()
    refs = tuple(('sym', cs.new_concept()) for _ in range(3))
    for ref in refs:
        force_order(cs, ref[1], 0)
        cs._csw_concept_row(0, ref[1])
    store = LTM0(8, percept_width=0)
    from index_fixtures import one_hot_unfold
    idea = replace(_meaning(), role_refs=(refs[0], None, None))
    first = store.append_meaning(idea, kind="fact", trust=0.6, sentence_index=len(store))
    second = store.append_meaning(idea, kind="fact", trust=-0.4, sentence_index=len(store))
    store.set_origin(first, store.ORIGIN_PROVISIONED, text="teacher")
    store.set_origin(second, store.ORIGIN_USER, text="witness")
    result = _signature('query', 'argin1').invoke(
        _context(cs, store=store), replace(idea))
    assert sorted(item['trust'] for item in result['value']) == pytest.approx([.6])
    for item in result['value']:
        torch.testing.assert_close(item['meaning'].roles, idea.roles)
    assert {item["text"] for item in result["value"]} == {"teacher"}
    assert {item['occurrence'] for item in result['value']} == {
        store.occurrence_of(first)}
    assert len(store) == 2



def test_part_and_converse_executor_read_native_taxonomy_without_writes():
    cs = _cs()
    a, b = cs.new_concept(), cs.new_concept()
    for concept in (a, b):
        force_order(cs, concept, 0)
        cs._csw_concept_row(0, concept)
    cs.add_whole(a, ("sym", b))
    context = _context(cs)
    grammar = Grammar()
    grammar.configure({'compose': {'rule': [
        'isPart_argout1 = isPart.forward(isPart_argin1, isPart_argin2)',
        {'_': 'whole_argout1 = whole.forward(whole_argin1, whole_argin2)',
         'family': 'isPart', 'permutation': 'argin2,argin1'},
    ]}, 'thought': {'rule': [
        'isPart_argout1 = isPart.thought(isPart_argin1, isPart_argin2)',
        {'_': 'whole_argout1 = whole.thought(whole_argin1, whole_argin2)',
         'family': 'isPart', 'permutation': 'argin2,argin1'},
    ]}})
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    result = registry.execute(registry.form('isPart', ('sym', a), ('sym', b)), context)
    assert result.support_true == 1
    assert result.evidence["path"][0].owner == ("sym", cs._concept_allocator.parts.find(a, b))
    # ``whole(B, A)`` is a grammar permutation of canonical part(A, B).
    assert registry.execute(
        registry.form('whole', ('sym', b), ('sym', a)), context).support_true == 1
    parts = registry.execute(
        registry.form('isPart', ('sym', b), open_roles=('argin1',)), context)
    wholes = registry.execute(
        registry.form('isPart', ('sym', a), open_roles=('argin2',)), context)
    assert parts.value[0]["reference"] == ("sym", a)
    assert wholes.value[0]["reference"] == ("sym", b)



def test_removed_quantize_raises_without_symbol_snapping():
    cs=_cs()
    cs.new_concept()
    before=dict(cs._concept_allocator.placement)
    with pytest.raises(ValueError,match='retired'):
        Grammar().configure({'thought':{'rule':'quantize_argout1 = quantize.thought(quantize_argin1)'}})
    assert cs._concept_allocator.placement==before


def test_query_does_not_fill_an_unconstrained_name_from_a_held_row():
    from meaning_fixtures import unsupported as question
    cs = _cs()
    store = LTM0(8, percept_width=0)
    retained = ConceptualMeaning.from_description(torch.ones(8))
    row = store.append_meaning(retained, kind='observation', evidence=(1., 0.))
    memory = SimpleNamespace(retrieved_frames=lambda **kwargs: (store.row(row),))
    context = _context(cs, store=store, memory=memory)
    pending = question(retained, (('referent', 0),))
    result = _signature('query', 'argin1').invoke(context, pending)
    assert result['frames'] == () and result['witnesses'] == ()
    assert result['support_true'] == result['support_false'] == 0.
    assert len(store) == 1


def test_ask_preserves_the_complete_open_question_and_schedules_same_controller():
    from meaning_fixtures import unsupported as opened
    question=opened(_meaning(),(('referent',0),))
    calls=[]
    result=_signature('ask','argin1').invoke(_context(_cs(),continuation=lambda value:
        calls.append(value) or dict(value=value,meaning=value,support_true=0.,support_false=0.,result_kind='subgoal')),question)
    assert len(calls)==1
    torch.testing.assert_close(calls[0].roles,question.roles)
    torch.testing.assert_close(result['value'].roles,question.roles)
    assert result['result_kind']=='subgoal'
    empty=_signature('ask','argin1').invoke(_context(_cs()),question)
    assert empty['frames']==() and empty['support_true']==empty['support_false']==0.


def test_wrong_domain_and_types_fail_before_execution():
    from dataclasses import replace
    calls = []
    descriptor = replace(
        THOUGHT_EXECUTORS['part'], executor=lambda *args: calls.append(args))
    signature = ThoughtSignature(
        SimpleNamespace(semantic_id='part', operand_roles=('argin1', 'argin2')),
        descriptor, ('argin1', 'argin2'))
    # The structural operation fixes its descriptor domain; callers cannot
    # select another one. Invalid operands fail before the executor.
    with pytest.raises(TypeError, match="reference|argument"):
        signature.invoke(_context(_cs()), 1, 2)
    assert calls == []


def test_equality_executor_rejects_width_truncation_and_nonfinite_values():
    context = _context(_cs())
    result = _signature('equal', 'argin1', 'argin2').invoke(
        context, torch.ones(8), torch.ones(8))
    assert result["support_true"] == 1
    with pytest.raises(ValueError, match="width"):
        _signature('equal', 'argin1', 'argin2').invoke(
            context, torch.ones(8), torch.ones(4))
    with pytest.raises((ValueError, FloatingPointError), match="finite"):
        _signature('equal', 'argin1', 'argin2').invoke(
            context, torch.ones(8), torch.full((8,), float("nan")))


def test_global_expectation_retains_all_roles_without_a_thought_operation():
    from test_sentence_expectation import layer as make_layer
    layer=make_layer()
    roles=torch.arange(3*layer.concept_dim,dtype=torch.float32).reshape(3,-1)/100
    layer.predict_and_observe_stm_end_state([3],[roles],layout='infix')
    prediction=layer.expect_next_meaning(0)
    marker=layer._inter_last_meaning[0]
    value=layer.expect_next_meaning(0,record=False)
    assert value is prediction
    assert value.roles.shape==(3,layer.concept_dim) and value.presence_logits.shape==(3,)
    assert layer._inter_last_meaning[0] is marker
    assert 'arma' not in THOUGHT_EXECUTORS


@pytest.mark.parametrize('reverse', (False, True))
def test_query_returns_only_the_completed_direction_of_an_unresolved_equality(monkeypatch, reverse):
    from Queries import ThoughtLTMCapability
    from ThoughtReferences import bindings
    from meaning_fixtures import unsupported as question
    from test_thought_binding import equality
    name, unresolved = ('sym', 1), ('ltm', 'test', 2)
    meaning = equality(name, unresolved)
    meaning = replace(meaning, bindings=dict(bindings(meaning),
        _unresolved_operands=((2, unresolved),)))
    frame = dict(meaning=meaning, occurrence=('ltm', 'test', 3),
                 kind='observation', evidence=(.8, 0.), match=1.)
    monkeypatch.setattr(ThoughtLTMCapability, '_cued', lambda *args, **kwargs:
        dict(value=(frame,), records_scanned=1, incomplete=()))
    target = unresolved if reverse else name
    pattern = question(equality(None, target), (('referent', 0),))
    result = _context(_cs()).ltm.best_match(pattern, max_records=10, work=None)
    assert result['records_scanned'] == 1
    if reverse:
        assert result['frames'] == (frame,) and result['support_true'] == .8
    else:
        assert result['frames'] == () and result['witnesses'] == ()
        assert result['support_true'] == result['support_false'] == 0.
