"""A selected checked thought preserves its declared object and effects."""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from Meaning import ConceptualMeaning
from Layers import TernaryTruthStore
from test_cs_symbol_table import _cs
from Language import Grammar
from Queries import GrammaticalThoughtRegistry, THOUGHT_EXECUTORS, ThoughtSignature, _existing_row
from test_query_vp_boundaries import _context, _signature


def _meaning():
    return ConceptualMeaning(torch.eye(8)[:3], torch.ones(3, dtype=torch.bool),
                             scope={"place": "workshop"})


def test_exist_executor_keeps_all_roles_and_conflicting_fact_sources():
    cs = _cs()
    refs = tuple(('sym', cs.new_concept()) for _ in range(3))
    for ref in refs:
        cs._csw_concept_row(0, ref[1])
    store = TernaryTruthStore(8)
    store.configure_leaf_index(code_row=lambda ref: _existing_row(cs, ref))
    idea = replace(_meaning(), role_refs=refs)
    first = store.append_meaning(idea, kind="fact", trust=0.6)
    second = store.append_meaning(idea, kind="fact", trust=-0.4)
    store.set_origin(first, store.ORIGIN_PROVISIONED, text="teacher")
    store.set_origin(second, store.ORIGIN_USER, text="witness")
    result = _signature('exist', 'I1').invoke(_context(cs, store=store), idea)
    assert result["support_true"] == pytest.approx(0.6)
    assert result["support_false"] == pytest.approx(0.4)
    torch.testing.assert_close(result["meaning"].roles, idea.roles)
    assert {item["text"] for item in result["candidates"]} == {"teacher", "witness"}
    assert {item['occurrence'] for item in result['candidates']} == {
        store.occurrence_of(first), store.occurrence_of(second)}
    assert all('meaning' not in item for item in result['candidates'])
    assert len(store) == 2


def test_part_and_converse_executor_read_native_taxonomy_without_writes():
    cs = _cs()
    a, b = cs.new_concept(), cs.new_concept()
    for concept in (a, b):
        cs._csw_concept_row(0, concept)
    cs.add_whole(a, ("sym", b))
    context = _context(cs)
    grammar = Grammar()
    grammar.configure({'compose': {'rule': [
        'part_O1 = part.forward(part_I1, part_I2)',
        {'_': 'whole_O1 = whole.forward(whole_I1, whole_I2)',
         'family': 'part', 'permutation': 'I2,I1'},
    ]}, 'thought': {'rule': [
        'part_O1 = part.thought(part_I1, part_I2)',
        {'_': 'whole_O1 = whole.thought(whole_I1, whole_I2)',
         'family': 'part', 'permutation': 'I2,I1'},
    ]}})
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    result = registry.execute(registry.form('part', ('sym', a), ('sym', b)), context)
    assert result.support_true == 1
    assert result.evidence["path"][0].owner == ("sym", a)
    # ``whole(B, A)`` is a grammar permutation of canonical part(A, B).
    assert registry.execute(
        registry.form('whole', ('sym', b), ('sym', a)), context).support_true == 1
    parts = registry.execute(
        registry.form('part', ('sym', b), open_roles=('I1',)), context)
    wholes = registry.execute(
        registry.form('part', ('sym', a), open_roles=('I2',)), context)
    assert parts.value[0]["reference"] == ("sym", a)
    assert wholes.value[0]["reference"] == ("sym", b)


def test_quantize_reads_allocated_conceptual_atoms_without_symbol_snapping():
    cs = _cs()
    a = cs.mint_frozen_concept("test-quantize-existing-concept")
    row = cs._csw_row_of(a)
    atom = cs.similarity_codebook.getW()[row].clone()
    before = dict(cs._concept_allocator.placement)
    result = _signature('quantize', 'I1').invoke(_context(cs), atom)
    assert result["reference"] == ("sym", a)
    torch.testing.assert_close(result["value"], atom)
    assert cs._concept_allocator.placement == before


def test_what_preserves_the_complete_question_and_schedules_same_controller():
    from dataclasses import replace
    question = replace(_meaning(), mode='interrogative')
    calls = []
    result = _signature('what', 'I1').invoke(
        _context(_cs(), continuation=lambda value: calls.append(value) or value), question)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0].roles, question.roles)
    assert result["value"] is not question
    torch.testing.assert_close(result["value"].roles, question.roles)
    assert result["result_kind"] == "subgoal"
    empty = _signature('what', 'I1').invoke(_context(_cs()), question)
    assert empty['result_kind'] == 'set' and empty['frames'] == ()
    assert 'unavailable_ltm' in empty['incomplete']


def test_wrong_domain_and_types_fail_before_execution():
    from dataclasses import replace
    calls = []
    descriptor = replace(
        THOUGHT_EXECUTORS['part'], executor=lambda *args: calls.append(args))
    signature = ThoughtSignature(
        SimpleNamespace(semantic_id='part', operand_roles=('I1', 'I2')),
        descriptor, ('I1', 'I2'))
    # The structural operation fixes its descriptor domain; callers cannot
    # select another one. Invalid operands fail before the executor.
    with pytest.raises(TypeError, match="reference|argument"):
        signature.invoke(_context(_cs()), 1, 2)
    assert calls == []


def test_equality_executor_rejects_width_truncation_and_nonfinite_values():
    context = _context(_cs())
    result = _signature('equal', 'I1', 'I2').invoke(
        context, torch.ones(8), torch.ones(8))
    assert result["support_true"] == 1
    with pytest.raises(ValueError, match="width"):
        _signature('equal', 'I1', 'I2').invoke(
            context, torch.ones(8), torch.ones(4))
    with pytest.raises((ValueError, FloatingPointError), match="finite"):
        _signature('equal', 'I1', 'I2').invoke(
            context, torch.ones(8), torch.full((8,), float("nan")))


def test_arma_returns_all_roles_as_an_estimate_without_overwriting_pending_prediction():
    # Native fixture helper constructs the current production predictor.
    from test_sentence_expectation import layer as make_layer
    layer = make_layer()
    roles = torch.arange(3 * layer.concept_dim, dtype=torch.float32).reshape(3, -1) / 100
    layer.predict_and_observe_stm_end_state([3], [roles], layout="infix")
    marker = object()
    layer._inter_last_meaning[0] = marker
    space = SimpleNamespace(outputShape=(1, layer.concept_dim))
    model = SimpleNamespace(symbolSpace=SimpleNamespace(discourse=layer))
    result = _signature('arma', 'I1').invoke(
        _context(space, model=model, discourse=layer),
        ConceptualMeaning.from_description(roles))
    assert result["evidence_kind"] == "estimate"
    assert result["value"].roles.shape == (3, layer.concept_dim)
    assert result["value"].presence_logits.shape == (3,)
    assert layer._inter_last_meaning[0] is marker
