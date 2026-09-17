"""A selected checked query preserves its declared object and effects."""
from types import SimpleNamespace

import pytest
import torch

from Meaning import ConceptualMeaning
from Layers import TernaryTruthStore
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


def _invoke(name, args, *, reasoner=None, **kwargs):
    from Queries import BUILTIN_QUERIES, QueryContext
    context = QueryContext(reasoner=reasoner or TruthGroundedReasoner(), **kwargs)
    return BUILTIN_QUERIES[name].invoke(context, *args)


def _meaning():
    return ConceptualMeaning(torch.eye(8)[:3], torch.ones(3, dtype=torch.bool),
                             scope={"place": "workshop"})


def test_exist_executor_keeps_all_roles_and_conflicting_fact_sources():
    store = TernaryTruthStore(8)
    idea = _meaning()
    first = store.append_meaning(idea, kind="fact", trust=0.6)
    second = store.append_meaning(idea, kind="fact", trust=-0.4)
    store.set_origin(first, store.ORIGIN_PROVISIONED, text="teacher")
    store.set_origin(second, store.ORIGIN_USER, text="witness")
    result = _invoke("isTrue", (idea,), reasoner=TruthGroundedReasoner(store=store))
    assert result["support_true"] == pytest.approx(0.6)
    assert result["support_false"] == pytest.approx(0.4)
    assert result["meaning"] is idea
    assert {item["text"] for item in result["candidates"]} == {"teacher", "witness"}
    assert len(store) == 2


def test_part_and_converse_executor_read_native_taxonomy_without_writes():
    cs = _cs()
    a, b = cs.new_concept(), cs.new_concept()
    cs.add_whole(a, ("sym", b))
    reasoner = TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs))
    for name, args in [("isPart", (("sym", a), ("sym", b))),
                       ("isWhole", (("sym", b), ("sym", a)))]:
        result = _invoke(name, args, reasoner=reasoner)
        assert result["support_true"] == 1
        assert result["path"][0].owner == ("sym", a)
    assert _invoke("parts", (("sym", b),), reasoner=reasoner)["value"][0]["reference"] == ("sym", a)
    assert _invoke("wholes", (("sym", a),), reasoner=reasoner)["value"][0]["reference"] == ("sym", b)


def test_quantize_reads_allocated_conceptual_atoms_without_symbol_snapping():
    cs = _cs()
    a = cs.mint_frozen_concept("test-quantize-existing-concept")
    row = cs._csw_row_of(a)
    atom = cs.similarity_codebook.getW()[row].clone()
    reasoner = TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs))
    before = dict(cs._concept_allocator.placement)
    result = _invoke("quantize", (atom,), reasoner=reasoner)
    assert result["reference"] == ("sym", a)
    torch.testing.assert_close(result["value"], atom)
    assert cs._concept_allocator.placement == before


def test_what_preserves_the_complete_question_and_schedules_same_controller():
    from dataclasses import replace
    question = replace(_meaning(), mode='interrogative')
    calls = []
    result = _invoke("what", (question,), schedule_subgoal=lambda value: calls.append(value) or value)
    assert calls == [question]
    assert result["value"] is question
    assert result["result_kind"] == "subgoal"
    with pytest.raises(RuntimeError, match="controller"):
        _invoke("what", (question,))


def test_wrong_domain_and_types_fail_before_execution():
    from Queries import BUILTIN_QUERIES, QueryContext
    from dataclasses import replace
    calls = []
    signature = replace(BUILTIN_QUERIES["isPart"], executor=lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="domain"):
        signature.invoke(QueryContext(TruthGroundedReasoner()), ("sym", 1), ("sym", 2), domain="perceptual-mereonomy")
    with pytest.raises(TypeError, match="reference|argument"):
        signature.invoke(QueryContext(TruthGroundedReasoner()), 1, 2)
    assert calls == []


def test_equality_executor_rejects_width_truncation_and_nonfinite_values():
    result = _invoke("isEqual", (torch.ones(8), torch.ones(8)))
    assert result["support_true"] == 1
    with pytest.raises(ValueError, match="width"):
        _invoke("isEqual", (torch.ones(8), torch.ones(4)))
    with pytest.raises((ValueError, FloatingPointError), match="finite"):
        _invoke("isEqual", (torch.ones(8), torch.full((8,), float("nan"))))


def test_arma_returns_all_roles_as_an_estimate_without_overwriting_pending_prediction():
    # Native fixture helper constructs the current production predictor.
    from test_sentence_expectation import layer as make_layer
    layer = make_layer()
    roles = torch.arange(3 * layer.concept_dim, dtype=torch.float32).reshape(3, -1) / 100
    layer.predict_and_observe_stm_end_state([3], [roles], layout="infix")
    marker = object()
    layer._inter_last_meaning[0] = marker
    model = SimpleNamespace(symbolSpace=SimpleNamespace(discourse=layer))
    result = _invoke("arma", (ConceptualMeaning.from_description(roles),),
                     reasoner=TruthGroundedReasoner(model=model))
    assert result["evidence_kind"] == "estimate"
    assert result["value"].roles.shape == (3, layer.concept_dim)
    assert result["value"].presence_logits.shape == (3,)
    assert layer._inter_last_meaning[0] is marker
