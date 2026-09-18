"""Query preparation and nested readers share actual cost, not renewed caps."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Queries import BUILTIN_QUERIES, QueryContext
from QueryWork import QueryWorkBudget, QueryWorkExhausted
from reasoning import TruthGroundedReasoner
from test_grammatical_query_vps import _world


def test_candidate_native_reads_obey_budget_before_payload_access(monkeypatch):
    cs, registry, a, b, context = _world()
    context = replace(context, work=QueryWorkBudget(0))
    monkeypatch.setattr(
        cs.similarity_codebook,
        "active_prototypes",
        lambda: pytest.fail("read before budget"),
    )
    with pytest.raises(QueryWorkExhausted):
        registry.form("isPart", a, b, context=context)
    assert context.work.spent == 0


def test_selected_vp_validation_cannot_read_before_its_allowance(monkeypatch):
    cs, registry, a, b, context = _world()
    question = registry.form("isPart", a, b)
    owner = cs._concept_allocator._layers[0]
    monkeypatch.setattr(owner, "row_of", lambda *args: pytest.fail("read before budget"))
    result = registry.execute(question, replace(context, work=QueryWorkBudget(0)))
    assert result["meaning"] is question and result["support_true"] == 0
    assert "work_budget" in result["incomplete"]


def test_registry_description_resolution_and_fact_scan_use_one_budget():
    cs, registry, a, b, context = _world()
    del a, b, context
    store = TernaryTruthStore(8)
    description = ConceptualMeaning.from_description(torch.eye(8)[:3])
    slot = store.append_meaning(description, kind="fact", trust=.6)
    context = QueryContext(
        TruthGroundedReasoner(
            model=SimpleNamespace(conceptualSpace=cs), store=store))
    question = registry.form("isTrue", store.occurrence_of(slot), context=context)
    meter = QueryWorkBudget(3)
    result = registry.execute(question, replace(context, work=meter))
    assert result["support_true"] == 0 and "work_budget" in result["incomplete"]
    assert meter.spent == 3
    assert result["resolution_records_scanned"] == 1
    assert result["records_scanned"] == 0


def test_nested_executor_keeps_same_meter_and_does_not_start_an_allowance():
    meaning = ConceptualMeaning.from_description(torch.eye(4)[:3])
    store = TernaryTruthStore(4)
    store.append_meaning(meaning, kind="fact", trust=.7)
    meter = QueryWorkBudget(2)
    context = QueryContext(TruthGroundedReasoner(store=store), work=meter)
    context = replace(
        context,
        schedule_subgoal=lambda value: BUILTIN_QUERIES["exist"].invoke(context, value),
    )
    result = BUILTIN_QUERIES["what"].invoke(
        context, replace(meaning, mode="interrogative"))
    assert result["value"]["support_true"] == 0
    assert result["value"]["records_scanned"] == 0
    assert meter.spent == 2 and dict(meter.counts) == {"operation": 2}


def test_arma_reserves_context_reads_before_running_the_predictor(monkeypatch):
    from test_sentence_expectation import layer as make_layer

    layer = make_layer()
    roles = torch.eye(layer.concept_dim)[:3]
    for _ in range(2):
        layer.predict_and_observe_stm_end_state([3], [roles], layout="infix")
    marker = object()
    layer._inter_last_meaning[0] = marker
    monkeypatch.setattr(
        layer._inter_predictor,
        "forward",
        lambda *args: pytest.fail("prediction beyond work"),
    )
    context = QueryContext(
        TruthGroundedReasoner(
            model=SimpleNamespace(symbolSpace=SimpleNamespace(discourse=layer))),
        work=QueryWorkBudget(2),
    )
    result = BUILTIN_QUERIES["arma"].invoke(
        context, ConceptualMeaning.from_description(roles))
    assert result["value"] is None and "work_budget" in result["incomplete"]
    assert layer._inter_last_meaning[0] is marker
    assert context.work.spent == 1


def test_quantize_does_not_materialize_basis_after_operation_uses_budget(monkeypatch):
    cs, registry, a, b, context = _world()
    del registry, a, b
    monkeypatch.setattr(
        cs.similarity_codebook,
        "active_prototypes",
        lambda: pytest.fail("basis read after budget"),
    )
    result = BUILTIN_QUERIES["quantize"].invoke(
        replace(context, work=QueryWorkBudget(1)), torch.ones(8))
    assert result["value"] is None and result["nodes_scanned"] == 0
    assert "work_budget" in result["incomplete"]


def test_taxonomy_query_reserves_work_to_use_the_evidence_it_captures():
    from test_taxonomy_view import _ref, _world as taxonomy_world

    cs, (a, b, c) = taxonomy_world()
    del c
    cs.add_whole(a, _ref(b))
    context = QueryContext(
        TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs)),
        work=QueryWorkBudget(5),
    )
    result = BUILTIN_QUERIES["isPart"].invoke(context, _ref(a), _ref(b))
    assert result["support_true"] == 1, "capture must leave work for the direct proof"
    assert context.work.spent <= 5 and result["edges_expanded"] == 1
