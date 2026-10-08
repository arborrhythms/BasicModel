"""Thought preparation and nested readers share actual cost, not renewed caps."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Queries import THOUGHT_EXECUTORS, ThoughtSignature
from QueryWork import QueryWorkBudget, QueryWorkExhausted
from test_grammatical_query_vps import _world
from test_cs_symbol_table import _cs
from test_query_vp_boundaries import _context, _signature


def test_candidate_native_reads_obey_budget_before_payload_access(monkeypatch):
    cs, registry, a, b, context = _world()
    context = replace(context, work=QueryWorkBudget(0))
    monkeypatch.setattr(
        cs.similarity_codebook,
        "active_prototypes",
        lambda: pytest.fail("read before budget"),
    )
    with pytest.raises(QueryWorkExhausted):
        registry.form("part", a, b, context=context)
    assert context.work.spent == 0


def test_selected_vp_validation_cannot_read_before_its_allowance(monkeypatch):
    cs, registry, a, b, context = _world()
    question = registry.form("part", a, b)
    owner = cs._concept_allocator._layers[0]
    monkeypatch.setattr(owner, "row_of", lambda *args: pytest.fail("read before budget"))
    result = registry.execute(question, replace(context, work=QueryWorkBudget(0)))
    assert result.support_true == 0
    assert "work_budget" in result.incomplete


def test_registry_description_resolution_and_fact_scan_use_one_budget():
    cs, registry, a, b, context = _world()
    del a, b, context
    store = TernaryTruthStore(8)
    description = ConceptualMeaning.from_description(torch.eye(8)[:3])
    slot = store.append_meaning(description, kind="fact", trust=.6)
    context = _context(cs, store=store)
    question = registry.form("ask", store.occurrence_of(slot), context=context)
    meter = QueryWorkBudget(3)
    result = registry.execute(question, replace(context, work=meter))
    assert result.support_true == 0 and "work_budget" in result.incomplete
    assert meter.spent == 3
    assert result.evidence["resolution_records_scanned"] == 1
    assert result.evidence["records_scanned"] == 0


def test_nested_executor_keeps_same_meter_and_does_not_start_an_allowance():
    from ThoughtReferences import question
    meaning = question(ConceptualMeaning.from_description(torch.eye(8)[:3]), (('evidence', -1),))
    meter = QueryWorkBudget(2)
    context = _context(_cs(), work=meter)
    query = _signature('ask', 'I1')
    context = replace(context, continuation=lambda value: query.invoke(context, value))
    result = query.invoke(context, meaning)
    assert result['support_true'] == 0 and 'work_budget' in result['incomplete']
    assert meter.spent == 2 and dict(meter.counts) == {'operation': 2}


def test_arma_reserves_context_reads_before_running_the_predictor(monkeypatch):
    cs, registry, a, b, context = _world()
    before = context.work.spent
    for name in ('arma', 'expect'):
        with pytest.raises(ValueError, match='sentenceExpectation / gain'):
            registry.form(name, ConceptualMeaning.from_description(torch.ones(8)), context=context)
    assert context.work.spent == before


def test_quantize_does_not_materialize_basis_after_operation_uses_budget(monkeypatch):
    cs, registry, a, b, context = _world()
    monkeypatch.setattr(cs.similarity_codebook, 'active_prototypes', lambda: pytest.fail('retired read'))
    with pytest.raises(ValueError, match='symbolization'):
        registry.form('quantize', torch.ones(8), context=context)
    assert context.work.spent == 0


def test_taxonomy_query_reserves_work_to_use_the_evidence_it_captures():
    from test_taxonomy_view import _ref, _world as taxonomy_world

    cs, (a, b, c) = taxonomy_world()
    del c
    cs.add_whole(a, _ref(b))
    context = _context(cs, work=QueryWorkBudget(5))
    result = _signature('isPart', 'I1', 'I2').invoke(context, _ref(a), _ref(b))
    assert result["support_true"] == 1, "capture must leave work for the direct proof"
    assert context.work.spent <= 5 and result["edges_expanded"] == 1
