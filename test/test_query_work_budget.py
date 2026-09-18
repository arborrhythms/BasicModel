"""All selected-query work consumes one shared allowance before its read."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Queries import BUILTIN_QUERIES, QueryContext
from Taxonomy import capture_taxonomy
from reasoning import TruthGroundedReasoner
from test_taxonomy_view import _ref, _world


def budget(limit):
    from QueryWork import QueryWorkBudget
    return QueryWorkBudget(limit)


def test_taxonomy_capture_and_expansion_share_one_meter(monkeypatch):
    cs, (a, b, c) = _world()
    cs.add_whole(a, _ref(b))
    cs.add_whole(b, _ref(c))
    owner = cs._concept_allocator.store_of(a)
    original = owner.iter_constituents
    read = []

    def tracked(cid):
        for item in original(cid):
            read.append(item)
            yield item

    monkeypatch.setattr(owner, "iter_constituents", tracked)
    meter = budget(2)
    view = capture_taxonomy(
        cs, max_nodes=50, max_records=50, focus=(_ref(a),), work=meter)
    result = view.part_of(_ref(a), _ref(b), max_expansions=50, work=meter)
    assert view.nodes_scanned == 1 and view.records_scanned == 1 and len(read) == 1
    assert result["edges_expanded"] == 0 and result["support_true"] == 0
    assert "work_budget" in result["incomplete"]
    assert meter.spent == 2 and meter.remaining == 0
    later = capture_taxonomy(cs, max_nodes=50, max_records=50, work=meter)
    assert later.nodes_scanned == later.records_scanned == 0
    assert len(read) == 1


def test_selected_operation_is_charged_before_its_executor():
    calls = []
    signature = replace(
        BUILTIN_QUERIES["what"],
        executor=lambda context, arguments: calls.append(arguments) or {"value": None},
    )
    question = ConceptualMeaning.from_description(torch.ones(4))
    context = QueryContext(TruthGroundedReasoner(), work=budget(0))
    result = signature.invoke(context, question)
    assert calls == [] and "work_budget" in result["incomplete"]
    assert context.work.spent == 0


def test_invalid_call_spends_no_work_and_never_executes():
    calls = []
    signature = replace(
        BUILTIN_QUERIES["isPart"],
        executor=lambda context, arguments: calls.append(arguments) or {},
    )
    meter = budget(5)
    context = QueryContext(TruthGroundedReasoner(), work=meter)
    with pytest.raises(TypeError):
        signature.invoke(context, 1, 2)
    assert calls == [] and meter.spent == 0


def test_fact_reads_stop_at_shared_budget_and_preserve_partial_evidence(monkeypatch):
    meaning = ConceptualMeaning.from_description(torch.ones(4))
    store = TernaryTruthStore(4)
    store.append_meaning(meaning, kind="fact", trust=.6)
    store.append_meaning(meaning, kind="fact", trust=-.4)
    read, original = [], store.row

    def row(index):
        read.append(index)
        return original(index)

    monkeypatch.setattr(store, "row", row)
    meter = budget(2)
    result = BUILTIN_QUERIES["exist"].invoke(
        QueryContext(TruthGroundedReasoner(store=store), work=meter), meaning)
    assert read == [0] and result["records_scanned"] == 1
    assert result["support_true"] == pytest.approx(.6)
    assert result["support_false"] == 0 and "work_budget" in result["incomplete"]
    assert meter.spent == 2


def test_taxonomy_executor_cannot_renew_node_record_and_expansion_allowances():
    cs, (a, b, c) = _world()
    del c
    cs.add_whole(a, _ref(b))
    meter = budget(3)
    context = QueryContext(
        TruthGroundedReasoner(model=SimpleNamespace(conceptualSpace=cs)),
        max_nodes=99,
        max_records=99,
        max_expansions=99,
        work=meter,
    )
    result = BUILTIN_QUERIES["isPart"].invoke(context, _ref(a), _ref(b))
    assert (result["nodes_scanned"] + result["records_scanned"]
            + result["edges_expanded"] == 2)
    assert result["support_true"] == 0 and meter.spent == 3
    assert "work_budget" in result["incomplete"]


def test_budget_validation_and_overdraw_are_atomic():
    from QueryWork import QueryWorkBudget, QueryWorkExhausted

    for value in (-1, True, 1.5, float("inf")):
        with pytest.raises((ValueError, TypeError)):
            QueryWorkBudget(value)
    meter = QueryWorkBudget(3)
    meter.require("record", 2)
    with pytest.raises(QueryWorkExhausted):
        meter.require("record", 2)
    assert meter.spent == 2 and meter.remaining == 1
    assert dict(meter.counts) == {"record": 2}
    assert meter.consume("node") and not meter.consume("node")
    assert meter.spent == 3
