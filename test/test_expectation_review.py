"""Failing review probes from integrated specification section 11."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Layers import WhatInteractionMemory
from Models import BasicModel
from What import LTMSlot
from test_sentence_expectation import layer, observe


def test_soft_reset_preserves_stream_and_bare_reset_is_hard():
    disc = layer(batch=2)
    disc.predict_and_observe_stm_end_state(
        [3, 3], [torch.ones(3, 4), torch.full((3, 4), 2.)],
        documents=["a", "b"], layout="infix")
    prior = disc.expect_next_meaning(0).roles.detach().clone()
    disc.Reset(batch=0, hard=False)
    assert len(disc._inter_context[0]) == 1
    assert disc._expectation_documents == ["a", "b"]
    torch.testing.assert_close(disc.expect_next_meaning(0).roles, prior)
    disc.Reset(batch=0)
    assert disc.expect_next_meaning(0) is None
    assert len(disc._inter_context[1]) == 1


def test_unaddressed_driver_rows_form_independent_streams():
    disc = layer(batch=2)
    host = SimpleNamespace(
        inputSpace=SimpleNamespace(data=SimpleNamespace(
            source_addresses={"train": [{"document": "a"}]})),
        symbolSpace=SimpleNamespace(discourse=disc))
    for sources in ([[7, 8], [-1, None]], [[9, 10], [11, 12]]):
        BasicModel._stage_expectation_documents(host, "train", sources, 2)
        assert host._expectation_documents == ((None, None), (None, None))
        disc.predict_and_observe_stm_end_state(
            [3, 3], [torch.ones(3, 4), torch.full((3, 4), 2.)],
            documents=[None, None], layout="infix")
    assert disc._inter_loss_count == 2


def test_address_present_without_document_still_fails():
    host = SimpleNamespace(inputSpace=SimpleNamespace(data=SimpleNamespace(
        source_addresses={"train": [{}]})))
    with pytest.raises(ValueError, match="document"):
        BasicModel._stage_expectation_documents(host, "train", [0], 1)


def test_one_interaction_owner_even_when_expectation_is_enabled(tmp_path):
    from test_meronomy_ladder import _build_ladder_variant
    model = _build_ladder_variant(tmp_path, "one_memory", [
        ("<sentenceExpectation>false</sentenceExpectation>",
         "<sentenceExpectation>true</sentenceExpectation>")])
    try:
        assert isinstance(model.symbolSpace.what_memory, WhatInteractionMemory)
        assert model._what_memory() is model.symbolSpace.what_memory
        assert not hasattr(model.symbolSpace.discourse, "what_memory")
        assert not hasattr(model.symbolSpace.discourse, "append_what_slot")
    finally:
        model.End()


def test_expectation_names_and_production_defaults():
    import Layers
    import xml.etree.ElementTree as ET
    assert hasattr(Layers, "SentenceExpectation")
    assert hasattr(Layers, "MeaningExpectation")
    root = Path(__file__).resolve().parents[1]
    for name in ("model.xml", "BasicModel.xml"):
        training = ET.parse(root / "data" / name).getroot().find("architecture/training")
        assert training.findtext("sentenceExpectation") == "true"
        assert float(training.findtext("interLossWeight")) == .1


def test_provisioning_keeps_its_existing_hard_reset_of_what_episode():
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model = _make_model(_SERIAL_CONFIG)
    memory = model._what_memory()
    memory.begin_what_episode(0)
    memory.append_what_slot(LTMSlot(input=torch.ones(4)))
    assert memory.in_episode(0)
    model.provision_ltm()
    assert not memory.in_episode(0)
    assert memory.get_what_slots(b=0) == []


def test_real_packed_bricks_share_one_document_stream(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setenv("MODEL_COMPILE", "none")
    model = _build_ladder_variant(tmp_path, "brick_stream", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<sentenceExpectation>false</sentenceExpectation>", "<sentenceExpectation>true</sentenceExpectation>"),
        ("<interLossWeight>0.0</interLossWeight>", "<interLossWeight>0.1</interLossWeight>"),
    ])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    model.train()
    data = model.inputSpace.data
    monkeypatch.setattr(data, "has_supervised_outputs", False)
    optimizer = model.getOptimizer(lr=1e-5)
    disc = model.symbolSpace.discourse
    observe_meanings = disc._observe_meanings
    pairs = []
    def capture(*args, **kwargs):
        before = disc._inter_loss_count
        result = observe_meanings(*args, **kwargs)
        pairs.append(disc._inter_loss_count - before)
        return result
    monkeypatch.setattr(disc, "_observe_meanings", capture)
    try:
        for rows in ([["1 plus 2", "3 plus 4"]], [["5 plus 6", "7 plus 8"]]):
            inputs = model.inputSpace.prepPackedInput(rows)
            model.runBatch(train=True, batchSize=1, optimizer=optimizer,
                           source_rows=[[10**9, 10**9 + 1]],
                           batch_override=(inputs, torch.zeros(1, 1, 0)))
            model.dispatch_packed_soft_reset(hard_eos=[False])
        assert sum(pairs) == 3, pairs
        model.symbolSpace.Reset(batch=0, hard=True)
        assert disc.expect_next_meaning(0) is None
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
