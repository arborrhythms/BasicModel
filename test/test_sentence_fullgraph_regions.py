"""Regression gate: sentence regions compile fullgraph=True with zero graph breaks."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MODEL_DEBUG", "0")
os.environ["MODEL_COMPILE"] = "eager"
os.environ["BASICMODEL_DEVICE"] = "cpu"

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent / "bin"))
sys.path.insert(0, str(_ROOT / "bin"))

import pytest
import torch
import torch._dynamo
from data import Data
from Models import BaseModel
from util import init_config, init_device
from What import What


def _build_xor_model():
    init_device("cpu")
    cfg = str(_ROOT / "data" / "XOR_grammar.xml")
    init_config(path=cfg, defaults_path=str(_ROOT / "data" / "model.xml"))
    data = Data()
    model, _ = BaseModel.from_config(cfg, data=data)
    model = model.to("cpu")
    optimizer = model.getOptimizer()
    model._optimizer = optimizer
    model.train()
    return model, optimizer, data


def _step(model, optimizer, data, texts):
    with data.runtime_batch(texts):
        data.has_supervised_outputs = False
        source = model.inputSpace.prepInput(texts)
        questions = tuple(What.present(i, split="runtime") for i in range(len(texts)))
        batch = source, torch.empty(len(texts), 0, device=source.device)
        model.runBatch(
            train=True,
            optimizer=optimizer,
            batchSize=len(texts),
            split="runtime",
            questions=questions,
            batch_override=batch,
        )
    model.End()


def test_sentence_regions_compile_without_graph_breaks(monkeypatch):
    import LookForecast
    import RequestRead
    import WholePaths

    model, optimizer, data = _build_xor_model()
    model.enable_compiled_step()

    monkeypatch.setattr(WholePaths, "credit", torch.compile(WholePaths.credit, backend="eager", fullgraph=True))
    monkeypatch.setattr(RequestRead.RequestRead, "forward", torch.compile(RequestRead.RequestRead.forward, backend="eager", fullgraph=True))
    monkeypatch.setattr(RequestRead.RequestRead, "given", torch.compile(RequestRead.RequestRead.given, backend="eager", fullgraph=True))
    monkeypatch.setattr(RequestRead.RequestRead, "units", torch.compile(RequestRead.RequestRead.units, backend="eager", fullgraph=True))
    monkeypatch.setattr(RequestRead.RequestRead, "stopping", torch.compile(RequestRead.RequestRead.stopping, backend="eager", fullgraph=True))
    monkeypatch.setattr(LookForecast.OccurrenceCredit, "forecast", torch.compile(LookForecast.OccurrenceCredit.forecast, backend="eager", fullgraph=True))

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    texts = ["hello world hello", "world hello world", "red blue red", "blue red blue"]
    _step(model, optimizer, data, texts)

    stats = dict(torch._dynamo.utils.counters.get("stats", {}))
    breaks = dict(torch._dynamo.utils.counters.get("graph_break", {}))
    assert stats.get("unique_graphs", 0) > 0, f"no compiled graphs captured: {stats}"
    assert stats.get("calls_captured", 0) > 0, f"no ops captured: {stats}"
    assert not breaks, f"unexpected Dynamo graph breaks: {breaks}"


def test_whole_sentence_forward_fullgraph_no_breaks(monkeypatch):
    model, optimizer, data = _build_xor_model()
    # Single-look mode exercises the entire forward() sentence graph in ~75s
    # while unrolling hardwire_reading=True (attention_budget=32) takes ~38m.
    model.hardwire_reading = False
    monkeypatch.setattr(type(model), "forward", torch.compile(type(model).forward, backend="eager", fullgraph=True))

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    texts = ["hello world hello", "world hello world", "red blue red", "blue red blue"]
    _step(model, optimizer, data, texts)

    stats = dict(torch._dynamo.utils.counters.get("stats", {}))
    breaks = dict(torch._dynamo.utils.counters.get("graph_break", {}))
    assert stats.get("unique_graphs", 0) == 1, f"expected 1 graph for whole sentence, got: {stats}"
    assert not breaks, f"unexpected Dynamo graph breaks: {breaks}"


@pytest.mark.slow
def test_whole_sentence_forward_fullgraph_over_several_looks(monkeypatch):
    """The loop of looks the default configuration runs, compiled whole: the
    gate above reads one look, so a break from the second look on passes it.
    A budget of two looks keeps the unrolled graph small."""
    model, optimizer, data = _build_xor_model()
    assert model.hardwire_reading
    model.attention_budget = 2
    monkeypatch.setattr(type(model), "forward", torch.compile(type(model).forward, backend="eager", fullgraph=True))

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    texts = ["hello world hello", "world hello world", "red blue red", "blue red blue"]
    _step(model, optimizer, data, texts)

    stats = dict(torch._dynamo.utils.counters.get("stats", {}))
    breaks = dict(torch._dynamo.utils.counters.get("graph_break", {}))
    assert stats.get("unique_graphs", 0) == 1, f"expected 1 graph for whole sentence, got: {stats}"
    assert not breaks, f"unexpected Dynamo graph breaks: {breaks}"
