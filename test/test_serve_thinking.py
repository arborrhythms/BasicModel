"""Phase 5 of the mathematical thinking plan: the serve endpoint runs a
bounded thinking episode for a thinking-configured model and reports it;
a thought-free (Shamatha) request never opens an internal dialogue."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
_DATA = _ROOT / "data"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Language import WhatStepChooser  # noqa: E402

_MATH_DAT = {"mathRange": 16, "mathDepths": "1-2", "mathTestDepths": "3",
             "mathDistractors": 1, "mathProblems": 64, "mathSeed": 0}


@pytest.fixture(scope="module")
def served():
    import Language
    from util import init_config
    from data import TheData
    import Models
    import serve

    init_config(path=str(_DATA / "MM_math.xml"), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("math", dat=dict(_MATH_DAT))
    torch.manual_seed(0)
    m, cfg = Models.BaseModel.from_config(str(_DATA / "MM_math.xml"), data=TheData)
    m = m.to("cpu")
    m.eval()
    serve._model = m
    serve._model_config = cfg
    serve._rate_limit = 0
    return serve, m


def _post(serve, payload):
    client = serve.app.test_client()
    return client.post("/chat/completions", json=payload)


def test_thinking_payload_is_attached_for_a_thinking_model(served, monkeypatch):
    serve, model = served
    problem = model.inputSpace.data.math_problems["train"][0]
    # Script the chooser: open the first chain variable, answer it, then
    # answer the root (values may stay unbound; the mechanism is what is
    # under test here).
    labels = [f"open:{problem.order[0]}", "answer", "answer"]

    def choose(self, context, candidates, *, pressure=0.0, sample=False,
               temperature=1.0):
        names = [c["label"] for c in candidates]
        want = labels.pop(0) if labels else "answer"
        index = names.index(want) if want in names else 0
        return index, torch.log_softmax(
            self.logits(context, candidates, pressure=pressure), dim=-1)[index]

    monkeypatch.setattr(WhatStepChooser, "choose", choose)
    resp = _post(serve, {"messages": [{"role": "user", "content": problem.surface()}]})
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    thinking = body.get("thinking")
    assert thinking is not None and thinking["thought_free"] is False
    assert thinking["iterations"] >= 2 and thinking["forced_closures"] >= 0
    assert thinking["slots"][0] == "open"
    assert any(s["choice"].startswith("open:") for s in thinking["steps"])
    assert "content" in body["choices"][0]["message"]


def test_thought_free_request_opens_no_dialogue(served):
    serve, model = served
    problem = model.inputSpace.data.math_problems["train"][1]
    calls = []
    real_think = model.think
    model.think = lambda *a, **k: calls.append(1) or real_think(*a, **k)
    try:
        resp = _post(serve, {"messages": [{"role": "user", "content": problem.surface()}],
                             "thought_free": True})
    finally:
        model.think = real_think
    assert resp.status_code == 200, resp.get_json()
    thinking = resp.get_json().get("thinking")
    assert thinking == {"thought_free": True, "iterations": 1, "forced_closures": 0,
                        "slots": [], "steps": [], "primitives": 0, "value": None}
    assert not calls
    memory = model._what_memory()
    assert memory.what_at_parity(b=0)


def test_non_thinking_model_has_no_payload(served):
    serve, model = served
    saved = (model.what_thinking_iterations, model.what_thinking_primitives)
    model.what_thinking_iterations, model.what_thinking_primitives = 1, 0
    try:
        resp = _post(serve, {"messages": [{"role": "user", "content": "a = 1 ; what is a ?"}]})
    finally:
        model.what_thinking_iterations, model.what_thinking_primitives = saved
    assert resp.status_code == 200
    assert "thinking" not in resp.get_json()
