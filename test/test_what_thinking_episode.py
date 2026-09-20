"""Single-presentation thinking, target isolation and retained numerical schedules.

Recurrent child execution and cutoff are exercised on the normal controller in
 test_unified_thought_controller.py; no parity chooser is scripted here.
"""
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

from What import What, WhatSlotOperation  # noqa: E402

_MATH_DAT = {"mathRange": 16, "mathDepths": "1-2", "mathTestDepths": "3",
             "mathDistractors": 1, "mathProblems": 64, "mathSeed": 0}


def _build(config_path=None):
    import Language
    from util import init_config
    from data import TheData
    import Models

    config_path = str(config_path or (_DATA / "MM_math.xml"))
    init_config(path=config_path, defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("math", dat=dict(_MATH_DAT))
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(config_path, data=TheData)
    return m.to("cpu")


def _batch(m, rows=2):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=rows)
    inp_items, out_items = next(iter(loader))
    return (m.inputSpace.prepInput(inp_items),
            m.outputSpace.prepOutput(out_items))


@pytest.fixture(scope="module")
def model():
    return _build()


def _problem(m, row):
    return m.inputSpace.data.math_problems["train"][row]


def test_neutral_chooser_is_single_step_and_iterations_one_builds_no_chooser(model, tmp_path):
    x, _ = _batch(model, rows=2)
    with torch.no_grad():
        result = model.think((What.supervised(0), What.supervised(1)), x)
    assert result.iterations == 1 and result.forced_closures == 0
    assert [s.operation for s in result.slots] == [WhatSlotOperation.COMPLETE] * 2
    # The presentation wrapper has no separate parity chooser.
    trace = model._last_answer_construction.derivation.grammar_trace
    steps = [e for e in trace if e.get("operation", "").startswith("step:")]
    assert steps == []
    assert not hasattr(model, "what_step_chooser")
    # The legacy iteration knob never installs another controller.
    src = (_DATA / "MM_math.xml").read_text()
    off = src.replace("<whatThinkingIterations>8</whatThinkingIterations>",
                      "<whatThinkingIterations>1</whatThinkingIterations>")
    path = tmp_path / "MM_math_off.xml"
    path.write_text(off)
    m_off = _build(path)
    assert not m_off._thinking_enabled()
    with torch.no_grad():
        m_off.what(What.supervised(0), _batch(m_off, rows=1)[0])
    assert not any(k.startswith("what_step_chooser") for k in m_off.state_dict())
    assert m_off._last_answer_construction.derivation.step == ()


def test_think_never_consults_the_desired_answer(model, monkeypatch):
    data = model.inputSpace.data

    def forbidden(question):
        raise AssertionError("Data.what() reached during thinking")

    monkeypatch.setattr(data, "what", forbidden)
    x, _ = _batch(model, rows=2)
    with torch.no_grad():
        result = model.think((What.supervised(0), What.supervised(1)), x[1:2])
    assert result.answer.available
    # The presented surface (the INPUT side) is what the referents come from.
    assert set(model._what_referents[0]) <= set(
        model._presented_surface(What.supervised(0)).replace(";", " ").split())


def test_pressure_schedules_are_monotone_and_end_at_one(model):
    for mode in ("linear", "quadratic", "step"):
        model.what_thinking_pressure = mode
        series = [model._thinking_pressure(i, 8) for i in range(8)]
        assert all(a <= b for a, b in zip(series, series[1:])), (mode, series)
        assert series[0] == 0.0 and series[-1] == 1.0
    model.what_thinking_pressure = "linear"
