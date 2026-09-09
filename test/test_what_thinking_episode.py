"""Phase 3 of the mathematical thinking plan: the resolve step
(``WhatStepChooser`` over ANSWER / OPEN a presented referent) inside
``Model.think()`` on the ``MM_math`` fixture.  Spec section 11 invariants
1, 2, 3, 5, 9 and the checkpoint round trip.  The runtime carries no
arithmetic (Alec 2026-09-09); scripted choosers exercise the MECHANISM."""
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
from Layers import WhatInteractionMemory  # noqa: E402
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


def _script(monkeypatch, chooser, script):
    """Drive the chooser by candidate LABEL: ``script`` is a list of labels
    consumed in order; a missing label falls back to ``answer``."""
    remaining = list(script)
    chosen = []

    def choose(self, context, candidates, *, pressure=0.0, sample=False,
               temperature=1.0):
        labels = [c["label"] for c in candidates]
        want = remaining.pop(0) if remaining else "answer"
        if want not in labels:
            want = "answer"
        index = labels.index(want)
        chosen.append(want)
        return index, torch.log_softmax(self.logits(
            context, candidates, pressure=pressure), dim=-1)[index]

    monkeypatch.setattr(WhatStepChooser, "choose", choose)
    return chosen


def _problem(m, row):
    return m.inputSpace.data.math_problems["train"][row]


def _solver_labels(problem):
    """Labels of a chain-shaped dialogue under the LIFO discipline: from
    the root open each chain variable in order, answer it, return to the
    (still open) root, and finally answer the root."""
    labels = []
    for v in problem.order[:-1]:
        labels += [f"open:{v}", "answer"]
    labels.append("answer")
    return labels


def _expected_ops(problem):
    n = len(problem.order) - 1
    if n == 0:
        return [WhatSlotOperation.COMPLETE]
    return ([WhatSlotOperation.OPEN] + [WhatSlotOperation.COMPLETE] * n
            + [WhatSlotOperation.CLOSE])


# -- invariant 1: byte-identical when thinking is off --------------------------

def test_neutral_chooser_is_single_step_and_iterations_one_builds_no_chooser(model, tmp_path):
    x, _ = _batch(model, rows=2)
    with torch.no_grad():
        result = model.think((What.supervised(0), What.supervised(1)), x)
    assert result.iterations == 1 and result.forced_closures == 0
    assert [s.operation for s in result.slots] == [WhatSlotOperation.COMPLETE] * 2
    # The untrained head scores every candidate equally: ANSWER (first).
    trace = model._last_answer_construction.derivation.grammar_trace
    steps = [e for e in trace if e.get("operation", "").startswith("step:")]
    assert steps and all(e["choice"] == "answer" and e["index"] == 0 for e in steps)
    # A configuration with the limit at 1 never builds the chooser: its
    # state dict is that of the established model.
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


# -- invariants 2, 8, 9: one forward, budgets, replayable trace ----------------

def test_scripted_episode_opens_executes_and_closes_in_lifo_order(model, monkeypatch):
    problem = _problem(model, 0)
    chosen = _script(monkeypatch, WhatStepChooser, _solver_labels(problem))
    x, _ = _batch(model, rows=1)
    forwards = []
    original = model.forward

    def counting_forward(*a, **k):
        forwards.append(1)
        return original(*a, **k)

    monkeypatch.setattr(model, "forward", counting_forward)
    with torch.no_grad():
        result = model.think(What.supervised(0), x)
    n = len(problem.order) - 1
    assert len(forwards) == 1                              # invariant 2
    assert result.iterations == 2 * n + 1 and result.forced_closures == 0
    ops = [s.operation for s in result.slots]
    assert ops == _expected_ops(problem), (ops, chosen)
    memory = model._what_memory()
    assert memory.what_at_parity(b=0)
    # Every step choice is in the replayable trace with its candidates
    # (invariant 9); the referents are the presented WORDS, nothing more.
    steps = [e for e in model._what_episode_steps if e.get("operation", "").startswith("step:")]
    assert steps and all("candidates" in e and "choice" in e for e in steps)
    words = model._what_referents[0]
    assert all(c == "answer" or c[5:] in words
               for e in steps for c in e["candidates"])
    # Slot traces carry the referents so the open stack is derivable from
    # LTM alone; the root's answer is not a computed value.
    referents = [e.get("referent") for s in result.slots for e in s.grammar_trace
                 if isinstance(e, dict) and e.get("operation") == "what_answer_referent"]
    assert referents[-1] is None                       # the root
    assert set(referents[:-1]) == set(problem.order[:-1])
    assert "value" not in [k for s in result.slots for e in s.grammar_trace
                           if isinstance(e, dict) for k in e]


# -- completed subquestions transform the root answer (spec 7.3) ---------------

def test_subquestion_answers_condition_the_root_through_ltm(model, monkeypatch):
    """The root's answer symbol attends over the row's LTM outputs; with a
    non-neutral attention, a different subquestion answer changes the root
    answer, and editing the trace alone does not."""
    problem = _problem(model, 0)
    x, _ = _batch(model, rows=1)
    _script(monkeypatch, WhatStepChooser, _solver_labels(problem))
    with torch.no_grad():
        model.think(What.supervised(0), x)
    attention = model.ltm_attention
    with torch.no_grad():
        attention["out"].weight.add_(0.1 * torch.randn_like(attention["out"].weight))
    _script(monkeypatch, WhatStepChooser, _solver_labels(problem))
    with torch.no_grad():
        model.think(What.supervised(0), x)
    honest = model._last_answer_construction.actual.detach().clone()
    # Edit the trace afterwards: nothing changes.
    model._what_episode_steps[-1] = {**model._what_episode_steps[-1], "choice": "edited"}
    assert torch.equal(model._last_answer_construction.actual, honest)
    # Corrupt the stored subquestion outputs in LTM before the root answers:
    # the root construction changes.
    memory = model._what_memory()
    real_context = memory.what_context

    def corrupted(question=None, b=0):
        ctx = dict(real_context(question=question, b=b))
        ctx["output_representations"] = tuple(
            (v * -3.0 + 1.0) if torch.is_tensor(v) else v
            for v in ctx["output_representations"])
        return ctx

    monkeypatch.setattr(memory, "what_context", corrupted)
    _script(monkeypatch, WhatStepChooser, _solver_labels(problem))
    with torch.no_grad():
        model.think(What.supervised(0), x)
    assert not torch.equal(model._last_answer_construction.actual, honest)
    with torch.no_grad():
        attention["out"].weight.zero_()


def _subanswer_for(model, monkeypatch, x, word):
    """Open the root, answer the subquestion about ``word``, answer the
    root; return the subquestion's answer (the COMPLETE slot's output)."""
    model.symbolSpace.Reset(batch=0, hard=True)
    _script(monkeypatch, WhatStepChooser, [f"open:{word}", "answer", "answer"])
    with torch.no_grad():
        result = model.think(What.supervised(0), x)
    ops = [s.operation for s in result.slots]
    assert ops == [WhatSlotOperation.OPEN, WhatSlotOperation.COMPLETE,
                   WhatSlotOperation.CLOSE], ops
    return result.slots[1].output.detach().clone()


def _two_subanswers(monkeypatch, first_word, second_word):
    """Two consecutive episodes on a FRESH model (the forward path primes
    across calls, so only same-position runs are comparable)."""
    m = _build()
    m.eval()
    x, _ = _batch(m, rows=1)
    return (_subanswer_for(m, monkeypatch, x, first_word),
            _subanswer_for(m, monkeypatch, x, second_word))


def test_subanswer_is_conditioned_on_the_active_subquestion(model, monkeypatch):
    """With the input and the memory fixed, answering a different
    subquestion yields a different subanswer: the ANSWER step for a
    pending subquestion starts from QUERY(w), not from the root idea."""
    problem = _problem(model, 0)
    words = [w for w in problem.order[:-1]] or []
    presented = [w for w in model.inputSpace.data.math_problems["train"][0].surface().split()
                 if w not in ("equals", ";", "what", "is", "plus", "minus", "times")]
    words = list(dict.fromkeys(words + presented))
    assert len(words) >= 2, words
    control_a, control_b = _two_subanswers(monkeypatch, words[0], words[0])
    assert torch.equal(control_a, control_b)          # same word: same answer
    first, other = _two_subanswers(monkeypatch, words[0], words[1])
    assert torch.equal(first, control_a)               # same position: same answer
    assert not torch.equal(first, other), (words[0], words[1])


# -- invariant 3: the desired answer is unreachable ----------------------------

def test_think_never_consults_the_desired_answer(model, monkeypatch):
    data = model.inputSpace.data
    problem = _problem(model, 1)
    _script(monkeypatch, WhatStepChooser, _solver_labels(problem))

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


# -- invariant 5: forced closure ------------------------------------------------

def test_limit_forces_lifo_closure_with_a_scoreable_root(model, monkeypatch):
    problem = _problem(model, 0)
    # Always open something new; never answer.
    variables = sorted(set().union(*(e.vars for e in problem.equations)))
    _script(monkeypatch, WhatStepChooser, [f"open:{v}" for v in variables] * 4)
    x, _ = _batch(model, rows=1)
    with torch.no_grad():
        result = model.think(What.supervised(0), x, max_iterations=3)
    memory = model._what_memory()
    assert result.iterations == 3
    assert result.forced_closures >= 1
    assert memory.what_at_parity(b=0)
    assert result.answer.available and torch.is_tensor(result.answer.what)
    assert all(s.forced and s.operation is WhatSlotOperation.CLOSE
               for s in result.slots[-result.forced_closures:])
    # closure pressure is monotone and reaches 1 at the limit
    ps = result.closure_pressures
    assert all(a <= b for a, b in zip(ps, ps[1:])) and ps[-1] == 1.0
    # the forced root answer traversed reverseOutput (the construction of
    # the last iteration is what the forced answer replays)
    assert model._last_answer_construction is not None


def test_pressure_schedules_are_monotone_and_end_at_one(model):
    for mode in ("linear", "quadratic", "step"):
        model.what_thinking_pressure = mode
        series = [model._thinking_pressure(i, 8) for i in range(8)]
        assert all(a <= b for a, b in zip(series, series[1:])), (mode, series)
        assert series[0] == 0.0 and series[-1] == 1.0
    model.what_thinking_pressure = "linear"


# -- per-row episodes -----------------------------------------------------------

def test_rows_reach_parity_independently(model, monkeypatch):
    p0, p1 = _problem(model, 0), _problem(model, 1)
    # Row 0 opens once and answers; row 1 answers immediately.  The script
    # is consumed in row order within each iteration.
    v0 = p0.order[0]
    labels = [f"open:{v0}", "answer",            # iteration 0: row 0, row 1
              "answer",                          # iteration 1: row 0 (pending v0)
              "answer"]                          # iteration 2: row 0 (root)
    _script(monkeypatch, WhatStepChooser, labels)
    x, _ = _batch(model, rows=2)
    with torch.no_grad():
        result = model.think((What.supervised(0), What.supervised(1)), x)
    memory = model._what_memory()
    assert memory.what_at_parity(b=0) and memory.what_at_parity(b=1)
    assert [s.operation for s in memory.get_what_slots(b=1)] == [WhatSlotOperation.COMPLETE]
    assert [s.operation for s in memory.get_what_slots(b=0)] == [
        WhatSlotOperation.OPEN, WhatSlotOperation.COMPLETE, WhatSlotOperation.CLOSE]
    assert len(result.answers) == 2 and all(a.available for a in result.answers)
    assert result.forced_closures == 0


# -- checkpoint round trip -------------------------------------------------------

def test_checkpoint_round_trips_the_step_chooser(model, tmp_path):
    x, _ = _batch(model, rows=1)
    with torch.no_grad():
        model.think(What.supervised(0), x)
    chooser = model.what_step_chooser
    with torch.no_grad():
        chooser.mlp[-1].weight.add_(0.01)               # make it non-neutral
        chooser.mlp[-1].bias.add_(0.02)
    path = tmp_path / "math.ckpt"
    model.save_weights(str(path))
    saved = model.state_dict()
    assert any(k.startswith("what_step_chooser.") for k in saved)
    fresh = _build()
    assert getattr(fresh, "what_step_chooser", None) is None       # built on demand
    assert fresh.load_weights(str(path), require_match=True)       # the established path
    loaded = fresh.state_dict()
    assert set(saved) == set(loaded)
    for k in saved:
        assert torch.equal(saved[k], loaded[k]), k
    assert torch.equal(fresh.what_step_chooser.mlp[-1].weight, chooser.mlp[-1].weight)
    with torch.no_grad():
        a = model.think(What.supervised(0), x).answer.what
        b = fresh.think(What.supervised(0), x).answer.what
    assert torch.allclose(a, b)


def test_forced_closure_answers_are_row_shaped_in_a_batch(model, monkeypatch):
    """A forced best-effort answer is the ROW's constructed response, not
    the whole batch tensor (found by the stage-1 runs: a forced row's
    answer had B * R elements)."""
    p0, p1 = _problem(model, 0), _problem(model, 1)
    variables = sorted(set().union(*(e.vars for e in p0.equations)) | set().union(*(e.vars for e in p1.equations)))
    _script(monkeypatch, WhatStepChooser, [f"open:{v}" for v in variables] * 6)
    x, y = _batch(model, rows=2)
    with torch.no_grad():
        result = model.think((What.supervised(0), What.supervised(1)), x, max_iterations=2)
    assert result.forced_closures >= 1
    for b, answer in enumerate(result.answers):
        assert torch.is_tensor(answer.what)
        assert answer.what.shape[0] == 1 and answer.what.numel() == y[b].numel()
