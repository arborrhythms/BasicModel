"""Phase 3 of the mathematical thinking plan: the resolve step
(``WhatStepChooser`` + exact primitives) inside ``Model.think()`` on the
``MM_math`` fixture.  Spec section 11 invariants 1, 2, 3, 5, 6, 8, 9 and the
checkpoint round trip.  Scripted choosers exercise the MECHANISM; nothing
here claims learned thinking."""
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
from exact import ExactVerifier, numeral_code, referent_code  # noqa: E402

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
    """Labels of a solver-order derivation under the LIFO discipline: from
    the root open the first chain variable, evaluate / bind / answer it,
    return to the (still open) root, open the next one, ... and finally
    evaluate / bind / answer the query, which closes the root."""
    order = list(problem.order)          # [x0, x1, ..., query]
    labels = []
    for v in order[:-1]:
        i = problem.index_of(v)
        labels += [f"open:{v}", f"evaluate:{i}",
                   f"bind:{v}={problem.solution[v]}", "answer"]
    q = order[-1]
    labels += [f"evaluate:{problem.index_of(q)}",
               f"bind:{q}={problem.solution[q]}", "answer"]
    return labels


def _expected_ops(problem):
    n = len(problem.order) - 1           # chain variables before the query
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
    # A configuration with the limit at 1 and primitives off never builds
    # the chooser: its state dict is that of the established model.
    src = (_DATA / "MM_math.xml").read_text()
    off = src.replace("<whatThinkingIterations>8</whatThinkingIterations>",
                      "<whatThinkingIterations>1</whatThinkingIterations>")
    off = off.replace("<whatThinkingPrimitives>4</whatThinkingPrimitives>",
                      "<whatThinkingPrimitives>0</whatThinkingPrimitives>")
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
    # Every choice and every primitive execution is in the trace with
    # operands and results (invariant 9); executions per iteration are
    # within the budget (invariant 8); the verifier accepts the derivation.
    steps = model._what_exact_states[0][0].trace
    assert all(s["operation"].startswith("exact:") and "operands" in s
               and "result" in s for s in steps)
    per_iteration = {}
    for s in steps:
        per_iteration[s["iteration"]] = per_iteration.get(s["iteration"], 0) + 1
    assert max(per_iteration.values()) <= model.what_thinking_primitives
    report = ExactVerifier().check(steps, problem, answer=problem.answer)
    assert report.rejected == 0 and report.valid
    # The root answer symbol's root slot is the exact numeral code of the
    # bound query value.
    symbol = model._last_answer_construction.derivation.answer_symbol
    D = symbol.shape[-1]
    code = numeral_code(problem.answer, D)
    # (the question conditioner adds a zero-initialised delta: exact match)
    assert torch.allclose(symbol[0, 0, :], code)
    # Slot traces carry the referents so the open stack is derivable from
    # LTM alone; the answers name their values.
    referents = [e.get("referent") for s in result.slots for e in s.grammar_trace
                 if isinstance(e, dict) and e.get("operation") == "what_answer_referent"]
    assert referents[-1] == problem.query
    values = [e.get("value") for s in result.slots for e in s.grammar_trace
              if isinstance(e, dict) and e.get("operation") == "what_answer_referent"]
    assert values[-1] == problem.answer


# -- invariant 6: answer causality --------------------------------------------

def test_changing_a_necessary_intermediate_changes_the_answer(model, monkeypatch):
    problem = _problem(model, 0)
    x, _ = _batch(model, rows=1)
    _script(monkeypatch, WhatStepChooser, _solver_labels(problem))
    with torch.no_grad():
        model.think(What.supervised(0), x)
    honest = model._last_answer_construction
    honest_actual = honest.actual.detach().clone()
    honest_symbol = honest.derivation.answer_symbol.detach().clone()
    # Editing the trace afterwards changes nothing about the emitted answer.
    trace_edit = list(model._what_exact_states[0][0].trace)
    trace_edit[-1] = {**trace_edit[-1], "result": 99}
    assert torch.equal(model._last_answer_construction.actual, honest_actual)
    # Corrupt the exact evaluation of the first chain variable: the bound
    # values, the answer symbol and the emitted answer all change.
    from exact import ExactState
    real_evaluate = ExactState.evaluate

    def wrong_evaluate(self, i):
        value = real_evaluate(self, i)
        return (value + 1) % self.range if isinstance(value, int) else value

    monkeypatch.setattr(ExactState, "evaluate", wrong_evaluate)
    _script(monkeypatch, WhatStepChooser, [
        lbl if not lbl.startswith("bind:") else
        f"bind:{lbl[5:].split('=')[0]}={(int(lbl.split('=')[1]) + 1) % 16}"
        for lbl in _solver_labels(problem)])
    with torch.no_grad():
        model.think(What.supervised(0), x)
    corrupted = model._last_answer_construction
    # the first chain variable's bind differs, so the derivation differs
    assert model._what_exact_states[0][0].bindings[problem.order[0]] != problem.solution[problem.order[0]]
    assert not torch.equal(corrupted.derivation.answer_symbol[0, 0, :], honest_symbol[0, 0, :])
    assert not torch.equal(corrupted.actual, honest_actual)
    report = ExactVerifier().check(model._what_exact_states[0][0].trace, problem,
                                   answer=problem.answer)
    assert not report.valid                                 # verifier catches it


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
    # The presented surface (the INPUT side) is what the scratchpad lexed.
    state, query = model._what_exact_states[0]
    assert query == _problem(model, 0).query or query == problem.query


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


def test_referent_and_numeral_codes_are_distinct():
    D = 14
    numerals = {tuple(numeral_code(n, D, bits=6).tolist()) for n in range(16)}
    referents = {tuple(referent_code(v, D).tolist()) for v in "abcxyz"}
    assert not (numerals & referents)
    assert len(referents) == 6


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
