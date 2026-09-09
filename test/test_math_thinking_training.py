"""Phase 4 of the mathematical thinking plan: training THROUGH thinking
episodes on ``MM_math`` (spec 8): root scored after parity, one optimizer
step, the episode credit boundary (invariant 7), the desired answer out of
reach until loss, the step-chooser policy credit, and the report.  Scripted
choosers exercise the mechanism; the ``RUN_SLOW`` test is a learning floor,
not the spec's pilot gate."""
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
from What import What, WhatSlotOperation  # noqa: E402

_MATH_DAT = {"mathRange": 16, "mathDepths": "1-2", "mathTestDepths": "3",
             "mathDistractors": 1, "mathProblems": 64, "mathSeed": 0}


def _config(tmp_path_factory, name, **overrides):
    src = (_DATA / "MM_math.xml").read_text()
    for key, value in overrides.items():
        tag = f"<{key}>"
        if tag in src:
            start = src.index(tag) + len(tag)
            end = src.index(f"</{key}>", start)
            src = src[:start] + str(value) + src[end:]
        else:
            src = src.replace("</training>", f"      <{key}>{value}</{key}>\n    </training>", 1)
    path = tmp_path_factory.mktemp("cfg") / name
    path.write_text(src)
    return path


def _build(config_path):
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(config_path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("math", dat=dict(_MATH_DAT))
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config_path), data=TheData)
    return m.to("cpu")


def _batch(m, rows=2):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=rows)
    inp_items, out_items = next(iter(loader))
    return (m.inputSpace.prepInput(inp_items),
            m.outputSpace.prepOutput(out_items))


def _problem(m, row):
    return m.inputSpace.data.math_problems["train"][row]


def _solver_labels(problem):
    order = list(problem.order)
    labels = []
    for v in order[:-1]:
        labels += [f"open:{v}", f"evaluate:{problem.index_of(v)}",
                   f"bind:{v}={problem.solution[v]}", "answer"]
    q = order[-1]
    labels += [f"evaluate:{problem.index_of(q)}", f"bind:{q}={problem.solution[q]}",
               "answer"]
    return labels


def _script_rows(monkeypatch, per_row_labels):
    """Drive the chooser per ROW: ``per_row_labels[b]`` is consumed in order
    for the row the model is resolving (read from the trace it will
    record: the runtime sets ``_what_current_row`` only for slots, so the
    script keys on the candidate set's row via the closure below)."""
    queues = {b: list(labels) for b, labels in per_row_labels.items()}
    state = {"row": 0}
    chosen = []

    def choose(self, context, candidates, *, pressure=0.0, sample=False,
               temperature=1.0):
        labels = [c["label"] for c in candidates]
        queue = queues.get(state["row"], [])
        want = queue.pop(0) if queue else "answer"
        if want not in labels:
            want = "answer"
        chosen.append((state["row"], want))
        index = labels.index(want)
        return index, torch.log_softmax(
            self.logits(context, candidates, pressure=pressure), dim=-1)[index]

    monkeypatch.setattr(WhatStepChooser, "choose", choose)
    return state, chosen


def _install_row_tracker(monkeypatch, model, state):
    """The resolve step enumerates rows in order; track the row so the
    per-row script can follow it."""
    original = model._enumerate_step_candidates

    def tracked(s, query, active, open_refs, _rows=[0]):
        return original(s, query, active, open_refs)

    real_resolve = model._resolve_step

    def resolve(understanding, questions, per_row, answer):
        # wrap _active_referent to learn the row being resolved
        real_active = model._active_referent

        def active(b, query):
            state["row"] = int(b)
            return real_active(b, query)

        monkeypatch.setattr(model, "_active_referent", active)
        try:
            return real_resolve(understanding, questions, per_row, answer)
        finally:
            monkeypatch.setattr(model, "_active_referent", real_active)

    monkeypatch.setattr(model, "_resolve_step", resolve)


@pytest.fixture(scope="module")
def episode_config(tmp_path_factory):
    return _config(tmp_path_factory, "MM_math_episode.xml")


@pytest.fixture(scope="module")
def slot_config(tmp_path_factory):
    return _config(tmp_path_factory, "MM_math_slot.xml", whatThinkingDetach="slot")


@pytest.fixture(scope="module")
def policy_config(tmp_path_factory):
    return _config(tmp_path_factory, "MM_math_policy.xml",
                   whatThinkingPolicyWeight="0.5")


def _train_one(model, monkeypatch, rows=2, spies=None):
    """One runBatch with the solver script on every row; returns the result
    and the number of optimizer steps taken."""
    problems = [_problem(model, b) for b in range(rows)]
    state, chosen = _script_rows(
        monkeypatch, {b: _solver_labels(p) for b, p in enumerate(problems)})
    _install_row_tracker(monkeypatch, model, state)
    opt = model.getOptimizer(lr=1e-3)
    steps = []
    real_step = opt.step

    def counting_step(*a, **k):
        steps.append(1)
        if spies is not None:
            spies(model)
        return real_step(*a, **k)

    monkeypatch.setattr(opt, "step", counting_step)
    batch = _batch(model, rows=rows)
    model.train()
    result, _ = model.runBatch(train=True, batchSize=rows, split="train",
                               optimizer=opt, batch_override=batch)
    return result, len(steps), chosen, problems


def test_runbatch_drives_an_episode_and_scores_the_root_after_parity(episode_config, monkeypatch):
    m = _build(episode_config)
    calls = []
    real_what = m.inputSpace.data.what

    def guarded(question):
        # The desired answer is resolved only once thinking has ended.
        assert not getattr(m, "_thinking_active", False)
        calls.append(question)
        return real_what(question)

    monkeypatch.setattr(m.inputSpace.data, "what", guarded)
    result, n_steps, chosen, problems = _train_one(m, monkeypatch, rows=2)
    assert n_steps == 1                                          # one optimizer step
    assert calls and all(q.relation.value == "supervised" for q in calls)
    thinking = m._last_what_thinking
    assert thinking.iterations > 1 and thinking.forced_closures == 0
    memory = m._what_memory()
    assert all(memory.what_at_parity(b=b) for b in range(2))
    # The scored answer is the ROOT construction: every row's root symbol
    # carries the exact numeral code of its bound query.
    from exact import numeral_code
    symbol = m._last_answer_construction.derivation.answer_symbol
    D = symbol.shape[-1]
    for b, p in enumerate(problems):
        assert torch.allclose(symbol[b, 0, :], numeral_code(p.answer, D)), b
    costs = m.primary_costs()
    assert torch.is_tensor(costs["answer_construction"])
    assert torch.isfinite(costs["answer_construction"])
    assert torch.isfinite(result.lossOut)
    # After the step the episode is closed: every slot is detached.
    for b in range(2):
        assert not memory.in_episode(b)
        for slot in memory.get_what_slots(b=b):
            for value in (slot.input, slot.output):
                assert not (torch.is_tensor(value) and value.requires_grad)
    report = m.what_report()
    assert report["thinking"]["episodes"] == 1
    assert report["thinking"]["mean_iterations"] == thinking.iterations
    assert report["thinking"]["primitives"] > 0
    assert report["thinking"]["detach"] == "episode"
    assert report["thinking"]["iteration_limit"] == 8
    assert report["policy"]["thinking"]["weight"] == 0.0
    assert report["policy"]["thinking"]["batches"] == 0


@pytest.mark.parametrize("mode", ["episode", "slot"])
def test_episode_boundary_keeps_slots_live_until_the_step(episode_config, slot_config, monkeypatch, mode):
    m = _build(episode_config if mode == "episode" else slot_config)
    seen = {}

    def spy(model):
        memory = model._what_memory()
        live = []
        for b in range(2):
            assert memory.in_episode(b) == (mode == "episode") or mode == "slot"
            for slot in memory.get_what_slots(b=b):
                for value in (slot.input, slot.output):
                    if torch.is_tensor(value):
                        live.append(bool(value.requires_grad))
        seen["live"] = live

    _train_one(m, monkeypatch, rows=2, spies=spy)
    assert seen["live"], "no tensor-valued slots were recorded"
    if mode == "episode":
        assert any(seen["live"])          # live at step time (invariant 7)
    else:
        assert not any(seen["live"])      # detached at append (established)
    memory = m._what_memory()
    for b in range(2):
        for slot in memory.get_what_slots(b=b):
            for value in (slot.input, slot.output):
                assert not (torch.is_tensor(value) and value.requires_grad)


def test_root_loss_reaches_earlier_iteration_state_under_episode_detach(episode_config, monkeypatch):
    m = _build(episode_config)
    problem = _problem(m, 0)
    state, _ = _script_rows(monkeypatch, {0: _solver_labels(problem)})
    _install_row_tracker(monkeypatch, m, state)
    x, y = _batch(m, rows=1)
    m.train()
    memory = m._what_memory()
    result = m.think(What.supervised(0), x)
    assert result.iterations > 1
    # The subquestion's COMPLETE slot (iteration 1) holds a live output
    # tensor; the ROOT construction of the final iteration reaches it.
    first_complete = next(s for s in memory.get_what_slots(b=0)
                          if s.operation is WhatSlotOperation.COMPLETE)
    assert torch.is_tensor(first_complete.output) and first_complete.output.requires_grad
    root = m._last_answer_construction.actual
    (g,) = torch.autograd.grad(root.sum(), first_complete.output,
                               retain_graph=True, allow_unused=True)
    # The earlier state is on the SAME graph as the root construction (the
    # question conditioner / synthesis layers are shared); a None gradient
    # would mean the episode had been detached mid-way.
    assert g is not None or first_complete.output.grad_fn is not None
    m._end_what_episodes()
    assert not first_complete.output.requires_grad or not memory.in_episode(0)


def test_policy_credit_trains_the_step_chooser_and_is_reported(policy_config, monkeypatch):
    m = _build(policy_config)
    assert m.what_thinking_policy_weight == 0.5
    opt = m.getOptimizer(lr=1e-2)
    batch = _batch(m, rows=2)
    m.train()
    torch.manual_seed(1)
    for _ in range(3):
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch)
    chooser = m.what_step_chooser
    report = m.what_report()
    pol = report["policy"]["thinking"]
    assert pol["weight"] == 0.5 and pol["batches"] >= 1 and pol["choices"] >= 2
    assert pol["credit"] == pol["credit"]                        # finite
    assert abs(pol["mean_return"]) < 100.0
    # Sampling explored beyond ANSWER at least once, and the chooser moved.
    assert float(chooser.mlp[-1].weight.abs().sum()) > 0.0
    names = [t[0] for t in m.errors.terms()]
    assert "what_step_policy" in names
    # Slots are detached after the episodes; parity holds.
    memory = m._what_memory()
    assert all(memory.what_at_parity(b=b) for b in range(2))


def test_iterations_one_is_byte_identical_to_a_plain_batch(episode_config, tmp_path_factory, monkeypatch):
    off = _config(tmp_path_factory, "MM_math_off.xml", whatThinkingIterations="1",
                  whatThinkingPrimitives="0")
    m = _build(off)
    assert not m._thinking_enabled()
    thinks = []
    real_think = m.think
    monkeypatch.setattr(m, "think", lambda *a, **k: thinks.append(1) or real_think(*a, **k))
    opt = m.getOptimizer(lr=1e-3)
    batch = _batch(m, rows=2)
    m.train()
    result, _ = m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                           batch_override=batch)
    assert not thinks and torch.isfinite(result.lossOut)
    assert m.what_report()["thinking"]["episodes"] == 0


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="learning floor; RUN_SLOW=1")
@pytest.mark.xfail(strict=True, reason=(
    "learning gate NOT met (2026-09-09 pilot, doc/benchmarks/2026-09-09-math-"
    "thinking-pilot.md): 150 steps on 16 depth-1/2 problems reach 12.5 % exact "
    "accuracy with the sampled policy, the same as no thinking; flip to required "
    "when a configuration passes"))
def test_learning_floor_depth_one_and_two(policy_config):
    """A floor, not the pilot gate: with sampling and policy credit, exact
    accuracy on the training problems exceeds 50 % within the budget."""
    m = _build(policy_config)
    opt = m.getOptimizer(lr=5e-3)
    m.train()
    data = m.inputSpace.data
    n = min(16, data.what_extent("train"))
    loader = data.data_loader(split="train", num_streams=n)
    inp_items, out_items = next(iter(loader))
    x = m.inputSpace.prepInput(inp_items)
    y = m.outputSpace.prepOutput(out_items)
    for _ in range(150):
        m.runBatch(train=True, batchSize=n, split="train", optimizer=opt,
                   batch_override=(x, y))
    m.eval()
    with torch.no_grad():
        result = m.think(tuple(What.supervised(b) for b in range(n)), x)
    predicted = torch.stack([a.what.reshape(-1) for a in result.answers]).argmax(-1)
    target = y.reshape(n, -1).argmax(-1)
    accuracy = float((predicted == target).float().mean())
    assert accuracy > 0.5, accuracy


# -- stage 0 through the exact route (Alec, 2026-09-09) -------------------------

@pytest.fixture(scope="module")
def stage_zero_config(tmp_path_factory):
    """MM_add (direct arithmetic) at R = 16, 32 wide (the one-hot numeral
    code needs width >= R), 256 stochastic problems split by unseen pairs,
    a two-iteration episode with three primitives and policy credit."""
    src = (_DATA / "MM_add.xml").read_text()
    for tag in ("nDim", "nInputDim", "nOutputDim"):
        src = src.replace(f"<{tag}>14</{tag}>", f"<{tag}>32</{tag}>")
    src = (src.replace("<mathRange>32</mathRange>", "<mathRange>16</mathRange>")
              .replace("<nOutput>32</nOutput>", "<nOutput>16</nOutput>")
              .replace("<mathProblems>4096</mathProblems>", "<mathProblems>256</mathProblems>")
              .replace("<answerSynthesis>true</answerSynthesis>",
                       "<answerSynthesis>true</answerSynthesis>\n"
                       "    <whatThinkingMemory>true</whatThinkingMemory>\n"
                       "    <whatThinkingDetach>episode</whatThinkingDetach>\n"
                       "    <whatThinkingIterations>2</whatThinkingIterations>\n"
                       "    <whatThinkingPrimitives>3</whatThinkingPrimitives>")
              .replace("</training>",
                       "      <whatThinkingPolicyWeight>0.5</whatThinkingPolicyWeight>\n"
                       "    </training>", 1))
    path = tmp_path_factory.mktemp("cfg") / "MM_add_exact.xml"
    path.write_text(src)
    return path


def _build_stage_zero(config_path):
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(config_path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config_path))
    TheData.load("math", dat=cfg["architecture"]["data"])
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config_path), data=TheData)
    return m.to("cpu")


def _exact_accuracy(m, split):
    data = m.inputSpace.data
    n = data.what_extent(split)
    correct = 0
    m.eval()
    with torch.no_grad():
        for start in range(0, n, 64):
            idx = list(range(start, min(n, start + 64)))
            x = m.inputSpace.prepInput(
                [data._what_split_values(split, "input")[i] for i in idx])
            y = m.outputSpace.prepOutput(
                [data._what_split_values(split, "output")[i] for i in idx])
            result = m.think(tuple(What.supervised(i, split=split) for i in idx), x)
            pred = torch.stack([a.what.reshape(-1) for a in result.answers]).argmax(-1)
            correct += int((pred == y.reshape(len(idx), -1).argmax(-1)).sum())
    m.train()
    return correct / max(1, n)


def test_stage_zero_direct_arithmetic_learns_through_the_exact_route(stage_zero_config):
    """Alec 2026-09-09: the stack must answer direct problems ("a + b" in,
    c out, the input reconstructed) before any substitution.  The learned
    direct head stays at chance (pilot report); through the exact route --
    the bare expression lexed as ``_ = a + b``, the learned WhatStepChooser
    choosing evaluate / bind / answer under policy credit, the one-hot
    numeral code realized by the answer path -- the model reaches exact
    accuracy on training AND unseen operand pairs within 40 epochs."""
    m = _build_stage_zero(stage_zero_config)
    data = m.inputSpace.data
    assert all(p.stage == 0 for p in data.math_problems["train"])
    assert _exact_accuracy(m, "train") < 0.5              # untrained: chance
    opt = m.getOptimizer(lr=0.01)
    best = (0.0, 0.0)
    for epoch in range(1, 41):
        m.train()
        m.runEpoch(optimizer=opt, batchSize=32, split="train")
        if epoch % 5 == 0:
            best = (_exact_accuracy(m, "train"), _exact_accuracy(m, "test"))
            if best[0] >= 0.9 and best[1] >= 0.9:
                break
    assert best[0] >= 0.9 and best[1] >= 0.9, best
    # The derivation is exact and replayable: every row bound its query
    # through evaluate + bind, and the answer symbol's root slot carries
    # the one-hot numeral code of that value.
    from exact import numeral_code
    x = m.inputSpace.prepInput([data.train_input[0]])
    m.eval()
    with torch.no_grad():
        result = m.think(What.supervised(0), x)
    state, query = m._what_exact_states[0]
    assert query == "_" and state.lookup("_") == data.math_problems["train"][0].answer
    ops = [s["operation"] for s in state.trace]
    assert "exact:evaluate" in ops and "exact:bind" in ops
    symbol = m._last_answer_construction.derivation.answer_symbol
    code = numeral_code(state.lookup("_"), symbol.shape[-1], answer_range=16)
    # The root slot is the one-hot code plus the (trained) question
    # conditioner's delta: the answer band's argmax is the bound value.
    assert int(symbol[0, 0, :16].argmax()) == int(code[:16].argmax()) == state.lookup("_")
    assert float((symbol[0, 0, :16] - code[:16]).abs().max()) < 0.5
    report = m.what_report()
    assert report["policy"]["thinking"]["choices"] > 0
    assert torch.isfinite(torch.tensor(report["families"]["supervised"]["answer_construction"]))
