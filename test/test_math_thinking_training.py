"""Phase 4 of the mathematical thinking plan: training THROUGH thinking
episodes on ``MM_math`` (spec 8): root scored after parity, one optimizer
step, the episode credit boundary (invariant 7), the desired answer out of
reach until loss, the step-chooser policy credit, and the report.

The runtime carries no arithmetic (Alec 2026-09-09): the resolve step
chooses ANSWER or OPEN a subquestion about a presented word, and the root
answer is conditioned on the row's LTM outputs.  Scripted choosers
exercise the mechanism; the stage-0 / stage-1 learning tests are strict
xfails until a syntactic configuration learns ``plus`` as a verb."""
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


def _build(config_path, dat=None):
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(config_path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("math", dat=dict(_MATH_DAT) if dat is None else dat)
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
    """A chain-shaped dialogue: open each chain word from the root, answer
    it, return, and finally answer the root."""
    labels = []
    for v in problem.order[:-1]:
        labels += [f"open:{v}", "answer"]
    labels.append("answer")
    return labels


def _script_rows(monkeypatch, per_row_labels):
    """Drive the chooser per ROW (``_active_referent`` is called once per
    row by the runtime; the tracker below uses it to learn the row)."""
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
    real_active = model._active_referent

    def active(b):
        state["row"] = int(b)
        return real_active(b)

    monkeypatch.setattr(model, "_active_referent", active)


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
    """One runBatch with the chain script on every row; returns the result
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
    # The scored answer is the ROOT construction of every row.
    steps = m._last_answer_construction.derivation.step
    assert all(s is None or s.role == "root" for s in steps)
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
    x, y = _batch(m, rows=1)
    m.train()
    memory = m._what_memory()
    state, _ = _script_rows(monkeypatch, {0: _solver_labels(problem)})
    _install_row_tracker(monkeypatch, m, state)
    result = m.think(What.supervised(0), x)
    assert result.iterations > 1
    first_complete = next(s for s in memory.get_what_slots(b=0)
                          if s.operation is WhatSlotOperation.COMPLETE)
    assert torch.is_tensor(first_complete.output) and first_complete.output.requires_grad
    # A non-neutral LTM attention makes the root's dependence on the stored
    # subquestion outputs visible in the graph.
    with torch.no_grad():
        m.ltm_attention["out"].weight.add_(0.1)
    m._end_what_episodes()
    memory.reset()
    state, _ = _script_rows(monkeypatch, {0: _solver_labels(problem)})
    _install_row_tracker(monkeypatch, m, state)
    result = m.think(What.supervised(0), x)
    first_complete = next(s for s in memory.get_what_slots(b=0)
                          if s.operation is WhatSlotOperation.COMPLETE)
    root = m._last_answer_construction.actual
    (g,) = torch.autograd.grad(root.sum(), first_complete.output,
                               retain_graph=True, allow_unused=True)
    assert g is not None and torch.isfinite(g).all()     # the root reaches iteration 1
    m._end_what_episodes()
    assert not memory.in_episode(0)


def test_policy_credit_trains_the_step_chooser_and_is_reported(policy_config):
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
    assert float(chooser.mlp[-1].weight.abs().sum()) > 0.0
    names = [t[0] for t in m.errors.terms()]
    assert "what_step_policy" in names
    memory = m._what_memory()
    assert all(memory.what_at_parity(b=b) for b in range(2))


def test_iterations_one_is_byte_identical_to_a_plain_batch(episode_config, tmp_path_factory, monkeypatch):
    off = _config(tmp_path_factory, "MM_math_off.xml", whatThinkingIterations="1")
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


def test_primitives_knob_is_retired(tmp_path_factory):
    """<whatThinkingPrimitives> is gone from the schema (and the model
    raises 'retired' should a config slip past validation)."""
    path = _config(tmp_path_factory, "MM_math_prims.xml", whatThinkingPrimitives="4")
    with pytest.raises(Exception, match="whatThinkingPrimitives"):
        _build(path)


# -- the learning gates on the syntactic route (strict xfail) --------------------

def _exact_accuracy(m, split, limit=128):
    data = m.inputSpace.data
    n = min(limit, data.what_extent(split))
    correct = 0
    m.eval()
    with torch.no_grad():
        for start in range(0, n, 32):
            idx = list(range(start, min(n, start + 32)))
            x = m.inputSpace.prepInput(
                [data._what_split_values(split, "input")[i] for i in idx])
            y = m.outputSpace.prepOutput(
                [data._what_split_values(split, "output")[i] for i in idx])
            result = m.think(tuple(What.supervised(i, split=split) for i in idx), x)
            pred = torch.stack([a.what.reshape(-1) for a in result.answers]).argmax(-1)
            correct += int((pred == y.reshape(len(idx), -1).argmax(-1)).sum())
    m.train()
    return correct / max(1, n)


def _learn(config_path, dat, epochs, lr=0.01):
    m = _build(config_path, dat=dat)
    assert _exact_accuracy(m, "train") < 0.5
    opt = m.getOptimizer(lr=lr)
    best = (0.0, 0.0)
    for epoch in range(1, epochs + 1):
        m.train()
        m.runEpoch(optimizer=opt, batchSize=32, split="train")
        if epoch % 5 == 0:
            best = (_exact_accuracy(m, "train"), _exact_accuracy(m, "test"))
            if min(best) >= 0.9:
                break
    return best


@pytest.fixture(scope="module")
def stage_zero_config(tmp_path_factory):
    """MM_add (direct arithmetic, ``3 plus 4`` -> 7) at R = 16, 32 wide,
    256 stochastic problems split by unseen pairs, a two-iteration
    episode with policy credit."""
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
                       "    <whatThinkingIterations>2</whatThinkingIterations>")
              .replace("</training>",
                       "      <whatThinkingPolicyWeight>0.5</whatThinkingPolicyWeight>\n"
                       "    </training>", 1))
    path = tmp_path_factory.mktemp("cfg") / "MM_add_syntax.xml"
    path.write_text(src)
    return path


@pytest.mark.xfail(strict=True, reason=(
    "learning gate (Alec 2026-09-09: no mathematical machinery in the runtime; "
    "plus is a transitive verb the grammar learns): the small syntactic "
    "configuration does not yet learn direct arithmetic -- the large stage-0 "
    "run is the experiment; flip to required when a configuration passes"))
def test_stage_zero_direct_arithmetic_learns_as_syntax(stage_zero_config):
    """Alec 2026-09-09: the stack must answer direct problems ("3 plus 4"
    in, 7 out, the input reconstructed) before any substitution, with
    ``plus`` learned as a transitive verb over numeral nouns.  Gate: at
    least 90 % exact accuracy on training AND unseen operand pairs within
    40 epochs of the small configuration."""
    best = _learn(stage_zero_config, {"mathStage": 0, "mathRange": 16,
                                      "mathProblems": 256, "mathSeed": 0}, 40)
    assert min(best) >= 0.9, best


@pytest.fixture(scope="module")
def stage_one_config(tmp_path_factory):
    """MM_math (dependency chains, depths 1-2, depth 3 held out) at
    R = 16, 32 wide, 384 problems, eight-iteration episodes with policy
    credit; codebooks sized for the corpus."""
    src = (_DATA / "MM_math.xml").read_text()
    for tag in ("nDim", "nInputDim", "nOutputDim"):
        src = src.replace(f"<{tag}>14</{tag}>", f"<{tag}>32</{tag}>")
    src = (src.replace("<mathProblems>64</mathProblems>", "<mathProblems>384</mathProblems>")
              .replace("</training>",
                       "      <whatThinkingPolicyWeight>0.5</whatThinkingPolicyWeight>\n"
                       "    </training>", 1)
              .replace("<nVectors>8</nVectors>\n    <nDim>", "<nVectors>512</nVectors>\n    <nDim>", 1)
              .replace("<nVectors>128</nVectors>", "<nVectors>2048</nVectors>")
              .replace("<nVectors>200</nVectors>", "<nVectors>2048</nVectors>"))
    path = tmp_path_factory.mktemp("cfg") / "MM_math_syntax.xml"
    path.write_text(src)
    return path


@pytest.mark.xfail(strict=True, reason=(
    "learning gate (Alec 2026-09-09: no mathematical machinery in the runtime): "
    "dependency chains must be resolved by subquestions whose answers live in "
    "LTM and by the grammar's verbs; not yet learned -- flip when a "
    "configuration passes"))
def test_stage_one_dependency_chains_learn_as_syntax(stage_one_config):
    """The original variable-substitution problems (spec section 1, stage
    1) on the syntactic route: subquestions about presented words, their
    answers in LTM, the root conditioned on them.  Gate: at least 90 %
    exact accuracy on training and on unseen structures / the held-out
    depth within 30 epochs of the small configuration."""
    best = _learn(stage_one_config, {"mathRange": 16, "mathDepths": "1,2",
                                     "mathTestDepths": "3", "mathDistractors": 1,
                                     "mathProblems": 384, "mathSeed": 0}, 30)
    assert min(best) >= 0.9, best


# -- the first learned rung: the successor as a verb (RUN_SLOW) -----------------

@pytest.fixture(scope="module")
def successor_config(tmp_path_factory):
    """``MM_add_verb`` (the serial verb grammar, 230M parameters) on the
    successor corpus: ``n plus one`` -> the next numeral, single-digit
    facts only (R = 10), 256 presentations per epoch."""
    src = (_DATA / "MM_add_verb.xml").read_text()
    src = (src.replace("<mathRange>16</mathRange>", "<mathRange>10</mathRange>")
              .replace("<mathProblems>2048</mathProblems>", "<mathProblems>256</mathProblems>")
              .replace("<mathSeed>0</mathSeed>",
                       "<mathSeed>0</mathSeed>\n      <mathOperators>succ</mathOperators>"))
    assert "<mathOperators>succ</mathOperators>" in src
    path = tmp_path_factory.mktemp("cfg") / "MM_succ_verb.xml"
    path.write_text(src)
    return path


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="RUN_SLOW: ~20 min on CPU")
def test_successor_is_learned_as_a_verb(successor_config):
    """Alec 2026-09-09: addition is iterated succession, and the successor
    must be learnable by the existing VP.  On the verb grammar with the
    answer path seeded from the root idea, every single-digit fact
    ``n plus one`` is answered (pilot report, "The successor is learned as
    a VP").  Gate: at least 90 % exact accuracy on the presented facts
    within 30 epochs at lr 1e-3 (two-digit numerals are excluded: they
    are read as their digit parts until the multi-digit rung lands)."""
    m = _build(successor_config, dat={"mathStage": 0, "mathRange": 10,
                                      "mathProblems": 256, "mathSeed": 0,
                                      "mathOperators": "succ"})
    assert _exact_accuracy(m, "train") < 0.5
    opt = m.getOptimizer(lr=1e-3)
    best = 0.0
    for epoch in range(1, 31):
        m.train()
        m.runEpoch(optimizer=opt, batchSize=8, split="train")
        if epoch % 5 == 0:
            best = max(best, _exact_accuracy(m, "train", limit=256))
            if best >= 0.9:
                break
    assert best >= 0.9, best
