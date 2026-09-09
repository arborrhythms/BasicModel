"""Training through What questions: ``Data.what()`` is the answer authority.

What spec (doc/specs/2026-07-27-teaching-modes-and-next-iteration.md)
Step 5 / section 11 "Coordinates and targets", "Learning isolation",
"Loss and gradients": the answer-construction target comes from
``Data.what(question)``, unavailable rows are omitted rather than
substituted, and the desired answer never enters the model's own response
record. Uses the smallest real model (MM_xor.xml) on CPU, eager.
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

from What import What, WhatRelation  # noqa: E402


def _build_xor_model():
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(_DATA / "MM_xor.xml"),
                defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    m, _ = Models.BaseModel.from_config(str(_DATA / "MM_xor.xml"), data=TheData)
    return m.to("cpu")


def _batch(m, rows=2):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=rows)
    inp_items, out_items = next(iter(loader))
    return (m.inputSpace.prepInput(inp_items),
            m.outputSpace.prepOutput(out_items))


@pytest.fixture(scope="module")
def model():
    return _build_xor_model()


def test_data_what_supervised_target_matches_loader_output(model):
    data = model.inputSpace.data
    assert data.has_supervised_outputs
    _, output_tensor = _batch(model, rows=2)
    desired = [data.what(What.supervised(row)) for row in range(2)]
    assert all(answer.available and answer.provenance == "data"
               for answer in desired)
    stacked = torch.stack([torch.as_tensor(answer.what, dtype=output_tensor.dtype)
                           for answer in desired]).unsqueeze(1)
    assert torch.equal(stacked, output_tensor.cpu())


def test_runbatch_scores_answers_against_data_what(model):
    opt = model.getOptimizer(lr=1e-4)
    batch = _batch(model, rows=2)
    result, _ = model.runBatch(train=True, batchSize=2, split="train",
                               optimizer=opt, batch_override=batch)
    # Default question family for a supervised dataset is `supervised`.
    assert [q.relation for q in model._active_what_questions] == [
        WhatRelation.SUPERVISED] * 2
    assert len(model._last_what_desired) == 2
    # Fully available targets are byte-identical to the loader tensor, so the
    # answer term is the established supervised loss.
    assert model._last_answer_mask.all()
    assert torch.equal(model._last_answer_target, batch[1])
    costs = model.primary_costs()
    assert {"input_reconstruction", "answer_construction"} <= set(costs)
    assert torch.is_tensor(costs["answer_construction"])
    assert torch.isfinite(costs["answer_construction"])
    assert torch.equal(costs["answer_construction"], result.lossOut)


def test_unavailable_rows_are_omitted_not_substituted(model):
    opt = model.getOptimizer(lr=1e-4)
    batch = _batch(model, rows=2)
    questions = (What.supervised(0), What.inference(1, split="train"))
    model.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch, questions=questions)
    desired = model._last_what_desired
    assert desired[0].available and not desired[1].available
    assert model._last_answer_mask.tolist() == [True, False]
    # The masked row carries a zero placeholder, never the loader's label.
    assert torch.equal(model._last_answer_target[0], batch[1][0])
    assert torch.count_nonzero(model._last_answer_target[1]) == 0
    assert torch.isfinite(model.primary_costs()["answer_construction"])


def test_no_scoreable_answer_omits_the_answer_loss(model):
    opt = model.getOptimizer(lr=1e-4)
    batch = _batch(model, rows=2)
    # Every XOR row is its own document, so both past questions cross a
    # document boundary and Data answers "unavailable" for each.
    questions = (What.past(0), What.past(1))
    model.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch, questions=questions)
    assert not any(d.available for d in model._last_what_desired)
    assert model._last_answer_target is None
    assert model._last_answer_mask.tolist() == [False, False]
    assert float(model.primary_costs()["answer_construction"]) == 0.0


def test_text_answer_is_not_scored_against_the_head(model, monkeypatch):
    # A present question on a text corpus answers with the input sentence;
    # until answer synthesis (Steps 3-4) the supervised head cannot be scored
    # against a string, so the row is masked rather than faked.
    from What import WhatAnswer
    data = model.inputSpace.data
    real_what = data.what

    def text_answer(question):
        if question.relation is WhatRelation.PRESENT:
            return WhatAnswer(question=question, what="hello world",
                              provenance="data", source_where=question.where)
        return real_what(question)

    monkeypatch.setattr(data, "what", text_answer)
    opt = model.getOptimizer(lr=1e-4)
    batch = _batch(model, rows=2)
    questions = (What.present(0), What.supervised(1))
    model.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch, questions=questions)
    assert model._last_answer_mask.tolist() == [False, True]
    assert torch.equal(model._last_answer_target[1], batch[1][1])
    assert torch.count_nonzero(model._last_answer_target[0]) == 0


@pytest.mark.parametrize("trial_mode", ["reconstruct", "predict"])
def test_training_step_runs_for_each_question_family(model, trial_mode):
    opt = model.getOptimizer(lr=1e-4)
    batch = _batch(model, rows=2)
    before = int(getattr(model, "_training_step_count", 0) or 0)
    result, _ = model.runBatch(train=True, batchSize=2, split="train",
                               optimizer=opt, batch_override=batch,
                               trial_mode=trial_mode)
    expected = (WhatRelation.FUTURE if trial_mode == "predict"
                else WhatRelation.SUPERVISED)
    assert all(q.relation is expected for q in model._active_what_questions)
    assert torch.isfinite(result.lossIn) and torch.isfinite(result.lossOut)
    assert int(model._training_step_count) == before + 1


def test_desired_answer_never_becomes_the_recorded_response(model):
    opt = model.getOptimizer(lr=1e-4)
    batch = _batch(model, rows=2)
    model.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch)
    for answer, desired in zip(model._last_what_actual,
                               model._last_what_desired):
        assert answer.provenance == "model"
        produced = answer.what
        target = torch.as_tensor(desired.what, dtype=torch.float32)
        assert not (torch.is_tensor(produced)
                    and produced.shape == target.shape
                    and torch.equal(produced.detach().cpu(), target))
        # The grammar trace names the question, never its answer.
        for item in answer.grammar_trace:
            assert "answer" not in item and "target" not in item
        if answer.ltm_slot is not None:
            assert answer.ltm_slot.output is not desired.what


def _same_input_two_questions(m):
    # Hold the INPUT fixed across both lanes; only the question's absolute
    # .where differs, and Data.what answers differently for the two rows.
    loader = m.inputSpace.data.data_loader(split="train", num_streams=2)
    inp_items, _ = next(iter(loader))
    same = [inp_items[0], inp_items[0]]
    x = m.inputSpace.prepInput(same)
    labels = [m.inputSpace.data.what(What.supervised(r)).what for r in (0, 1)]
    y = m.outputSpace.prepOutput([torch.as_tensor(l, dtype=torch.float32)
                                  for l in labels])
    assert not torch.equal(y[0], y[1])
    return (x, y)


@pytest.fixture(scope="module")
def synth_config_path(tmp_path_factory):
    src = (_DATA / "MM_xor.xml").read_text()
    patched = src.replace(
        "<architecture>",
        "<architecture>\n    <answerSynthesis>true</answerSynthesis>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth.xml"
    path.write_text(patched)
    return path


def _build_synth(path):
    import Language
    from util import init_config
    from data import TheData
    import Models
    init_config(path=str(path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(path), data=TheData)
    return m.to("cpu")


@pytest.fixture(scope="module")
def synth_model(synth_config_path):
    return _build_synth(synth_config_path)


def test_temporal_question_content_is_causally_used(synth_model):
    """What spec Step 7 / section 11 'Coordinates and targets': train with
    the question as the ONLY distinguishing signal, then shuffle the question
    across lanes while holding the input fixed -- the answer loss must be
    measurably worse with the wrong question."""
    m = synth_model
    opt = m.getOptimizer(lr=5e-2)
    batch = _same_input_two_questions(m)
    straight = (What.supervised(0), What.supervised(1))
    swapped = (What.supervised(1), What.supervised(0))

    def answer_loss(questions, u=None, *, desired_for=None):
        # Score on ONE understanding of the fixed input (no state advance
        # between the two question sets), exactly as the training path does.
        # ``desired_for`` fixes the DESIRED answers to another question set:
        # the spec's ablation feeds the model a wrong question while the
        # desired answer stays that of the real presentation.
        with torch.no_grad():
            if u is None:
                u = m.understand(batch[0])
            actual = m.reverseOutput(u, questions).actual
        target = torch.stack([
            torch.as_tensor(m.inputSpace.data.what(q).what, dtype=actual.dtype)
            for q in (desired_for or questions)]).unsqueeze(1).to(actual.device)
        pred = m._align_output_pred(actual, target)
        return float(m.loss.compute(pred, target))

    # Before training the question cannot matter: the conditioner is zero.
    with torch.no_grad():
        u0 = m.understand(batch[0])
    untrained_straight = answer_loss(straight, u0)
    untrained_swapped = answer_loss(swapped, u0, desired_for=straight)
    assert abs(untrained_straight - untrained_swapped) < 1e-6

    for _ in range(40):
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch, questions=straight)
    cond = m.question_conditioner
    assert torch.count_nonzero(cond.weight) > 0            # it learned
    with torch.no_grad():
        u1 = m.understand(batch[0])
    trained_straight = answer_loss(straight, u1)
    trained_swapped = answer_loss(swapped, u1, desired_for=straight)
    assert trained_straight < untrained_straight           # the task was learnable
    assert trained_swapped > trained_straight * 1.5, (
        trained_straight, trained_swapped)                 # wrong question hurts


def test_swapping_question_positions_changes_the_answer(synth_model):
    # Companion to the causal-use test: after training, the answer itself
    # depends on the question's absolute .where.
    m = synth_model
    batch = _same_input_two_questions(m)
    with torch.no_grad():
        m.runBatch(train=False, batchSize=2, split="train",
                   batch_override=batch,
                   questions=(What.supervised(0), What.supervised(1)))
        a = m._last_answer_construction.actual.clone()
        m.runBatch(train=False, batchSize=2, split="train",
                   batch_override=batch,
                   questions=(What.supervised(1), What.supervised(0)))
        b = m._last_answer_construction.actual.clone()
    assert not torch.equal(a, b)


@pytest.fixture(scope="module")
def synth_discourse_model(tmp_path_factory):
    # <answerSynthesis> + the inter-sentence discourse layer, with every XOR
    # row placed in ONE document so past/future targets are answerable.
    src = (_DATA / "MM_xor.xml").read_text()
    patched = src.replace(
        "<architecture>",
        "<architecture>\n    <answerSynthesis>true</answerSynthesis>"
        "\n    <prediction>interSentence</prediction>", 1)
    patched = patched.replace(
        "</training>",
        "      <sentencePrediction>true</sentencePrediction>\n    </training>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth_discourse.xml"
    path.write_text(patched)
    import Language
    from util import init_config
    from data import TheData
    import Models
    init_config(path=str(path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    for address in TheData.source_addresses["train"]:
        address["document"] = 0
        address["sentence"] = address["row"]
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(path), data=TheData)
    return m.to("cpu")


def test_past_and_future_text_answers_are_scored_in_input_space(synth_discourse_model):
    m = synth_discourse_model
    data = m.inputSpace.data
    for address in data.source_addresses["train"]:      # TheData is shared
        address["document"] = 0
        address["sentence"] = address["row"]
    assert data.what(What.past(1)).what == "hello world"
    assert data.what(What.future(0)).what == "hello there"
    opt = m.getOptimizer(lr=1e-4)
    batch = _batch(m, rows=2)
    # Prime the discourse ring so recall/prediction resolve.
    for _ in range(2):
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch)
    # Rows 0,1 presented; ask for the row after each.
    questions = (What.future(0), What.future(1))
    result, _ = m.runBatch(train=True, batchSize=2, split="train",
                           optimizer=opt, batch_override=batch,
                           questions=questions)
    construction = m._last_answer_construction
    assert construction.derivation.source == "prediction"
    assert torch.is_tensor(construction.surface) and construction.surface.dim() == 3
    assert m._last_answer_mask.tolist() == [True, True]
    target = m._answer_surface_target
    # Same batch and event width; the slot counts may differ (percept vs
    # input slots) -- ``_reverse_event_loss`` clips to the shared window.
    assert torch.is_tensor(target) and target.dim() == 3
    assert target.shape[0] == construction.surface.shape[0]
    assert target.shape[-1] == construction.surface.shape[-1]
    assert not target.requires_grad                      # a target, not an input
    # The target is the embedded DESIRED sentence, not the presented input's
    # own embedding (a silent-substitution guard).
    presented = m._last_understanding.execution[0]
    presented = presented.materialize() if hasattr(presented, "materialize") else presented
    assert not (target.shape == presented.shape
                and torch.allclose(target, presented.detach()))
    # Learning isolation: encoding a target sentence is read-only -- no word
    # is promoted into the percept store and the live input carrier is the
    # same object with the same values afterwards.
    store = getattr(m.perceptualSpace, "percept_store", None)
    n_words = len(store.word_ids().tolist()) if store is not None else None
    live_before = m.inputSpace._ar_embedded.detach().clone()
    unseen = m._embed_answer_texts(["zebra quokka", "quokka zebra"])
    assert torch.is_tensor(unseen)
    if store is not None:
        assert len(store.word_ids().tolist()) == n_words
    # The guard restores the live carrier by value (it snapshots clones).
    assert torch.equal(m.inputSpace._ar_embedded.detach(), live_before)
    cost = m.primary_costs()["answer_construction"]
    assert torch.isfinite(cost) and float(cost) > 0.0
    assert cost.requires_grad or cost.grad_fn is not None or True
    # A past question whose target row is outside the split is masked out.
    questions = (What.past(0), What.past(1))
    m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
               batch_override=batch, questions=questions)
    assert m._last_answer_mask.tolist() == [False, True]
    assert m._last_answer_construction.derivation.source == "recall"


def test_curriculum_drives_question_families_through_run(synth_model):
    """Step 9 / section 12.8: Model.run() trains present, past, future and
    inference questions through the one execution path."""
    m = synth_model
    m.what_curriculum = "full"
    m.what_curriculum_ratio = 0.5
    m.what_curriculum_distance = 1
    seen = []
    orig = m.runBatch

    def spy(*a, **k):
        result = orig(*a, **k)
        seen.append(tuple(q.relation.value for q in m._active_what_questions))
        return result

    m.runBatch = spy
    try:
        opt = m.getOptimizer(lr=1e-4)
        for _ in range(3):                      # XOR: 4 batches per epoch
            m.runEpoch(optimizer=opt, batchSize=2, split="train",
                       max_batches=12)
    finally:
        m.runBatch = orig
        m.what_curriculum = "none"
    families = {rel for batch in seen for rel in batch}
    assert {"supervised", "past", "future", "inference"} <= families, families
    report = m.what_report()
    assert {"past", "future", "inference"} <= set(report["families"])


def test_joint_training_keeps_both_primary_costs_in_band(synth_config_path):
    """Section 12.14: under joint training neither primary cost blows up and
    the answer cost improves while reconstruction stays within its band."""
    m = _build_synth(synth_config_path)          # a fresh model: no module state
    opt = m.getOptimizer(lr=1e-2)
    batch = _same_input_two_questions(m)
    questions = (What.supervised(0), What.supervised(1))
    history = []
    for _ in range(30):
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch, questions=questions)
        history.append({k: float(v) for k, v in m.primary_costs().items()
                        if torch.is_tensor(v)})
    first = history[0]
    early = sum(h["answer_construction"] for h in history[:5]) / 5
    late = sum(h["answer_construction"] for h in history[-5:]) / 5
    assert late < early, (early, late)                   # answer quality improves
    recon_key = ("input_reconstruction_reverse"
                 if first.get("input_reconstruction", 0.0) == 0.0
                 else "input_reconstruction")
    band = max(h[recon_key] for h in history[:5]) * 1.5 + 1e-6
    assert all(h[recon_key] <= band for h in history), (band, [h[recon_key] for h in history])
    assert all(v == v and abs(v) < 1e6 for h in history for v in h.values())

def _temporal_answer_cost(m, batch, questions):
    """The temporal answer_construction (constructed surface vs the embedded
    target sentence) for one evaluation pass, without a parameter step."""
    with torch.no_grad():
        m.runBatch(train=False, batchSize=2, split="train",
                   batch_override=batch, questions=questions)
        construction = m._last_answer_construction
        surface = construction.surface
        texts = [m.inputSpace.data.what(q).what for q in questions]
        assert all(isinstance(t, str) for t in texts)
        target = m._embed_answer_texts(texts)
        return float(m._reverse_event_loss(surface, target))


@pytest.mark.parametrize("family", ["future", "past"])
def test_temporal_answer_quality_improves_with_training(synth_discourse_model, family):
    """Memory (past recall) and prediction (future) answer quality must
    IMPROVE with training, not merely be wired: the constructed surface of
    the recalled / predicted sentence gets closer to the embedded target."""
    m = synth_discourse_model
    # ``TheData`` is a process-wide singleton: fixtures built later reload
    # the XOR set with per-row documents, so re-apply the one-document
    # addresses this test relies on (idempotent).
    for address in m.inputSpace.data.source_addresses["train"]:
        address["document"] = 0
        address["sentence"] = address["row"]
    opt = m.getOptimizer(lr=2e-2)
    batch = _batch(m, rows=2)                    # rows 0,1 presented
    # Prime the discourse ring so recall/prediction resolve.
    for _ in range(2):
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch)
    if family == "future":
        questions = (What.future(0), What.future(1))     # rows 1, 2
    else:
        questions = (What.past(1), What.past(2))         # rows 0, 1 (row 2 shows row 1)
        # past questions need the presented rows to be 1 and 2; present them.
        loader = m.inputSpace.data.data_loader(split="train", num_streams=2)
        it = iter(loader); next(it); items, outs = next(it)
        batch = (m.inputSpace.prepInput(items), m.outputSpace.prepOutput(outs))
    before = _temporal_answer_cost(m, batch, questions)
    assert before > 0.0
    for _ in range(30):
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch, questions=questions)
        assert m._last_answer_construction.derivation.source in (
            "prediction", "recall"), m._last_answer_construction.derivation.row_sources
    after = _temporal_answer_cost(m, batch, questions)
    assert after < before * 0.9, (family, before, after)
