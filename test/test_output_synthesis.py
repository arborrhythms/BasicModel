"""What spec Steps 3-4: the separate ``Model.reverseOutput()`` path.

Section 11 "Reconstruction and answer paths": a generated answer starts at
the resolved symbol and traverses conceptual synthesis, perceptual synthesis
and ``OutputSpace``; reconstruction does not consume an answer carrier and
output synthesis does not consume a reconstruction carrier; either path can
run first with identical results; the compatibility path is byte-identical
while ``<answerSynthesis>`` is off.
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

from Output import AnswerConstruction, AnswerDerivation  # noqa: E402
from Understanding import Understanding  # noqa: E402
from What import What  # noqa: E402


def _build(config_path):
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(config_path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config_path), data=TheData)
    return m.to("cpu")


def _batch(m, rows=2):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=rows)
    inp_items, out_items = next(iter(loader))
    return (m.inputSpace.prepInput(inp_items),
            m.outputSpace.prepOutput(out_items))


@pytest.fixture(scope="module")
def synth_config(tmp_path_factory):
    src = (_DATA / "MM_xor.xml").read_text()
    assert src.count("<architecture>") == 1
    patched = src.replace(
        "<architecture>",
        "<architecture>\n    <answerSynthesis>true</answerSynthesis>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth.xml"
    path.write_text(patched)
    return path


@pytest.fixture(scope="module")
def synth_discourse_config(tmp_path_factory):
    # Same, plus the inter-sentence discourse layer (ARMA ring + predictor)
    # that past recall and future prediction resolve through.
    src = (_DATA / "MM_xor.xml").read_text()
    assert "<prediction>" not in src and "<sentencePrediction>" not in src
    patched = src.replace(
        "<architecture>",
        "<architecture>\n    <answerSynthesis>true</answerSynthesis>"
        "\n    <prediction>interSentence</prediction>", 1)
    patched = patched.replace(
        "</training>",
        "      <sentencePrediction>true</sentencePrediction>\n    </training>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth_discourse.xml"
    path.write_text(patched)
    return path


def test_gate_off_keeps_the_projected_head_and_state_dict(model_off=None):
    m = _build(_DATA / "MM_xor.xml")
    assert not m.answer_synthesis
    x, _ = _batch(m)
    with torch.no_grad():
        answer = m.what(What.supervised(0), input_data=x)
    assert answer.what is answer.execution[2]
    assert m._last_answer_construction is None
    assert not any(k.endswith("percept_adapter.W") or "percept_adapter" in k
                   for k in m.state_dict())


def test_output_traverses_symbol_concepts_percepts_output(synth_config):
    m = _build(synth_config)
    assert m.answer_synthesis
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        construction = m.reverseOutput(u, What.supervised(0))
    assert isinstance(construction, AnswerConstruction)
    assert isinstance(construction.derivation, AnswerDerivation)
    assert torch.equal(construction.derivation.answer_symbol, u.symbolic_state)
    B = x.shape[0]
    assert construction.concepts.dim() == 3 and construction.concepts.shape[0] == B
    assert construction.percepts.dim() == 3 and construction.percepts.shape[0] == B
    assert tuple(construction.actual.shape) == (
        B, int(m.outputSpace.outputShape[0]), int(m.outputSpace.outputShape[1]))
    ops = [item["operation"] for item in construction.trace]
    assert ops == ["resolve:identity", "synthesize:conceptual",
                   "synthesize:perceptual", "output:from_percepts"]
    # The adapter now exists and is registered (checkpointable).
    assert any("percept_adapter" in k for k in m.state_dict())


def test_what_reaches_output_not_the_legacy_head(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    with torch.no_grad():
        answer = m.what(What.supervised(0), input_data=x)
    head = answer.execution[2]
    assert answer.what is m._last_answer_construction.actual
    assert answer.what is not head
    assert tuple(answer.what.shape) == tuple(head.shape)


def test_reconstruction_and_output_are_order_independent(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    q = What.supervised(0)
    with torch.no_grad():
        u = m.understand(x)
        rev_first, _ = m.reverseReconstruct(u)
        out_after = m.reverseOutput(u, q).actual
        rev_again, _ = m.reverseReconstruct(u)
    with torch.no_grad():
        u2 = m.understand(x)
        out_first = m.reverseOutput(u2, q).actual
        rev_after, _ = m.reverseReconstruct(u2)
        out_again = m.reverseOutput(u2, q).actual
    assert torch.equal(rev_first, rev_again)        # output did not disturb reconstruction
    assert torch.equal(out_first, out_again)        # reconstruction did not disturb output
    assert torch.allclose(rev_first, rev_after)     # same understanding, either order
    assert torch.allclose(out_first, out_after)


def test_output_does_not_consume_reconstruction_carriers(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    q = What.supervised(0)
    with torch.no_grad():
        u = m.understand(x)
        baseline = m.reverseOutput(u, q).actual
        # Strip every reconstruction-only carrier from the understanding and
        # the live stages; the answer must not change.
        stripped = Understanding(
            perceptual_context=u.perceptual_context,
            conceptual_state=u.conceptual_state,
            symbolic_state=u.symbolic_state,
            reconstruction_carriers={"ir_mask_positions": None,
                                     "terminal_idea": None,
                                     "combine_last_cs_sub": None,
                                     "merge_diffs": None},
            execution=u.execution)
        for stage in m.body_stages:
            if "merge" in stage:
                stage["merge"]._merge_diff = None
        object.__setattr__(m, "_combine_carriers", None)
        again = m.reverseOutput(stripped, q).actual
    assert torch.equal(baseline, again)


def test_training_step_scores_the_constructed_answer(synth_config):
    m = _build(synth_config)
    opt = m.getOptimizer(lr=1e-3)
    batch = _batch(m)
    result, _ = m.runBatch(train=True, batchSize=2, split="train",
                           optimizer=opt, batch_override=batch)
    construction = m._last_answer_construction
    assert construction is not None
    assert result.outputPred is construction.actual
    costs = m.primary_costs()
    assert torch.isfinite(costs["answer_construction"])
    adapter = m.outputSpace.percept_adapter
    grads = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert grads and any(torch.count_nonzero(g) > 0 for g in grads)


def test_think_constructs_the_root_answer_through_output(synth_config):
    # Step 8: the final root response is built by the same reverseOutput path.
    m = _build(synth_config)
    x, _ = _batch(m)
    with torch.no_grad():
        result = m.think(What.supervised(0, prompt="what is your name?"), x,
                         max_iterations=3)
    construction = m._last_answer_construction
    assert construction is not None
    assert result.answer.available
    assert result.answer.what is construction.actual
    assert construction.trace[-1]["operation"] == "output:from_percepts"


def test_branch_gradient_diagnostics_read_without_update(synth_config):
    # Step 6: norms and cosine at the branch points, computed from the two
    # primary costs before backward, and the optimizer step is unchanged.
    m = _build(synth_config)
    m.branch_diagnostics_every = 1
    opt = m.getOptimizer(lr=1e-3)
    batch = _batch(m)
    before = {k: v.detach().clone() for k, v in m.named_parameters()}
    m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
               batch_override=batch)
    report = m._last_branch_diagnostics
    assert set(report) == {"conceptual", "symbolic"}
    for entry in report.values():
        assert set(entry) == {"reconstruction_norm", "answer_norm", "cosine"}
    assert any(entry["answer_norm"] is not None for entry in report.values())
    # Diagnostics are a read: a second model without sampling takes the same
    # step from the same state.
    m2 = _build(synth_config)
    m2.branch_diagnostics_every = 0
    opt2 = m2.getOptimizer(lr=1e-3)
    m2.runBatch(train=True, batchSize=2, split="train", optimizer=opt2,
                batch_override=_batch(m2))
    after = dict(m.named_parameters())
    after2 = dict(m2.named_parameters())
    for k in after:
        if k in after2:
            assert torch.allclose(after[k], after2[k]), k


def _prime_discourse(m, batch, sentences=2):
    # The discourse layer observes a sentence rep on TRAINING batches only,
    # so prime the ARMA ring with a couple of training steps.
    opt = m.getOptimizer(lr=1e-6)
    for _ in range(sentences):
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch)


def test_present_and_supervised_resolve_by_identity(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        for q in (What.present(1), What.supervised(1),
                  What.inference(1, split="train")):
            d = m._resolve_answer(u, q)
            assert d.source == "identity" and d.resolved
            # The zero-initialised question conditioner leaves the symbol
            # numerically identical (a clone, so the understanding is never
            # mutated in place).
            assert torch.equal(d.answer_symbol, u.symbolic_state)
            assert d.answer_symbol is not u.symbolic_state


def test_past_resolves_by_recall_from_discourse_memory(synth_discourse_config):
    m = _build(synth_discourse_config)
    batch = _batch(m)
    memory = m._what_memory()
    assert memory is not None and getattr(memory, "_s_history", None) is not None
    _prime_discourse(m, batch, sentences=2)
    with torch.no_grad():
        u = m.understand(batch[0])
        # Priming trained the question conditioner a little; silence it so
        # the root slot is exactly the recalled rep.
        cond = getattr(m, "question_conditioner", None)
        if cond is not None:
            cond.weight.zero_()
        d = m._resolve_answer(u, What.past(1, -1))
    assert d.source == "recall" and d.resolved
    assert d.answer_symbol is not u.symbolic_state
    assert d.answer_symbol.shape == u.symbolic_state.shape
    # Recall is the model's chronological per-row history (newest last), NOT
    # the ARMA ring tail (whose fill-phase layout is newest-at-low-end).
    history = m._recall_history()
    width = min(history[0][-1].shape[-1], u.symbolic_state.shape[-1])
    for b in range(u.symbolic_state.shape[0]):
        assert torch.equal(d.answer_symbol[b, 0, :width], history[b][-1][:width])
    assert torch.equal(d.answer_symbol[:, 1:], u.symbolic_state[:, 1:])
    # Deeper than the history -> cold, unresolved identity.
    far = m._resolve_answer(u, What.past(9, -(len(history[0]) + 1)))
    assert far.source == "identity:cold-memory" and not far.resolved


def test_future_resolves_by_prediction_without_committing_memory(synth_discourse_config):
    m = _build(synth_discourse_config)
    batch = _batch(m)
    memory = m._what_memory()
    assert memory is not None and getattr(memory, "predictor", None) is not None
    _prime_discourse(m, batch, sentences=2)
    before = memory._s_history.clone()
    with torch.no_grad():
        u = m.understand(batch[0])
        d1 = m._resolve_answer(u, What.future(1, 1))
        d2 = m._resolve_answer(u, What.future(1, 2))
    assert d1.source == "prediction" and d1.resolved
    assert torch.equal(memory._s_history, before)          # ring untouched
    assert not torch.equal(d1.answer_symbol[:, 0], d2.answer_symbol[:, 0])
    with torch.no_grad():
        construction = m.reverseOutput(u, What.future(1, 1))
    assert construction.trace[0]["operation"] == "resolve:prediction"


def test_cold_memory_resolution_is_flagged_unresolved(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        d = m._resolve_answer(u, What.past(3, -1))
    assert not d.resolved and d.source == "identity:cold-memory"


def test_dedicated_synthesis_operators_start_at_identity_and_are_answer_only(synth_config):
    # Alec 2026-09-09: reverseOutput() has its own weights.  They start at
    # identity (so the answer path begins exactly at the shared inverse) and
    # reconstruction never sees them.
    m = _build(synth_config)
    x, _ = _batch(m)
    q = What.supervised(0)
    with torch.no_grad():
        u = m.understand(x)
        first = m.reverseOutput(u, q)
    cs_layer = m.conceptualSpace.synthesis_layer
    ps_layer = m.perceptualSpace.synthesis_layer
    for layer in (cs_layer, ps_layer):
        eye = torch.eye(layer.nInput)
        probe = torch.randn(3, layer.nInput)
        assert torch.allclose(layer.forward(probe), probe, atol=1e-6)   # identity at init
        assert torch.allclose(layer.compute_W().detach(), eye, atol=1e-6)
    with torch.no_grad():
        rev_before, _ = m.reverseReconstruct(u)
        for layer in (cs_layer, ps_layer):
            layer.raw_L.add_(0.3)
            layer.d.mul_(1.7)
        out_after = m.reverseOutput(u, q).actual
        rev_after, _ = m.reverseReconstruct(u)
    assert not torch.allclose(first.actual, out_after)        # answer path moved
    assert torch.equal(rev_before, rev_after)                  # reconstruction untouched
    assert all(any(p is q2 for q2 in m.synthesis_parameters())
               for layer in (cs_layer, ps_layer) for p in layer.parameters())


def test_answer_path_modules_join_the_live_optimizer_and_move(synth_config):
    m = _build(synth_config)
    opt = m.getOptimizer(lr=1e-2)
    batch = _batch(m)
    # First step builds the modules and registers them; second step moves them.
    m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
               batch_override=batch)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    synth = m.synthesis_parameters()
    assert synth and all(id(p) in ids for p in synth)
    before = [p.detach().clone() for p in synth]
    m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
               batch_override=batch)
    assert any(not torch.equal(p.detach(), b) for p, b in zip(synth, before))
    # Registration happens once: no duplicate groups on later steps.
    n_groups = len(opt.param_groups)
    m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
               batch_override=batch)
    assert len(opt.param_groups) == n_groups


# -- named perceptual bindings (spec 5.3 / section 11) ----------------------

@pytest.fixture(scope="module")
def synth_bindings_config(tmp_path_factory):
    src = (_DATA / "MM_xor.xml").read_text()
    patched = src.replace(
        "<architecture>",
        "<architecture>\n    <answerSynthesis>true</answerSynthesis>"
        "\n    <synthesisBindings>2</synthesisBindings>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth_bind.xml"
    path.write_text(patched)
    return path


def test_perceptual_context_reaches_output_only_through_named_bindings(synth_bindings_config):
    m = _build(synth_bindings_config)
    x, _ = _batch(m)
    q = What.supervised(0)
    with torch.no_grad():
        u = m.understand(x)
        construction = m.reverseOutput(u, q)
    refs = construction.derivation.synthesis_references
    assert len(refs) == 2 and refs == tuple(sorted(refs))
    assert construction.trace[0]["bindings"] == refs
    assert construction.derivation.bindings["perceptual_slots"] == refs
    N = u.perceptual_context.shape[1]
    unbound = next(i for i in range(N) if i not in refs)
    bound = refs[0]

    def realized_with(field):
        alt = Understanding(
            perceptual_context=field, conceptual_state=u.conceptual_state,
            symbolic_state=u.symbolic_state,
            reconstruction_carriers=dict(u.reconstruction_carriers),
            execution=u.execution)
        with torch.no_grad():
            return m.reverseOutput(alt, q).percepts

    base = construction.percepts
    # Perturb WITHOUT changing which slots are the most salient (the named
    # selection must stay fixed): shrink the unbound slot, scale the bound one.
    changed_unbound = u.perceptual_context.clone()
    changed_unbound[:, unbound] *= 0.5
    changed_bound = u.perceptual_context.clone()
    changed_bound[:, bound] *= 3.0
    with torch.no_grad():
        alt_refs = m._select_perceptual_bindings(Understanding(
            perceptual_context=changed_unbound, symbolic_state=u.symbolic_state))
        alt_refs_b = m._select_perceptual_bindings(Understanding(
            perceptual_context=changed_bound, symbolic_state=u.symbolic_state))
    assert alt_refs == refs and alt_refs_b == refs
    # Changing unrelated live activation cannot reach the answer ...
    assert torch.equal(realized_with(changed_unbound), base)
    # ... changing a NAMED binding can.
    assert not torch.equal(realized_with(changed_bound), base)


def test_no_bindings_by_default_keeps_context_out_of_the_answer(synth_config):
    m = _build(synth_config)
    assert m.synthesis_bindings == 0
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        d = m._resolve_answer(u, What.supervised(0))
    assert d.synthesis_references == ()


# -- reasoning hook, identity guard, reporting --------------------------------

def test_prompted_question_records_reasoner_posture(synth_config, monkeypatch):
    m = _build(synth_config)
    monkeypatch.setattr(m, "reasoning_iterations", 3, raising=False)
    monkeypatch.setattr(
        m, "answer_query",
        lambda prompt, **kw: {"posture": "true", "confidence": 0.9,
                              "support_true": 0.9, "support_false": 0.1,
                              "sentences": [], "trace": ()},
        raising=False)
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        d = m._resolve_answer(u, What.inference(0, split="train",
                                                prompt="is the cat black?"))
    assert d.source == "reasoning"
    assert d.grammar_trace[-1]["operation"] == "reason"
    assert d.grammar_trace[-1]["posture"] == "true"


def test_exact_identity_reconstruction_is_flagged_not_counted(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        rev, cost = m.reverseReconstruct(u, target=u.execution[0])
        assert cost is not None and float(cost) > 0.0
        assert m._reconstruction_identity_flag is False
        # Score the reconstruction against ITSELF: an exact identity.
        _, zero = m.reverseReconstruct(u, target=rev)
    assert float(zero) == 0.0
    assert m._reconstruction_identity_flag is True


def test_what_report_separates_families_and_thinking(synth_config):
    m = _build(synth_config)
    opt = m.getOptimizer(lr=1e-3)
    batch = _batch(m)
    m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
               batch_override=batch)
    m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
               batch_override=batch, trial_mode="predict")
    with torch.no_grad():
        m.think(What.supervised(0, prompt="what is your name?"), batch[0],
                max_iterations=2)
    report = m.what_report()
    assert {"supervised", "future"} <= set(report["families"])
    for fam in report["families"].values():
        assert set(fam) == {"batches", "answer_construction",
                            "input_reconstruction"}
    assert report["thinking"]["episodes"] == 1
    assert report["thinking"]["mean_iterations"] >= 1
    assert 0.0 <= report["thinking"]["forced_closure_rate"] <= 1.0
    assert report["throughput"]["batches"] == 2
    assert report["throughput"]["sentences"] == 4


# -- Codex review 2026-09-09: checkpoint reload, per-row resolution, recall ---

def test_synthesized_answer_checkpoint_reloads_into_a_fresh_model(synth_config, tmp_path):
    m = _build(synth_config)
    opt = m.getOptimizer(lr=1e-2)
    batch = _batch(m)
    for _ in range(2):                          # builds + trains the answer path
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch)
    assert m.synthesis_parameters()
    with torch.no_grad():
        u = m.understand(batch[0])
        before = m.reverseOutput(u, What.supervised(0)).actual.clone()
    path = tmp_path / "synth.ckpt"
    m.save_weights(str(path))
    fresh = _build(synth_config)
    assert fresh.load_weights(str(path), require_match=True)
    saved = m.state_dict()
    loaded = fresh.state_dict()
    assert set(saved) == set(loaded)
    for k in saved:
        assert torch.equal(saved[k], loaded[k]), k
    assert fresh.synthesis_parameters()
    for a, b in zip(m.synthesis_parameters(), fresh.synthesis_parameters()):
        assert torch.equal(a.detach(), b.detach())
    # Two cold models loading the same checkpoint answer identically (the
    # trained model itself also carries runtime STM/ring state, so it is not
    # the comparison point).
    twin = _build(synth_config)
    assert twin.load_weights(str(path), require_match=True)
    with torch.no_grad():
        u2 = fresh.understand(batch[0])
        a = fresh.reverseOutput(u2, What.supervised(0)).actual
        u3 = twin.understand(batch[0])
        b = twin.reverseOutput(u3, What.supervised(0)).actual
    assert torch.equal(a, b)
    assert not torch.equal(a, torch.zeros_like(a))
    # A fresh model with the gate on already carries the width-known modules,
    # so its optimizer includes them from the start and no duplicate group is
    # added later.
    opt2 = fresh.getOptimizer(lr=1e-2)
    n_groups = len(opt2.param_groups)
    fresh.runBatch(train=True, batchSize=2, split="train", optimizer=opt2,
                   batch_override=_batch(fresh))
    ids = {id(p) for g in opt2.param_groups for p in g["params"]}
    assert all(id(p) in ids for p in fresh.synthesis_parameters())
    assert len(opt2.param_groups) <= n_groups + 1


def test_batched_questions_resolve_per_row(synth_discourse_config):
    m = _build(synth_discourse_config)
    batch = _batch(m)
    _prime_discourse(m, batch, sentences=3)
    history = m._recall_history()
    assert all(len(history[b]) >= 3 for b in range(2))
    with torch.no_grad():
        u = m.understand(batch[0])
        cond = getattr(m, "question_conditioner", None)
        if cond is not None:
            cond.weight.zero_()
        d = m._resolve_answer(u, (What.past(2, -1), What.past(2, -2)))
    assert d.row_sources == ("recall", "recall") and d.resolved
    width = min(u.symbolic_state.shape[-1], history[0][-1].shape[-1])
    assert torch.equal(d.answer_symbol[0, 0, :width], history[0][-1][:width])
    assert torch.equal(d.answer_symbol[1, 0, :width], history[1][-2][:width])
    # Mixed relations: row 0 present (identity), row 1 past recall.
    with torch.no_grad():
        d2 = m._resolve_answer(u, (What.present(2), What.past(2, -1)))
    assert d2.row_sources == ("identity", "recall") and d2.source == "mixed"
    assert torch.equal(d2.answer_symbol[0], u.symbolic_state[0])
    # A cold lane invalidates ONLY that lane.
    with torch.no_grad():
        d3 = m._resolve_answer(u, (What.past(2, -1), What.past(2, -9)))
    assert d3.row_sources == ("recall", "identity:cold-memory")
    assert not d3.resolved
    assert torch.equal(d3.answer_symbol[1], u.symbolic_state[1])


def test_recall_returns_the_most_recent_sentence_during_ring_fill(synth_discourse_config):
    # The ARMA ring's fill phase writes new rows at cap-count-1 (newest at the
    # LOW end), so indexing the ring tail is wrong; recall uses its own
    # chronological history instead.
    m = _build(synth_discourse_config)
    memory = m._what_memory()
    batch = _batch(m)
    seen = []
    orig = m._observe_discourse

    def spy(disc, sentence, mask=None):
        seen.append(disc._pool_sentence_rep(sentence).detach().clone())
        return orig(disc, sentence, mask=mask)

    m._observe_discourse = spy
    try:
        _prime_discourse(m, batch, sentences=2)
    finally:
        m._observe_discourse = orig
    assert len(seen) == 2 and int(memory._s_count[0]) == 2 < int(memory.p)
    with torch.no_grad():
        u = m.understand(batch[0])
        if getattr(m, "question_conditioner", None) is not None:
            m.question_conditioner.weight.zero_()
        d = m._resolve_answer(u, What.past(3, -1))
    width = min(u.symbolic_state.shape[-1], seen[-1].shape[-1])
    assert torch.equal(d.answer_symbol[:, 0, :width], seen[-1][:, :width])   # newest
    assert not torch.equal(d.answer_symbol[:, 0, :width], seen[0][:, :width])
    # The ring tail still holds the FIRST observation during fill: the bug.
    assert torch.equal(memory._s_history[:, -1, :width], seen[0][:, :width])
