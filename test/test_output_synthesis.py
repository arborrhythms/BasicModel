"""What spec Steps 3-4: the separate ``Model.reverseOutput()`` path.

Section 11 "Reconstruction and answer paths": a generated answer starts at
the resolved symbol and traverses generate, owned occurrences and the shared
order-zero reverse; reconstruction does not consume an answer carrier and
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
        "<architecture>\n    <answerSynthesis>false</answerSynthesis>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth.xml"
    path.write_text(patched)
    return path


@pytest.fixture(scope="module")
def synth_discourse_config(tmp_path_factory):
    # Same, plus the inter-sentence discourse layer (ARMA ring + predictor)
    # that past recall and future prediction resolve through.
    src = (_DATA / "MM_xor.xml").read_text()
    assert "<prediction>" not in src and "<sentenceExpectation>" not in src
    patched = src.replace(
        "<architecture>",
        "<architecture>\n    <answerSynthesis>false</answerSynthesis>"
        , 1)
    patched = patched.replace(
        "</training>",
        "      <sentenceExpectation>true</sentenceExpectation>\n    </training>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth_discourse.xml"
    path.write_text(patched)
    return path



@pytest.fixture
def loop_model(tmp_path, monkeypatch):
    from test_native_attention_field import _model
    import Models
    loader = Models.BaseModel.load_weights
    model = _model(tmp_path, monkeypatch)
    monkeypatch.setattr(Models.BaseModel, 'load_weights', loader)
    data = model.inputSpace.data
    data.has_supervised_outputs = True
    data.text_answers = {'train': ['alpha', 'beta']}
    yield model
    model.End()
    model.symbolSpace.soft_reset()


def _loop_batch(model):
    return model.inputSpace.prepInput(['alpha beta', 'beta alpha']), torch.empty(2, 0)


def _loop_understanding(model):
    with torch.no_grad():
        return model.understand(_loop_batch(model)[0])


def _loop_answer(model, understanding):
    return model.reverseOutput(understanding, model.resolveAnswer(understanding,
        (What.supervised(0), What.supervised(1))))

def test_gate_off_keeps_the_projected_head_and_state_dict(model_off=None):
    m = _build(_DATA / "MM_xor.xml")
    assert not m.answer_synthesis
    x, _ = _batch(m)
    with torch.no_grad():
        answer = m.ask(What.supervised(0), input_data=x)
    assert answer.what is answer.execution[2]
    assert m._last_answer_construction is None
    assert not any(k.endswith("percept_adapter.W") or "percept_adapter" in k
                   for k in m.state_dict())


def test_output_traverses_symbol_concepts_percepts_output(loop_model):
    m = loop_model
    u = _loop_understanding(m)
    with torch.no_grad():
        construction = _loop_answer(m, u)
    assert isinstance(construction, AnswerConstruction)
    assert construction.concepts.shape[0] == construction.percepts.shape[0] == 2
    assert construction.actual is construction.surface
    assert [item['operation'] for item in construction.trace[-3:]] == [
        'generate:grammar', 'generate:occurrences', 'reverse:order_zero']
    assert not any('synthesis_layer' in key or 'percept_adapter' in key for key in m.state_dict())



def test_what_reaches_output_not_the_legacy_head(loop_model):
    m = loop_model
    with torch.no_grad():
        answer = m.ask(What.supervised(0), input_data=_loop_batch(m)[0])
    assert answer.what is m._last_answer_construction.actual
    assert answer.what is not answer.execution[2]



def test_reconstruction_and_output_are_order_independent(loop_model):
    m = loop_model
    u = _loop_understanding(m)
    with torch.no_grad():
        first, _ = m.reverseReconstruct(u)
        out1 = _loop_answer(m, u).actual
        second, _ = m.reverseReconstruct(u)
        out2 = _loop_answer(m, u).actual
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)



def test_output_does_not_consume_reconstruction_carriers(loop_model):
    from dataclasses import replace
    m = loop_model
    u = _loop_understanding(m)
    with torch.no_grad():
        baseline = _loop_answer(m, u).actual
        again = _loop_answer(m, replace(u, reconstruction_carriers={})).actual
    assert torch.equal(baseline, again)



def test_training_step_scores_the_constructed_answer(loop_model):
    m = loop_model
    result, _ = m.runBatch(train=True, optimizer=m.getOptimizer(lr=.001),
        batchSize=2, batch_override=_loop_batch(m))
    assert result.outputPred is m._last_answer_construction.actual
    assert torch.isfinite(m.primary_costs()['answer_construction'])
    assert m._last_answer_construction.percepts is not None



def test_think_constructs_the_root_answer_through_output(loop_model):
    m = loop_model
    with torch.no_grad():
        result = m.think(What.supervised(0), _loop_batch(m)[0], max_iterations=3)
    assert result.answer.available
    assert result.answer.what is m._last_answer_construction.actual
    assert m._last_answer_construction.trace[-1]['operation'] == 'reverse:order_zero'



def test_branch_gradient_diagnostics_read_without_update(loop_model):
    m = loop_model
    optimizer = m.getOptimizer(lr=.001)
    m.runBatch(train=True, optimizer=optimizer, batchSize=2, batch_override=_loop_batch(m))
    before = [p.detach().clone() for p in m.parameters()]
    report = m.ownership_gradient_diagnostics(optimizer)
    assert report['conflicts'] == 0 and report['parameters']
    assert all(set(row['writers']) <= {'R', 'A'} for row in report['parameters'])
    assert all(torch.equal(p, old) for p, old in zip(m.parameters(), before))



def _prime_discourse(m, batch, sentences=2):
    # Explicit external meanings for temporal API tests. These fixtures do
    # not manufacture observations as a side effect of non-bracket forward.
    memory = m.symbolSpace.expectation
    memory.ensure_batch(2)
    # This isolated non-bracket temporal fixture has no durable observations.
    memory._ltm_store = None
    memory.set_symbol_bank(torch.eye(memory.concept_dim))
    for turn in range(sentences):
        values = torch.eye(memory.concept_dim)[[turn % memory.concept_dim,
                                                   (turn+1) % memory.concept_dim]]
        memory.predict_and_observe_stm_end_state([1,1], [value[None] for value in values],layout='infix')


def test_present_and_supervised_resolve_by_identity(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        for q in (What.present(1), What.supervised(1),
                  What.inference(1, split="train")):
            d = m.resolveAnswer(u, q)
            assert d.source == "identity" and d.resolved
            # The zero-initialised question conditioner leaves the symbol
            # numerically identical (a clone, so the understanding is never
            # mutated in place).
            assert torch.equal(d.answer_symbol, u.symbolic_state)
            assert d.answer_symbol is not u.symbolic_state


def test_past_resolves_by_recall_from_discourse_memory(synth_discourse_config):
    m = _build(synth_discourse_config)
    batch = _batch(m)
    memory = m.symbolSpace.expectation
    assert memory is not None and getattr(memory, "_inter_context", None) is not None
    _prime_discourse(m, batch, sentences=2)
    with torch.no_grad():
        u = m.understand(batch[0])
        # Priming trained the question conditioner a little; silence it so
        # the root slot is exactly the recalled rep.
        cond = getattr(m, "question_conditioner", None)
        if cond is not None:
            cond.weight.zero_()
        d = m.resolveAnswer(u, What.past(1, -1))
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
    far = m.resolveAnswer(u, What.past(9, -(len(history[0]) + 1)))
    assert far.source == "identity:cold-memory" and not far.resolved


def test_future_resolves_by_prediction_without_committing_memory(synth_discourse_config):
    m = _build(synth_discourse_config)
    batch = _batch(m)
    memory = m.symbolSpace.expectation
    assert memory is not None and getattr(memory, "_inter_predictor", None) is not None
    _prime_discourse(m, batch, sentences=2)
    before = [[frame[1].clone() for frame in row] for row in memory._inter_context]
    with torch.no_grad():
        u = m.understand(batch[0])
        d1 = m.resolveAnswer(u, What.future(1, 1))
        d2 = m.resolveAnswer(u, What.future(1, 2))
    assert d1.source == "prediction" and d1.resolved
    assert all(torch.equal(frame[1], old) for row, saved in zip(memory._inter_context, before)
               for frame, old in zip(row, saved))
    assert not d2.resolved  # future forecasts never become factual context
    with torch.no_grad():
        construction = m.reverseOutput(u, m.resolveAnswer(u, What.future(1, 1)))
    assert construction.trace[0]["operation"] == "resolve:prediction"


def test_cold_memory_resolution_is_flagged_unresolved(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        d = m.resolveAnswer(u, What.past(3, -1))
    assert not d.resolved and d.source == "identity:cold-memory"


def test_retired_synthesis_operators_are_absent_and_checkpoint_keys_are_reported(loop_model):
    m = loop_model
    assert not hasattr(m.conceptualSpace, 'synthesis_layer')
    assert not hasattr(m.perceptualSpace, 'synthesis_layer')
    for space in (m.conceptualSpace, m.perceptualSpace):
        for name in ('synthesis_operator', '_apply_synthesis_operator', 'synthesize', 'synthesize_idea'):
            assert not hasattr(space, name)
    state = dict(m.state_dict())
    retired = ('conceptualSpace.synthesis_layer.raw_L', 'perceptualSpace.synthesis_layer.d')
    for key in retired:
        state[key] = torch.ones(2, 2)
    with pytest.warns(UserWarning, match='retired answer synthesis checkpoint keys') as warnings:
        m._materialize_answer_path_from_checkpoint(state)
    assert all(key not in state for key in retired)
    assert set(m._retired_answer_checkpoint_keys) == set(retired)
    assert set(retired) <= m._pending_thought_policy_reset
    assert all(key in str(warnings[0].message) for key in retired)
    m.load_state_dict(state, strict=True)



def test_answer_path_modules_join_the_live_optimizer_and_move(loop_model):
    m = loop_model
    optimizer = m.getOptimizer(lr=.001)
    m.runBatch(train=True, optimizer=optimizer, batchSize=2, batch_override=_loop_batch(m))
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    policy = tuple(m.languageSpace.generate_policy.parameters())
    assert all(ids.count(id(p)) == 1 for p in policy)
    groups = m.objective_parameter_groups(optimizer)
    assert {id(p) for p in policy} <= {id(p) for p in groups['output']}
    m.to(dtype=torch.float64)
    assert all(p.dtype == torch.float64 for p in policy)



# -- named perceptual bindings (spec 5.3 / section 11) ----------------------

@pytest.fixture(scope="module")
def synth_bindings_config(tmp_path_factory):
    src = (_DATA / "MM_xor.xml").read_text()
    patched = src.replace(
        "<architecture>",
        "<architecture>\n    <answerSynthesis>false</answerSynthesis>"
        "\n    <synthesisBindings>2</synthesisBindings>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth_bind.xml"
    path.write_text(patched)
    return path


def test_perceptual_context_is_not_an_operand_of_the_common_decoder(loop_model):
    from dataclasses import replace
    m = loop_model
    u = _loop_understanding(m)
    with torch.no_grad():
        baseline = _loop_answer(m, u).actual
        altered = replace(u, perceptual_context=torch.zeros_like(u.perceptual_context))
        again = _loop_answer(m, altered).actual
    assert torch.equal(baseline, again)



def test_no_bindings_by_default_keeps_context_out_of_the_answer(synth_config):
    m = _build(synth_config)
    assert m.synthesis_bindings == 0
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        d = m.resolveAnswer(u, What.supervised(0))
    assert d.synthesis_references == ()


# -- normal-boundary quarantine, identity guard, reporting ---------------------

def test_prompted_question_cannot_route_normal_resolution_to_legacy_reasoner(
        synth_config, monkeypatch):
    """Surface text is not a production selector for the legacy controller."""
    m = _build(synth_config)
    monkeypatch.setattr(m, "reasoning_iterations", 3, raising=False)
    monkeypatch.setattr(
        m, "answer_query",
        lambda *_args, **_kwargs: pytest.fail(
            "normal answer resolution invoked answer_query"),
        raising=False)
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        d = m.resolveAnswer(u, What.inference(0, split="train",
                                                prompt="is the cat black?"))
    assert d.source == "identity"
    assert all(item["operation"] != "reason" for item in d.grammar_trace)


def test_numeric_field_inverse_scores_identity_and_a_changed_target(synth_config):
    m = _build(synth_config)
    x, _ = _batch(m)
    with torch.no_grad():
        u = m.understand(x)
        rev, cost = m.reverseReconstruct(u, target=u.execution[0])
        assert cost is not None and float(cost) == 0.
        _, zero = m.reverseReconstruct(u, target=rev)
        _, changed = m.reverseReconstruct(u, target=rev+1.)
    assert float(zero) == 0. and float(changed) == 1.


def test_what_report_separates_families_and_thinking(loop_model):
    m = loop_model
    optimizer = m.getOptimizer(lr=.001)
    m.runBatch(train=True, optimizer=optimizer, batchSize=2, batch_override=_loop_batch(m))
    with torch.no_grad():
        m.think(What.supervised(0), _loop_batch(m)[0], max_iterations=2)
    report = m.what_report()
    assert 'supervised' in report['families']
    for family in report['families'].values():
        assert set(family) == {'batches', 'answer_construction', 'input_reconstruction'}
    assert report['thinking']['episodes'] == 1
    assert report['thinking']['mean_iterations'] >= 1
    assert 0 <= report['thinking']['forced_closure_rate'] <= 1
    assert report['throughput']['batches'] == 1
    assert report['throughput']['sentences'] == 2



# -- Codex review 2026-09-09: checkpoint reload, per-row resolution, recall ---

def test_synthesized_answer_checkpoint_reloads_into_a_fresh_model(loop_model, tmp_path, monkeypatch):
    from test_native_attention_field import _model
    import Models
    m = loop_model
    m.runBatch(train=True, optimizer=m.getOptimizer(lr=.001), batchSize=2, batch_override=_loop_batch(m))
    path = tmp_path / 'common-reverse.ckpt'
    m.save_weights(str(path))
    expected = {key: value.detach().clone() for key, value in m.languageSpace.generate_policy.state_dict().items()}
    loader = Models.BaseModel.load_weights
    fresh = _model(tmp_path, monkeypatch)
    monkeypatch.setattr(Models.BaseModel, 'load_weights', loader)
    try:
        assert fresh.load_weights(str(path), require_match=True)
        for key, value in fresh.languageSpace.generate_policy.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        assert not hasattr(fresh.conceptualSpace, 'synthesis_layer')
        assert not hasattr(fresh.perceptualSpace, 'synthesis_layer')
        with torch.no_grad():
            u = fresh.understand(_loop_batch(fresh)[0])
            a = _loop_answer(fresh, u).actual
            b = _loop_answer(fresh, u).actual
        assert torch.equal(a, b)
    finally:
        fresh.End(); fresh.symbolSpace.soft_reset()



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
        d = m.resolveAnswer(u, (What.past(2, -1), What.past(2, -2)))
    assert d.row_sources == ("recall", "recall") and d.resolved
    width = min(u.symbolic_state.shape[-1], history[0][-1].shape[-1])
    assert torch.equal(d.answer_symbol[0, 0, :width], history[0][-1][:width])
    assert torch.equal(d.answer_symbol[1, 0, :width], history[1][-2][:width])
    # Mixed relations: row 0 present (identity), row 1 past recall.
    with torch.no_grad():
        d2 = m.resolveAnswer(u, (What.present(2), What.past(2, -1)))
    assert d2.row_sources == ("identity", "recall") and d2.source == "mixed"
    assert torch.equal(d2.answer_symbol[0], u.symbolic_state[0])
    # A cold lane invalidates ONLY that lane.
    with torch.no_grad():
        d3 = m.resolveAnswer(u, (What.past(2, -1), What.past(2, -9)))
    assert d3.row_sources == ("recall", "identity:cold-memory")
    assert not d3.resolved
    assert torch.equal(d3.answer_symbol[1], u.symbolic_state[1])


def test_recall_returns_the_most_recent_sentence_during_ring_fill(synth_discourse_config):
    m = _build(synth_discourse_config)
    memory = m.symbolSpace.expectation
    batch = _batch(m)
    _prime_discourse(m,batch,sentences=2)
    seen = m._recall_history()
    assert len(seen[0]) == 2 < memory._inter_chain_window
    with torch.no_grad():
        u=m.understand(batch[0])
        if getattr(m,'question_conditioner',None) is not None:
            m.question_conditioner.weight.zero_()
        d=m.resolveAnswer(u,What.past(3,-1))
    width=min(u.symbolic_state.shape[-1],seen[0][-1].shape[-1])
    expected=torch.stack([seen[b][-1][:width] for b in range(2)])
    torch.testing.assert_close(d.answer_symbol[:,0,:width],expected,rtol=0,atol=0)
    assert not torch.equal(seen[0][0],seen[0][-1])
    assert not hasattr(memory,'_s_history')


# -- the three items Alec asked to close (2026-09-09) --------------------------

def _live_state_fingerprint(m):
    """Every piece of live Space state a downward path could touch."""
    import types
    out = {}
    def tensor_fp(t):
        return ("T", tuple(t.shape), float(t.detach().float().sum()) if t.numel() else 0.0)
    spaces = {"inputSpace": m.inputSpace, "perceptualSpace": m.perceptualSpace,
              "conceptualSpace": m.conceptualSpace, "wholeSpace": m.wholeSpace,
              "outputSpace": m.outputSpace}
    for i, sp in enumerate(m.conceptualSpaces or []):
        spaces[f"conceptualSpaces[{i}]"] = sp
    for name, sp in spaces.items():
        sub = getattr(sp, "subspace", None)
        if sub is not None:
            ev = sub.materialize() if hasattr(sub, "materialize") else None
            out[f"{name}.event"] = tensor_fp(ev) if torch.is_tensor(ev) else None
            for k, v in vars(sub).items():
                if torch.is_tensor(v):
                    out[f"{name}.sub.{k}"] = tensor_fp(v)
            out[f"{name}.sub.batch"] = getattr(sub, "batch", None)
        for k in ("concepts", "reconstructed", "input", "_recovered_input",
                  "_recovered_input_thunk", "_embedded_input", "_ar_embedded"):
            v = getattr(sp, k, None) if k in vars(sp) else None
            out[f"{name}.{k}"] = tensor_fp(v) if torch.is_tensor(v) else (
                None if v is None else type(v).__name__)
        for role in ("event", "what", "where", "when", "activation"):
            basis = getattr(sub, role, None) if sub is not None else None
            W = basis.getW() if basis is not None and hasattr(basis, "getW") else None
            if torch.is_tensor(W):
                out[f"{name}.basis.{role}"] = tensor_fp(W)
    for t, stage in enumerate(m.body_stages):
        if "merge" in stage:
            d = stage["merge"]._merge_diff
            out[f"merge[{t}]"] = tensor_fp(d) if torch.is_tensor(d) else None
    out["_combine_carriers"] = type(getattr(m, "_combine_carriers", None)).__name__
    ss = m.symbolSpace.subspace
    out["generate_rules"] = repr(sorted((k, len(v) if hasattr(v, "__len__") else v)
                                        for k, v in (getattr(ss, "generate_rules", {}) or {}).items()))
    out["_generate_generation"] = getattr(ss, "_generate_generation", None)
    return out


def test_reverse_output_is_carrier_pure_without_the_guard(loop_model, monkeypatch):
    from contextlib import nullcontext
    m = loop_model
    u = _loop_understanding(m)
    with torch.no_grad():
        _loop_answer(m, u)
    monkeypatch.setattr(m, '_synthesis_guard', lambda: nullcontext())
    before = _live_state_fingerprint(m)
    with torch.no_grad():
        _loop_answer(m, u)
        _loop_answer(m, u)
    after = _live_state_fingerprint(m)
    changed = {key: (before.get(key), after.get(key)) for key in set(before) | set(after)
               if before.get(key) != after.get(key)}
    assert not changed, changed



def test_model_trains_with_teacher_detached(synth_config):
    """12.16 closed: the model owns its data authority, loss registry and loss
    composition; Teacher is an optional provenance adapter."""
    m = _build(synth_config)
    assert m.errors is not None and m.loss is not None and m.data is not None
    m.teacher = None
    opt = m.getOptimizer(lr=1e-3)
    batch = _batch(m)
    result, _ = m.runBatch(train=True, batchSize=2, split="train",
                           optimizer=opt, batch_override=batch)
    assert torch.isfinite(result.lossOut)
    names = [t[0] for t in m.errors.terms()]
    assert "output" in names and "reconstruction" in names
    m.runEpoch(optimizer=opt, batchSize=2, split="train", max_batches=2)
    with torch.no_grad():
        answer = m.ask(What.supervised(0), input_data=batch[0])
    assert answer.available
    assert m.inputSpace.data.what(What.supervised(0)).provenance == "data"


def test_compose_auxiliary_policy_setting_is_retired(tmp_path):
    from util import init_config
    import Language, Models
    src = (_DATA / 'MM_ladder.xml').read_text()
    src = src.replace('</training>', '<forwardGrammarWeight>0.5</forwardGrammarWeight></training>', 1)
    path = tmp_path / 'retired-policy.xml'
    path.write_text(src)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match='forwardGrammarWeight|schema|validate'):
        init_config(path=str(path), defaults_path=str(_DATA / 'model.xml'))
        Language.TheGrammar._configured = False
        Models.BasicModel.from_config(str(path))


def test_fresh_model_deep_copies_after_another_model_trained(synth_config):
    # The shared loss registry keeps the previous batch's (non-leaf) loss
    # tensors; a fresh model must not expose it as an instance attribute or
    # ``copy.deepcopy`` (used by the compiled word-chunk tests) fails.
    import copy
    trained = _build(synth_config)
    opt = trained.getOptimizer(lr=1e-3)
    trained.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                     batch_override=_batch(trained))
    fresh = _build(synth_config)
    assert "errors" not in vars(fresh)
    assert fresh.errors is trained.errors            # one process registry
    clone = copy.deepcopy(fresh)
    assert clone is not fresh and clone.errors is not None   # Teacher's deepcopy gives the clone its own registry
