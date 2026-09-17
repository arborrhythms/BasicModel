"""Supervised training on the ``reverseOutput()`` path (Alec 2026-09-09:
"this will need supervised training on the output() path; do we have
tests that provide that?").

On the parallel XOR topology the answer path memorizes supervised labels
(the Step-7 causal test already trained it).  On the SERIAL grammar
topology (``MM_phrase_decode``: symbolicOrder 1, complete.grammar, word
analysis, 1024 wide) it did not until the answer path was seeded from
the grammar's ROOT IDEA (``Understanding.answer_seed``): the ``symbols``
tensor there is the symbol-space activation over a codebook with two
active rows at initialization and is the same for every sentence
(relative spread ~0.03 against ~0.5 for the root idea).  With the seed
the path memorizes eight labels at lr 1e-3 (5e-3 diverges on this
1024-wide topology)."""
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

from What import What  # noqa: E402


def _build(config_path, dataset):
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(config_path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load(dataset)
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config_path), data=TheData)
    return m.to("cpu")


def _spread(t):
    n = t.shape[0]
    t = t.reshape(n, -1)
    d = torch.cdist(t, t)
    return float(d[~torch.eye(n, dtype=torch.bool)].mean() / (t.norm(dim=-1).mean() + 1e-9))


def _train_accuracy(m):
    data = m.inputSpace.data
    n = data.what_extent("train")
    m.eval()
    with torch.no_grad():
        x = m.inputSpace.prepInput(list(data.train_input))
        y = m.outputSpace.prepOutput(list(data.train_output))
        answers = m.what(tuple(What.supervised(i) for i in range(n)), x)
        pred = torch.stack([a.what.reshape(-1) for a in answers])
        target = y.reshape(n, -1)
    m.train()
    if target.shape[-1] == 1:
        return float(((pred > 0.5).float() == target).float().mean())
    return float((pred.argmax(-1) == target.argmax(-1)).float().mean())


def _memorize(m, epochs, lr=5e-3, batch=8):
    opt = m.getOptimizer(lr=lr)
    best = 0.0
    for epoch in range(1, epochs + 1):
        m.train()
        m.runEpoch(optimizer=opt, batchSize=batch, split="train")
        if epoch % 10 == 0:
            best = max(best, _train_accuracy(m))
            if best >= 1.0:
                break
    return best


@pytest.fixture(scope="module")
def xor_synth_config(tmp_path_factory):
    src = (_DATA / "MM_xor.xml").read_text()
    src = src.replace("<architecture>", "<architecture>\n    <answerSynthesis>true</answerSynthesis>", 1)
    path = tmp_path_factory.mktemp("cfg") / "MM_xor_synth.xml"
    path.write_text(src)
    return path


@pytest.fixture(scope="module")
def serial_synth_config(tmp_path_factory):
    src = (_DATA / "MM_phrase_decode.xml").read_text()
    src = src.replace("<ideaDecode>true</ideaDecode>",
                      "<ideaDecode>true</ideaDecode>\n    <answerSynthesis>true</answerSynthesis>\n"
                      "    <stmReduceTau>0.05</stmReduceTau>\n    <transformChooser>mlp</transformChooser>")
    path = tmp_path_factory.mktemp("cfg") / "MM_phrase_synth.xml"
    path.write_text(src)
    return path


@pytest.mark.slow
def test_output_path_memorizes_supervised_labels_on_the_parallel_topology(xor_synth_config):
    m = _build(xor_synth_config, "xor")
    assert m.answer_synthesis
    assert _memorize(m, epochs=60) >= 1.0


@pytest.mark.slow
def test_serial_answer_path_varies_with_the_input(serial_synth_config):
    """The answer path can only be trained on what varies with the input:
    the symbolic root that seeds synthesis must differ across different
    sentences at least as much as the conceptual state does."""
    m = _build(serial_synth_config, "phrases")
    m.eval()
    data = m.inputSpace.data
    with torch.no_grad():
        x = m.inputSpace.prepInput(list(data.train_input))
        u = m.understand(x)
        c = m.reverseOutput(u, m.resolveAnswer(u, tuple(What.supervised(i) for i in range(len(data.train_input)))))
    conceptual = _spread(u.conceptual_state)
    seed = _spread(u.answer_seed if torch.is_tensor(u.answer_seed) else u.symbolic_state)
    actual = _spread(c.actual)
    assert conceptual > 0.3, conceptual                    # the understanding varies
    assert seed > 0.3, seed                                # so does what seeds the answer
    assert actual > 0.1, actual                            # and the emitted answer


@pytest.mark.slow
def test_output_path_memorizes_supervised_labels_on_the_serial_topology(serial_synth_config):
    m = _build(serial_synth_config, "phrases")
    assert torch.is_tensor(m.understand(m.inputSpace.prepInput(
        list(m.inputSpace.data.train_input[:2]))).answer_seed)
    assert _memorize(m, epochs=40, lr=1e-3) >= 1.0



_NATIVE_CONCEPT_WIDTHS = [
    pytest.param(264, id="development"),
    pytest.param(1032, id="production", marks=pytest.mark.slow),
]


def _native_answer_model(tmp_path, output_loop, *, concept_width=264):
    """Distinct concept/percept widths; production width is an explicit slow case."""
    from test_meronomy_ladder import _build_ladder_variant
    if concept_width <= 136:
        raise ValueError("native fixture requires concepts wider than percepts")
    src = (_DATA / "MM_ladder.xml").read_text()
    def block(name):
        start = src.index("<" + name + ">")
        end = src.index("</" + name + ">", start) + len(name) + 3
        return src[start:end]
    cs, ws, output = block("ConceptualSpace"), block("WholeSpace"), block("OutputSpace")
    native_cs = cs
    for tag in ("nInputDim", "nDim", "nOutputDim"):
        native_cs = native_cs.replace(f"<{tag}>136</{tag}>", f"<{tag}>{concept_width}</{tag}>")
    native_cs = (native_cs.replace("<nVectors>4096</nVectors>", "<nVectors>256</nVectors>")
                 .replace("<activeVectors>4096</activeVectors>", "<activeVectors>256</activeVectors>"))
    native_ws = ws.replace("<nInputDim>136</nInputDim>", f"<nInputDim>{concept_width}</nInputDim>")
    native_output = output.replace("<nInputDim>136</nInputDim>", f"<nInputDim>{concept_width}</nInputDim>")
    replacements = [(cs, native_cs), (ws, native_ws), (output, native_output)]
    if output_loop:
        replacements.append(("<training>", "<training>\n      <outputInLoop>true</outputInLoop>"))
    m = _build_ladder_variant(tmp_path, "native_answer", replacements)
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.synthesis_bindings = 0  # named percepts must not hide a dead answer operand
    assert m.answer_synthesis and m.output_in_loop is output_loop
    assert m.wholeSpace.subspace.muxedSize == 136
    assert m.conceptualSpace.stm.concept_dim == concept_width
    return m


@pytest.mark.parametrize("concept_width", _NATIVE_CONCEPT_WIDTHS)
@pytest.mark.parametrize("output_loop", [False, True])
def test_native_answer_uses_owned_ideas_without_dense_symbol_state(tmp_path, output_loop, concept_width):
    from dataclasses import replace
    from test_output_walk import _capture_program_probe, _stop
    m = _native_answer_model(tmp_path, output_loop, concept_width=concept_width)
    m.eval()
    if output_loop:
        _stop(m)
    try:
        with torch.no_grad():
            u = _capture_program_probe(m, ["1 plus 2", "3 plus 4"])
            assert len(u.answer_program) == 2 and all(p is not None for p in u.answer_program)
            # These dense compatibility carriers cannot own a production answer.
            owned = replace(u, symbolic_state=None, conceptual_state=None, answer_seed=None)
            c = m.reverseOutput(owned, m.resolveAnswer(owned, (What.supervised(0), What.supervised(1))))
        assert c.derivation.resolved
        assert c.concepts.shape[0] == 2 and c.concepts.shape[-1] == concept_width
        assert not torch.equal(c.concepts[0], c.concepts[1])
        assert bool(torch.isfinite(c.actual).all())
        assert not torch.equal(c.actual[0], c.actual[1])
        for saved, selected in zip(owned.answer_program, c.derivation.program):
            assert selected is saved
    finally:
        m.End()
        m.symbolSpace.soft_reset()
        torch._dynamo.reset()


def _dedicated_answer_parameters(m):
    modules = list(getattr(m, "question_conditioners", {}).values())
    modules.extend([getattr(m, "what_step_chooser", None),
                    getattr(m, "ltm_attention", None),
                    getattr(m.languageSpace, "generate_policy", None)])
    for space in (m.conceptualSpace, m.perceptualSpace, m.outputSpace):
        modules.extend(getattr(space, name, None)
                       for name in ("synthesis_layer", "percept_adapter"))
    return list({id(p): p for module in modules if module is not None
                 for p in module.parameters() if p.requires_grad}.values())


def _answer_training_probe(m, opt, questions):
    """Observe real answer loss, the trained total, and the resulting step."""
    observed, recorded = {}, {}
    backward, record = m._backward_training_loss, m.record_loss

    def record_probe(name, value, **kwargs):
        recorded[name] = value
        return record(name, value, **kwargs)

    def backward_probe(total, objectives, optimizer, amp_scaler=None):
        params = _dedicated_answer_parameters(m)
        observed["params"] = params
        observed["before"] = [p.detach().clone() for p in params]
        observed["total_grads"] = torch.autograd.grad(
            total, params, retain_graph=True, allow_unused=True)
        c = m._last_answer_construction
        answer_loss = recorded["output"]
        observed["answer_loss"] = float(answer_loss.detach())
        observed["answer_requires_grad"] = answer_loss.requires_grad
        active = m.question_conditioners[str(m.conceptualSpace.stm.concept_dim)].weight
        observed["active_index"] = next(i for i, p in enumerate(params) if p is active)
        if answer_loss.requires_grad:
            observed["realization_grad"] = torch.autograd.grad(
                c.percepts.sum(), active, retain_graph=True, allow_unused=True)[0]
            observed["actual_grad"] = torch.autograd.grad(
                answer_loss, c.actual, retain_graph=True, allow_unused=True)[0]
            if torch.is_tensor(c.surface) and c.surface.requires_grad:
                observed["surface_grad"] = torch.autograd.grad(
                    answer_loss, c.surface, retain_graph=True, allow_unused=True)[0]
        return backward(total, objectives, optimizer, amp_scaler)

    m.record_loss, m._backward_training_loss = record_probe, backward_probe
    try:
        batch = (m.inputSpace.prepInput(["1 plus 2", "3 plus 4"]),
                 torch.zeros(2, 1, 1))
        torch.manual_seed(11)
        m.runBatch(train=True, batchSize=2, split="train", optimizer=opt,
                   batch_override=batch, questions=questions)
    finally:
        m.record_loss, m._backward_training_loss = record, backward
    observed["changed"] = [not torch.equal(p, old) for p, old in
                           zip(observed["params"], observed["before"])]
    observed["mask"] = m._last_answer_mask.tolist()
    observed["recorded"] = recorded
    return observed


@pytest.mark.slow
@pytest.mark.parametrize("concept_width", _NATIVE_CONCEPT_WIDTHS)
@pytest.mark.parametrize("output_loop", [False, True])
def test_native_realized_answer_loss_trains_the_active_conditioner(tmp_path, output_loop, concept_width):
    m = _native_answer_model(tmp_path, output_loop, concept_width=concept_width)
    opt = m.getOptimizer(lr=1e-3)
    try:
        got = _answer_training_probe(m, opt, (What.supervised(0), What.supervised(1)))
        i = got["active_index"]
        assert got["answer_loss"] > 0 and got["answer_requires_grad"]
        assert got["total_grads"][i] is not None
        assert bool(got["total_grads"][i].abs().sum() > 0)
        # At identity the small label head reads only its initial coordinates.
        # All concept coordinates must still reach the percept realization.
        assert bool(got["realization_grad"][136:].abs().sum() > 0)
        assert m._last_answer_construction.percepts.shape[-1] == 136
        assert got["changed"][i]
        assert got["actual_grad"] is not None and bool(got["actual_grad"].abs().sum() > 0)
        owned = [id(p) for group in opt.param_groups for p in group["params"]]
        assert owned.count(id(got["params"][i])) == 1
        for b, answer in enumerate(m._last_what_answers):
            root_input = answer.ltm_slot.input
            assert root_input.shape[-1] == concept_width
            expected = m._last_understanding.answer_program[b].end_state
            torch.testing.assert_close(root_input.reshape_as(expected), expected)
    finally:
        m.End()
        m.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.slow
@pytest.mark.parametrize("output_loop", [False, True])
def test_available_input_targets_do_not_train_answer_modules_after_adam(output_loop):
    from copy import deepcopy
    from Layers import WhatInteractionMemory
    from test_output_walk import _model
    m = _model()
    m.output_in_loop = output_loop
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.what_thinking_iterations = 2
    m.what_thinking_policy_weight = 1.0
    if m._what_memory() is None:
        object.__setattr__(m.symbolSpace, "what_memory",
                           WhatInteractionMemory(batch=2, capacity=64))
    data = m.inputSpace.data
    saved_addresses = deepcopy(data.source_addresses["train"])
    had_outputs = data.has_supervised_outputs
    for address in data.source_addresses["train"]:
        address["document"] = 0
        address["sentence"] = address["row"]
    opt = m.getOptimizer(lr=1e-3)
    try:
        supplied = (What.supervised(0), What.supervised(1))
        got = _answer_training_probe(m, opt, supplied)
        assert got["changed"][got["active_index"]]
        assert "what_step_policy" in got["recorded"]
        for family, questions in (
                ("present", (What.present(0), What.present(1))),
                ("past", (What.past(1), What.past(2))),
                ("future", (What.future(0), What.future(1))),
                ("missing-label", supplied)):
            if family == "missing-label":
                data.has_supervised_outputs = False
            else:
                assert all(data.what(q).available for q in questions)
            baselines = (m.__dict__.get("_output_policy_baseline"),
                         m.__dict__.get("_what_policy_baseline"))
            got = _answer_training_probe(m, opt, questions)
            assert got["mask"] == [False, False], family
            assert got["answer_loss"] == 0 and not got["answer_requires_grad"], family
            assert all(g is None for g in got["total_grads"]), family
            assert not any(got["changed"]), family
            assert baselines == (m.__dict__.get("_output_policy_baseline"),
                                 m.__dict__.get("_what_policy_baseline")), family
    finally:
        data.has_supervised_outputs = had_outputs
        data.source_addresses["train"] = saved_addresses
        m.End()
        m.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_mixed_supplied_numeric_and_automatic_text_trains_only_supplied_row():
    from copy import deepcopy
    from test_output_walk import _model
    m = _model()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    data = m.inputSpace.data
    saved_addresses = deepcopy(data.source_addresses["train"])
    for address in data.source_addresses["train"]:
        address["document"] = 0
        address["sentence"] = address["row"]
    opt = m.getOptimizer(lr=1e-3)
    try:
        # Put the automatic text first: it must not select the text modality
        # and displace the second row's explicitly supplied numeric target.
        questions = (What.future(0), What.supervised(1))
        assert all(data.what(q).available for q in questions)
        got = _answer_training_probe(m, opt, questions)
        assert got["mask"] == [False, True]
        assert got["actual_grad"] is not None
        assert torch.count_nonzero(got["actual_grad"][0]) == 0
        assert bool(got["actual_grad"][1].abs().sum() > 0)
        assert got["changed"][got["active_index"]]
    finally:
        data.source_addresses["train"] = saved_addresses
        m.End()
        m.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_supplied_text_scores_fixed_surface_and_masks_automatic_text():
    from test_output_walk import _model
    m = _model()
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.output_policy_weight = 1.0
    data = m.inputSpace.data
    saved_outputs = data.train_output
    opt = m.getOptimizer(lr=1e-3)
    questions = (What.supervised(0), What.present(1))
    try:
        data.train_output = ["9 minus 8", "2 times 3"] + list(saved_outputs[2:])
        got = _answer_training_probe(m, opt, questions)
        assert got["mask"] == [True, False]
        assert got["answer_loss"] > 0 and got["answer_requires_grad"]
        assert got["surface_grad"] is not None
        assert bool(got["surface_grad"][0].abs().sum() > 0)
        assert torch.count_nonzero(got["surface_grad"][1]) == 0
        assert "output_policy" in got["recorded"]
        c = m._last_answer_construction
        before = {name: getattr(c, name).detach().clone()
                  for name in ("actual", "concepts", "percepts", "surface")}
        with torch.no_grad():
            before_target, before_mask = m._what_answer_target(
                questions, torch.zeros(2, 1, 1), supervised_only=True)
            before_cost = float(m._reverse_event_loss(
                c.surface[before_mask], before_target[before_mask]))
        # A different desired answer changes scoring only. Reuse exactly
        # the existing construction and do not run forward/output again.
        data.train_output = ["1 plus 2", "3 plus 4"] + list(saved_outputs[2:])
        m._last_what_desired = tuple(data.what(q) for q in questions)
        with torch.no_grad():
            after_target, mask = m._what_answer_target(
                questions, torch.zeros(2, 1, 1), supervised_only=True)
            after_cost = float(m._reverse_event_loss(
                c.surface[mask], after_target[mask]))
        assert mask.tolist() == [True, False]
        assert not torch.equal(before_target, after_target)
        assert abs(after_cost - before_cost) > 1e-6
        for name, value in before.items():
            torch.testing.assert_close(getattr(m._last_answer_construction, name),
                                       value, rtol=0, atol=0)
    finally:
        data.train_output = saved_outputs
        m.End()
        m.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.parametrize("concept_width", _NATIVE_CONCEPT_WIDTHS)
def test_native_thinking_changes_the_owned_conceptual_answer(tmp_path, concept_width):
    from What import LTMSlot
    from Layers import WhatInteractionMemory
    from test_output_walk import _capture_program_probe
    m = _native_answer_model(tmp_path, False, concept_width=concept_width)
    m.eval()
    m.what_thinking_iterations = 2
    memory = WhatInteractionMemory(batch=2, capacity=8)
    object.__setattr__(m.symbolSpace, "what_memory", memory)
    questions = (What.supervised(0), What.supervised(1))
    try:
        with torch.no_grad():
            u = _capture_program_probe(m, ["1 plus 2", "3 plus 4"])
            base, _ = m._materialize_entries(
                u.answer_program, torch.zeros(2, 3, concept_width), m._walk_budget())
            module = m._ltm_attention(concept_width, 136, device=base.device, dtype=base.dtype)
            module["value"].weight.fill_(0.003)
            module["out"].weight.copy_(torch.eye(concept_width) * 0.1)
            response = torch.linspace(0.1, 0.3, 136)
            for b in range(2):
                memory.append_what_slot(LTMSlot(input=torch.zeros(136), output=response), b=b)
            deltas = torch.stack([m._attend_ltm(base[b, 0], [response]) for b in range(2)])
            assert bool(deltas.abs().sum() > 0)
            d = m._resolve_answer(u, questions)
            assert m.ltm_attention is module
            assert d.conceptual_answer.shape == (2, 3, concept_width)
            torch.testing.assert_close(d.conceptual_answer[:, 0], base[:, 0] + deltas)
            torch.testing.assert_close(d.conceptual_answer[:, 1:], base[:, 1:])
            idea = m._materialize_answer_idea(u, d, questions)[0]
            torch.testing.assert_close(idea, d.conceptual_answer)
            # The referent is an owned full-width leaf identified by its
            # WORD row, not a perceptual position padded up to concept_width.
            entry = u.answer_program[0]
            owner = m._concept_owner()
            word_id = owner._concept_allocator.word_obj_meta["plus"][0]
            word_row = owner._csw_row_of(word_id)
            found = (entry.word_rows == word_row).nonzero().reshape(-1)
            assert found.numel() == 1
            query = m._referent_representation(
                u, 0, ("1", "plus", "2"), "plus", base[:1], detach=False)
            torch.testing.assert_close(query[0, 0], entry.leaves[int(found[0])])
            # Later state changes cannot revise the resolved thinking result.
            _capture_program_probe(m, ["5 plus 6", "7 plus 8"])
            memory.append_what_slot(LTMSlot(input=torch.ones(136),
                                           output=torch.full((136,), 8.0)), b=0)
            again = m._materialize_answer_idea(u, d, questions)[0]
            torch.testing.assert_close(again, idea, rtol=0, atol=0)
    finally:
        m.End()
        m.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.parametrize("concept_width", _NATIVE_CONCEPT_WIDTHS)
@pytest.mark.parametrize("legacy_only", [False, True])
def test_native_checkpoint_restores_active_answer_widths(tmp_path, legacy_only, concept_width):
    from test_output_walk import _capture_program_probe
    m = _native_answer_model(tmp_path, False, concept_width=concept_width)
    fresh = None
    try:
        with torch.no_grad():
            u = _capture_program_probe(m, ["1 plus 2", "3 plus 4"])
            m.reverseOutput(u, m.resolveAnswer(u, (What.supervised(0), What.supervised(1))))
            m.question_conditioners[str(concept_width)].weight.fill_(0.02)
            m.conceptualSpace.synthesis_layer.raw_L[1, 0] = 0.17
            m.outputSpace.percept_adapter.raw_L[32, 3] = -0.11
            if legacy_only:
                narrow = m._question_conditioner(136, device=torch.device("cpu"), dtype=torch.float32)
                narrow.weight.fill_(0.03)
                del m.question_conditioners[str(concept_width)]
        path = tmp_path / "native.ckpt"
        m.save_weights(str(path))
        folder = tmp_path / "fresh"
        folder.mkdir()
        fresh = _native_answer_model(folder, False, concept_width=concept_width)
        assert fresh.load_weights(str(path), strict=True)
        for before, after in (
                (m.conceptualSpace.synthesis_layer, fresh.conceptualSpace.synthesis_layer),
                (m.perceptualSpace.synthesis_layer, fresh.perceptualSpace.synthesis_layer),
                (m.outputSpace.percept_adapter, fresh.outputSpace.percept_adapter)):
            assert before.nInput == after.nInput
            for key, value in before.state_dict().items():
                torch.testing.assert_close(after.state_dict()[key], value, rtol=0, atol=0)
        if legacy_only:
            assert torch.count_nonzero(fresh.question_conditioners[str(concept_width)].weight) == 0
            torch.testing.assert_close(fresh.question_conditioners["136"].weight,
                                       m.question_conditioners["136"].weight, rtol=0, atol=0)
        else:
            torch.testing.assert_close(fresh.question_conditioners[str(concept_width)].weight,
                                       m.question_conditioners[str(concept_width)].weight, rtol=0, atol=0)
        opt = fresh.getOptimizer(lr=1e-3)
        ids = [id(p) for group in opt.param_groups for p in group["params"]]
        assert all(ids.count(id(p)) == 1 for p in fresh.synthesis_parameters())
    finally:
        for model in (m, fresh):
            if model is not None:
                model.End()
                model.symbolSpace.soft_reset()
        torch._dynamo.reset()
