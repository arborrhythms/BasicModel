"""Prepared answer reasoning must finish before surface realization begins."""
import pytest
import torch

from What import What
from test_output_path_supervised import _native_answer_model
from test_output_walk import _capture_program_probe, _stop


def _with_identity_bank(understanding, codes):
    """Supply the lexical support of an explicit concluded working idea."""
    from dataclasses import replace

    record = understanding.sentence_records[-1]
    bank = record.primed
    batch, count = codes.shape[:2]
    bank = replace(bank, codes=codes, rows=torch.arange(count)[None].expand(batch, -1),
                   weights=codes.new_ones(batch, count),
                   own=torch.ones(batch, count, dtype=torch.bool),
                   bytes=bank.bytes[:, :1].expand(-1, count, -1),
                   byte_valid=bank.byte_valid[:, :1].expand(-1, count, -1))
    return replace(understanding, sentence_records=(replace(record, primed=bank),))


@pytest.mark.parametrize("output_loop", [False, True])
def test_realization_uses_a_held_derivation_without_running_resolution(
        tmp_path, monkeypatch, output_loop, eager_reading):
    model = _native_answer_model(tmp_path, output_loop)
    model.eval()
    if output_loop:
        _stop(model)
    questions = (What.supervised(0), What.supervised(1))
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ["1 plus 2", "3 plus 4"])
            held = model._resolve_answer(understanding, questions)

            def forbidden(*args, **kwargs):
                raise AssertionError("output realization must not run query or answer resolution")

            monkeypatch.setattr(model, "_resolve_answer", forbidden)
            monkeypatch.setattr(model, "_resolve_step", forbidden, raising=False)
            monkeypatch.setattr(model, "answer_query", forbidden)
            first = model.reverseOutput(understanding, held)
            _capture_program_probe(model, ["5 plus 6", "7 plus 8"])
            second = model.reverseOutput(understanding, held)
        assert first.derivation is held and second.derivation is held
        torch.testing.assert_close(second.actual, first.actual, rtol=0, atol=0)
        torch.testing.assert_close(second.concepts, first.concepts, rtol=0, atol=0)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_prepared_answer_generates_more_words_than_the_captured_input(tmp_path, eager_reading):
    """A fixed generate policy expands its answer; input actions supply no teacher.

    This verifies execution and independent termination, not learned language
    quality. The native benchmark separately measures sampled output lengths.
    """
    from dataclasses import replace

    model = _native_answer_model(tmp_path, True)
    model.eval()
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ["1", "2"])
            held = model.resolveAnswer(
                understanding, (What.supervised(0), What.supervised(1)))
            # An independently concluded four-leaf union and one terminal.
            # Paired forms are centroids; no additive lift inverse exists.
            language = model.languageSpace
            policy = language.generate_policy
            choice = list(language._generate_binary_names).index("union")
            operator = language._generate_binary_ops[choice]
            form_width = operator.meaning_layout[0]
            leaves = held.conceptual_answer.new_zeros(4, held.conceptual_answer.shape[-1])
            leaves[torch.arange(4), torch.arange(4)] = .6
            leaves[torch.arange(4), form_width + torch.arange(4)] = .5
            halves = operator.compose(leaves[::2], leaves[1::2])
            direction = operator.compose(halves[0], halves[1])
            assert bool(direction.abs().sum() > 0)
            ideas = torch.zeros_like(held.conceptual_answer)
            ideas[0, 0] = direction
            ideas[1, 0] = leaves[0]
            held = replace(held, conceptual_answer=ideas)
            policy.weight.zero_()
            policy.bias.fill_(-1000)
            policy.bias[-1] = 0
            policy.bias[choice] = 1
            codes = torch.cat((leaves, halves))[None].expand(2, -1, -1)
            understanding = _with_identity_bank(understanding, codes)
            construction = model.reverseOutput(understanding, held)
            lengths = (construction.concepts.abs().amax(-1) > 0).sum(-1)
        input_lengths = model.inputSpace._word_active_mask.sum(-1).tolist()
        assert input_lengths == [1, 1]
        assert lengths.tolist() == [4, 1]
        assert int(lengths[0]) > input_lengths[0]
        assert not bool(model._output_truncated.any())
        assert construction.derivation is held
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_public_boundary_prepares_target_free_metadata_before_realization(tmp_path, eager_reading):
    model = _native_answer_model(tmp_path, False)
    model.eval()
    questions = (What.supervised(0), What.supervised(1))
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ["1 plus 2", "3 plus 4"])
            held = model.resolveAnswer(understanding, questions)
            assert held.questions == questions
            constructed = model.reverseOutput(understanding, held)
            assert constructed.derivation is held
            with pytest.raises(TypeError, match="AnswerDerivation"):
                model.reverseOutput(understanding, questions)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.parametrize("output_loop", [False, True])
def test_prepared_concept_handoff_cuts_answer_state_gradient(tmp_path, monkeypatch, output_loop, eager_reading):
    from dataclasses import replace
    import Models

    monkeypatch.setattr(Models, "_GRAD_ANCHORS", {})
    model = _native_answer_model(tmp_path, output_loop)
    model.eval()
    if output_loop:
        _stop(model)
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ["1", "2"])
            resolved = model.resolveAnswer(
                understanding, (What.supervised(0), What.supervised(1)))
        # The gradient boundary consumes a given answer. An untrained input
        # derivation can erase its root, so supply a readable concluded idea.
        source = torch.full_like(resolved.conceptual_answer, .2).requires_grad_()
        understanding = _with_identity_bank(understanding, source[:, :1].detach())
        held = replace(resolved, conceptual_answer=source)
        construction = model.reverseOutput(understanding, held)
        loss = construction.percepts.square().mean()
        gradient = torch.autograd.grad(loss, source, allow_unused=True, retain_graph=True)[0]
        assert gradient is None or not bool(gradient.abs().any()), (
            "every answer reader treats the concluded idea as given")
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        assert any(g is not None and bool(g.abs().any()) for g in gradients), (
            "given state must still train generation's parameters")
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_percept_answer_keeps_meaning_inverse_and_state_out_of_credit(tmp_path, monkeypatch, eager_reading):
    from dataclasses import replace

    model = _native_answer_model(tmp_path, True)
    model.eval()
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ["1", "2"])
            held = model.resolveAnswer(understanding, (What.supervised(0), What.supervised(1)))
            language = model.languageSpace
            index = list(language._generate_binary_names).index("lower")
            operator = language._generate_binary_ops[index]
            form_width = operator.meaning_layout[0]
            marker = torch.full((2, held.conceptual_answer.shape[-1]), .5)
            leaf = marker.clone()
            marker[:, :form_width] = 0
            leaf[:, :form_width] = 0
            marker[:, 0] = .6
            leaf[:, 1] = .6
            operator.meaning_kernel.bias.fill_(-1)
            parent = operator.compose(marker, leaf)
            children = operator.reverse_blocks(parent, marker, reference_right=False)
            # Distinct forms make the centroid split progress. The declared
            # marker and content, not an invented balanced split, are recovered.
            torch.testing.assert_close(children[0], marker, rtol=0, atol=0)
            torch.testing.assert_close(children[1], leaf, rtol=0, atol=0)
            understanding = _with_identity_bank(understanding, torch.stack((marker, leaf), 1))
            threshold = ((parent-.5).norm(dim=-1).min()
                         + torch.stack([(child-.5).norm(dim=-1).max() for child in children]).max()) / 2
            assert all(bool(((child-.5).norm(dim=-1) < threshold).all()) for child in children)
            assert bool(((parent-.5).norm(dim=-1) > threshold).all())
            state = torch.zeros_like(held.conceptual_answer)
            state[:, 0] = parent
        state.requires_grad_()

        def one_split(top):
            # A tensor-only fixed policy exercises the real shared inverse.
            # Root expands once; its smaller children are emitted.
            scores = top.new_full((top.shape[0], language.generate_policy.out_features), -1000.)
            scores[:, -1] = 0.
            scores[:, index] = (top-.5).norm(dim=-1) - threshold
            return scores

        monkeypatch.setattr(language, "generate_policy_logits", one_split)
        construction = model.reverseOutput(understanding, replace(held, conceptual_answer=state))
        assert (construction.concepts.abs().amax(-1) > 0).sum(-1).tolist() == [2, 2]
        loss = construction.percepts.sum()
        parameters = tuple(operator.parameters())
        optimizer = model.getOptimizer(lr=.001)
        owners = model.objective_parameter_groups(optimizer)
        owned = {id(p) for p in owners['reconstruction']}
        assert parameters and all(id(p) in owned for p in parameters)
        gradients = torch.autograd.grad(loss, (state, *parameters), allow_unused=True, retain_graph=True)
        assert gradients[0] is None
        assert all(g is None or not bool(g.abs().any()) for g in gradients[1:])
        # Meaning inverses remain reconstruction-owned. With fixed selected
        # forms, a percept-only loss has no path through the meaning block.
        from ObjectiveOwnership import backward_owned
        backward_owned({'output': loss}, owners)
        assert all(p.grad is None for p in parameters)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.slow
def test_what_reconstructs_then_resolves_then_realizes(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    import util

    monkeypatch.setattr(util, "TheCompileBackend", "eager")
    model = _build_ladder_variant(tmp_path, "prepared_phase_order", [
        ("<training>", "<training><teacherReconstruction>true</teacherReconstruction>"
         "<reconstructInLoop>true</reconstructInLoop>"
         "<reconstructionPlacement>compiled</reconstructionPlacement>")])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.eval()
    events = []
    factory = model._compiled_reconstruct
    resolve = model.resolveAnswer
    realize = model.reverseOutput
    private_resolve = model._resolve_answer
    phase = ["understand"]

    def compilation():
        compiled = factory()

        def reconstruct(*args, **kwargs):
            assert phase[0] == "understand"
            events.append("reconstruct")
            return compiled(*args, **kwargs)
        return reconstruct

    def prepare(understanding, questions):
        assert understanding.input_reconstruction is not None
        assert events == ["reconstruct"]
        events.append("resolve")
        phase[0] = "resolve"
        result = resolve(understanding, questions)
        phase[0] = "prepared"
        return result

    def checked_private(*args, **kwargs):
        assert phase[0] == "resolve", "sentence paths must not resolve queries"
        return private_resolve(*args, **kwargs)

    def generate(understanding, derivation):
        assert phase[0] in ("prepared", "generated")
        events.append("generate")
        phase[0] = "generate"
        result = realize(understanding, derivation)
        phase[0] = "generated"
        return result

    monkeypatch.setattr(model, "_compiled_reconstruct", compilation)
    monkeypatch.setattr(model, "resolveAnswer", prepare)
    monkeypatch.setattr(model, "_resolve_answer", checked_private)
    monkeypatch.setattr(model, "reverseOutput", generate)
    try:
        with torch.no_grad():
            questions = (What.supervised(0), What.supervised(1))
            model.ask(questions, model.inputSpace.prepInput(["1 plus 2", "3 plus 4"]))
            construction = model._last_answer_construction
            model.reverseOutput(model._last_understanding, construction.derivation)
        assert events == ["reconstruct", "resolve", "generate", "generate"]
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
