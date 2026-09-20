"""Prepared answer reasoning must finish before surface realization begins."""
import pytest
import torch

from What import What
from test_output_path_supervised import _native_answer_model
from test_output_walk import _capture_program_probe, _stop


@pytest.mark.parametrize("output_loop", [False, True])
def test_realization_uses_a_held_derivation_without_running_resolution(
        tmp_path, monkeypatch, output_loop):
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


def test_prepared_answer_generates_more_words_than_the_captured_input(tmp_path):
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
            # Make an explicit concluded idea. A semantic input to generation
            # can differ from the input sentence; its direction comes from the
            # actual understood first row, rather than an input surface witness.
            direction = held.conceptual_answer[0].sum(0)
            direction = torch.nn.functional.normalize(direction, dim=0)
            assert bool(direction.abs().sum() > 0)
            ideas = torch.zeros_like(held.conceptual_answer)
            ideas[0, 0] = direction
            ideas[1, 0] = direction * .25
            held = replace(held, conceptual_answer=ideas)
            language = model.languageSpace
            policy = language.generate_policy
            choice = list(language._generate_binary_names).index("sum")
            # Sum's witness-free inverse halves its parent. Expand values of
            # length 1 and .5, then emit .25: four leaves in row 0, one in row 1.
            policy.weight.zero_()
            policy.bias.fill_(-1000)
            policy.bias[-1] = 0
            policy.weight[choice].copy_(4 * direction[:policy.in_features])
            policy.bias[choice] = -1.5
            construction = model.reverseOutput(understanding, held)
            lengths = (construction.concepts.abs().amax(-1) > 0).sum(-1)
        input_lengths = [int(entry.leaves.shape[0])
                         for entry in understanding.answer_program]
        assert input_lengths == [1, 1]
        assert lengths.tolist() == [4, 1]
        assert int(lengths[0]) > input_lengths[0]
        assert not bool(model._output_truncated.any())
        assert construction.derivation is held
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_public_boundary_prepares_target_free_metadata_before_realization(tmp_path):
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
def test_prepared_concept_handoff_cuts_answer_state_gradient(tmp_path, monkeypatch, output_loop):
    from dataclasses import replace
    import Models

    monkeypatch.setattr(Models, "_GRAD_ANCHORS", {})
    model = _native_answer_model(tmp_path, output_loop)
    model.eval()
    if output_loop:
        _stop(model)
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ["1 plus 2", "3 plus 4"])
            resolved = model.resolveAnswer(
                understanding, (What.supervised(0), What.supervised(1)))
        source = resolved.conceptual_answer.detach().clone().requires_grad_()
        held = replace(resolved, conceptual_answer=source)
        construction = model.reverseOutput(understanding, held)
        loss = construction.percepts.square().mean()
        gradient = torch.autograd.grad(loss, source, allow_unused=True, retain_graph=True)[0]
        assert gradient is None or not bool(gradient.abs().any()), (
            "output matching must treat the concluded idea as given")
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        assert any(g is not None and bool(g.abs().any()) for g in gradients), (
            "given state must still train generation's parameters")
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_answer_loss_trains_an_executed_shared_inverse_without_state_credit(tmp_path, monkeypatch):
    from dataclasses import replace

    model = _native_answer_model(tmp_path, True)
    model.eval()
    try:
        with torch.no_grad():
            understanding = _capture_program_probe(model, ["1 plus 2", "3 plus 4"])
            held = model.resolveAnswer(understanding, (What.supervised(0), What.supervised(1)))
            language = model.languageSpace
            index = list(language._generate_binary_names).index("lower")
            operator = language._generate_binary_ops[index]
            leaf = torch.full((2, held.conceptual_answer.shape[-1]), .08)
            parent = operator.compose(leaf, leaf)
            children = operator.generate(parent)
            threshold = (parent.norm(dim=-1).min()
                         + torch.stack([child.norm(dim=-1).max() for child in children]).max()) / 2
            assert all(bool((child.norm(dim=-1) < threshold).all()) for child in children)
            assert bool((parent.norm(dim=-1) > threshold).all())
            state = torch.zeros_like(held.conceptual_answer)
            state[:, 0] = parent
        state.requires_grad_()

        def one_split(top):
            # A tensor-only fixed policy exercises the real shared inverse.
            # Root expands once; its smaller children are emitted.
            scores = top.new_full((top.shape[0], language.generate_policy.out_features), -1000.)
            scores[:, -1] = 0.
            scores[:, index] = top.norm(dim=-1) - threshold
            return scores

        monkeypatch.setattr(language, "generate_policy_logits", one_split)
        construction = model.reverseOutput(understanding, replace(held, conceptual_answer=state))
        assert (construction.concepts.abs().amax(-1) > 0).sum(-1).tolist() == [2, 2]
        loss = construction.percepts.sum()
        parameters = tuple(operator.parameters())
        owned = {id(p) for p in model._shared_representation_parameters(model.getOptimizer(lr=.001))}
        assert parameters and all(id(p) in owned for p in parameters)
        gradients = torch.autograd.grad(loss, (state, *parameters), allow_unused=True)
        assert gradients[0] is None
        assert any(g is not None and bool(g.abs().any()) for g in gradients[1:])
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
            model.what(questions, model.inputSpace.prepInput(["1 plus 2", "3 plus 4"]))
            construction = model._last_answer_construction
            model.reverseOutput(model._last_understanding, construction.derivation)
        assert events == ["reconstruct", "resolve", "generate", "generate"]
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
