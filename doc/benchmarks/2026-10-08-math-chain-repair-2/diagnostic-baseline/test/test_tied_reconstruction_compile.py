"""The separate reconstruction graph must cache its backward program."""
import importlib

import pytest
import torch

from test_tied_reconstruction_migration import _model
from test_reverse_traversal import _run


def test_separate_reconstruction_resolves_the_auto_backend(monkeypatch):
    from types import SimpleNamespace
    from Models import BasicModel
    import util

    monkeypatch.setattr(util, "TheCompileBackend", "auto")
    objective = lambda x: (x.sin() + x.cos()).square().sum()
    owner = SimpleNamespace(_reconstruct_sentences=objective,
                            _recon_placement=lambda: "compiled")
    x = torch.linspace(-.5, .6, 16, device="cpu", requires_grad=True)
    try:
        expected, = torch.autograd.grad(objective(x), (x,))
        value = BasicModel._compiled_reconstruct(owner)(x)
        for retain in (True, False):
            gradient, = torch.autograd.grad(value, (x,), retain_graph=retain)
            torch.testing.assert_close(gradient, expected)
    finally:
        torch._dynamo.reset()


def test_word_end_scoring_accepts_dynamic_target_windows():
    """A staged candidate window must not constrain the dynamic input P axis."""
    from types import SimpleNamespace
    from Models import BasicModel

    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=0.1)
    bank_n = torch.eye(2, device="cpu")
    bank_bytes = torch.tensor([[[97, 98, 99, 0, 0], [97, 98, 99, 100, 0]]], device="cpu")
    bank_valid = torch.tensor([[[True, True, True, False, False],
                                [True, True, True, True, False]]], device="cpu")
    word = torch.tensor(0, device="cpu")

    def score(idea, target, valid):
        return BasicModel._byte_word_cost(
            owner, idea, word, bank_n.unsqueeze(0), bank_bytes, bank_valid,
            target, valid, True)

    try:
        compiled = torch.compile(score, backend="eager", fullgraph=True)
        for width in (3, 8):
            idea = torch.tensor([[1., .25]], device="cpu", requires_grad=True)
            target = torch.zeros(1, 1, width, dtype=torch.long, device="cpu")
            target[0, 0, :3] = torch.tensor([97, 98, 99], device="cpu")
            valid = target != 0
            expected = score(idea, target, valid)
            expected_gradient, = torch.autograd.grad(expected.sum(), (idea,))
            for tensor in (target, valid):
                torch._dynamo.mark_dynamic(tensor, 2, min=3, max=8192)
            actual = compiled(idea, target, valid)
            gradient, = torch.autograd.grad(actual.sum(), (idea,))
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(gradient, expected_gradient)
    finally:
        torch._dynamo.reset()


def test_cached_backward_supports_retained_reads_after_an_ordinary_step(monkeypatch):
    """A cached backward must not assume every future call is single-use."""
    from types import SimpleNamespace
    from Models import BasicModel
    from torch._functorch import config
    import util

    monkeypatch.setattr(util, "TheCompileBackend", "eager")
    def objective(x):
        return (x.sin() + x.cos()).tanh().square().sum()
    owner = SimpleNamespace(_reconstruct_sentences=objective,
                            _recon_placement=lambda: "compiled")
    x = torch.linspace(-.5, .6, 32, device="cpu", requires_grad=True)
    try:
        with config.patch(donated_buffer=True):
            compiled = BasicModel._compiled_reconstruct(owner)
            torch.autograd.grad(compiled(x), (x,))  # populate the backward cache
            value = compiled(x)
            assert config.donated_buffer is True
            expected, = torch.autograd.grad(objective(x), (x,))
            for retain in (True, True, False):
                gradient, = torch.autograd.grad(value, (x,), retain_graph=retain)
                torch.testing.assert_close(gradient, expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.slow
def test_reconstruction_compile_retains_buffers_for_joint_gradient_reads(tmp_path, monkeypatch):
    """Reconstruction is differentiated more than once by the joint rule."""
    from Models import _ensure_grad_anchors
    from torch._functorch import config
    from test_meronomy_ladder import _build_ladder_variant
    from test_reverse_traversal import _stage_packed
    import util

    monkeypatch.setattr(util, "TheCompileBackend", "eager")
    monkeypatch.setenv("BASICMODEL_RECON_PLACEMENT", "compiled")
    _ensure_grad_anchors(torch.device("cpu"))
    model = _build_ladder_variant(tmp_path, "retained_recon", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>32</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>32</serialWordBuckets>"),
        ("<packSentences>false</packSentences>",
         "<packSentences>false</packSentences>\n      <reconstructInLoop>true</reconstructInLoop>")])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    try:
        _stage_packed(model, [["aa bb cc", "dd ee ff", "gg hh ii"]])
        with torch.no_grad():
            state = model._forward_with_compiled_sentence_state(None)
        model._publish_compiled_sentence_state(state)
        roots = model._tensor_sentence_roots_live.detach().clone().requires_grad_()
        end = model._tensor_final_end_slots.detach().clone().requires_grad_()
        with config.patch(donated_buffer=True):
            result = model._compiled_reconstruct()(
                model._stm_single_S.detach().clone(), model._tensor_pushed_ideas.detach().clone(),
                roots, model._tensor_sentence_roots_depth.detach().clone(),
                end, model._tensor_final_end_depth.detach().clone())
            value = result[2].sum()
            assert config.donated_buffer is True  # no global configuration mutation
            expected = None
            for retain in (True, True, False):
                gradients = torch.autograd.grad(value, (roots, end), retain_graph=retain)
                assert all(g.isfinite().all() and g.abs().sum() > 0 for g in gradients)
                if expected is not None:
                    for gradient, previous in zip(gradients, expected):
                        torch.testing.assert_close(gradient, previous)
                expected = gradients
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.slow
def test_separate_reconstruction_does_not_retrace_backward_every_step(tmp_path, monkeypatch):
    import util

    monkeypatch.setenv("BASICMODEL_RECON_PLACEMENT", "compiled")
    monkeypatch.setattr(util, "TheCompileBackend", "eager")
    model = _model(tmp_path)
    calls = []
    try:
        _run(model, ["aa bb cc"])
        for module_name in ("cond", "while_loop"):
            module = importlib.import_module("torch._higher_order_ops." + module_name)
            original = module.materialize_as_graph

            def observe(*args, _original=original, _name=module_name, **kwargs):
                calls.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(module, "materialize_as_graph", observe)
        reconstruct = model._compiled_reconstruct()
        counts = []
        for index in range(3):
            end = (model._tensor_final_end_slots.detach().clone() + index * .001).requires_grad_()
            result = reconstruct(
                model._stm_single_S.detach().clone(),
                model._tensor_pushed_ideas.detach().clone(),
                model._tensor_sentence_roots_live.detach().clone(),
                model._tensor_sentence_roots_depth.detach().clone(),
                end, model._tensor_final_end_depth.detach().clone())
            compose = tuple(p for p in model.languageSpace._tree_layer(2).parameters()
                            if p.requires_grad)
            gradient, *operator_gradients = torch.autograd.grad(
                result[2].sum(), (end, *compose), allow_unused=True)
            assert gradient.isfinite().all() and gradient.abs().sum() > 0
            used = [g for g in operator_gradients if g is not None]
            assert used and all(g.isfinite().all() for g in used)
            assert any(g.abs().sum() > 0 for g in used)
            counts.append(len(calls))
        assert counts[2] == counts[1], (
            "a warmed reconstruction should execute its backward program, "
            f"not materialize it again on every optimizer call: {counts}")
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
