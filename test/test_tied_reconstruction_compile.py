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


def _held_final_field(model, roots, offset=0.):
    """Keep the last presented field as an independent differentiable input."""
    ids = model.inputSpace._packed_sentence_ids
    index = ids.amax(-1).clamp_min(0)
    field = roots.gather(1, index[:, None, None].expand(-1, 1, roots.shape[-1]))[:, 0]
    end = (field.detach().reshape_as(model._tensor_final_end_slots).clone() + offset).requires_grad_()
    positions = torch.arange(roots.shape[1], device=roots.device)[None]
    held = torch.where((positions == index[:, None])[..., None], end.flatten(1)[:, None], roots)
    depth = model._tensor_sentence_roots_depth.gather(1, index[:, None])[:, 0].clone()
    return held, end, depth


def _supplied_inverse_fixture(model, monkeypatch):
    """Declare a differentiable numerical reading for the compiler contract.

    Fixed-identity byte readback is a discrete audit; legacy free search emits
    detached dictionary rows. Neither promises a nonzero numerical adjoint.
    This fixture instead supplies a known co-operand to the real tied inverse,
    as the existing affine-inverse certificate does. The model's production
    reconstruction compiler still owns capture, donation and backward caching.
    """
    language = model.languageSpace
    binary = language._tree_layer(2)
    index = list(binary.op_names).index('lift')
    op = getattr(binary.ops[index], 'gl', binary.ops[index])
    monkeypatch.setattr(op, 'percept_band', None)  # numerical content only
    width = int(model.conceptualSpace.stm.concept_dim)
    left = torch.linspace(.4, .65, width).reshape(1, width)
    right = torch.linspace(.55, .425, width).reshape(1, width)
    parent = op.compose(left, right).detach()
    witness = right.detach().clone()
    fields = model._tensor_sentence_roots_live
    model._tensor_sentence_roots_live = parent[:, None, None].expand(
        fields.shape[0], fields.shape[1], 3, width).clone().flatten(2)
    model._tensor_final_end_slots = parent[:, None].expand(
        fields.shape[0], 3, width).clone()

    def reconstruct(S, reference, roots, depths, end, end_depth):
        values = roots.reshape(roots.shape[0], -1, width)
        def condition(i, total):
            return i < values.shape[1]
        def body(i, total):
            value = values.gather(1, i.reshape(1, 1, 1).expand(
                values.shape[0], 1, width))[:, 0]
            older, newer = language.reverse_binary_step(value,
                torch.full((values.shape[0],), index),
                torch.ones(values.shape[0], dtype=torch.bool),
                reference=witness.expand(values.shape[0], -1))
            return i + 1, total + (older.square() + newer.square()).mean(-1)
        from Models import _carries_with_grad
        _, cost = torch.while_loop(condition, body, _carries_with_grad((
            torch.zeros((), dtype=torch.long), values.new_zeros(values.shape[0]))))
        return reference, S.sum(-1) * 0, cost, cost.bool(), cost[:, None]

    monkeypatch.setattr(model, '_reconstruct_sentences', reconstruct)
    model.__dict__.pop('_reconstruct_compiled', None)


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
        from reading_fixtures import use_eager_reading
        with pytest.MonkeyPatch.context() as staging, torch.no_grad():
            use_eager_reading(staging)
            state = model._forward_with_compiled_sentence_state(None)
        model._publish_compiled_sentence_state(state)
        _supplied_inverse_fixture(model, monkeypatch)
        roots = model._tensor_sentence_roots_live.detach().clone().requires_grad_()
        # Packed reconstruction owns each held field, not the later STM end.
        # Supply its final held field as an independent input alongside the others.
        reading_roots, end, end_depth = _held_final_field(model, roots)
        with config.patch(donated_buffer=True):
            result = model._compiled_reconstruct()(
                model._stm_single_S.detach().clone(), model._tensor_pushed_ideas.detach().clone(),
                reading_roots, model._tensor_sentence_roots_depth.detach().clone(),
                end, end_depth)
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
        from reading_fixtures import use_eager_reading
        with pytest.MonkeyPatch.context() as staging:
            use_eager_reading(staging)
            _run(model, ["aa bb cc"])
        _supplied_inverse_fixture(model, monkeypatch)
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
            held = model._tensor_sentence_roots_live.detach().clone()
            held, end, end_depth = _held_final_field(model, held, index * .001)
            result = reconstruct(
                end[:, 0],
                model._tensor_pushed_ideas.detach().clone(),
                held,
                model._tensor_sentence_roots_depth.detach().clone(),
                end, end_depth)
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
