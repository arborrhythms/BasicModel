"""The identity path stays in the cube after learned chart maps move."""
import torch
import pytest

from Language import LiftLayer, LowerLayer, SurfaceLayer, VerbLayer


def test_native_synthesis_adds_in_chart_and_keeps_its_inverse():
    from Language import SynthesizeLayer
    from Layers import cube_chart, cube_unchart
    layer = SynthesizeLayer(4, 4)
    layer.meaning_layout = (2, 4)
    layer.configure_identity_domain()
    left, right = torch.tensor([[.2, .8, 0., 1.]]), torch.tensor([[.3, .7, .5, .5]])
    parent = layer.compose(left, right)
    assert bool(((parent >= 0) & (parent <= 1)).all())
    a, b = layer.generate(parent)
    torch.testing.assert_close(layer.compose(a, b), parent, atol=1e-6, rtol=1e-5)
    recovered = cube_unchart(cube_chart(parent[..., :2]) - cube_chart(left[..., :2]))
    torch.testing.assert_close(recovered, right[..., :2], atol=1e-6, rtol=1e-5)


def test_lift_chart_preserves_cube_under_signed_linear_updates():
    layer = LiftLayer(4, 4)
    with torch.no_grad():
        layer._sigma.layer.bias.fill_(-.25)
    left = torch.tensor([[0., .2, .4, .6]], requires_grad=True)
    right = torch.tensor([[0., .3, .5, .7]], requires_grad=True)
    value = layer.compose(left, right)
    assert torch.isfinite(value).all() and ((value >= 0) & (value <= 1)).all()
    # The declared inverse is recomposition, not recovery of both unknown
    # operands. This includes the old negative-output initialization cause.
    a, b = layer.generate(value)
    torch.testing.assert_close(layer.compose(a, b), value, atol=1e-6, rtol=1e-5)
    value.sum().backward()
    assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()


def test_surface_content_chart_and_witness_inverse():
    layer = SurfaceLayer(4, 4)
    with torch.no_grad():
        layer.marker_map.bias.fill_(-.3)
    marker = torch.tensor([[.2, .3, .4, .5]])
    content = torch.tensor([[.3, .4, .5, .6]])
    value = layer.compose(marker, content)
    assert ((value >= 0) & (value <= 1)).all()
    for reference, side in ((marker, 'left'), (content, 'right')):
        a, b = layer.reconstruct(value, reference, side)
        torch.testing.assert_close(a, marker, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(b, content, atol=1e-6, rtol=1e-5)
    a, b = layer.generate(value)
    torch.testing.assert_close(layer.compose(a, b), value, atol=1e-6, rtol=1e-5)


def test_verb_translation_is_inside_cube_chart_and_inverts():
    layer = VerbLayer(4, 4)
    with torch.no_grad():
        layer._verb_shift.fill_(-.1)
        layer._verb_spec.bias.fill_(.3)
    value = torch.tensor([[.2, .3, .4, .5]])
    verb = torch.tensor([[.4, .5, .6, .7]])
    mapped = layer.apply_verb(value, verb)
    assert ((mapped >= 0) & (mapped <= 1)).all()
    torch.testing.assert_close(layer.unapply_verb(mapped, verb), value, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('value', [-.001, 1.001, float('nan')])
def test_lower_rejects_non_cube_identity_input(value):
    with pytest.raises(RuntimeError, match='identity map'):
        LowerLayer(4, 4).compose(torch.zeros(1, 4), torch.full((1, 4), value))


def test_chart_consumers_stay_in_cube_after_both_signs_of_learning():
    from Layers import cube_translate
    from Attention import PrimedSymbolReader
    from Models import BasicModel
    values = torch.tensor([[[0., .2, .5, .8, 1.]]])
    reader = PrimedSymbolReader()
    model = BasicModel()
    module = model._question_conditioner(5, device=values.device, dtype=values.dtype)
    context = torch.ones(1, module.in_features)
    torch.testing.assert_close(reader.consume(values, values[:, 0]), values, rtol=0, atol=0)
    torch.testing.assert_close(model._condition_answer_on_question(values, context), values, rtol=0, atol=0)
    for shift in (-2., 2.):
        with torch.no_grad():
            reader.consume_gate.fill_(shift)
            module.weight.fill_(shift)
        for result in (reader.consume(values, values[:, 0]),
                       model._condition_answer_on_question(values, context)):
            assert bool((torch.isfinite(result) & (result >= 0) & (result <= 1)).all())
    x = values[..., 1:-1]
    torch.testing.assert_close(cube_translate(cube_translate(x, .2), -.2), x,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('gain', (0., .5, 1.))
def test_wide_cube_identity_length_is_not_a_compose_activation(gain):
    from types import SimpleNamespace
    from Language import ConjunctionLayer, DisjunctionLayer, OperationSelectionLayer, LanguageSpace
    from Expectation import SymbolRead
    layer = OperationSelectionLayer(d_model=4, ops=[ConjunctionLayer(), DisjunctionLayer()], chooser='mlp')
    values = torch.tensor([[[1., 1., 1., .5], [1., .8, 1., 1.]]])
    layer._symbol_expectation_read = SymbolRead(values, torch.tensor([1., 0.]).expand(1,2,2),
        gain, 1e-6, 'clip', identities=torch.tensor([1,2]))
    host = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=layer),
        _structural_context=lambda **kw: None, _compose_binary_rules=(), _compose_unary_rules=())
    state = (values, torch.tensor([2]), torch.ones(1,2,dtype=torch.long), torch.zeros(1,2,dtype=torch.long),
             torch.full((1,2),-1,dtype=torch.long), torch.zeros(1,2))
    calls = []
    hook = layer.chooser.register_forward_pre_hook(lambda *_: None)
    original = layer._stacked_reduced
    def record(x, context=None, references=None, activations=None):
        calls.append(activations.clone())
        result = original(x, context, references, activations)
        assert bool(((result >= 0) & (result <= 1)).all())
        return result
    layer._stacked_reduced = record
    try:
        LanguageSpace.choose_operation(host, state, torch.tensor([True]), sample=False)
    finally:
        hook.remove()
    torch.testing.assert_close(calls[0], torch.ones(1,2), atol=0, rtol=0)
    if gain:
        torch.testing.assert_close(calls[1], torch.full((1,2),1-gain), atol=0, rtol=0)
    torch.testing.assert_close(state[0], values, rtol=0, atol=0)


def test_union_is_an_exact_join_including_cube_rails():
    from Language import UnionLayer
    x = torch.tensor([[1., .99, 0., .3]], requires_grad=True)
    y = torch.tensor([[1., .98, .1, .4]], requires_grad=True)
    result = UnionLayer().compose(x, y)
    torch.testing.assert_close(result, torch.maximum(x, y), rtol=0, atol=0)
    result.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
