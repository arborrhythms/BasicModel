"""Code-specific, masked concept readouts and connection-wise lasso."""
import math

import pytest
import torch

from Layers import (
    ConceptsFromPercepts,
    GatedConceptsFromPercepts,
    Layer,
    SigmaConceptsFromPercepts,
    SigmaLayer,
)


KINDS = [GatedConceptsFromPercepts, SigmaConceptsFromPercepts]


def test_gated_formula_includes_signed_evidence_weights_and_bias():
    layer = GatedConceptsFromPercepts(3, 2, context_dim=2).double()
    with torch.no_grad():
        layer.input_weights.copy_(torch.tensor([[1., -1.], [2., 3.], [-2., .5]]))
        layer.concept_bias.copy_(torch.tensor([[.3, -.7]]))
        layer.gate_logits.copy_(torch.tensor([[.2, -.3], [.5, .1], [-.4, .8]]))
        layer.context_weights.copy_(torch.arange(12).reshape(2, 3, 2) / 10.)
    evidence = torch.tensor([[[1., -.6, .4], [.2, .5, -1.]]], dtype=torch.float64)
    context = torch.tensor([[[.3, .5], [-.2, .7]]], dtype=torch.float64)
    logits = layer.gate_logits + torch.einsum("...d,dic->...ic", context,
                                             layer.context_weights)
    expected = torch.tanh(layer.concept_bias + (
        evidence.unsqueeze(-1) * layer.input_weights * logits.sigmoid()).sum(-2))
    torch.testing.assert_close(layer(evidence, context=context), expected)
    assert isinstance(layer, (ConceptsFromPercepts, Layer))


def test_sigma_is_existing_linear_sigma_plus_affine_bias_and_output_tanh():
    layer = SigmaConceptsFromPercepts(3, 2).double()
    assert isinstance(layer.sigma, SigmaLayer)
    with torch.no_grad():
        layer.input_weights.copy_(torch.tensor([[1., -1.], [.5, 2.], [-.2, .3]]))
        layer.concept_bias.copy_(torch.tensor([[.7, -.1]]))
    # Exact +/-1 poles are legal evidence; no atanh input chart is used.
    evidence = torch.tensor([[1., -1., 0.]], dtype=torch.float64)
    expected = torch.tanh(evidence @ layer.input_weights + layer.concept_bias)
    torch.testing.assert_close(layer(evidence), expected)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4), (2, 1, 3, 4)])
def test_arbitrary_leading_dimensions_and_named_entry_point(kind, shape):
    layer = kind(4, 3)
    evidence = torch.rand(shape) * 2 - 1
    seen = []
    hook = layer.register_forward_hook(lambda *args: seen.append(True))
    actual = layer.conceptsFromPercepts(evidence)
    hook.remove()
    assert seen == [True]
    assert actual.shape == (*shape[:-1], 3)
    assert ((actual >= -1) & (actual <= 1)).all()
    torch.testing.assert_close(actual, layer(evidence))


@pytest.mark.parametrize("kind", KINDS)
def test_masked_nan_padding_is_neutral_in_value_and_gradient(kind):
    layer = kind(4, 2).double()
    with torch.no_grad():
        layer.input_weights.fill_(.7)
        layer.concept_bias.fill_(-.3)
    evidence = torch.tensor([[1., .4, float("nan"), float("nan")]],
                            dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True, True, False, False])
    result = layer(evidence, mask=mask)
    clean = torch.tensor([[1., .4, 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(result, layer(clean))
    result.sum().backward()
    assert torch.isfinite(evidence.grad).all()
    assert (evidence.grad[..., 2:] == 0).all()
    assert (layer.input_weights.grad[2:] == 0).all()
    assert torch.isfinite(layer.input_weights.grad).all()


@pytest.mark.parametrize("kind", KINDS)
def test_missing_is_unknown_not_negative_evidence_or_bias_prior(kind):
    layer = kind(2, 1)
    with torch.no_grad():
        layer.input_weights.fill_(1.)
        layer.concept_bias.fill_(-.5)
    evidence = -torch.ones(2, 2, requires_grad=True)
    mask = torch.tensor([[False, False], [True, True]])
    result = layer(evidence, mask=mask)
    assert result[0, 0] == 0
    assert result[1, 0] < -.5
    layer.zero_grad()
    result[0].sum().backward()
    assert torch.count_nonzero(layer.input_weights.grad) == 0
    assert torch.count_nonzero(layer.concept_bias.grad) == 0


@pytest.mark.parametrize("kind", KINDS)
def test_two_references_suffice_and_eight_can_add_parallel_detail(kind):
    layer = kind(8, 2)
    with torch.no_grad():
        layer.input_weights.fill_(.25)
        layer.concept_bias.zero_()
    evidence = torch.ones(1, 8)
    serial = layer(evidence, mask=torch.tensor([True, True] + [False] * 6))
    parallel = layer(evidence)
    assert (serial > 0).all()  # no normalization by an eight-member budget
    assert (parallel > serial).all()
    # Car and tire may both be present: no competition across concepts.
    assert parallel.sum() > 1


@pytest.mark.parametrize("kind", KINDS)
def test_equal_rms_evidence_for_different_codes_is_distinguished(kind):
    layer = kind(3, 1)
    with torch.no_grad():
        layer.input_weights.copy_(torch.tensor([[1.], [-1.], [0.]]))
        layer.concept_bias.zero_()
    evidence = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    assert evidence[0].square().mean() == evidence[1].square().mean()
    result = layer(evidence)
    assert result[0, 0] > 0 and result[1, 0] < 0


def test_local_context_changes_participation_but_is_not_an_output_shortcut():
    layer = GatedConceptsFromPercepts(2, 2, context_dim=1)
    with torch.no_grad():
        layer.input_weights.copy_(torch.eye(2))
        layer.concept_bias.zero_()
        layer.context_weights.copy_(torch.tensor([[[4., 0.], [0., -4.]]]))
    evidence = torch.ones(2, 2, requires_grad=True)
    context = torch.tensor([[1.], [-1.]], requires_grad=True)
    result = layer(evidence, context=context)
    assert result[0, 0] > result[0, 1]
    assert result[1, 1] > result[1, 0]
    assert torch.count_nonzero(layer(torch.zeros_like(evidence), context=context)) == 0
    result.sum().backward()
    for value in (evidence, context, layer.context_weights, layer.gate_logits,
                  layer.input_weights, layer.concept_bias):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0


def test_neutral_and_broadcast_context():
    layer = GatedConceptsFromPercepts(3, 2, context_dim=2)
    evidence = torch.rand(2, 4, 3)
    context = torch.tensor([.3, -.2])
    torch.testing.assert_close(layer(evidence), layer(evidence, context=torch.zeros(2)))
    torch.testing.assert_close(layer(evidence, context=context),
                               layer(evidence, context=context.expand(2, 4, 2)))


@pytest.mark.parametrize("kind", KINDS)
def test_lasso_penalizes_connections_not_percepts_or_bias(kind):
    layer = (kind(3, 2, l1_lambda=.4) if kind is SigmaConceptsFromPercepts
             else kind(3, 2))
    with torch.no_grad():
        layer.input_weights.copy_(torch.tensor([[1., -2.], [0., .5], [-1., 0.]]))
    evidence = torch.ones(1, 3, requires_grad=True)
    before = evidence.detach().clone()
    layer(evidence)
    penalty = layer.regularization_loss()
    assert penalty.ndim == 0 and penalty.device == layer.input_weights.device
    if kind is SigmaConceptsFromPercepts:
        torch.testing.assert_close(penalty, torch.tensor(.9))  # .4 * 4.5 / 2
        penalty.backward()
        torch.testing.assert_close(layer.input_weights.grad,
                                   layer.input_weights.sign() * .2)
        assert layer.concept_bias.grad is None and evidence.grad is None
    else:
        assert penalty == 0  # no automatic pressure toward a two-reference field
    torch.testing.assert_close(evidence, before)


def test_proximal_step_zeros_edges_per_output_with_correct_lr_normalization():
    layer = SigmaConceptsFromPercepts(3, 2, l1_lambda=.4)
    with torch.no_grad():
        layer.input_weights.copy_(torch.tensor([[.05, .3], [-.1, -.8], [0., .1]]))
        layer.concept_bias.fill_(.2)
    assert layer.proximal_step_(.5) is layer
    expected = torch.tensor([[0., .2], [0., -.7], [0., 0.]])
    torch.testing.assert_close(layer.input_weights, expected)
    torch.testing.assert_close(layer.concept_bias, torch.full((1, 2), .2))
    # The same percept has no effect on concept 0 and does affect concept 1.
    delta = layer(torch.tensor([[1., 0., 0.]])) - layer(torch.zeros(1, 3))
    assert delta[0, 0] == 0 and delta[0, 1] > 0
    # A zero edge is not a permanently deleted relationship.
    optimizer = torch.optim.SGD(layer.parameters(), lr=.5)
    optimizer.zero_grad()
    (-layer(torch.tensor([[1., 0., 0.]]))[0, 0]).backward()
    optimizer.step()
    layer.proximal_step_(.5)
    assert layer.input_weights[0, 0] > 0


def test_disabled_prox_and_zero_lr_preserve_parameters_exactly():
    for strength, lr in [(0., .2), (.3, 0.)]:
        layer = SigmaConceptsFromPercepts(4, 2, l1_lambda=strength)
        before = {name: value.clone() for name, value in layer.state_dict().items()}
        layer.proximal_step_(lr)
        for name, value in layer.state_dict().items():
            assert torch.equal(value, before[name])


@pytest.mark.parametrize("value", [-1., math.inf, math.nan])
def test_invalid_lasso_strength_and_step_size_rejected(value):
    with pytest.raises(ValueError, match="l1_lambda"):
        SigmaConceptsFromPercepts(2, 2, l1_lambda=value)
    with pytest.raises(ValueError, match="learning_rate"):
        SigmaConceptsFromPercepts(2, 2).proximal_step_(value)


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_shapes_and_non_boolean_masks_rejected(kind):
    layer = kind(3, 2)
    with pytest.raises(ValueError, match="nInput"):
        kind(0, 2)
    with pytest.raises(ValueError, match="nOutput"):
        kind(2, 0)
    with pytest.raises(ValueError, match="nInput"):
        layer(torch.ones(2, 4))
    with pytest.raises(TypeError, match="floating"):
        layer(torch.ones(2, 3, dtype=torch.long))
    with pytest.raises(TypeError, match="mask must be boolean"):
        layer(torch.ones(2, 3), mask=torch.ones(3))
    with pytest.raises(NotImplementedError, match="collective field"):
        layer.reverse(torch.ones(2, 2))


def test_invalid_context_rejected_not_silently_broadcast_or_ignored():
    evidence = torch.ones(2, 3)
    with pytest.raises(ValueError, match="context_dim"):
        GatedConceptsFromPercepts(3, 2)(evidence, context=torch.ones(1))
    with pytest.raises(ValueError, match="context_dim"):
        GatedConceptsFromPercepts(3, 2, context_dim=2)(evidence, context=torch.ones(1))
    with pytest.raises(RuntimeError):
        GatedConceptsFromPercepts(3, 2, context_dim=2)(
            evidence, context=torch.ones(4, 2))
    with pytest.raises(ValueError, match="no context"):
        SigmaConceptsFromPercepts(3, 2)(evidence, context=torch.ones(1))
    with pytest.raises(ValueError, match="context_dim"):
        GatedConceptsFromPercepts(3, 2, context_dim=-1)


@pytest.mark.parametrize("kind", KINDS)
def test_state_dict_roundtrip_gradcheck_and_fullgraph_compile(kind):
    kwargs = ({"context_dim": 2} if kind is GatedConceptsFromPercepts
              else {"l1_lambda": .01})
    layer = kind(3, 2, **kwargs).double()
    clone = kind(3, 2, **kwargs).double()
    clone.load_state_dict(layer.state_dict(), strict=True)
    evidence = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True], [True, True, True]])
    context = (torch.rand(2, 2, dtype=torch.float64)
               if kind is GatedConceptsFromPercepts else None)
    torch.testing.assert_close(layer(evidence, mask=mask, context=context),
                               clone(evidence, mask=mask, context=context))
    assert torch.autograd.gradcheck(lambda x: layer(x, mask=mask, context=context),
                                    (evidence,))
    compiled = torch.compile(clone, backend="eager", fullgraph=True)
    eager_result = layer(evidence, mask=mask, context=context)
    compiled_result = compiled(evidence, mask=mask, context=context)
    torch.testing.assert_close(eager_result, compiled_result)
    eager_result.sum().backward()
    compiled_result.sum().backward()
    for original, copied in zip(layer.parameters(), clone.parameters()):
        torch.testing.assert_close(original.grad, copied.grad)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("kind", KINDS)
def test_mps_forward_backward_and_prox(kind):
    layer = kind(4, 2).to("mps")
    evidence = torch.rand(2, 4, device="mps", requires_grad=True)
    layer(evidence).square().mean().backward()
    assert torch.isfinite(evidence.grad).all()
    if isinstance(layer, SigmaConceptsFromPercepts):
        layer.l1_lambda = .1
        layer.proximal_step_(.01)
    assert layer.regularization_loss().device.type == "mps"
