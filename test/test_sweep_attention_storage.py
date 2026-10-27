"""Candidate scoring preserves the dense affine while retaining shared input once."""
import copy
import pytest
import torch
from CandidateAttention import CandidateAttention


def dense(layer, context, candidates):
    context = context / context.norm(dim=-1, keepdim=True).clamp_min(1.)
    inputs = torch.cat((context[:, None].expand(-1, candidates.shape[1], -1), candidates), -1)
    return layer.readout(layer.hidden(inputs.detach())).squeeze(-1)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_shared_reader_matches_dense_scores_and_every_parameter_gradient(compiled, dtype):
    layer = CandidateAttention(17, 7, hidden_width=9, hidden_layers=2).to(dtype=dtype)
    with torch.no_grad():
        for value in layer.parameters():
            value.copy_(torch.linspace(-.13, .17, value.numel(), dtype=dtype).reshape_as(value))
    reference = copy.deepcopy(layer)
    context = torch.arange(51, dtype=dtype).reshape(3, 17).requires_grad_()
    candidates = torch.linspace(-1., 1., 105, dtype=dtype).reshape(3, 5, 7).requires_grad_()
    expected = dense(reference, context, candidates)
    call = torch.compile(layer, backend='aot_eager', fullgraph=True) if compiled else layer
    try:
        actual = call(context, candidates)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for parameter, original in zip(layer.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0)
        assert context.grad is candidates.grad is None
    finally:
        if compiled: torch._dynamo.reset()


def test_native_reader_does_not_retain_repeated_context_for_each_candidate():
    # Recorded native affine width is 1,160,764; each key has D+8=1,008 values.
    width, key = 1160764, 1008
    layer = CandidateAttention(width-key, key, hidden_width=4)
    context = torch.ones(28, width-key)
    candidates = torch.ones(28, 3, key)
    saved = []
    def pack(value):
        saved.append(tuple(value.shape))
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = layer(context, candidates)
    assert result.shape == (28, 3)
    assert (84, width) not in saved and (28, 3, width) not in saved


def test_reader_input_reuse_keeps_the_original_parameter_gradient():
    layer = CandidateAttention(17, 7, hidden_width=9).double()
    with torch.no_grad():
        layer.readout.weight.fill_(.2)
    reference = copy.deepcopy(layer)
    context = torch.arange(34, dtype=torch.float64).reshape(2, 17).requires_grad_()
    candidates = torch.linspace(-1., 1., 42, dtype=torch.float64).reshape(2, 3, 7).requires_grad_()
    expected = dense(reference, context, candidates)
    actual = layer(context, candidates)
    with torch.no_grad():
        context.add_(5.)
        candidates.mul_(2.)
    expected.sum().backward()
    actual.sum().backward()
    for parameter, original in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0)
    assert context.grad is candidates.grad is None
