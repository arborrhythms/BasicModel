"""Candidate scores are learned; support and discrete admission are distinct."""
import torch
import pytest
from CandidateAttention import CandidateAttention
from Layers import Layer
from AttentionTraversal import FieldTraversal


def test_layer_scores_one_value_per_candidate_and_keeps_reader_inputs_fixed():
    layer = CandidateAttention(3, 4, hidden_width=8, hidden_layers=2)
    assert isinstance(layer, Layer)
    context = torch.randn(2, 3, requires_grad=True)
    keys = torch.randn(2, 5, 4, requires_grad=True)
    scores = layer(context, keys)
    assert scores.shape == (2, 5)
    scores.sum().backward()
    assert context.grad is keys.grad is None
    assert layer.readout.bias.grad.item() == 10
    restored = CandidateAttention(3, 4, hidden_width=8, hidden_layers=2)
    restored.load_state_dict(layer.state_dict())
    torch.testing.assert_close(restored(context, keys), scores)


def test_scorer_compiles_with_ordinary_parameter_gradients():
    layer = CandidateAttention(3, 4, hidden_width=8)
    compiled = torch.compile(layer, backend='aot_eager', fullgraph=True)
    compiled(torch.ones(2, 3), torch.ones(2, 5, 4)).sum().backward()
    assert layer.readout.weight.grad is not None


def test_support_admits_negative_evidence_and_zero_has_no_pathwise_choice_gradient():
    lanes = torch.tensor([[[[0., .7], [0., 0.]], [[.4, 0.], [0., 0.]]]], requires_grad=True)
    spans = torch.tensor([[[0., 1.], [1., 2.]]])
    field = FieldTraversal(lanes, spans, spans, torch.ones(1, 2, dtype=torch.bool), torch.tensor([1]))
    scores = torch.tensor([[2., 1.]], requires_grad=True)
    admitted = field.select(scores)
    assert admitted.tolist() == [[True, False]]
    reconstructed = (field.support * admitted[..., None, None]).sum()
    reconstructed.backward()
    assert scores.grad is None
    torch.testing.assert_close(lanes.grad, torch.tensor([[[[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]]]]))
    assert field.support[0, 0, 1].count_nonzero() == 0


@pytest.mark.parametrize('width', [0, -1])
def test_invalid_scorer_width(width):
    with pytest.raises(ValueError):
        CandidateAttention(width, 4)
