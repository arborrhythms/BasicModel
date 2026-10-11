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


def test_retained_two_word_scorer_preserves_candidate_differences():
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / 'doc/benchmarks/2026-10-10-item6'
            '/identity-rst/validation/retained-scorer.pt')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    context, candidates = saved['context'], saved['candidates']
    layer = CandidateAttention(context.shape[-1], candidates.shape[-1]).cpu()
    layer.load_state_dict(saved['state_dict'])
    raw = torch.cat((context[:, None].expand(-1, 2, -1), candidates), -1)
    # The archived checkpoint reproduces the original obstruction exactly.
    assert layer.hidden(raw).abs().eq(1).all()
    hidden = []
    handle = layer.hidden.register_forward_hook(lambda module, args, result: hidden.append(result))
    try:
        scores = layer(context, candidates)
    finally:
        handle.remove()
    assert scores[:, 0].ne(scores[:, 1]).all()
    assert hidden[0].abs().lt(1).all()
    loss = torch.nn.functional.cross_entropy(scores, torch.zeros(len(scores), dtype=torch.long))
    loss.backward()
    assert layer.hidden[0].weight.grad.abs().sum() > 0


def test_shared_context_magnitude_cannot_saturate_the_retained_scorer():
    layer = CandidateAttention(2048, 1, hidden_width=8)
    with torch.no_grad():
        layer.hidden[0].weight.fill_(.1)
        layer.hidden[0].bias.zero_()
        layer.readout.weight.fill_(1.)
    context = torch.ones(1, 2048)
    candidates = torch.tensor([[[-1.], [1.]]])
    ordinary = layer(context, candidates)
    large = layer(1e6 * context, candidates)
    torch.testing.assert_close(ordinary, large)
    assert large[0, 1] > large[0, 0]


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
