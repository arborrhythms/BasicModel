"""Immediate unary inverses are ineligible only at the same occupied slot."""
from types import SimpleNamespace
import torch
import pytest


def _prefer_unary(monkeypatch, both=False):
    from Language import OperationSelectionLayer, NotLayer, NonLayer, SumLayer
    layer = OperationSelectionLayer(d_model=2, ops=[SumLayer()],
        unary_ops=[NotLayer(), NonLayer()] if both else [NotLayer()], chooser='mlp')
    def binary(content, candidates, *args, **kwargs):
        B, N, _ = content.shape
        return content.new_zeros(B, N, 1), content.new_zeros(B, N-1, 1)
    def unary(content, candidates, *args, **kwargs):
        B, N, _ = content.shape
        return content.new_zeros(B, N, 1), content.new_full((B, N, layer.r_apply), 20.)
    monkeypatch.setattr(layer.chooser, 'score_binary', binary)
    monkeypatch.setattr(layer.chooser, 'score_unary', unary)
    return layer


def test_closing_does_not_spend_rounds_repeating_not_at_one_slot(monkeypatch):
    layer = _prefer_unary(monkeypatch)
    value = torch.tensor([[[.2, .4]]])
    result = layer.derive(value, rounds=6, slots=1, greedy=True)
    assert result['used'].tolist() == [2]  # not, then STOP
    assert [int(route['kind'][0]) for route in result['traces'][:2]] == [2, 0]
    torch.testing.assert_close(result['value'], -value)


@pytest.mark.parametrize('compiled', [False, True])
def test_inverse_exclusion_is_local_and_leaves_other_unaries_eligible(monkeypatch, compiled):
    layer = _prefer_unary(monkeypatch, both=True)
    function = torch.compile(layer, backend='eager', fullgraph=True) if compiled else layer
    value = torch.tensor([[[.2, .4], [.7, .3]]], requires_grad=True)
    _, path, route = function(value, slots=2, previous_unary=torch.tensor([[0, -1]]))
    logits = route['logits'][0]
    # One binary, then [not, non] at each of two occupied locations.
    assert torch.isneginf(logits[1])
    assert torch.isfinite(logits[[0, 2, 3, 4, 5]]).all()
    path.sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_serial_window_excludes_the_inverse_at_the_same_physical_slot(monkeypatch):
    from Language import LanguageSpace
    layer = _prefer_unary(monkeypatch)
    owner = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=layer),
                            _structural_context=lambda **kwargs: None)
    buffer = torch.tensor([[[.2, .4], [.7, .3], [0., 0.]]])
    state = (buffer, torch.tensor([2]))
    choice = LanguageSpace.choose_operation(owner, state, torch.tensor([True]), slots=2,
        sample=False, previous_unary=torch.tensor([[0, -1, -1]]))
    assert choice.kind.tolist() == [2]
    assert choice.position.tolist() == [1]


def test_exploration_forces_only_a_round_with_a_legal_alternative(monkeypatch):
    layer = _prefer_unary(monkeypatch)
    value = torch.tensor([[[.2, .4]]])
    monkeypatch.setattr(torch, 'rand', lambda *shape, **kw: torch.full(shape, .9, **kw))
    exploit, explore = layer.derive_pair(value, rounds=6, slots=1, greedy=True)
    assert exploit['used'].tolist() == [2]
    assert explore['forced_round'].tolist() == [0]
    assert explore['different'].all()


def test_single_legal_derivation_is_compared_without_inventing_an_alternative():
    from Language import OperationSelectionLayer, SumLayer
    layer = OperationSelectionLayer(d_model=2, ops=[SumLayer()], chooser='mlp')
    value = torch.tensor([[[.2, .4], [.7, .3], [.1, .8]]])
    exploit, explore = layer.derive_pair(value, rounds=6, slots=1, greedy=True)
    assert exploit['complete'].all() and explore['complete'].all()
    assert explore['forced_round'].tolist() == [-1]
    assert not explore['different'].any()
    torch.testing.assert_close(explore['value'], exploit['value'], rtol=0, atol=0)
