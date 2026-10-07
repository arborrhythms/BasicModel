"""Value-distinct compose proposals and their reconstruction-only credit."""
import torch
from torch import nn
from Language import OperationSelectionLayer
from test_compose_operations import Add, Negate


class Product(nn.Module):
    def forward(self, left, right):
        return left * right


def test_departure_values_are_unique_and_identity_and_exact_stop_are_excluded():
    layer = OperationSelectionLayer(d_model=1, ops=[Add(), Add(), Product(), Product()],
        unary_ops=[nn.Identity(), Negate()])
    with torch.no_grad():
        layer.reduce_anchor.fill_(0)
        layer.reduce_anchor[0].fill_(1)
        layer.apply_anchor.fill_(0)
        layer.stop_anchor.fill_(0)
    _, _, route = layer(torch.tensor([[[2.], [3.]]]), slots=2, stop_exact=torch.tensor([True]))
    assert route['action'].tolist() == [0]
    assert route['departure_eligible'].tolist() == [[False, False, True, False,
                                                  False, True, False, True, False]]
    _, _, readable = layer(torch.tensor([[[2.], [3.]]]), slots=2, stop_exact=torch.tensor([False]))
    assert readable['departure_eligible'][0, -1]


def test_uniform_departure_is_independent_of_logits_and_credits_the_original_probability(monkeypatch):
    layer = OperationSelectionLayer(d_model=1, ops=[Add()])
    logits = torch.tensor([[9., 8., -4., -torch.inf]]).expand(6, -1)
    eligible = torch.tensor([[False, True, True, False]]).expand(6, -1)
    draws = torch.tensor([[0.], [.25], [.499], [.5], [.75], [.999]])
    monkeypatch.setattr(torch, 'rand', lambda *args, **kwargs: draws.to(device=kwargs.get('device')))
    action, probability, legal = layer.select_logits(logits,
        structural=(True,)*4, masked_action=torch.zeros(6, dtype=torch.long),
        departure_eligible=eligible)
    assert action.tolist() == [1, 1, 1, 2, 2, 2]
    assert legal.all()
    torch.testing.assert_close(probability, logits.softmax(-1))


def test_greedy_stop_has_operand_gradient_and_no_chooser_gradient():
    layer = OperationSelectionLayer(d_model=1, ops=[Add()], unary_ops=[Negate()])
    with torch.no_grad():
        layer.stop_anchor.fill_(100.)
    x = torch.tensor([[[2.], [3.]]], requires_grad=True)
    _, path, route = layer(x, slots=2)
    assert route['stopped'].all()
    path.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert all(p.grad is None for p in layer.parameters())
