"""Joint operation layer shape and gradient contracts."""
import torch
from torch import nn
from Language import OperationSelectionLayer

class Add(nn.Module):
    def forward(self, left, right):
        return left + right

class Mul(nn.Module):
    def forward(self, left, right):
        return left * right

def test_layer_forward_shapes():
    step = OperationSelectionLayer(d_model=4, ops=[Add(), Mul()])
    x = torch.randn(2, 5, 4)
    hard, path, route = step(x)
    assert hard.shape == path.shape == x.shape
    assert route['probabilities'].shape == (2, 9)
    assert route['binary_probabilities'].shape == (2, 4, 2)
    assert route['depth'].tolist() == [4, 4]

def test_chosen_operator_gradient_is_probability_weighted():
    x = torch.tensor([[[1.], [2.]]], requires_grad=True)
    step = OperationSelectionLayer(d_model=1, ops=[Add(), Mul()])
    with torch.no_grad():
        step.reduce_anchor.zero_()
    _, path, route = step(x)
    path.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, .5))
    torch.testing.assert_close(route['probabilities'][0], torch.tensor([.5, .5, 0.]))
    assert step.reduce_anchor.grad.abs().sum() > 0

def test_layer_n_one_stops_without_unary_ops():
    step = OperationSelectionLayer(d_model=3, ops=[Add()])
    x = torch.randn(2, 1, 3)
    hard, path, route = step(x)
    torch.testing.assert_close(hard, x)
    torch.testing.assert_close(path, x)
    assert route['stopped'].all()
