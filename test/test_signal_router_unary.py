"""Unary candidates share one round and one location selector."""
import torch
from torch import nn
from Language import OperationSelectionLayer

class Negate(nn.Module):
    def forward(self, x):
        return -x

class Abs(nn.Module):
    def forward(self, x):
        return x.abs()

def test_unary_output_shape_and_one_global_softmax():
    step = OperationSelectionLayer(d_model=4, unary_ops=[Negate(), Abs()])
    x = torch.randn(2, 5, 4)
    hard, path, route = step(x)
    assert hard.shape == path.shape == x.shape
    assert route['probabilities'].shape == (2, 11)
    torch.testing.assert_close(route['probabilities'].sum(-1), torch.ones(2))
    assert route['depth'].tolist() == [5, 5]

def test_exactly_one_unary_position_changes():
    step = OperationSelectionLayer(d_model=1, unary_ops=[Negate()])
    x = torch.tensor([[[1.], [2.], [3.], [4.]]])
    with torch.no_grad():
        step.apply_anchor.fill_(-1)
    _, path, route = step(x)
    assert (path != x).any(-1).sum(-1).tolist() == [1]
    assert route['position'].tolist() == [3]

def test_unary_and_stop_anchors_receive_only_score_function_credit():
    step = OperationSelectionLayer(d_model=1, unary_ops=[Negate()])
    with torch.no_grad():
        step.stop_anchor.zero_(); step.apply_anchor.fill_(-1)
    x = torch.tensor([[[1.]]], requires_grad=True)
    _, path, route = step(x)
    path.sum().backward()
    assert x.grad.abs().sum() > 0
    assert step.stop_anchor.grad is None
    assert step.apply_anchor.grad is None
    operand_gradient = x.grad.clone()
    (-route['probability'].sum()).backward()
    assert step.stop_anchor.grad.abs().sum() > 0
    assert step.apply_anchor.grad.abs().sum() > 0
    torch.testing.assert_close(x.grad, operand_gradient, atol=0, rtol=0)
