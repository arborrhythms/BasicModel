"""The explicit candidate scores enumerate every operation and location."""
import torch
from torch import nn
from Language import OperationSelectionLayer
from chooser_steering import stand_in

class Add(nn.Module):
    def forward(self, left, right):
        return left + right
class Negate(nn.Module):
    def forward(self, x):
        return -x

def test_unified_anchor_shapes():
    step = OperationSelectionLayer(d_model=4, ops=[Add(), Add()], unary_ops=[Negate()])
    stand_in(step)
    assert step.stand_in.stop.shape == (1, 4)
    assert step.stand_in.reduce.shape == (2, 4)
    assert step.stand_in.unary.shape == (1, 4)

def test_scores_and_probabilities_match_explicit_enumeration():
    step = OperationSelectionLayer(d_model=1, ops=[Add()], unary_ops=[Negate()], temperature=2)
    stand_in(step)
    with torch.no_grad():
        step.stand_in.reduce.fill_(2); step.stand_in.unary.fill_(3); step.stand_in.stop.fill_(1)
    x = torch.tensor([[[1.], [2.], [4.]]])
    _, _, route = step(x, slots=3)
    expected = torch.tensor([[6., 12., -3., -6., -12., 7./3]])
    expected[:, :-1] -= step.operator_cost
    torch.testing.assert_close(route['logits'], expected)
    torch.testing.assert_close(route['probabilities'], expected.softmax(-1))
