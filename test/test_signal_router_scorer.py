"""The explicit candidate scores enumerate every operation and location."""
import torch
from torch import nn
from Language import OperationSelectionLayer

class Add(nn.Module):
    def forward(self, left, right):
        return left + right
class Negate(nn.Module):
    def forward(self, x):
        return -x

def test_unified_anchor_shapes():
    step = OperationSelectionLayer(d_model=4, ops=[Add(), Add()], unary_ops=[Negate()])
    assert step.stop_anchor.shape == (1, 4)
    assert step.reduce_anchor.shape == (2, 4)
    assert step.apply_anchor.shape == (1, 4)

def test_scores_and_probabilities_match_explicit_enumeration():
    step = OperationSelectionLayer(d_model=1, ops=[Add()], unary_ops=[Negate()], temperature=2)
    with torch.no_grad():
        step.reduce_anchor.fill_(2); step.apply_anchor.fill_(3); step.stop_anchor.fill_(1)
    x = torch.tensor([[[1.], [2.], [4.]]])
    _, _, route = step(x, slots=3)
    expected = torch.tensor([[6., 12., -3., -6., -12., 7./3]])
    torch.testing.assert_close(route['logits'], expected)
    torch.testing.assert_close(route['probabilities'], expected.softmax(-1))
