"""Activation supplies binding magnitude; code length supplies no certainty."""
import pytest
import torch
from Language import ConjunctionLayer, DisjunctionLayer, OperationSelectionLayer


@pytest.mark.parametrize('cls,magnitude', [(ConjunctionLayer, .3), (DisjunctionLayer, .8)])
def test_kernel_uses_activation_and_is_invariant_to_form_length(cls, magnitude):
    x = torch.tensor([[1., .5, .2]], dtype=torch.float64)
    y = torch.tensor([[.2, .4, 1.]], dtype=torch.float64)
    op = cls()
    activation = dict(left_activation=.5, right_activation=.6)
    value = op.compose(x, y, **activation)
    torch.testing.assert_close(value.norm(dim=-1), torch.tensor([magnitude], dtype=x.dtype))
    torch.testing.assert_close(op.compose(x * 12, y * .03, **activation), value)
    torch.testing.assert_close(op.compose(x, y).norm(dim=-1), torch.ones(1, dtype=x.dtype))
    # Presence changes magnitude and not the identity direction.
    torch.testing.assert_close(op.compose(x, y) * magnitude, value)


@pytest.mark.parametrize('cls', [ConjunctionLayer, DisjunctionLayer])
def test_operation_candidates_receive_leaf_activation_and_keep_code_directions(cls):
    x = torch.tensor([[[.8, .2, .4], [.1, .7, .3]]])
    layer = OperationSelectionLayer(d_model=3, ops=[cls()], chooser='mlp')
    activations = torch.tensor([[.5, .6]])
    expected = cls().compose(x[:, 0], x[:, 1], left_activation=.5, right_activation=.6)
    result = layer._stacked_reduced(x, activations=activations)
    torch.testing.assert_close(result[:, 0, 0], expected)
    # An arbitrary positive change of either form's scale cannot change it.
    result = layer._stacked_reduced(x * torch.tensor([[[20.], [.01]]]), activations=activations)
    torch.testing.assert_close(result[:, 0, 0], expected)


def test_repeated_reference_preserves_its_activation_and_direction():
    x = torch.tensor([[2., 1.]])
    value = ConjunctionLayer().compose(x, x, left_activation=.4, right_activation=.4)
    torch.testing.assert_close(value, .4 * x / x.norm(dim=-1, keepdim=True))
