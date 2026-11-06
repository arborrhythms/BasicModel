"""Unused proposal corrections cannot contaminate a credited departure."""
import pytest
import torch
from SentenceCredit import score_function


@pytest.mark.parametrize('unused', [float('inf'), float('nan'), 0.])
def test_mask_precedes_product_in_value_and_gradient(unused):
    probability = torch.tensor([[.3, .2]], requires_grad=True)
    value, _ = score_function(probability, torch.tensor([[2., unused]]),
        torch.tensor([[True, False]]), torch.tensor([[1., 2.]]))
    value.sum().backward()
    torch.testing.assert_close(value, torch.tensor([.6]))
    torch.testing.assert_close(probability.grad, torch.tensor([[2., 0.]]))


@pytest.mark.parametrize('correction', [0., -1., float('inf'), float('nan')])
def test_invalid_credited_correction_still_fails_loud(correction):
    with pytest.raises(RuntimeError, match='finite proposal correction'):
        score_function(torch.tensor([[.3]], requires_grad=True),
            torch.tensor([[correction]]), torch.tensor([[True]]), torch.tensor([[1., 2.]]))


def test_nonfinite_credited_cost_still_fails_loud():
    with pytest.raises(RuntimeError, match='finite paired advantage'):
        score_function(torch.tensor([[.3]], requires_grad=True), torch.tensor([[2.]]),
            torch.tensor([[True]]), torch.tensor([[1., float('inf')]]))
