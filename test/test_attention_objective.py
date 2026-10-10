"""Whole-field reconstruction is an objective, with exact forward admission."""
import pytest
import torch

from AttentionObjective import field_cost, tolerate_heterogeneity


def test_real_reconstruction_and_floor_count_each_valid_item_once():
    observed = torch.tensor([[[2., 1.], [4., 2.], [100., 100.]]], requires_grad=True)
    recovered = torch.tensor([[[1., .5], [0., 0.], [0., 0.]]], requires_grad=True)
    admission = torch.tensor([[1., 0., 0.]], requires_grad=True)
    costs = field_cost(observed, recovered, admission, torch.tensor([[True, True, False]]),
                       floor=.2, iterations=torch.tensor([2]), step_cost=.01)
    assert costs['inside'].item() == pytest.approx(1.5)
    assert costs['outside'].item() == pytest.approx(1.2)
    assert costs['work'].item() == pytest.approx(.02)
    costs['total'].sum().backward()
    assert observed.grad is None
    torch.testing.assert_close(recovered.grad, torch.tensor([[[-1., -1.], [0., 0.], [0., 0.]]]))
    assert admission.grad[0, 2] == 0


def test_future_negative_image_changes_only_the_outside_residual():
    observed = torch.tensor([[[.8, .6]]])
    image = torch.tensor([[[-.3, -.1]]])
    costs = field_cost(observed, torch.zeros_like(observed), torch.zeros(1, 1),
                       torch.ones(1, 1, dtype=torch.bool), floor=.2, negative_image=image)
    assert costs['outside'].item() == pytest.approx(.2)


def test_undefined_inverse_coordinates_have_no_inside_charge_or_gradient():
    observed = torch.tensor([[[2., 50.], [30., 40.], [4., 6.]]], requires_grad=True)
    recovered = torch.zeros_like(observed, requires_grad=True)
    defined = torch.tensor([[[True, False], [False, False], [False, False]]])
    costs = field_cost(observed, recovered, torch.tensor([[1., 1., 0.]]),
        torch.ones(1, 3, dtype=torch.bool), floor=.2, defined=defined,
        iterations=torch.tensor([2]), step_cost=.1)
    assert costs['inside'].item() == 2.
    assert costs['outside'].item() == 2.
    assert costs['work'].item() == pytest.approx(.2)
    costs['total'].sum().backward()
    torch.testing.assert_close(recovered.grad, torch.tensor([[[-1., 0.], [0., 0.], [0., 0.]]]))
    assert observed.grad is None


@pytest.mark.parametrize('floor', [0., 1., -1., float('nan')])
def test_floor_must_be_strictly_between_zero_and_one(floor):
    with pytest.raises(ValueError, match='attentionFloor'):
        field_cost(torch.ones(1, 1, 2), torch.ones(1, 1, 2), torch.ones(1, 1),
                   torch.ones(1, 1, dtype=torch.bool), floor=floor)


def test_heterogeneity_retains_both_lanes_and_default_is_identity():
    lanes = torch.tensor([[[.8, .6], [0., 0.], [1., 0.], [0., 1.]]])
    torch.testing.assert_close(tolerate_heterogeneity(lanes, 1.), lanes, rtol=0, atol=0)
    torch.testing.assert_close(tolerate_heterogeneity(lanes, 0.),
        torch.tensor([[[.2, 0.], [0., 0.], [1., 0.], [0., 1.]]]))
    with pytest.raises(ValueError, match='hetTolerance'):
        tolerate_heterogeneity(lanes, 1.1)
