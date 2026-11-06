"""Case folding keeps an empty lane zero through the ordinary pullback."""
import pytest
import torch
from CaseSelection import CaseView
from MeaningKernels import LaneKernel


@pytest.mark.parametrize('empty_lane', [0, 1])
@pytest.mark.parametrize('cotangent', [0., 1., 8., -8.])
def test_empty_case_lane_has_finite_zero_kernel_gradient(empty_lane, cotangent):
    layer = LaneKernel(32)
    head = torch.full((1, 2, 16), .7)
    head[:, empty_lane] = 0.
    head = head.flatten(-2).requires_grad_()
    codes = head.detach()[:, None].expand(1, 5, 32).clone()
    weights = torch.ones(1, 5, 2)
    test = layer(head)
    actual = CaseView(codes, weights, test, torch.tensor([True])).fold(meaning_start=0)
    # On the occupied lane all five cases agree; the other lane is empty.
    torch.testing.assert_close(actual, head, rtol=0, atol=0)
    (actual.sum() * cotangent).backward()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)
    assert torch.isfinite(head.grad).all()
    assert head.grad.unflatten(-1, (2, 16))[:, empty_lane].eq(0).all()


def test_nonempty_case_lanes_retain_weighted_mean_and_native_derivative():
    codes = torch.tensor([[[.2, .4, .6, .8], [.8, .6, .4, .2]]], requires_grad=True)
    weights = torch.tensor([[[2., 3.], [4., 1.]]], requires_grad=True)
    test = torch.ones(1, 4, requires_grad=True)
    actual = CaseView(codes, weights, test, torch.tensor([True])).fold(meaning_start=0)
    shaped = codes.unflatten(-1, (2, 2))
    expected = ((shaped * weights[..., None]).sum(-3) / weights.sum(-2)[..., None]).flatten(-2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (codes, weights, test)
    got = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    wanted = torch.autograd.grad(expected.square().sum(), parameters, allow_unused=True)
    for gradient, reference in zip(got, wanted):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient) if reference is None else reference, rtol=0, atol=0)
