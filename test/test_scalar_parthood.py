"""The shared-size scalar part contract of item 1.8 §70.17."""
import pytest
import torch

from ConceptualMatch import match
from Layers import Ops, epsilon


def lanes(value):
    return torch.cat((value.clamp_min(0), (-value).clamp_min(0)), -1)


@pytest.mark.parametrize('case,expected', [
    ('apart', (0., 0.)), ('contains', (.3, 1.)),
    ('contained', (1., .3)), ('same', (1., 1.)), ('overlap', (.2, .2))])
def test_five_regions(case, expected):
    x, y = torch.zeros(20), torch.zeros(20)
    x[:10] = 1
    if case == 'apart':
        y[10:] = 1
    elif case == 'contains':
        y[:3] = 1
    elif case == 'contained':
        y[:10] = 1; x[3:] = 0
    elif case == 'same':
        y[:10] = 1
    else:
        y[8:18] = 1
    assert (Ops.part(x, y, scalar=True).item(), Ops.part(y, x, scalar=True).item()) == pytest.approx(expected)


def test_signed_part_is_shared_lane_coverage_with_the_same_gradients():
    x, y = (torch.randn(32, 8, requires_grad=True) for _ in range(2))
    backward, forward = match(lanes(x), lanes(y))
    for a, b, expected in ((x, y, forward), (y, x, backward)):
        actual = Ops.part(a, b, scalar=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        before = torch.autograd.grad(expected.sum(), (x, y), retain_graph=True)
        after = torch.autograd.grad(actual.sum(), (x, y), retain_graph=True)
        for new, old in zip(after, before):
            torch.testing.assert_close(new, old, rtol=0, atol=0)
        assert ((actual >= 0) & (actual <= 1)).all()


def test_self_negation_and_simultaneous_sign_reversal():
    x, y = torch.randn(32, 8), torch.randn(32, 8)
    torch.testing.assert_close(Ops.part(x, x, scalar=True), torch.ones(32), rtol=0, atol=0)
    torch.testing.assert_close(Ops.part(x, -x, scalar=True), torch.zeros(32), rtol=0, atol=0)
    torch.testing.assert_close(Ops.part(x, y, scalar=True), Ops.part(-x, -y, scalar=True))


def test_full_part_exactly_means_between_zero_and_the_other_coordinate():
    points = torch.cartesian_prod(torch.arange(-2., 3.), torch.arange(-2., 3.))
    x, y = points[:, None], points[None, :]
    between = ((x >= torch.minimum(y, torch.zeros_like(y)))
               & (x <= torch.maximum(y, torch.zeros_like(y)))).all(-1)
    assert torch.equal(Ops.part(x, y, scalar=True) == 1, between)


@pytest.mark.parametrize('left,right,expected', [(0., 1., 1.), (1., 0., 0.), (0., 0., 1.)])
def test_empty_operands_keep_the_scalar_parts_contract(left, right, expected):
    x, y = torch.full((4,), left), torch.full((4,), right)
    assert Ops.part(x, y, scalar=True).item() == expected
    # The existing near-empty norm threshold is unchanged too.
    x = x + (epsilon/8 if left == 0 else 0)
    y = y + (epsilon/8 if right == 0 else 0)
    assert Ops.part(x, y, scalar=True).item() == expected


def test_copart_is_the_unaccounted_share_of_the_second_operand():
    x, y = torch.tensor([1., 0.]), torch.tensor([2., 0.])
    assert Ops.copart(x, y, scalar=True).item() == .5
    assert Ops.copart(y, x, scalar=True).item() == 0.
    torch.testing.assert_close(Ops.copart(x, y), y-x, rtol=0, atol=0)


@pytest.mark.parametrize('reader', ('thought', 'query'))
def test_part_support_preserves_the_direction_through_both_readers(reader):
    from test_query_vp_boundaries import _world
    from ThoughtFaces import part
    from Queries import _part
    _, _, _, _, _, context = _world()
    x = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
    y = 2*x
    read = part if reader == 'thought' else _part
    forward = read(context, dict(argin1=x, argin2=y))
    backward = read(context, dict(argin1=y, argin2=x))
    assert (forward['support_true'], backward['support_true']) == (1., .5)
    for a, b, result in ((x, y, forward), (y, x, backward)):
        torch.testing.assert_close(result['value'], Ops.part(a, b), rtol=0, atol=0)
