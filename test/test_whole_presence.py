"""§44 item 74: whole presence bounds its forward contribution and pullback."""
from pathlib import Path

import pytest
import torch

import FormCentroids
from MeaningKernels import centroid, split_centroid, uncenter


def native_read(weights, codes):
    # The fallback lets this certificate exercise the frozen producer's
    # actual old equation, rather than fail because a helper was added.
    return getattr(FormCentroids, 'presence_read', FormCentroids.weighted_mean)(weights, codes)


def compose(parts, weights, codes):
    return centroid((parts,), whole=native_read(weights, codes))


def captured(dtype):
    path = (Path(__file__).resolve().parents[1] / 'doc/benchmarks/'
            '2026-10-31-item2-5-completion/addendum-10/native-mean/whole-pullback-1.pt')
    value = torch.load(path, map_location='cpu', weights_only=False)
    return value['weights'].to(dtype), value['codes'].to(dtype)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('mass', ('captured', 'smallest_normal'))
def test_whole_presence_forward_and_weight_pullback_are_bounded(dtype, mass):
    weights, codes = captured(dtype)
    if mass == 'smallest_normal':
        weights = torch.zeros_like(weights)
        weights[1] = torch.finfo(dtype).tiny
    weights.requires_grad_()
    codes.requires_grad_()
    # The capture contains the whole and its old mean cotangent, not the
    # consuming parts. This certificate supplies explicit interior parts.
    parts = torch.full_like(codes[0], .25)
    gradient = torch.linspace(.25, 1., codes.shape[-1], dtype=dtype)
    total = weights.sum()
    assert 0 < total < 1
    result = compose(parts, weights, codes)
    assert bool(((result-parts).abs() <= total/2).all())
    # A float result can round an increment upward by one ULP. Compare in
    # exact input arithmetic, using zero parts to make tiny changes visible.
    zero_result = compose(torch.zeros_like(parts), weights, codes)
    assert bool((zero_result.abs() <= total/2).all())
    expected = torch.lerp(parts, (weights @ codes / total).detach(), total.detach()/2)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    actual_weight, actual_code = torch.autograd.grad((result*gradient).sum(), (weights, codes))
    expected_weight = ((codes.detach()-parts)*gradient).sum(-1)/2
    expected_code = weights.detach()[:, None]*gradient/2
    torch.testing.assert_close(actual_weight, expected_weight)
    torch.testing.assert_close(actual_code, expected_code, rtol=8*torch.finfo(dtype).eps, atol=0)
    assert actual_code.count_nonzero() == expected_code.count_nonzero()
    bound = ((codes.detach()-parts).abs()*gradient.abs()).sum(-1)/2
    assert bool((actual_weight.abs() <= bound).all())


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_whole_presence_empty_has_no_consumer_gradient(dtype):
    weights = torch.zeros(3, dtype=dtype, requires_grad=True)
    codes = torch.tensor([[0., 1.], [.5, .5], [1., 0.]], dtype=dtype, requires_grad=True)
    parts = torch.tensor([.25, .75], dtype=dtype, requires_grad=True)
    result = compose(parts, weights, codes)
    torch.testing.assert_close(result, parts, rtol=0, atol=0)
    gw, gc, gp = torch.autograd.grad(result.sum(), (weights, codes, parts))
    assert gw.count_nonzero() == gc.count_nonzero() == 0
    torch.testing.assert_close(gp, torch.ones_like(parts), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('mass', (.3, 2.))
def test_whole_presence_lerp_and_inverse_are_the_same_equation(dtype, mass):
    weights = torch.tensor([mass*.3, mass*.7], dtype=dtype, requires_grad=True)
    codes = torch.tensor([[.1, .9], [.7, .2]], dtype=dtype, requires_grad=True)
    left, right = torch.tensor([.2, .6], dtype=dtype), torch.tensor([.8, .4], dtype=dtype)
    whole = native_read(weights, codes)
    parts = (left*3+right)/4
    result = centroid((left, right), (3, 1), whole=whole)
    expected = torch.lerp(parts, (weights @ codes/weights.sum()).detach(), min(1., mass)/2)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(split_centroid(result, right, (3, 1), whole=whole), left)
    g = torch.tensor([.3, -.7], dtype=dtype)
    gw, gc = torch.autograd.grad((result*g).sum(), (weights, codes))
    mean = weights.detach() @ codes.detach()/weights.detach().sum()
    expected_w = (((codes.detach()-parts) if mass < 1 else
                   (codes.detach()-mean)/weights.detach().sum())*g).sum(-1)/2
    torch.testing.assert_close(gw, expected_w)
    torch.testing.assert_close(gc, weights.detach()[:, None]*g/(2*max(1., mass)),
                               rtol=8*torch.finfo(dtype).eps, atol=0)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_whole_presence_common_cube_rails_are_exact(dtype):
    for mass in (.3, 2.):
        weights = torch.tensor([mass*.17, mass*.31, mass*.52], dtype=dtype)
        codes = torch.tensor([[0., 1.]]*3, dtype=dtype)
        parts = codes[0]
        torch.testing.assert_close(compose(parts, weights, codes), parts, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_whole_presence_gradcheck_and_jvp(dtype):
    weights = torch.tensor([.12, .23, .31], dtype=dtype, requires_grad=True)
    codes = torch.tensor([[.1, .9], [.7, .2], [.4, .6]], dtype=dtype, requires_grad=True)
    parts = torch.tensor([.25, .75], dtype=dtype)
    epsilon = 1e-3 if dtype == torch.float32 else 1e-6
    tolerance = 2e-4 if dtype == torch.float32 else 1e-7
    assert torch.autograd.gradcheck(lambda w, c: compose(parts, w, c), (weights, codes),
                                    eps=epsilon, atol=tolerance, rtol=tolerance)
    wt, ct = torch.tensor([.2, -.3, .1], dtype=dtype), torch.ones_like(codes)*.15
    _, tangent = torch.func.jvp(lambda w, c: compose(parts, w, c), (weights, codes), (wt, ct))
    expected = (((codes-parts)*wt[:, None]).sum(0)+(weights[:, None]*ct).sum(0))/2
    torch.testing.assert_close(tangent, expected)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('mass', ('captured', 'smallest_normal'))
def test_whole_presence_tiny_mass_jvp_and_weight_gradcheck(dtype, mass):
    weights, codes = captured(dtype)
    if mass == 'smallest_normal':
        weights = torch.zeros_like(weights)
        weights[1] = torch.finfo(dtype).tiny
    parts = torch.full_like(codes[0], .25)
    tangent_weights = torch.linspace(.1, .3, weights.numel(), dtype=dtype)
    _, tangent = torch.func.jvp(lambda w: compose(parts, w, codes),
                                (weights,), (tangent_weights,))
    expected = ((codes-parts)*tangent_weights[:, None]).sum(0)/2
    assert torch.isfinite(tangent).all()
    torch.testing.assert_close(tangent, expected)
    # At this scale an additive finite-difference step would cross zero.
    # Check the existing positive membership in its relative coordinate;
    # zero parts and output units of total make its small effect observable.
    index = weights.nonzero().flatten()
    total = weights.sum()
    coefficient = (weights[index]/total).requires_grad_()
    def normalized(value):
        native = torch.zeros_like(weights).scatter(0, index, value*total)
        return compose(torch.zeros_like(parts), native, codes)/total
    epsilon = 1e-3 if dtype == torch.float32 else 1e-6
    tolerance = 2e-4 if dtype == torch.float32 else 1e-7
    assert torch.autograd.gradcheck(normalized, (coefficient,), eps=epsilon,
                                    atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('mass', ('captured', 'smallest_normal'))
@pytest.mark.parametrize('backend', ('aot_eager', 'inductor'))
def test_whole_presence_compiled_pullback_matches_decided_bounds_at_tiny_mass(dtype, mass, backend):
    weights, codes = captured(dtype)
    if mass == 'smallest_normal':
        weights = torch.zeros_like(weights)
        weights[1] = torch.finfo(dtype).tiny
    parts = torch.full_like(codes[0], .25)
    direction = torch.linspace(2., 4., codes.shape[-1], dtype=dtype)
    total = weights.sum()
    eager = compose(parts, weights.requires_grad_(), codes.requires_grad_())
    expected_weight = ((codes.detach()-parts)*direction).sum(-1)/2
    expected_code = weights.detach()[:, None]*direction/2
    bound = ((codes.detach()-parts).abs()*direction.abs()).sum(-1)/2
    compiled = torch.compile(compose, backend=backend, fullgraph=True)
    actual = compiled(parts, weights, codes)
    actual_weight, actual_code = torch.autograd.grad((actual*direction).sum(), (weights, codes))
    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    assert ((actual-parts).abs() <= total/2).all()
    assert torch.isfinite(actual_weight).all() and torch.isfinite(actual_code).all()
    torch.testing.assert_close(actual_weight, expected_weight)
    torch.testing.assert_close(actual_code, expected_code, rtol=8*torch.finfo(dtype).eps, atol=0)
    assert actual_code.count_nonzero() == expected_code.count_nonzero()
    assert (actual_weight.abs() <= bound).all()
    # With zero parts even the smallest-normal forward increment is visible.
    actual_zero = compiled(torch.zeros_like(parts), weights, codes)
    assert (actual_zero.abs() <= total/2).all()


def rail_whole(dtype, fractional):
    weights = torch.tensor([1.] + [2.32e-5]*7 + [0.], dtype=dtype)
    if fractional:
        weights = weights*.3
    codes = torch.tensor([[1., .9, .6, 0.]]*9, dtype=dtype)
    return weights, codes


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('fractional', (False, True))
def test_whole_presence_inverse_preserves_multi_member_common_rails(dtype, fractional):
    weights, codes = rail_whole(dtype, fractional)
    whole = native_read(weights, codes)
    left = torch.tensor([1., .7, .55, 0.], dtype=dtype)
    right = torch.tensor([1., .3, .2, 0.], dtype=dtype)
    parent = centroid((left, right), (1, 1), whole=whole)
    recovered_parts = uncenter(parent, whole)
    recovered_left = split_centroid(parent, right, (1, 1), whole=whole)
    for recovered in (recovered_parts, recovered_left):
        assert torch.equal(recovered[[0, 3]], left[[0, 3]])
        assert bool(((recovered >= 0) & (recovered <= 1)).all())
    if not fractional:
        mean, _ = FormCentroids.weighted_mean(weights, codes)
        assert torch.equal(recovered_parts, parent*2-mean)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('fractional', (False, True))
def test_whole_presence_inverse_keeps_the_carrier_derivative(dtype, fractional):
    weights, codes = rail_whole(dtype, fractional)
    weights.requires_grad_()
    codes.requires_grad_()
    parent = torch.tensor([1., .4, .7, 0.], dtype=dtype, requires_grad=True)
    whole = native_read(weights, codes)
    _, total, carrier = whole
    presence = torch.where(total < 1, total, torch.ones_like(total))
    # The cf32ffd84391 expression is the derivative contract. Its forward
    # roundoff on common rails is deliberately not the expected new value.
    old = torch.where(total > 0, (parent-carrier*.5)/(1-presence*.5), parent)
    new = uncenter(parent, whole)
    direction = torch.tensor([.3, -.7, 1.2, .8], dtype=dtype)
    old_grad = torch.autograd.grad((old*direction).sum(), (parent, weights, codes), retain_graph=True)
    new_grad = torch.autograd.grad((new*direction).sum(), (parent, weights, codes))
    assert all(torch.equal(a, b) for a, b in zip(old_grad, new_grad))


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_whole_presence_free_unary_decodes_the_common_rail_candidate(monkeypatch, dtype):
    from types import SimpleNamespace
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import LanguageSpace, SurfaceLayer, SurfaceUnaryLayer
    from FormCentroids import CandidateWholes
    op = SurfaceUnaryLayer(SurfaceLayer(12, 12)).to(dtype)
    value = torch.tensor([[1., .7, .55, .3, .2, .3, .4, .5, .6, .4, .3, .2]], dtype=dtype)
    support = torch.zeros(1, 1, 256, dtype=torch.bool)
    support[..., 10] = True
    weights, codes = rail_whole(dtype, False)
    # The fourth form byte in the review's native free-unary fixture is .2.
    codes[:, 3] = .2
    properties = torch.zeros(9, 256, dtype=dtype)
    properties[:, 10] = weights
    bank = CandidateWholes(value[:, None, :4], support, torch.ones(1, 1, dtype=torch.bool), properties, codes)
    parent = op.compose_blocks(op.compose, value, whole=bank.read_presence(value))
    language = SimpleNamespace(_generate_unary_ops=(op,))
    actual = LanguageSpace.generate_unary_step(language, parent, torch.tensor([0]),
                                              torch.tensor([True]), whole_bank=bank)
    torch.testing.assert_close(actual, value, rtol=0, atol=torch.finfo(dtype).eps)
    assert actual[0, 0] == 1.


def test_native_unary_recompose_reads_the_presence_carrier(monkeypatch):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import SurfaceLayer, SurfaceUnaryLayer
    from Generative import native_recompose
    from FormCentroids import CandidateWholes
    op = SurfaceUnaryLayer(SurfaceLayer(12, 12))
    value = torch.tensor([[1., .7, .55, .3, .2, .3, .4, .5, .6, .4, .3, .2]])
    support = torch.zeros(1, 1, 256, dtype=torch.bool)
    support[..., 10] = True
    weights, codes = rail_whole(torch.float32, True)
    properties = torch.zeros(9, 256)
    properties[:, 10] = weights
    bank = CandidateWholes(value[:, None, :4], support, torch.ones(1, 1, dtype=torch.bool), properties, codes)
    expected = op.compose_blocks(op.compose, value, whole=bank.read_presence(value))
    literal_calls = []
    literal = CandidateWholes.read
    monkeypatch.setattr(CandidateWholes, 'read', lambda self, *parts: (literal_calls.append(parts), literal(self, *parts))[1])
    actual = native_recompose(op, expected, value, form_width=4, whole_bank=bank)
    assert not literal_calls
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
