"""Native property means stay inside the cube at learned fractional weights."""
import pytest
import torch

from FormCentroids import CandidateWholes, SpanWholes, weighted_mean


def test_native_whole_reads_preserve_common_rails_at_fractional_weights():
    # The previous matmul divided by a separately reduced sum exceeded one
    # for this valid property mixture (the native lesson's failure class).
    weights = torch.tensor([1.] + [2.32e-5]*7 + [0.])
    codes = torch.cat((torch.ones(9, 180), torch.zeros(9, 180)), -1)
    properties = weights[:, None].expand(-1, 256)
    prefix = torch.zeros(1, 3, 256)
    prefix[:, 1:, 65] = 1
    prefix[:, 2:, 66] = 1
    span = SpanWholes(prefix, properties, codes)
    forms = codes[:2][None]
    support = prefix[:, 1:].bool()
    candidates = CandidateWholes(forms, support, torch.ones(1, 2, dtype=torch.bool), properties, codes)
    for mean, count in (span.read(torch.tensor([[[0, 2]]])), candidates.read(forms)):
        assert torch.isfinite(mean).all() and ((mean >= 0) & (mean <= 1)).all()
        torch.testing.assert_close(mean, codes[0].expand_as(mean), rtol=0, atol=0)
        torch.testing.assert_close(count, weights.sum().expand_as(count), rtol=0, atol=0)


def test_whole_mean_matches_direct_value_and_both_gradients():
    weights = torch.tensor([[.12, .29, .59], [.8, .1, .04]], dtype=torch.double, requires_grad=True)
    codes = torch.tensor([[.1, .9], [.7, .2], [.4, .6]], dtype=torch.double, requires_grad=True)
    got, count = weighted_mean(weights, codes)
    expected = weights @ codes / weights.sum(-1, keepdim=True)
    torch.testing.assert_close(got, expected, rtol=1e-14, atol=1e-14)
    direction = torch.tensor([[.3, .7], [-.1, .2]], dtype=torch.double)
    actual_grad = torch.autograd.grad((got*direction).sum(), (weights, codes), retain_graph=True)
    expected_grad = torch.autograd.grad((expected*direction).sum(), (weights, codes))
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    empty, number = weighted_mean(torch.zeros_like(weights), codes)
    assert empty.count_nonzero() == number.count_nonzero() == 0


@pytest.mark.parametrize('invalid', (float('nan'), -.1, 1.1))
def test_invalid_whole_codes_still_reject(invalid):
    with pytest.raises(RuntimeError, match='finite coordinates'):
        weighted_mean(torch.ones(1, 1), torch.tensor([[invalid]]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('compiled', [False, True])
def test_whole_mean_zero_prefix_has_the_final_mean_derivative(dtype, compiled):
    # A whole property that becomes the first nonzero weight can have a
    # tiny prefix even though the complete span has ordinary mass.
    read = torch.compile(weighted_mean, backend='aot_eager', fullgraph=True) if compiled else weighted_mean
    weights = torch.tensor([[0., 1.23e-7, 0.], [0., 1.23e-7, 1.]],
                           dtype=dtype, requires_grad=True)
    codes = torch.tensor([[.25, 1.], [.25, 0.], [.25, .5]],
                         dtype=dtype, requires_grad=True)
    mean, total = read(weights, codes)
    direction = torch.tensor([[1000., 0.], [1., .75]], dtype=dtype)
    weight_grad, code_grad = torch.autograd.grad((mean * direction).sum(), (weights, codes))
    expected = ((codes[None] - mean[:, None]) * direction[:, None]).sum(-1) / total
    torch.testing.assert_close(weight_grad, expected)
    torch.testing.assert_close(code_grad, (weights / total).T @ direction)
    assert torch.equal(weight_grad[0], torch.zeros_like(weight_grad[0]))

    # A genuine inverse-total slope on a different code remains measurable.
    mean, total = read(weights, codes)
    gradient, = torch.autograd.grad(mean[0, 1], weights)
    torch.testing.assert_close(gradient[0, 0], total[0, 0].reciprocal())
