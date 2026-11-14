"""Incoming placement-review repairs, before the shared-chunk migration."""
import pytest
import torch


@pytest.mark.parametrize('has_cases', (False, True))
def test_unary_case_keeps_a_single_position_per_stream(monkeypatch, has_cases):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import CompoundLayer, CompoundUnaryLayer
    from CaseSelection import CaseView
    base = CompoundLayer().double()
    unary = CompoundUnaryLayer(base).double()
    values = torch.linspace(.1, .8, 24, dtype=torch.float64).reshape(2, 1, 12).requires_grad_()
    codes = torch.linspace(.15, .75, 72, dtype=torch.float64).reshape(2, 3, 12).requires_grad_()
    weights = torch.zeros(2, 3, 2, dtype=torch.float64)
    if has_cases:
        weights[0, 0] = 1.
        weights[1, 2] = 1.
    view = CaseView(codes, weights, torch.ones(2, 12, dtype=torch.float64),
                    torch.zeros(2, dtype=torch.bool), codes[:, 1])
    actual = unary.compose(values, case_view=view)
    expected = torch.stack([unary.compose(values[b], case_view=CaseView(
        codes[b:b+1], weights[b:b+1], view.test[b:b+1],
        view.selected[b:b+1], view.reference[b:b+1])) for b in range(2)])
    assert actual.shape == (2, 1, 12)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (values, codes, base.meaning_kernel.bias)
    wanted = torch.autograd.grad(expected.sum(), inputs, retain_graph=True, allow_unused=True)
    observed = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
    for first, second in zip(observed, wanted):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
