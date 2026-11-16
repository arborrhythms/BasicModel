"""Incoming placement-review repairs, before the shared-chunk migration."""
import pytest
import torch


def test_marker_keeps_a_single_position_per_stream(monkeypatch):
    from test_meaning_kernels import paired
    paired.__wrapped__(monkeypatch)
    from Language import MarkerLayer
    unary = MarkerLayer().candidates[0].double()
    values = torch.linspace(.1, .8, 24, dtype=torch.float64).reshape(2, 1, 12).requires_grad_()
    actual = unary.compose(values)
    expected = torch.stack([unary.compose(values[b]) for b in range(2)])
    assert actual.shape == (2, 1, 12)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (values, *unary.parameters())
    wanted = torch.autograd.grad(expected.sum(), inputs, retain_graph=True, allow_unused=True)
    observed = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
    for first, second in zip(observed, wanted):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
