"""Strict local inverse contracts; no promise to recover quantization loss."""
import os
import sys
from unittest.mock import patch

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin"))

import pytest
import torch

from Layers import InvertibleLinearLayer, MeronymicFoldAdapter, PiLayer, SigmaLayer
from embed import _unorm_ste


@pytest.mark.parametrize("naive", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_zero_ldu_diagonal_has_finite_matched_inverse(naive, gated):
    layer = InvertibleLinearLayer(3, 3, stable=True, hasBias=False, naive=naive).double()
    with torch.no_grad():
        layer.d.copy_(torch.tensor([0., -0.5, 1.], dtype=torch.float64))
    gate = torch.tensor([1., 0., -1.], dtype=torch.float64) if gated else None
    d = layer._d_effective_for_gate(gate)
    assert torch.isfinite(d).all() and (d.abs() > 0).all()
    assert d[0] > 0  # sign(0) must select a nonzero branch.
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    recovered = layer.reverse(layer(x, gate=gate), gate=gate)
    torch.testing.assert_close(recovered, x, rtol=1e-9, atol=1e-9)
    recovered.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    for param in layer.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()


def test_ergodic_noise_cancellation_cannot_make_stable_diagonal_zero():
    layer = InvertibleLinearLayer(3, 3, stable=True, ergodic=True, hasBias=False).double()
    with torch.no_grad():
        layer.bias.fill_(0.5)
        layer.var.fill_(0.5)
        layer.d.fill_(1.)
        layer.noise_d.fill_(-1.)
        layer.noise_raw_L.zero_()
        layer.noise_raw_U.zero_()
    assert (layer._d_eff() > 0).all()
    assert (layer._d_eff_for_gate(None) > 0).all()
    x = torch.randn(2, 3, dtype=torch.float64)
    with patch.object(layer, "resample_noise") as resample:
        recovered = layer.reverse(layer(x))
    assert resample.call_count == 2
    torch.testing.assert_close(recovered, x)


def test_noise_is_sampled_before_forward_and_after_reverse_only():
    layer = InvertibleLinearLayer(3, 3, stable=True, ergodic=True, hasBias=False).double()
    layer.bias.fill_(0.95)
    layer.var.fill_(0.05)
    x = torch.randn(2, 3, dtype=torch.float64)
    with patch.object(layer, "resample_noise", wraps=layer.resample_noise) as resample:
        y = layer(x)
        assert resample.call_count == 1
        before_reverse = layer.noise_d.clone()
        # Inverse computation sees exactly the forward sample.
        with patch.object(layer, "_d_eff", wraps=layer._d_eff) as effective:
            recovered = layer.reverse(y)
        assert effective.call_count == 1 and resample.call_count == 2
        assert not torch.equal(layer.noise_d, before_reverse)
    torch.testing.assert_close(recovered, x, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kind", [SigmaLayer, PiLayer])
@pytest.mark.parametrize("nominal,actual", [(3, 3), (7, 7), (8, 5), (8, 8)])
def test_trained_butterfly_preserves_live_subspace(kind, nominal, actual):
    layer = kind(3, 3, invertible=True, nonlinear=False,
                 butterfly=True, N=nominal).double()
    with torch.no_grad():
        layer.butterfly_L.fill_(0.35)
        layer.butterfly_U.fill_(-0.25)
        layer.butterfly_d.fill_(0.9)
    x = torch.randn(2, actual, dtype=torch.float64, requires_grad=True) * 0.1
    y = layer(x)
    torch.testing.assert_close(layer.reverse(y), x, rtol=1e-9, atol=1e-9)
    y.square().sum().backward()
    for level in range(layer.n_levels):
        pad = (layer.butterfly_perms[level].reshape(-1, 2) >= actual).any(-1)
        assert torch.count_nonzero(layer.butterfly_L.grad[level][pad]) == 0
        assert torch.count_nonzero(layer.butterfly_d.grad[level][pad]) == 0
        assert torch.count_nonzero(layer.butterfly_U.grad[level][pad]) == 0
    # No new mask keys: existing checkpoints still load strictly.
    clone = kind(3, 3, invertible=True, nonlinear=False,
                 butterfly=True, N=nominal).double()
    clone.load_state_dict(layer.state_dict(), strict=True)
    torch.testing.assert_close(clone(x), y)


@pytest.mark.parametrize("kind", ["sigma", "pi"])
def test_membership_field_roundtrip_does_not_require_one_summary_to_decode_it(kind):
    layer = MeronymicFoldAdapter(kind, 3, 3, legacy_N=3, butterfly=True)
    with torch.no_grad():
        layer.raw_bfly_L.fill_(0.4)
        layer.raw_bfly_U.fill_(0.3)
        layer.raw_bfly_d.fill_(0.2)
    # Independent activations (e.g. car and tire), retained together.
    field = torch.tensor([[[0., 0.25, 1.]], [[0.8, 0.5, 0.1]]])
    recovered = layer.reverse(layer(field))
    torch.testing.assert_close(recovered, field, rtol=0., atol=2e-6)


@pytest.mark.parametrize("kind", [SigmaLayer, PiLayer])
def test_trained_padded_butterfly_remains_fullgraph_traceable(kind):
    layer = kind(3, 3, invertible=True, nonlinear=False, butterfly=True, N=3)
    with torch.no_grad():
        layer.butterfly_L.fill_(0.35)
        layer.butterfly_U.fill_(-0.25)
        layer.butterfly_d.fill_(0.9)
    forward = torch.compile(layer, backend="eager", fullgraph=True)
    reverse = torch.compile(layer.reverse, backend="eager", fullgraph=True)
    x = torch.tensor([[0.1, 0.2, 0.4]])
    torch.testing.assert_close(reverse(forward(x)), x, rtol=1e-5, atol=1e-6)


def test_bounded_codebook_read_ste_retains_corrective_gradient():
    master = torch.tensor([-0.25, 0.5, 1.25], requires_grad=True)
    read = _unorm_ste(master)
    torch.testing.assert_close(read, master.detach().clamp(0., 1. - torch.finfo(master.dtype).eps))
    (read - 0.5).square().sum().backward()
    # Gradient descent can move both saturated coordinates back toward cube.
    assert master.grad[0] < 0 and master.grad[2] > 0


def test_hard_nearest_lookup_and_ste_have_same_values_different_encoder_credit():
    x = torch.tensor([[0.2, 0.1]], requires_grad=True)
    codes = torch.nn.Parameter(torch.tensor([[0., 0.], [1., 1.]]))
    index = (x.unsqueeze(1) - codes.unsqueeze(0)).square().sum(-1).argmin(-1)
    hard = codes[index]
    assert torch.autograd.grad(hard.sum(), x, allow_unused=True, retain_graph=True)[0] is None
    ste = hard + (x - x.detach())  # learnable-code variant in VectorQuantize
    torch.testing.assert_close(ste, hard)
    encoder_grad, selected_grad = torch.autograd.grad(ste.sum(), (x, codes))
    torch.testing.assert_close(encoder_grad, torch.ones_like(x))
    torch.testing.assert_close(selected_grad, torch.tensor([[1., 1.], [0., 0.]]))
