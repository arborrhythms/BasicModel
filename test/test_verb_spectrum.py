"""Verb eig-spectrum operator (Stage 1; doc/old/2026-06-20-idea-decoder.md
"VP parameterization"). The verb acts on the NP as Q·diag(e^w)·Qᵀ; this first
increment is the exp-diagonal (Q = I) in atanh-space:
    apply_verb(NP, verb)   = tanh(e^{w_v} ⊙ atanh(NP))
    unapply_verb(VP, verb) = tanh(e^{-w_v} ⊙ atanh(VP))   (exact inverse, given verb)
w_v = the verb's SPARSE log-eigenvalues from the verb code (soft-thresholded,
zero-init). INVERTIBLE by construction (e^w > 0). Gated <verbSpectrum>; flag off
-> byte-identical.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
import torch
from torch import nn

from util import TheXMLConfig
from Language import BinaryStructuredReductionLayer, LiftLayer, VerbLayer

_D = 8


def _set_flag(value):
    data = getattr(TheXMLConfig, "data", None)
    if not isinstance(data, dict):
        TheXMLConfig.data = data = {}
    data.setdefault("architecture", {})["verbSpectrum"] = bool(value)


def _np_and_verb(seed=0):
    g = torch.Generator().manual_seed(seed)
    np_ = torch.randn(2, 5, _D, generator=g).tanh() * 0.6      # moderate (atanh-safe)
    verb = torch.randn(2, 5, _D, generator=g).tanh() * 0.6
    return np_, verb


def _strong_spectrum(lyr, seed=1):
    """Force a non-trivial verb spectrum (zero-init thresholds to ~0 untrained)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        lyr._verb_spec.weight.copy_(
            torch.randn(lyr._verb_spec.weight.shape, generator=g) * 0.6)
        lyr._verb_spec.bias.fill_(0.3)
    return lyr


def test_flag_off_is_noop():
    _set_flag(False)
    lyr = LiftLayer(nInput=_D, nOutput=_D)
    assert lyr._verb_spectrum is False
    assert lyr._verb_spec is None
    np_, verb = _np_and_verb()
    assert torch.equal(lyr.apply_verb(np_, verb), np_)        # exact no-op
    assert torch.equal(lyr.unapply_verb(np_, verb), np_)


def test_flag_on_builds_projection():
    _set_flag(True)
    try:
        lyr = LiftLayer(nInput=_D, nOutput=_D)
        assert lyr._verb_spectrum is True
        assert lyr._verb_spec is not None
    finally:
        _set_flag(False)


def test_untrained_is_identity():
    """Zero-init => w_v = 0 => e^0 = 1 => apply_verb is the identity until trained."""
    _set_flag(True)
    try:
        lyr = LiftLayer(nInput=_D, nOutput=_D)
        np_, verb = _np_and_verb()
        assert torch.allclose(lyr.apply_verb(np_, verb), np_, atol=1e-5)
    finally:
        _set_flag(False)


def test_roundtrip_exact_and_nontrivial():
    """With a forced spectrum: apply_verb MOVES the NP, and unapply_verb recovers
    it exactly (invertible by construction)."""
    _set_flag(True)
    try:
        lyr = _strong_spectrum(LiftLayer(nInput=_D, nOutput=_D))
        np_, verb = _np_and_verb()
        vp = lyr.apply_verb(np_, verb)
        assert float((vp - np_).abs().max()) > 0.05            # the verb acted
        back = lyr.unapply_verb(vp, verb)
        assert torch.allclose(back, np_, atol=1e-5)            # exact inverse
    finally:
        _set_flag(False)


def test_per_verb_differentiation():
    """Two different verb codes produce different operators on the same NP."""
    _set_flag(True)
    try:
        lyr = _strong_spectrum(LiftLayer(nInput=_D, nOutput=_D))
        np_, verb_a = _np_and_verb(seed=0)
        _n, verb_b = _np_and_verb(seed=7)
        assert not torch.allclose(
            lyr.apply_verb(np_, verb_a), lyr.apply_verb(np_, verb_b), atol=1e-4)
    finally:
        _set_flag(False)


def test_spectrum_is_sparse():
    """w_v is soft-thresholded -> the verb touches few eigs (some w exactly 0)."""
    _set_flag(True)
    try:
        lyr = _strong_spectrum(LiftLayer(nInput=_D, nOutput=_D))
        _n, verb = _np_and_verb()
        w = lyr._verb_spectrum_w(verb)
        assert float((w.abs() < 1e-9).float().mean()) > 0.0
    finally:
        _set_flag(False)


def test_spectrum_log_gain_is_bounded_and_preserves_normal_range():
    """The condition bound is an identity inside +/-8 and clips only an
    already numerically destructive spectrum."""
    layer = VerbLayer(nInput=_D, nOutput=_D)
    raw = torch.tensor(
        [-20.0, -7.0, -0.05, 0.0, 0.05, 3.0, 7.0, 20.0])
    with torch.no_grad():
        layer._verb_spec.weight.zero_()
        layer._verb_spec.bias.copy_(raw)

    got = layer._verb_spectrum_w(torch.zeros(1, _D)).squeeze(0)
    sparse = torch.sign(raw) * (raw.abs() - 0.1).clamp_min(0.0)
    expected = sparse.clamp(-8.0, 8.0)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    # Explicitly pin the identity-on-normal-range part of the contract.
    torch.testing.assert_close(got[1:-1], sparse[1:-1], rtol=0.0, atol=0.0)


class _LeftOperand(nn.Module):
    """Finite binary control op for the eager-router adjoint regression."""

    def forward(self, left, right):
        return left


def _overflow_router_case(selected_op):
    """Build the historical failure: Verb's raw log gain is far beyond the
    float32 exp range, while the router hard-selects either Verb (0) or the
    finite control op (1)."""
    verb = VerbLayer(nInput=4, nOutput=4)
    with torch.no_grad():
        verb._verb_spec.weight.fill_(100.0)
        verb._verb_spec.bias.zero_()

    router = BinaryStructuredReductionLayer(
        d_model=4, ops=[verb, _LeftOperand()], r_copy=1,
        chooser="anchordot")
    with torch.no_grad():
        # Make REDUCE beat two COPY actions, then choose the requested op.
        router.copy_anchor.fill_(-100.0)
        router.reduce_anchor.zero_()
        router.reduce_anchor[selected_op].fill_(100.0)

    x = torch.tensor(
        [[[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]],
        requires_grad=True)
    hard, soft, routing = router(x)
    assert int(routing["reduce_mask"].argmax(-1).item()) == selected_op
    assert torch.isfinite(hard).all()
    assert torch.isfinite(soft).all()

    soft.sum().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in verb.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_selected_overflowing_verb_candidate_has_finite_adjoint():
    """A selected saturated verb remains finite in forward and backward."""
    _overflow_router_case(selected_op=0)


def test_unselected_overflowing_verb_candidate_has_finite_adjoint():
    """All grammar candidates are evaluated eagerly.  Before the log-spectrum
    bound, even a hard-unselected Verb returned finite +1 in forward but sent
    NaNs through ExpBackward's 0*inf product."""
    _overflow_router_case(selected_op=1)


_CPU_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _require_cpu_low_precision(dtype):
    """Skip only when the installed torch cannot construct this CPU dtype."""
    try:
        torch.ones(1, dtype=dtype).float()
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        pytest.skip(f"CPU {dtype} is unavailable: {exc}")


def _low_precision_spectrum_layer():
    layer = VerbLayer(nInput=_D, nOutput=_D)
    with torch.no_grad():
        layer._verb_spec.weight.zero_()
        layer._verb_spec.bias.copy_(torch.tensor(
            [0.35, -0.35, 0.20, -0.20, 0.50, -0.50, 0.10, -0.10]))
    return layer


@pytest.mark.parametrize("dtype", _CPU_LOW_PRECISION_DTYPES)
def test_low_precision_exact_rails_have_finite_forward_and_backward(dtype):
    """Half-precision 1-epsilon rounds to 1, so the rail chart itself must
    run in float32 rather than merely relying on the outer tanh to hide inf."""
    _require_cpu_low_precision(dtype)
    layer = _low_precision_spectrum_layer()
    np_what = torch.tensor(
        [[-1.0, 1.0, -0.5, 0.5, -0.25, 0.25, 0.0, 0.75]],
        dtype=dtype, requires_grad=True)
    verb_what = torch.zeros_like(np_what, requires_grad=True)

    vp_what = layer.apply_verb(np_what, verb_what)
    recovered = layer.unapply_verb(vp_what, verb_what)

    assert vp_what.dtype == dtype
    assert recovered.dtype == dtype
    assert torch.isfinite(vp_what).all()
    assert torch.isfinite(recovered).all()

    (vp_what.float().square().sum()
     + recovered.float().square().sum()).backward()
    assert np_what.grad is not None and torch.isfinite(np_what.grad).all()
    assert float(np_what.grad.abs().sum()) > 0.0
    assert verb_what.grad is not None and torch.isfinite(verb_what.grad).all()
    for parameter in layer.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dtype", _CPU_LOW_PRECISION_DTYPES)
def test_low_precision_roundtrip_is_accurate_to_output_quantization(dtype):
    """Forward and inverse share one float32 rail path; only their public
    low-precision outputs are quantized between the two operations."""
    _require_cpu_low_precision(dtype)
    layer = _low_precision_spectrum_layer()
    np_what = torch.tensor(
        [[-0.25, -0.125, -0.0625, 0.0625,
          0.125, 0.25, 0.375, -0.375]], dtype=dtype)
    verb_what = torch.zeros_like(np_what)

    vp_what = layer.apply_verb(np_what, verb_what)
    recovered = layer.unapply_verb(vp_what, verb_what)

    assert vp_what.dtype == dtype
    assert recovered.dtype == dtype
    torch.testing.assert_close(
        recovered, np_what, rtol=0.0, atol=torch.finfo(dtype).eps)
