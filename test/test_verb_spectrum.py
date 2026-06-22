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

import torch

from util import TheXMLConfig
from Language import LiftLayer

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
