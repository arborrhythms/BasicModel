"""Adverb sparse eigenvalue edit (the eig-based VERB edit was removed -- the verb
IS the lift operator; the ADVERB keeps the eigenmodifier: "ADV modifies the eigs
of VP").

When <adverbEigEdit> is on, ``LiftLayer.apply_adverb(vp, adv)`` modifies a
composed VP through a SPARSE eigenvalue edit of the verb, masked by the VP's OWN
eigen-signature (membership read from the VP -- no learned mask), preserving the
unused eigs: a2 = atanh(vp) + p_vp (.) delta_adv. The only per-adverb parameter
is the sparse edit delta_adv, built ONLY when the flag is on (flag-off
byte-identical). Crucially, lift.forward is now a PLAIN sigma fold (no verb edit).
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
    data.setdefault("architecture", {})["adverbEigEdit"] = bool(value)


def _vp_and_adv(seed=0):
    g = torch.Generator().manual_seed(seed)
    vp = torch.randn(2, 5, _D, generator=g).tanh() * 0.8
    adv = torch.randn(2, 5, _D, generator=g).tanh() * 0.8
    return vp, adv


def _strong_edit(lyr, seed=1):
    """Force a non-trivial, ADV-dependent edit (the zero-bias init thresholds to
    ~0 untrained). Returns the layer for chaining."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        lyr._adv_edit.weight.copy_(
            torch.randn(lyr._adv_edit.weight.shape, generator=g) * 0.8)
        lyr._adv_edit.bias.fill_(0.4)
    return lyr


def test_lift_forward_is_plain_sigma_fold():
    """The verb no longer eig-edits: lift.forward == the sigma fold, with the
    adverb flag off OR on."""
    for flag in (False, True):
        _set_flag(flag)
        try:
            lyr = LiftLayer(nInput=_D, nOutput=_D)
            left, right = _vp_and_adv()
            assert torch.equal(lyr.forward(left, right),
                               lyr._sigma.compose(left, right))
        finally:
            _set_flag(False)


def test_flag_off_apply_adverb_is_noop():
    _set_flag(False)
    lyr = LiftLayer(nInput=_D, nOutput=_D)
    assert lyr._adverb_eig_edit is False
    assert lyr._adv_edit is None
    vp, adv = _vp_and_adv()
    assert torch.equal(lyr.apply_adverb(vp, adv), vp)   # exact no-op


def test_flag_on_builds_edit_projection():
    _set_flag(True)
    try:
        lyr = LiftLayer(nInput=_D, nOutput=_D)
        assert lyr._adverb_eig_edit is True
        assert lyr._adv_edit is not None
    finally:
        _set_flag(False)


def test_untrained_adverb_is_noop():
    """Zero-init residual => an UNTRAINED adverb edit is exactly 0, so
    apply_adverb reproduces the VP (training shapes it)."""
    _set_flag(True)
    try:
        lyr = LiftLayer(nInput=_D, nOutput=_D)
        vp, adv = _vp_and_adv()
        out = lyr.apply_adverb(vp, adv)
        assert torch.allclose(out, vp, atol=1e-5)
        assert lyr._adverb_purchase is not None         # diagnostic stashed
        _strong_edit(lyr)
        out_strong = lyr.apply_adverb(vp, adv)
        assert float((out_strong - vp).abs().max()) > 0.1
    finally:
        _set_flag(False)


def test_strong_adverb_changes_output_but_preserves_complement():
    """With a forced edit: dims the VP uses are edited, but a dim the VP lacks
    (vp[..., k] == 0 -> p_vp[..., k] == 0) is preserved exactly."""
    _set_flag(True)
    try:
        lyr = _strong_edit(LiftLayer(nInput=_D, nOutput=_D))
        vp, adv = _vp_and_adv()
        vp = vp.clone()
        vp[..., 0] = 0.0                                 # VP lacks eig 0
        out = lyr.apply_adverb(vp, adv)
        assert not torch.allclose(out, vp, atol=1e-4)    # edited somewhere
        assert torch.allclose(out[..., 0], vp[..., 0], atol=1e-6)  # eig 0 preserved
    finally:
        _set_flag(False)


def test_per_adverb_differentiation():
    """Two different ADV codes produce different edits on the same VP."""
    _set_flag(True)
    try:
        lyr = _strong_edit(LiftLayer(nInput=_D, nOutput=_D))
        vp, adv_a = _vp_and_adv(seed=0)
        _v, adv_b = _vp_and_adv(seed=7)
        assert not torch.allclose(
            lyr.apply_adverb(vp, adv_a), lyr.apply_adverb(vp, adv_b), atol=1e-4)
    finally:
        _set_flag(False)


def test_delta_is_sparse():
    """The adverb's eigenvalue edit is soft-thresholded -> it touches few eigs
    (some delta components are exactly zero)."""
    _set_flag(True)
    try:
        lyr = _strong_edit(LiftLayer(nInput=_D, nOutput=_D))
        _v, adv = _vp_and_adv()
        delta = torch.tanh(lyr._adv_edit(adv.to(lyr._adv_edit.weight.dtype)))
        tau = 0.1
        delta = torch.sign(delta) * torch.clamp(delta.abs() - tau, min=0.0)
        assert float((delta.abs() < 1e-9).float().mean()) > 0.0
    finally:
        _set_flag(False)
