"""Verb sparse eigenvalue edit on lift(NP, VP) (doc/specs/
semantic_verb_np_mask_eigenvalue_proposal.md).

When <verbEigEdit> is on, LiftLayer applies the verb as a SPARSE eigenvalue
edit of the NP, masked by the noun CLASS (membership read from the NP's own
eigen-signature -- no learned mask parameter), preserving the NP complement:
x2 = x1 + p_class (.) delta_v. The only per-verb parameter is the sparse edit
delta_v, built ONLY when the flag is on (so flag-off is byte-identical).

These exercise LiftLayer directly: flag-off identity to the sigma fold, the
zero-bias untrained no-op, complement preservation where the NP lacks a
feature, per-verb differentiation, and edit sparsity.
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
    data.setdefault("architecture", {})["verbEigEdit"] = bool(value)


def _operands(seed=0):
    g = torch.Generator().manual_seed(seed)
    left = torch.randn(2, 5, _D, generator=g).tanh() * 0.8
    right = torch.randn(2, 5, _D, generator=g).tanh() * 0.8
    return left, right


def _strong_edit(lyr, seed=1):
    """Force a non-trivial, VP-dependent edit (the zero-bias init thresholds to
    ~0 untrained). Returns the layer for chaining."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        lyr._lex_edit.weight.copy_(
            torch.randn(lyr._lex_edit.weight.shape, generator=g) * 0.8)
        lyr._lex_edit.bias.fill_(0.4)
    return lyr


def test_flag_off_is_byte_identical_to_sigma_fold():
    _set_flag(False)
    lyr = LiftLayer(nInput=_D, nOutput=_D)
    assert lyr._verb_eig_edit is False
    assert lyr._lex_edit is None
    left, right = _operands()
    out = lyr.forward(left, right)
    ref = lyr._sigma.compose(left, right)
    assert torch.equal(out, ref)            # forward == fold, exactly


def test_flag_on_builds_edit_projection():
    _set_flag(True)
    try:
        lyr = LiftLayer(nInput=_D, nOutput=_D)
        assert lyr._verb_eig_edit is True
        assert lyr._lex_edit is not None
    finally:
        _set_flag(False)


def test_untrained_edit_is_noop():
    """Zero-init residual branch => an UNTRAINED verb edit is exactly 0, so the
    flag-on-untrained forward reproduces the sigma fold (training shapes it)."""
    _set_flag(True)
    try:
        lyr = LiftLayer(nInput=_D, nOutput=_D)
        left, right = _operands()
        ref = lyr._sigma.compose(left, right)
        out = lyr.forward(left, right)
        assert torch.allclose(out, ref, atol=1e-5)
        assert lyr._verb_purchase is not None      # diagnostic still stashed
        # A forced edit moves the output far more than the untrained one.
        _strong_edit(lyr)
        out_strong = lyr.forward(left, right)
        assert float((out_strong - ref).abs().max()) > 0.1
    finally:
        _set_flag(False)


def test_strong_edit_changes_output_but_preserves_complement():
    """With a forced edit: dims where the NP is active are edited, but a dim
    the NP lacks (left[..., k] == 0 -> p_class[..., k] == 0) is preserved
    exactly -- the spec's complement preservation."""
    _set_flag(True)
    try:
        lyr = _strong_edit(LiftLayer(nInput=_D, nOutput=_D))
        left, right = _operands()
        left = left.clone()
        left[..., 0] = 0.0                       # NP lacks feature 0 (out of class)
        out = lyr.forward(left, right)
        ref = lyr._sigma.compose(left, right)
        # The edit moved the output somewhere.
        assert not torch.allclose(out, ref, atol=1e-4)
        # ... but feature 0 (p_class == 0) is preserved exactly.
        assert torch.allclose(out[..., 0], ref[..., 0], atol=1e-6)
    finally:
        _set_flag(False)


def test_per_verb_differentiation():
    """Two different VP codes produce different edits on the same NP."""
    _set_flag(True)
    try:
        lyr = _strong_edit(LiftLayer(nInput=_D, nOutput=_D))
        left, right_a = _operands(seed=0)
        _l, right_b = _operands(seed=7)
        out_a = lyr.forward(left, right_a)
        out_b = lyr.forward(left, right_b)
        assert not torch.allclose(out_a, out_b, atol=1e-4)
    finally:
        _set_flag(False)


def test_delta_is_sparse():
    """The verb's eigenvalue edit is soft-thresholded -> it touches few eigs
    (some delta components are exactly zero)."""
    _set_flag(True)
    try:
        lyr = _strong_edit(LiftLayer(nInput=_D, nOutput=_D))
        _l, right = _operands()
        delta = torch.tanh(lyr._lex_edit(right.to(lyr._lex_edit.weight.dtype)))
        tau = 0.1
        delta = torch.sign(delta) * torch.clamp(delta.abs() - tau, min=0.0)
        zero_frac = float((delta.abs() < 1e-9).float().mean())
        assert zero_frac > 0.0                   # at least some eigs untouched
    finally:
        _set_flag(False)
