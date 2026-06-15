"""Ramsification table -- per-code Sigma/Pi fold record + inversion.

Phase 5 of the part/whole refactor. The Pi/Sigma folds carry a code onto
sortable mereological space but do not preserve their own ramsification.
This per-row sidecar (``Codebook.ramsification``, ``[V, max_order]`` uint8,
index-aligned with the codebook) records which fold each code was routed
through at each subsymbolic pass -- FOLD_NEITHER / FOLD_SIGMA / FOLD_PI --
so ``invert_ramsified`` can walk the inverse fold chain back to the
codebook row. The abstraction ORDER of a code is its fold count: proper
noun / prototype = 0, regular noun / type = 1, count noun = 2.

The table is an opt-in additive sidecar (plain attr, not a Parameter /
buffer): enabling it adds no state_dict keys and cannot move a pinned
basin.
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
import torch.nn as nn

from Spaces import Codebook
import Layers


def _codebook(V=6, D=4):
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(V, D))
    cb.nVectors = V
    return cb


# -- allocation / record / read -------------------------------------------

def test_enable_allocates_zeroed_table():
    cb = _codebook(V=6)
    cb.enable_ramsification(max_order=3)
    assert cb.ramsification.shape == (6, 3)
    assert cb.ramsification.dtype == torch.uint8
    assert int(cb.ramsification.sum()) == 0          # all FOLD_NEITHER


def test_enable_is_idempotent_widen_preserves():
    cb = _codebook(V=4)
    cb.enable_ramsification(max_order=2)
    cb.record_fold(torch.tensor([1]), pass_idx=0, route=Codebook.FOLD_SIGMA)
    cb.enable_ramsification(max_order=4)              # widen
    assert cb.ramsification.shape == (4, 4)
    assert int(cb.ramsification[1, 0]) == Codebook.FOLD_SIGMA   # preserved
    assert int(cb.ramsification[1, 3]) == Codebook.FOLD_NEITHER


def test_record_and_fold_sequence():
    cb = _codebook(V=5)
    cb.enable_ramsification(max_order=3)
    cb.record_fold(torch.tensor([2]), 0, Codebook.FOLD_SIGMA)
    cb.record_fold(torch.tensor([2]), 1, Codebook.FOLD_PI)
    seq = cb.fold_sequence(2)
    assert list(int(v) for v in seq) == [
        Codebook.FOLD_SIGMA, Codebook.FOLD_PI, Codebook.FOLD_NEITHER]


def test_record_out_of_range_pass_is_noop():
    cb = _codebook(V=3)
    cb.enable_ramsification(max_order=2)
    cb.record_fold(torch.tensor([0]), 5, Codebook.FOLD_PI)   # out of range
    assert int(cb.ramsification.sum()) == 0


# -- word abstraction order -----------------------------------------------

def test_abstraction_order_counts_folds():
    cb = _codebook(V=4)
    cb.enable_ramsification(max_order=3)
    # row 0: proper noun (order 0, no folds)
    # row 1: regular noun (order 1, one fold)
    cb.record_fold(torch.tensor([1]), 0, Codebook.FOLD_PI)
    # row 2: count noun (order 2, two folds)
    cb.record_fold(torch.tensor([2]), 0, Codebook.FOLD_SIGMA)
    cb.record_fold(torch.tensor([2]), 1, Codebook.FOLD_PI)
    assert cb.abstraction_order(0) == 0
    assert cb.abstraction_order(1) == 1
    assert cb.abstraction_order(2) == 2


def test_abstraction_order_zero_without_table():
    cb = _codebook()
    assert cb.abstraction_order(0) == 0


# -- inversion round-trip -------------------------------------------------

def test_invert_ramsified_round_trips_the_fold_chain():
    D = 4
    sigma = Layers.SigmaLayer(D, D, naive=True, invertible=True)
    pi = Layers.PiLayer(D, D, naive=True, invertible=True)
    sigma.set_sigma(0)
    pi.set_sigma(0)
    cb = _codebook(V=3, D=D)
    cb.enable_ramsification(max_order=2)
    # Row 1 was folded: pass 0 through Sigma, pass 1 through Pi.
    row = 1
    cb.record_fold(torch.tensor([row]), 0, Codebook.FOLD_SIGMA)
    cb.record_fold(torch.tensor([row]), 1, Codebook.FOLD_PI)

    x = torch.randn(2, 3, D).tanh()
    # Forward fold chain (pass 0 then pass 1):
    code = pi.forward(sigma.forward(x))
    # Invert via the recorded table -> back to the pre-fold value.
    x_rec = cb.invert_ramsified(code, row, sigma=sigma, pi=pi)
    err = (x - x_rec).abs().max().item()
    assert err < 1e-3, f"ramsified inversion did not round-trip (err={err:.2e})"


def test_invert_with_neither_pass_is_identity_on_that_pass():
    D = 4
    sigma = Layers.SigmaLayer(D, D, naive=True, invertible=True)
    sigma.set_sigma(0)
    cb = _codebook(V=2, D=D)
    cb.enable_ramsification(max_order=2)
    # Only pass 1 folds (Sigma); pass 0 is NEITHER (identity).
    cb.record_fold(torch.tensor([0]), 1, Codebook.FOLD_SIGMA)
    x = torch.randn(2, 3, D).tanh()
    code = sigma.forward(x)
    x_rec = cb.invert_ramsified(code, 0, sigma=sigma)
    assert (x - x_rec).abs().max().item() < 1e-3


def test_invert_without_table_returns_code_unchanged():
    cb = _codebook()
    code = torch.randn(2, 3, 4)
    out = cb.invert_ramsified(code, 0)
    assert torch.equal(out, code)


# -- basin safety: no state_dict key --------------------------------------

def test_ramsification_not_in_state_dict():
    cb = _codebook(V=4)
    cb.enable_ramsification(max_order=3)
    keys = list(cb.state_dict().keys())
    assert not any("ramsification" in k for k in keys), (
        f"ramsification must stay out of state_dict (got {keys})")


# -- resize keeps the table index-aligned ---------------------------------

def test_grow_to_preserves_and_extends_table():
    cb = _codebook(V=3, D=4)
    cb.enable_ramsification(max_order=2)
    cb.record_fold(torch.tensor([2]), 0, Codebook.FOLD_PI)
    cb.grow_to(6)
    assert cb.ramsification.shape == (6, 2)
    assert int(cb.ramsification[2, 0]) == Codebook.FOLD_PI    # preserved
    assert int(cb.ramsification[5].sum()) == 0               # new rows NEITHER
