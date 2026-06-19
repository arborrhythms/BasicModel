"""Stage 7 of doc/plans/MeronomyPlan.md: the interpret-as-word gate.

MeronomySpec §6 (rev 2026-06-11), §10.8 gate half: naming is
SEARCH-THEN-MINT on licensed miss — a first-class, loggable decision;
never first sight. Folds never create table rows (the cordon is
structural: a fold result is a fresh vector, not a binding). The gate
exists only with <architecture><meronomy>on (dark landing).
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import PiLayer2, SigmaLayer2
from Spaces import WholeSpace, ConceptualSpace

D = 4


def _knob(value):
    from util import TheXMLConfig
    if value is None:
        TheXMLConfig._data.get("architecture", {}).pop("meronomy", None)
    else:
        TheXMLConfig.set("architecture.meronomy", value)


def bare_ss():
    """A WholeSpace stub: the gate touches only its own table state."""
    return WholeSpace.__new__(WholeSpace)


# ---------------------------------------------------------------------------
# Dark landing: no knob, no gate.
# ---------------------------------------------------------------------------

def test_gate_requires_the_knob():
    ws = bare_ss()
    _knob(None)
    with pytest.raises(RuntimeError):
        ws.interpret_word(3)


# ---------------------------------------------------------------------------
# Search, then mint on licensed miss; every decision logged.
# ---------------------------------------------------------------------------

def test_unlicensed_miss_is_a_placeholder_never_a_row():
    ws = bare_ss()
    _knob("on")
    try:
        d = ws.interpret_word(7, licensed=False)
        assert d['action'] == 'placeholder' and d['a'] == 0.0
        assert len(ws.reference_table) == 0, (
            "naming never happens on first sight")
        assert ws.gate_log[-1] is d, "first-class loggable decision"
    finally:
        _knob(None)


def test_licensed_miss_mints_then_hits():
    ws = bare_ss()
    _knob("on")
    try:
        d1 = ws.interpret_word(7, licensed=True, object_id=2)
        assert d1['action'] == 'mint' and d1['object'] == 2
        assert ws.reference_table.deref(7) == 2
        d2 = ws.interpret_word(7)            # now a hit: use, no re-mint
        assert d2['action'] == 'use' and d2['object'] == 2
        assert len(ws.reference_table) == 1
        assert [e['action'] for e in ws.gate_log] == ['mint', 'use']
    finally:
        _knob(None)


def test_licensed_miss_without_object_stays_placeholder():
    # The license alone cannot mint: a binding needs both sides (full
    # rows only) — reuse-evidence without a referent is still ignorance.
    ws = bare_ss()
    _knob("on")
    try:
        d = ws.interpret_word(9, licensed=True, object_id=None)
        assert d['action'] == 'placeholder'
        assert len(ws.reference_table) == 0
    finally:
        _knob(None)


def test_mint_gauge_orients_object_row():
    ws = bare_ss()
    _knob("on")
    try:
        ref = torch.rand(D)
        u = -(ref / ref.norm())
        d = ws.interpret_word(5, licensed=True, object_id=0,
                              object_row=u, referent=ref,
                              extent=torch.rand(D) * 0.5 + 0.2)
        assert ws.reference_table.extent_of(5) is not None
        # Review fix 2026-06-11: the gauge-oriented row must ride the
        # decision (the table stores ids only; the caller owns the
        # codebook write-back) -- previously it was dropped.
        oriented = d.get('oriented_row')
        assert oriented is not None
        assert torch.allclose(oriented, -u), "the flip, oriented +u"
        assert (oriented * ref).sum() >= 0
    finally:
        _knob(None)


def test_mint_shift_pushes_the_oriented_row():
    # The immediate shift path must carry the gauge-fixed row, not the
    # un-oriented codebook representative (review fix 2026-06-11).
    import torch.nn as nn
    from Language import SymbolicSubSpace
    ss = object.__new__(SymbolicSubSpace)
    nn.Module.__init__(ss)
    ss._stm_payload_dim = D
    ss._idea_capacity = 8
    ss._idea_max_depth_host = 0
    ss._idea_buffer = torch.zeros(1, 8, D)
    ss._idea_depth = torch.zeros(1, dtype=torch.long)

    rows = torch.rand(3, D)
    ref = torch.rand(D)
    u = -(ref / ref.norm())
    rows[1] = u                              # un-oriented representative
    ws = bare_ss()
    _knob("on")
    try:
        d = ws.shift_word(ss, 0, 9, rows, licensed=True, object_id=1,
                          object_row=u, referent=ref)
        assert d['action'] == 'mint'
        pushed = ss._idea_buffer[0, 0]
        assert torch.allclose(pushed, -u), (
            "the mint-shift crosses the ORIENTED row")
        assert (pushed * ref).sum() >= 0
    finally:
        _knob(None)


# ---------------------------------------------------------------------------
# Cordon: folds never create rows (structural, tested not guarded).
# ---------------------------------------------------------------------------

def test_folds_never_create_rows():
    ws = bare_ss()
    _knob("on")
    try:
        ws.interpret_word(1, licensed=True, object_id=0)
        n_before = len(ws.reference_table)
        torch.manual_seed(1)
        pi = PiLayer2(2 * D, D, blocks=2)
        sig = SigmaLayer2(2 * D, D, blocks=2)
        A = torch.rand(3, D) * 0.6 + 0.2
        B = torch.rand(3, D) * 0.6 + 0.2
        pi.compose(A, B)
        sig.compose(A, B)
        pi.forward(torch.cat([A, B], dim=-1))
        ConceptualSpace.factor_percept(torch.rand(3, D),
                                       torch.rand(2, D))
        assert len(ws.reference_table) == n_before, (
            "a fold result is a fresh vector, not a binding — you "
            "cannot compute your way to a name")
        # Semantic use of an unbound code fails naturally: deref misses.
        assert ws.reference_table.deref(999) is None
    finally:
        _knob(None)
