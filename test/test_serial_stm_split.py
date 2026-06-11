"""Stage 7 of doc/plans/MeronomyPlan.md: the serial-mode duals.

MeronomySpec §6 (rev 2026-06-10c/11), §10.11. The two workspaces:
the PS-side idea stack (existing ``_idea_*`` — structurally unchanged,
semantic content) and the new SS-side constituent stack (symbolic codes
under analysis). Moves: split (SS analysis, serial π form), shift (the
callosum crossing at codebook words), reduce (PS synthesis, serial σ
form, via the existing driver). Exactly one workspace write per move;
shifted content is the semantic referent, never the word code (except
stop... mention-shifts, which carry the form verbatim — the zero-band
signature marks it); marker words bind the router and shift nothing;
parallel mode leaves both serial stacks untouched.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch
import torch.nn as nn

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

from Layers import SigmaLayer2
from References import symbol_code
from Spaces import SymbolicSpace

D = 4
CAP = 8


def _knob(value):
    from util import TheXMLConfig
    if value is None:
        TheXMLConfig._data.get("architecture", {}).pop("meronomy", None)
    else:
        TheXMLConfig.set("architecture.meronomy", value)


def make_ws(batch=1, dim=D, cap=CAP):
    """Bare WordSubSpace with idea-stack buffers (the established
    object.__new__ fixture idiom from _stm_test_fixtures)."""
    from Language import WordSubSpace
    ws = object.__new__(WordSubSpace)
    nn.Module.__init__(ws)
    ws._stm_payload_dim = int(dim)
    ws._idea_capacity = int(cap)
    ws._idea_max_depth_host = 0
    ws._idea_buffer = torch.zeros(batch, cap, dim)
    ws._idea_depth = torch.zeros(batch, dtype=torch.long)
    return ws


def snapshot(ws):
    """(idea, constituent) state snapshot for single-writer assertions."""
    idea = (ws._idea_buffer.clone(), ws._idea_depth.clone())
    cb = getattr(ws, '_constituent_buffer', None)
    cons = (None if cb is None
            else (cb.clone(), ws._constituent_depth.clone()))
    return idea, cons


def idea_changed(before, after):
    (b_buf, b_dep), _ = before
    (a_buf, a_dep), _ = after
    return not (torch.equal(b_buf, a_buf) and torch.equal(b_dep, a_dep))


def cons_changed(before, after):
    _, b = before
    _, a = after
    if b is None and a is None:
        return False
    if b is None or a is None:
        return a is not None and bool((a[1] != 0).any())
    return not (torch.equal(b[0], a[0]) and torch.equal(b[1], a[1]))


# ---------------------------------------------------------------------------
# The SS-side constituent stack.
# ---------------------------------------------------------------------------

def test_constituent_stack_mechanics():
    ws = make_ws()
    assert ws.constituent_depth_of(0) == 0, "dark until first use"
    c1, c2 = torch.rand(D), torch.rand(D)
    ws.constituent_push(0, c1)
    ws.constituent_push(0, c2)
    assert ws.constituent_depth_of(0) == 2
    assert torch.equal(ws.constituent_peek(0, 0), c2), "newest at slot 0"
    assert torch.equal(ws.constituent_peek(0, 1), c1)
    top = ws.constituent_pop(0)
    assert torch.equal(top, c2)
    assert ws.constituent_depth_of(0) == 1
    ws.constituent_clear()
    assert ws.constituent_depth_of(0) == 0


def test_constituent_capacity_is_the_miller_cap():
    ws = make_ws()
    for i in range(CAP):
        ws.constituent_push(0, torch.rand(D))
    with pytest.raises(RuntimeError):
        ws.constituent_push(0, torch.rand(D))


def test_split_replaces_whole_with_parts():
    ws = make_ws()
    whole, left, right = torch.rand(D), torch.rand(D), torch.rand(D)
    ws.constituent_push(0, whole)
    ws.constituent_split(0, left, right)
    assert ws.constituent_depth_of(0) == 2
    assert torch.equal(ws.constituent_peek(0, 0), left), (
        "left-to-right analysis: left is newest")
    assert torch.equal(ws.constituent_peek(0, 1), right)


# ---------------------------------------------------------------------------
# The shift: semantic referents cross; forms only under mention.
# ---------------------------------------------------------------------------

def _bound_ss(rows):
    ss = SymbolicSpace.__new__(SymbolicSpace)
    ss.interpret_word(7, licensed=True, object_id=1)   # bind word 7 -> row 1
    return ss


def test_shift_pushes_the_semantic_referent():
    rows = torch.rand(3, D)
    ws = make_ws()
    _knob("on")
    try:
        ss = _bound_ss(rows)
        d = ss.shift_word(ws, 0, 7, rows)
        assert d['action'] == 'use'
        assert torch.equal(ws._idea_buffer[0, 0], rows[1]), (
            "the dereferenced SEMANTIC row crosses -- the word is part "
            "of the sentence; the referent is not")
        assert int(ws._idea_depth[0].item()) == 1
    finally:
        _knob(None)


def test_unknown_word_shifts_ignorance_placeholder():
    rows = torch.rand(3, D)
    ws = make_ws()
    _knob("on")
    try:
        ss = SymbolicSpace.__new__(SymbolicSpace)
        d = ss.shift_word(ws, 0, 99, rows, licensed=False)
        assert d['action'] == 'placeholder'
        assert (ws._idea_buffer[0, 0] == 0).all(), "a = 0 placeholder"
        assert int(ws._idea_depth[0].item()) == 1, (
            "the placeholder still occupies the word's position")
    finally:
        _knob(None)


def test_marker_word_binds_router_and_shifts_nothing():
    rows = torch.rand(3, D)
    ws = make_ws()
    _knob("on")
    try:
        ss = SymbolicSpace.__new__(SymbolicSpace)
        before = snapshot(ws)
        d = ss.shift_word(ws, 0, 5, rows, marker=True)
        after = snapshot(ws)
        assert d['action'] == 'marker-bind'
        assert not idea_changed(before, after), "nothing shifts"
        assert not cons_changed(before, after)
        assert ss.gate_log[-1]['action'] == 'marker-bind', "logged"
    finally:
        _knob(None)


def test_mention_shifts_the_word_code_verbatim():
    rows = torch.rand(3, D)
    ws = make_ws()
    _knob("on")
    try:
        ss = SymbolicSpace.__new__(SymbolicSpace)
        # A zero-banded word code (the signature that marks form
        # content): what-part then zeroed where/when.
        code = symbol_code(7, n_what=2, n_where=1, n_when=1)
        d = ss.shift_word(ws, 0, 7, rows, mention=True, word_vec=code)
        assert d['action'] == 'mention-shift'
        assert torch.equal(ws._idea_buffer[0, 0], code), (
            "quotation: the form itself crosses, no deref")
        with pytest.raises(ValueError):
            ss.shift_word(ws, 0, 7, rows, mention=True)   # needs the code
    finally:
        _knob(None)


# ---------------------------------------------------------------------------
# Single-writer mutex: each move writes exactly one workspace.
# ---------------------------------------------------------------------------

def test_one_move_one_workspace_write():
    rows = torch.rand(3, D)
    ws = make_ws()
    _knob("on")
    try:
        ss = _bound_ss(rows)
        # SPLIT writes the constituent stack only.
        ws.constituent_push(0, torch.rand(D))
        before = snapshot(ws)
        ws.constituent_split(0, torch.rand(D), torch.rand(D))
        after = snapshot(ws)
        assert cons_changed(before, after) and not idea_changed(before, after)
        # SHIFT writes the idea stack only.
        before = snapshot(ws)
        ss.shift_word(ws, 0, 7, rows)
        after = snapshot(ws)
        assert idea_changed(before, after) and not cons_changed(before, after)
    finally:
        _knob(None)


# ---------------------------------------------------------------------------
# Deref/decode round-trip; reduce-chain vs parallel extent; parallel
# mode leaves the serial stacks untouched.
# ---------------------------------------------------------------------------

def test_deref_round_trip_preserves_extent():
    rows = torch.rand(3, D)
    ws = make_ws()
    _knob("on")
    try:
        ss = SymbolicSpace.__new__(SymbolicSpace)
        ext = torch.rand(D) * 0.5 + 0.2
        ss.interpret_word(7, licensed=True, object_id=1, extent=ext)
        ss.shift_word(ws, 0, 7, rows)
        assert torch.equal(ws._idea_buffer[0, 0], rows[1]), (
            "deref → decode lands the bound row exactly")
        assert torch.equal(ss.reference_table.extent_of(7), ext)
    finally:
        _knob(None)


def test_serial_reduce_chain_matches_parallel_sigma_extent():
    # §10.11: for associative content the serial reduce chain's extent
    # matches the parallel σ extent to tolerance. At near-identity init
    # the σ2 kernel is the probabilistic-sum family, whose parallel
    # n-ary extent is 1 − Π(1 − m_i).
    torch.manual_seed(4)
    sig = SigmaLayer2(2 * D, D, blocks=2)
    A = torch.rand(1, D) * 0.3 + 0.2
    B = torch.rand(1, D) * 0.3 + 0.2
    C = torch.rand(1, D) * 0.3 + 0.2
    chain = sig.compose(sig.compose(A, B), C)          # serial reduces
    parallel = 1.0 - (1 - A) * (1 - B) * (1 - C)        # parallel extent
    assert torch.allclose(chain, parallel, atol=0.08), (
        f"reduce chain {chain.tolist()} vs parallel {parallel.tolist()}")
    # And the chain is associative to tolerance (the content is).
    chain2 = sig.compose(sig.compose(A, C), B)
    assert torch.allclose(chain, chain2, atol=0.08)


def test_parallel_mode_leaves_serial_stacks_untouched():
    ws = make_ws(batch=2)
    # The parallel-mode push (idea_push_step) is the whole-slab path;
    # the SS-side analysis stack must stay dark through it.
    ws.idea_push_step(torch.rand(2, D))
    assert getattr(ws, '_constituent_buffer', None) is None
    assert ws.constituent_depth_of(0) == 0
    assert ws.constituent_depth_of(1) == 0
