"""The symbolic sigma-pyramid (dual-towers rev 2): depth, self-reference.

Pins the STRUCTURAL semantics of the single untyped square store: a depth-d
Gallistel vine completes structurally in ONE feedforward pass (rung d reads
rung d-1's winners); self-edges are the forbidden Quine atom; relate(x, x)
merges to ONE untyped edge.
(doc/plans/2026-07-10-conceptual-wave-ff-pyramid-design.md)
"""
# NOTE (dual-towers rev 2, 2026-07-11): the Kripke groundedness/cycle probe
# tests are RETIRED with cs_groundedness_probe -- a feedforward pyramid cannot
# represent self-sustaining loops (design doc, decision 5).

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

import Spaces
from test_basicmodel import _populate_test_config

_D = 8


def _cs(nS=64, order=3, serial=False):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    cs = Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])
    object.__setattr__(cs, "_symbolic_order", order)
    object.__setattr__(cs, "_serial", serial)
    return cs


def _layer(cs):
    return Spaces._concept_alloc_of(cs).layer(0)


def _row0(cs, cid):
    return _layer(cs).row_of(("snap", int(cid)))


def _rowp(cs, cid):
    """Global row of relation cid across the per-order namespaces (rev 2)."""
    for k, r in _layer(cs)._tensor_rows.items():
        if (isinstance(k, tuple) and len(k) == 2 and k[0] != "snap"
                and int(k[1]) == int(cid)):
            return r
    return None


def _set_edge_value(cs, row, col, w):
    """no_grad rewrite of an EXISTING edge's learnable value (a trained state)."""
    ly = _layer(cs)
    c = int(col)
    if c == int(cs.nVectors):
        c = int(ly.nOutput)              # global bias -> store coordinates
    with torch.no_grad():
        ly.values[ly._index[(int(row), c)]] = float(w)


def _mint_vine(cs, n=4):
    """n order-0 word mints + the bias-bounded Gallistel chain over them."""
    words = []
    for i in range(n):
        A, _b, _c = cs.create_word_object_meta(
            [2 * i + 1], 2 * i + 2, key=f"w{i}")
        words.append(A)
    head = cs.create_joint_concept(words, key=tuple(f"w{i}" for i in range(n)))
    return words, head


def _vine_links(cs, head, words):
    """[head, ..., tail] link ids, walking each link's part-role sym ref."""
    alloc = Spaces._concept_alloc_of(cs)
    wset = {int(w) for w in words}
    links, cur = [int(head)], int(head)
    while True:
        part = [x for (r, x) in alloc.records(cur)
                if r == "part" and isinstance(x, tuple) and x[0] == "sym"][0]
        if int(part[1]) in wset:
            return links                     # cur's rest IS a word: the tail
        cur = int(part[1])
        links.append(cur)


def _zero_bias(cs, rows):
    """Zero the EVERYTHING-bias values (store bias col == S, rev 2):
    isolates the pure chain path through the pyramid."""
    for r in rows:
        _set_edge_value(cs, r, int(_layer(cs).nOutput), 0.0)


# -- 1. structural depth --------------------------------------------------------

def test_depth_d_vine_completes_structurally():
    """dual-towers rev 2: 4 words -> 3 links across order blocks 1..3; the
    HEAD completes STRUCTURALLY in ONE feedforward pass (rung d reads rung
    d-1's winners) and zeroing the vine's edge values kills it."""
    cs = _cs(nS=64, order=3)
    words, head = _mint_vine(cs, 4)
    links = _vine_links(cs, head, words)             # [head, mid, tail]
    rows = [_rowp(cs, c) for c in links]
    h, m, tl = rows
    _zero_bias(cs, rows)                             # isolate the CHAIN path
    n_snap = cs._order_caps()[0]
    a_0 = torch.zeros(n_snap, 1)
    for w in words:
        a_0[_row0(cs, w), 0] = 0.75
    what = torch.randn(64, _D)
    _c, a = cs.cs_forward_content(a_0, what)         # ONE pass, no iteration
    assert float(a.detach()[tl, 0]) > 0.5            # tail: tanh(1.5)
    assert float(a.detach()[m, 0]) > 0.5             # mid reads the tail rung
    assert float(a.detach()[h, 0]) > 0.5             # head completes in-pass
    for r in rows:                                   # kill the vine's edges
        for c, _w in cs.concept_weights(r):
            _set_edge_value(cs, r, c, 0.0)
    _c, a2 = cs.cs_forward_content(a_0, what)
    assert abs(float(a2.detach()[h, 0])) < 1e-6      # the vine was the cause


# -- 2/3. self-reference at the store boundary ----------------------------------

def test_no_self_edge_via_populate():
    cs = _cs(nS=16, order=2)
    alloc = Spaces._concept_alloc_of(cs)
    other = alloc.new_concept()                      # min-support filler
    C = alloc.new_concept()
    alloc.add(C, "part", ("sym", C))                 # x = {x}: the Quine atom
    alloc.add(C, "part", ("sym", other))             # >= 2 sym constituents
    with pytest.raises(ValueError, match="self-edge"):
        cs._populate_concept_weights(C)


def test_relate_x_x_merges_to_one_edge():
    cs = _cs(nS=16, order=2)
    A, _b, _c = cs.create_word_object_meta([1], 2, key="w")
    sx = cs.singleton_concept(A)                     # the unit set {A}
    C = cs.reify_concept(sx, sx)                     # part- AND whole-leg on sx
    cs._populate_concept_weights(C)                  # relate() records only
    c_row, s_row = _rowp(cs, C), _rowp(cs, sx)
    got = cs.concept_weights(c_row)
    assert [c for (c, _w) in got].count(s_row) == 1  # merged: ONE untyped edge
    assert got == [(s_row, 1.0)]                     # ...and nothing else


# -- 4. cycles: observed, never policed ------------------------------------------
# RETIRED (dual-towers rev 2): test_cycle_flagged_by_wave_qe_not_settling --
# _cs_wave_qe is None; cycle observability was sacrificed with the settling
# dynamics (a feedforward pyramid has no settle residual).


# -- 5/6/7. Kripke groundedness ---------------------------------------------------