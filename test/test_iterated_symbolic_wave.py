"""The iterated symbolic wave (v3): depth schedule, cycles, groundedness.

Pins the temporal semantics of the single untyped square store: a depth-d
Gallistel vine completes at wave iteration d (tail links first); self-edges
are the forbidden Quine atom while LONGER cycles are legal, deliberate and
REPORTED (wave QE) never raised; the Kripke groundedness probe separates
source-answering structure (the vine) from free self-sustaining loops.
(doc/plans/2026-07-02-iterated-symbolic-loop.md)
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
    return _layer(cs).row_of(("pool", int(cid)))


def _set_edge_value(cs, row, col, w):
    """no_grad rewrite of an EXISTING edge's learnable value (a trained state)."""
    ly = _layer(cs)
    with torch.no_grad():
        ly.values[ly._index[(int(row), int(col))]] = float(w)


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
    """Zero the EVERYTHING-bias values: isolates the pure chain path in the
    WAVE schedule (the probe masks the bias column natively)."""
    for r in rows:
        _set_edge_value(cs, r, int(cs.nVectors), 0.0)


# -- 1. the depth schedule ------------------------------------------------------

def test_depth_d_vine_completes_at_iteration_d():
    """4 words -> 3 links; the TAIL lights at iteration 1, the HEAD only once
    the wave has walked the whole rest-chain (iteration 3): dark at K=1 AND
    K=2, decisive at K=3, monotone."""
    cs = _cs(nS=64, order=3)
    words, head = _mint_vine(cs, 4)
    links = _vine_links(cs, head, words)             # [head, mid, tail]
    rows = [_rowp(cs, c) for c in links]
    h, m, tl = rows
    _zero_bias(cs, rows)                             # isolate the CHAIN path
    n_snap = cs._order_caps()[0]
    a_0 = torch.zeros(n_snap, 1)
    # Source ONLY the tail's words: the head's sole live input is the chain
    # (its direct word edge reads a silent row) -- the pure depth schedule.
    for w in (words[-1], words[-2]):
        a_0[_row0(cs, w), 0] = 0.75
    what = torch.randn(64, _D)
    acts = {}
    for K in (1, 2, 3):
        object.__setattr__(cs, "_symbolic_order", K)
        _c, a = cs.cs_forward_content(a_0, what)
        acts[K] = a.detach()
    object.__setattr__(cs, "_symbolic_order", 3)
    assert float(acts[1][tl, 0]) > 0.5               # tail at K=1: tanh(1.5)
    assert abs(float(acts[1][m, 0])) < 1e-6          # mid dark at K=1...
    assert float(acts[2][m, 0]) > 0.5                # ...lights at K=2
    assert abs(float(acts[1][h, 0])) < 1e-6          # head dark at K=1
    assert abs(float(acts[2][h, 0])) < 1e-6          # ...AND K=2
    assert float(acts[3][h, 0]) > 0.5                # completes at K=3
    assert abs(float(acts[3][h, 0])) > abs(float(acts[2][h, 0]))   # monotone
    # Contrast (the docstring's "graded partial evidence"): with ALL words
    # sourced the head's DIRECT word edge lights it at K=1 already.
    a_full = torch.zeros(n_snap, 1)
    for w in words:
        a_full[_row0(cs, w), 0] = 0.75
    object.__setattr__(cs, "_symbolic_order", 1)
    _c, a1 = cs.cs_forward_content(a_full, what)
    object.__setattr__(cs, "_symbolic_order", 3)
    assert float(a1.detach()[h, 0]) > 0.5            # graded, not gated


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

def test_cycle_flagged_by_wave_qe_not_settling():
    """A 2-cycle is legal (only self-edges raise) and surfaces as a
    NON-SETTLED wave: qe[-1] stays decisively above tol. REPORTED, never
    raised -- cycles are documented fact (Alec 2026-07-03): observability,
    not enforcement."""
    cs = _cs(nS=8, order=4)                          # n_snap=4; K=4 steps
    rA, rB = 4, 5                                    # two pool rows
    cs.add_concept_edge(rA, rB, weight=3.0)          # decisively above the
    cs.add_concept_edge(rB, rA, weight=3.0)          # ~1.0 sustain threshold
    # WEAK seed (0.5 * 0.4): the cycle is still spinning up at step 4, so the
    # settle residual is macroscopic (a saturated cycle would sit at its
    # fixed point and read settled).
    cs.add_concept_edge(rA, 0, weight=0.5)
    a_0 = torch.zeros(4, 1)
    a_0[0, 0] = 0.4
    what = torch.randn(8, _D)
    cs.cs_forward_content(a_0, what)
    qe_cycle = cs._cs_wave_qe.clone()
    assert qe_cycle.shape == (4,)
    assert float(qe_cycle[-1]) > 1e-3                # ~6e-2: not settled
    # Causal contrast: zero the cycle legs -> the SAME space settles exactly
    # (the seed chain is feed-forward, constant from step 2).
    _set_edge_value(cs, rA, rB, 0.0)
    _set_edge_value(cs, rB, rA, 0.0)
    cs.cs_forward_content(a_0, what)
    qe_flat = cs._cs_wave_qe
    assert float(qe_flat[-1]) < 1e-6
    assert float(qe_flat[-1]) < float(qe_cycle[-1])  # the cycle is the cause


# -- 5/6/7. Kripke groundedness ---------------------------------------------------

def test_probe_informative_on_bias_bounded_vine():
    """Payoff of the probe's bias mask (Alec 2026-07-03): an UNMODIFIED
    production vine (EVERYTHING-bias edges intact) is probe-informative --
    the axiom pole neither lights run 1 nor sustains run 2."""
    cs = _cs(nS=64, order=3)
    words, head = _mint_vine(cs, 4)
    rows = [_rowp(cs, c) for c in _vine_links(cs, head, words)]
    n_snap = cs._order_caps()[0]
    a_0 = torch.zeros(n_snap, 1)
    for w in words:
        a_0[_row0(cs, w), 0] = 0.75
    g, u = cs.cs_groundedness_probe(a_0)
    for r in rows:
        assert bool(g[r])                            # sourced through the words
        assert not bool(u[r])                        # drains once released


def test_groundedness_probe_separates_vine_from_free_loop():
    cs = _cs(nS=64, order=3)                         # probe default k = 6
    words, head = _mint_vine(cs, 4)
    # Vine bias edges left INTACT: the probe masks the axiom pole itself.
    rows = [_rowp(cs, c) for c in _vine_links(cs, head, words)]
    rA, rB, seed = 50, 51, 30                        # pool rows past the mints; free snap row
    cs.add_concept_edge(rA, rB, weight=3.0)          # self-sustaining loop
    cs.add_concept_edge(rB, rA, weight=3.0)
    cs.add_concept_edge(rA, seed, weight=2.0)        # sourced: run 1 lights it
    n_snap = cs._order_caps()[0]
    a_0 = torch.zeros(n_snap, 1)
    for w in words:
        a_0[_row0(cs, w), 0] = 0.75
    a_0[seed, 0] = 0.8
    g, u = cs.cs_groundedness_probe(a_0)
    for r in rows:                                   # every link, head included:
        assert bool(g[r])                            # traces to the source...
        assert not bool(u[r])                        # ...and drains on release
    # A SOURCED loop is legitimately grounded AND ungrounded: assert only the
    # ungrounded half (persistence after release).
    assert bool(u[rA]) and bool(u[rB])


def test_weak_loop_decays_strong_loop_persists():
    """Temporal membership: a weak loop is an ECHO (fades once the source
    releases); the SAME rows rewritten strong are a reverberating assembly
    (membership persists without the source)."""
    cs = _cs(nS=16, order=3)                         # default k = 6 released steps
    rA, rB = 8, 9
    # Loop gain 0.05 << the ~1.0 sustain threshold: decays within k=6.
    cs.add_concept_edge(rA, rB, weight=0.05)
    cs.add_concept_edge(rB, rA, weight=0.05)
    cs.add_concept_edge(rA, 0, weight=2.0)           # seed off snap row 0
    a_0 = torch.zeros(8, 1)
    a_0[0, 0] = 0.8
    g, u = cs.cs_groundedness_probe(a_0)
    assert bool(g[rA]) and bool(g[rB])               # lit while sourced...
    assert not bool(u[rA]) and not bool(u[rB])       # ...gone on release
    _set_edge_value(cs, rA, rB, 3.0)                 # rewrite the SAME loop
    _set_edge_value(cs, rB, rA, 3.0)                 # decisively strong
    _g2, u2 = cs.cs_groundedness_probe(a_0)
    assert bool(u2[rA]) and bool(u2[rB])             # now it reverberates


def test_probe_inactive_returns_none():
    a_0 = torch.rand(8, 1)
    assert _cs(nS=16, order=3, serial=True).cs_groundedness_probe(a_0) is None
    assert _cs(nS=16, order=0).cs_groundedness_probe(a_0) is None
    assert _cs(nS=16, order=3).cs_groundedness_probe(None) is None
