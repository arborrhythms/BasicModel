"""Concept rows compose paired evidence over the bounded taper span S.

Each part matrix reads 2(S+1) sources: S positive poles and EVERYTHING,
then S negative poles and its absent counterpart. Codes are stored once.
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
import pytest

import Spaces
from test_basicmodel import _populate_test_config

_D = 8


def _cs(nS=64, order=3):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    cs = Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])
    object.__setattr__(cs, "_symbolic_order", order)
    object.__setattr__(cs, "_serial", False)
    return cs


def _evidence(positive, negative=None):
    """Fixture with explicit evidence; an omitted pole means unknown."""
    if negative is None:
        negative = torch.zeros_like(positive)
    return torch.stack((positive, negative), -1).unsqueeze(2)


def _mint_row(cs, order, cid):
    """First-seen allocated row of cid in its order block (dual-towers rev 2:
    only ALLOCATED rows enter a rung)."""
    return cs._csw_concept_row(order, cid)


def test_order_slice_taper_contract():
    """dual-towers rev 2: per-order TAPER caps [base, base>>1, ..], base
    halved until sum(caps) <= nVectors; contiguous cumulative blocks."""
    cs = _cs(nS=64, order=3)                         # base 64 -> 32
    assert cs._order_caps() == (32, 16, 8, 4)        # sum 60 <= 64
    assert cs.order_slice(0) == (0, 32)              # the snap block
    assert cs.order_slice(1) == (32, 48)
    assert cs.order_slice(2) == (48, 56)
    assert cs.order_slice(3) == (56, 60)
    assert cs.order_slice(9) == cs.order_slice(3)    # clamped to the taper
    cs2 = _cs(nS=16, order=1)                        # base 16 -> 8
    assert cs2._order_caps() == (8, 4)
    assert cs2.order_slice(0) == (0, 8)
    assert cs2.order_slice(1) == (8, 12)


# -- the shared untyped square store (v3) --------------------------------------

def test_add_concept_edge_dedup_and_query():
    cs = _cs()                                       # n_snap = 32
    r1 = cs.add_concept_edge(33, 5, weight=1.5)
    r2 = cs.add_concept_edge(33, 5, weight=9.0)      # repeat -> no-op
    assert r1 == r2
    cs.add_concept_edge(33, 40, weight=0.5)
    got = cs.concept_weights(33)
    assert (5, 1.5) in got and (40, 0.5) in got
    assert cs.concept_weights(34) == []              # different row


def test_shared_untyped_square_store():
    """dual-towers rev 2: every order shares ONE untyped [S+1 x S]
    ConceptualAttentionLayer, S = sum(taper caps) -- no role blocks; the bias is
    the store's trailing column S (callers pass col == nVectors); self-edges
    raise (the Quine atom); the learnable values register EXACTLY ONCE."""
    cs = _cs(nS=16, order=2)                         # taper (8, 4, 2): S = 14
    _p, s1 = cs._sparse_families(1)
    assert _p is s1.conjunctive                       # same concepts, conjunctive parts
    assert s1 is cs._sparse_families(2)[1]           # ONE shared store
    assert (s1.nOutput, s1.nInput) == (14, 30)       # [S+1 inputs x S outputs]
    assert s1.roles is None                          # untyped: no role blocks
    cs.add_concept_edge(8, 2)                        # order-1 row <- snap col
    cs.add_concept_edge(8, 16)                       # bias: col nVectors -> S
    assert (16, 0.0) in cs.concept_weights(8)        # read back as nVectors
    try:
        cs.add_concept_edge(9, 9)
        assert False, "self-edge must raise (the Quine atom)"
    except ValueError:
        pass
    assert s1.nnz == 2
    # The learnable values surface through getParameters ONCE (dedup: the
    # same layer is registered under several family keys).
    params = cs.getParameters()
    assert len([p for p in params if p is s1.values]) == 1


def test_any_edge_on_snap_row_fails_loud():
    cs = _cs(nS=16, order=1)                         # n_snap = 8
    for row in (0, 7):
        try:
            cs.add_concept_edge(row, 9)
            assert False, "snap-row edges must raise (codebook rows, not edges)"
        except ValueError:
            pass
    _p, s0 = cs._sparse_families(0)
    assert s0.nnz == 0


def test_family_values_grow_tail_preserving():
    cs = _cs()                                       # n_snap = 32
    cs.add_concept_edge(32, 0)
    cs.add_concept_edge(32, 1)
    _p, s = cs._sparse_families(0)
    with torch.no_grad():
        s.values.copy_(torch.tensor([3.0, 4.0]))
    cs.add_concept_edge(33, 2, weight=9.0)           # grow
    assert torch.allclose(s.values.detach(), torch.tensor([3.0, 4.0, 9.0]))


# -- source activation (still the snap's readout) + dictionary decoder --------

def test_source_code_activation_matches_dotproduct():
    cs = _cs()
    B, N, V = 2, 3, 5
    event = torch.randn(B, N, _D)
    W = torch.randn(V, _D)
    act = cs.source_code_activation(event, W, nonneg=False)
    assert act.shape == (V, B)
    ref = torch.einsum('bnd,vd->vb', event, W)        # sum over slots of <e,W>
    assert torch.allclose(act, ref, atol=1e-5)


def test_source_code_activation_is_nonneg_presence_by_default():
    cs = _cs()
    event = torch.randn(2, 3, _D)
    W = torch.randn(5, _D)
    act = cs.source_code_activation(event, W)          # default nonneg=True
    assert (act >= 0).all()                            # features are presences
    ref = torch.einsum('bnd,vd->vb', event, W).clamp(min=0.0)
    assert torch.allclose(act, ref, atol=1e-5)


def test_snap_order0_is_input_dependent_not_saturated():
    """Cube-valued events use diagonal units; an extent unions its slots.

    Fixed orthogonal directions test magnitude sensitivity without selecting
    a random draw that avoids saturation.
    """
    cs = _cs(nS=16, order=1)
    W = cs.similarity_codebook.getW()
    with torch.no_grad():
        W.zero_()
        W[:8, :8] = torch.eye(8)
    e1 = torch.full((2, 3, W.shape[-1]), .4)
    e2 = e1.clone()
    e2[..., 0] = -.2
    p1 = cs.cs_snap_order0(e1, chart='cube')
    p2 = cs.cs_snap_order0(e2, chart='cube')
    assert p1.shape == (8, 2, 1, 2)
    assert p1.min() >= 0. and p1.max() < .99
    assert not torch.allclose(p1, p2)
    p_half = cs.cs_snap_order0(.5 * e1, chart='cube')
    assert torch.all(p_half[p1 > 0] < p1[p1 > 0])


def test_snap_order0_ema_traces_winning_rows_training_only():
    """With ``ema=True`` the winning order-0 rows EMA toward their slot
    contents (no_grad identity/position trace); eval mode never writes."""
    cs = _cs(nS=16, order=1)
    torch.manual_seed(1)
    W = cs.similarity_codebook.getW()
    D_dict = int(W.shape[-1])
    ev = torch.rand(1, 2, D_dict)
    before = W.detach().clone()
    cs.eval()
    cs.cs_snap_order0(ev, ema=True)                    # eval -> no write
    assert torch.equal(W.detach(), before)
    cs.train()
    cs.cs_snap_order0(ev, ema=True)
    after = W.detach()
    start, end = cs.order_slice(0)
    assert not torch.equal(after[start:end], before[start:end])   # traced
    assert torch.equal(after[end:], before[end:])      # higher orders untouched


def test_snap_order0_is_differentiable_in_event():
    cs = _cs(nS=16, order=1)
    D_dict = int(cs.similarity_codebook.getW().shape[-1])
    ev = torch.rand(2, 3, D_dict, requires_grad=True)
    a0 = cs.cs_snap_order0(ev)
    a0.sum().backward()
    assert ev.grad is not None and torch.any(ev.grad != 0)


def test_cs_decode_scales_dictionary_atoms():
    cs = _cs()
    what = torch.randn(64, _D)
    a = torch.zeros(16, 2, 1, 2)
    a[0, 0, 0, 0] = .8
    a[1, 1, 0, 1] = .7
    code = cs.cs_decode(1, a, what)
    assert code.shape == (2, 32, _D)
    torch.testing.assert_close(code[0, 0], .8 * what[32])
    torch.testing.assert_close(code[1, 3], -.7 * what[33])
    assert code[0, 10].count_nonzero() == 0


def test_forward_content_shape_and_stacking():
    cs = _cs(nS=16, order=1)
    a0 = _evidence(torch.rand(8, 2))
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, 2.)
    content, a = cs.cs_forward_content(a0, torch.randn(16, _D))
    assert content.shape == (2, 24, _D)
    assert a.shape == (12, 2, 1, 2)
    torch.testing.assert_close(a[:8], a0)
    torch.testing.assert_close(a[row, :, 0, 0], 1 - (1 - a0[0, :, 0, 0]).square())
    assert a[row + 1:].count_nonzero() == 0


def test_forward_content_literal_poles_and_standing_bias():
    cs = _cs(nS=16, order=1)
    a0 = _evidence(torch.zeros(8, 1))
    a0[3, 0, 0, 0] = .5
    a0[5, 0, 0, 1] = .25
    row = _mint_row(cs, 1, 201)
    cs.add_concept_edge(row, 3, 3.)
    cs.add_concept_edge(row, 5, .5, negated=True)
    _, a = cs.cs_forward_content(a0, torch.randn(16, _D))
    assert float(a[row, 0, 0, 0]) == pytest.approx(1 - .5**3 * .75**.5)
    cs.add_concept_edge(row, 16, .25)
    _, a = cs.cs_forward_content(a0, torch.randn(16, _D))
    torch.testing.assert_close(a[row, 0, 0], torch.tensor([1., 0.]))


def test_forward_content_activation_is_bounded():
    cs = _cs(nS=16, order=2)
    r1, r2 = _mint_row(cs, 1, 101), _mint_row(cs, 2, 102)
    cs.add_concept_edge(r1, 0, 3.)
    cs.add_concept_edge(r2, r1, 4.)
    _, a = cs.cs_forward_content(_evidence(torch.full((8, 2), .99)), torch.randn(16, _D))
    assert a.min() >= 0 and a.max() <= 1
    assert a[r1, 0, 0, 0] > .9 and a[r2, 0, 0, 0] > .9


def test_forward_content_empty_store_passes_a0():
    cs = _cs(nS=16, order=1)
    a0 = _evidence(torch.rand(8, 2), torch.rand(8, 2))
    _, a = cs.cs_forward_content(a0, torch.randn(16, _D))
    assert torch.equal(a[:8], a0)
    assert a[8:].count_nonzero() == 0


def test_level_stats_are_report_only():
    cs = _cs(nS=16, order=2)
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, 1.)
    cs.cs_forward_content(_evidence(torch.rand(8, 2, requires_grad=True)), torch.randn(16, _D))
    assert cs._cs_wave_qe is None
    assert len(cs._cs_level_acts) == 3
    assert all(type(v) is float for v in cs._cs_level_acts)
    rows = cs._cs_level_rows
    assert len(rows) == 2
    assert all(r.dtype == torch.long and not r.requires_grad for r in rows)
    assert rows[0].shape == (8, 2)
    assert rows[1].shape == (1, 2)
    assert int(rows[1][0, 0]) == row


def test_forward_content_kernels_agree():
    cs = _cs(nS=16, order=1)
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, .4, negated=True)
    cs.add_concept_edge(row, 1, .7)
    a0 = _evidence(torch.rand(8, 2), torch.rand(8, 2))
    dictionary = torch.randn(16, _D)
    c1, a1 = cs.cs_forward_content(a0, dictionary)
    Spaces._concept_alloc_of(cs).layer().kernel = 'spmm'
    c2, a2 = cs.cs_forward_content(a0, dictionary)
    torch.testing.assert_close(c1, c2)
    torch.testing.assert_close(a1, a2)


def test_forward_content_keeps_two_symbol_rows_per_code():
    cs = _cs(nS=16, order=1)
    a0 = _evidence(torch.rand(8, 1), torch.rand(8, 1))
    what = torch.nn.functional.normalize(torch.randn(16, _D), dim=-1)
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, 1., negated=True)
    content, a = cs.cs_forward_content(a0, what)
    torch.testing.assert_close(a[row, 0, 0], a0[0, 0, 0].flip(0))
    torch.testing.assert_close(content[0, 2 * row], a[row, 0, 0, 0] * what[row])
    torch.testing.assert_close(content[0, 2 * row + 1], -a[row, 0, 0, 1] * what[row])


def test_forward_content_differentiable():
    cs = _cs(nS=16, order=2)
    r1, r2 = _mint_row(cs, 1, 101), _mint_row(cs, 2, 102)
    cs.add_concept_edge(r1, 0, .5)
    cs.add_concept_edge(r2, r1, .5)
    positive = torch.rand(8, 2, requires_grad=True)
    what = torch.randn(16, _D, requires_grad=True)
    content, _ = cs.cs_forward_content(_evidence(positive), what)
    content.sum().backward()
    assert what.grad is not None and what.grad.abs().sum() > 0
    assert positive.grad is not None and positive.grad.abs().sum() > 0
    ly = Spaces._concept_alloc_of(cs).layer()
    assert ly.values.grad is not None and ly.values.grad.abs().sum() > 0
