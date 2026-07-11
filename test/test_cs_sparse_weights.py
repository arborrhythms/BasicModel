"""The shared untyped square concept store via ConceptualAttentionLayer.

A concept is a stored ConceptDim atom. Rows split into per-order TAPER
blocks (dual-towers rev 2): the order-0 snap block [0, caps[0]) holds
codebook rows read by SIGNED normalized-sum presence (no edges); each
order-k block holds compositions on ONE shared untyped [S+1 x S] store
(S = sum(caps)) whose bias is the store's trailing column S (callers pass
col == nVectors). Direction/order live in the record store and the
nesting, never in typed columns.
(doc/plans/2026-07-10-conceptual-wave-ff-pyramid-design.md)
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
    assert _p is None                                # percept family RETIRED
    assert s1 is cs._sparse_families(2)[1]           # ONE shared store
    assert (s1.nOutput, s1.nInput) == (14, 15)       # [S+1 inputs x S outputs]
    assert s1.roles is None                          # untyped: no role blocks
    cs.add_concept_edge(8, 2)                        # order-1 row <- snap col
    cs.add_concept_edge(8, 16)                       # bias: col nVectors -> S
    assert (16, 1.0) in cs.concept_weights(8)        # read back as nVectors
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
    """The order-0 SNAP readout: normalized-sum presence of the settled
    field against the ORDER-0 codebook block -- slot-mean projection onto the
    unit atom direction in hypercube-diagonal units. SIGNED in (-1, 1)
    (dual-towers rev 2: the nonneg clamp annihilated the mean-negative
    epoch-1 readout), input-dependent, and (per Alec) MAGNITUDE-SENSITIVE:
    objects in the unit hypercube differentiate by magnitude, so events are
    NOT unit-normalized (no cosine). Raw dot-SUM presences would saturate
    tanh to an input-blind constant (the sO=1 mean-collapse root cause)."""
    cs = _cs(nS=16, order=1)                           # taper caps (8, 4)
    torch.manual_seed(0)
    D_dict = int(cs.similarity_codebook.getW().shape[-1])
    e1 = torch.rand(2, 3, D_dict)                      # unit-hypercube events
    e2 = torch.rand(2, 3, D_dict)
    p1 = cs.cs_snap_order0(e1)
    p2 = cs.cs_snap_order0(e2)
    assert p1.shape == (8, 2)                          # [caps[0], B]
    assert p1.min() > -1.0 and p1.max() < 1.0          # SIGNED tanh range
    assert float((p1.abs() > 0.99).float().mean()) == 0.0   # never saturates
    assert not torch.allclose(p1, p2)                  # input-dependent
    # Magnitude carries information: a half-magnitude object is LESS present.
    p_half = cs.cs_snap_order0(0.5 * e1)
    mask = p1 > 1e-6
    assert torch.all(p_half[mask] < p1[mask])


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
    cs = _cs(nS=64, order=3)                           # taper (32, 16, 8, 4)
    what = torch.randn(64, _D)
    # the order-1 block occupies rows [32:48); pick its first 2 active
    n_o = 16
    a = torch.zeros(n_o, 2)                            # [n_concepts_o, B]
    a[0, 0] = 2.0
    a[1, 1] = -1.0
    code = cs.cs_decode(1, a, what)
    assert code.shape == (2, n_o, _D)
    # batch 0, concept 0 = 2.0 * what[32]
    assert torch.allclose(code[0, 0], 2.0 * what[32], atol=1e-5)
    # batch 1, concept 1 = -1.0 * what[33]
    assert torch.allclose(code[1, 1], -1.0 * what[33], atol=1e-5)
    # inactive concept -> zero
    assert torch.allclose(code[0, 5], torch.zeros(_D), atol=1e-6)


# -- forward-content assembly (dual-towers rev 2 sigma-pyramid) -----------------
# a = a_0 padded to N; each rung k is ONE feedforward hop tanh(W [a|1])
# gathered to the ALLOCATED order-k rows, top-caps[k] winners scattered in;
# no source re-injection, no iteration.

def test_forward_content_shape_and_stacking():
    """dual-towers rev 2: order-0 rows equal the padded a_0 (1:1, no extra
    tanh); an allocated+selected rung row reads one feedforward hop."""
    cs = _cs(nS=16, order=1)                           # taper (8, 4), K=1
    B = 2
    a_0 = torch.rand(8, B).tanh()                      # snap presences
    what = torch.randn(16, _D)
    r = _mint_row(cs, 1, 101)                          # first order-1 row: 8
    cs.add_concept_edge(r, 0, weight=2.0)              # fires off a_0[0]
    content, a = cs.cs_forward_content(a_0, what)
    assert content.shape == (B, 16, _D)                # stacked [B, N, CDim]
    assert a.shape == (16, B)                          # full-inventory acts
    assert torch.equal(a[:8], a_0)                     # order-0 passes 1:1
    assert torch.allclose(a[r], torch.tanh(2.0 * a_0[0]), atol=1e-5)
    assert torch.equal(a[r + 1:], torch.zeros(16 - r - 1, B))   # unallocated


def test_forward_content_legs_and_bias_sum_pre_tanh():
    """dual-towers rev 2: a selected rung row is ONE feedforward hop
    tanh(sum_c w_c * a_0[c] + bias edge); expectation computed from
    concept_weights(r) -- the bias col is S in store coordinates."""
    cs = _cs(nS=16, order=1)                           # taper (8, 4): S = 12
    B = 1
    a_0 = torch.zeros(8, B)
    a_0[3, 0] = 0.5
    a_0[5, 0] = 0.25
    what = torch.randn(16, _D)
    r = _mint_row(cs, 1, 201)
    cs.add_concept_edge(r, 3, weight=3.0)              # constituent leg 1
    cs.add_concept_edge(r, 5, weight=-0.5)             # constituent leg 2
    cs.add_concept_edge(r, 16, weight=0.25)            # bias: col nVectors -> S
    _content, a = cs.cs_forward_content(a_0, what)
    nV = int(cs.nVectors)
    acc = sum(w * (1.0 if c == nV else float(a_0[c, 0]))
              for c, w in cs.concept_weights(r))       # 3*0.5 - 0.5*0.25 + 0.25
    want = torch.tanh(torch.tensor(acc))
    assert abs(float(a[r, 0]) - float(want)) < 1e-5


def test_forward_content_activation_is_bounded_tanh():
    cs = _cs(nS=16, order=2)                           # taper (8, 4, 2): 2 rungs
    r1 = _mint_row(cs, 1, 101)                         # order-1 row: 8
    r2 = _mint_row(cs, 2, 102)                         # order-2 row: 12
    cs.add_concept_edge(r1, 0, weight=3.0)             # big weight saturates
    cs.add_concept_edge(r2, r1, weight=4.0)            # second-rung pile-on
    a_0 = torch.tanh(torch.full((8, 2), 5.0))
    what = torch.randn(16, _D)
    content, a = cs.cs_forward_content(a_0, what)
    assert a.min() >= -1.0 and a.max() <= 1.0          # tanh-bounded
    assert 0.9 < float(a[r1, 0]) <= 1.0
    assert 0.9 < float(a[r2, 0]) <= 1.0


def test_forward_content_empty_store_passes_a0():
    """dual-towers rev 2: empty store -> a IS the padded a_0 EXACTLY
    (order-0 passes 1:1, no extra tanh; higher rows stay zero)."""
    cs = _cs(nS=16, order=1)                           # no edges anywhere
    a_0 = torch.rand(8, 2)
    content, a = cs.cs_forward_content(a_0, torch.randn(16, _D))
    assert torch.equal(a[:8], a_0)                     # exact pass-through
    assert torch.equal(a[8:], torch.zeros(8, 2))       # higher rows zero


# RETIRED (dual-towers rev 2): test_wave_depth2_vine_completes_at_k2 -- the
# K-iteration depth schedule was temporal semantics; structural one-pass
# completion is pinned in test_iterated_symbolic_wave.py.


def test_level_stats_are_report_only():
    """dual-towers rev 2: _cs_wave_qe is RETIRED (None -- no settle statistic
    without iteration); the per-level pyramid stats are report-only."""
    cs = _cs(nS=16, order=2)                           # taper (8, 4, 2)
    r1 = _mint_row(cs, 1, 101)
    cs.add_concept_edge(r1, 0, weight=1.0)
    a_0 = torch.rand(8, 2, requires_grad=True)
    cs.cs_forward_content(a_0, torch.randn(16, _D))
    assert cs._cs_wave_qe is None                      # settle stat retired
    acts = cs._cs_level_acts
    assert isinstance(acts, list) and len(acts) == 3   # [max|a_0|, rung 1, 2]
    assert all(type(v) is float for v in acts)         # plain floats, no grad
    rows = cs._cs_level_rows
    assert isinstance(rows, list) and len(rows) == 2   # order-0 + visited rung
    for r in rows:
        assert r.dtype == torch.long and not r.requires_grad
    assert rows[0].shape == (8, 2)                     # order-0 arange x B
    assert rows[1].shape == (1, 2)                     # rung-1 winner rows
    assert int(rows[1][0, 0]) == r1


def test_forward_content_kernels_agree():
    cs = _cs(nS=16, order=1)
    cs.add_concept_edge(10, 0, weight=-0.4)
    cs.add_concept_edge(10, 1, weight=0.7)
    a_0 = torch.rand(8, 2).tanh()
    d = torch.randn(16, _D)
    c1, a1 = cs.cs_forward_content(a_0, d)
    Spaces._concept_alloc_of(cs).layer(0).kernel = "spmm"   # flip kernel
    c2, a2 = cs.cs_forward_content(a_0, d)
    assert torch.allclose(c1, c2, atol=1e-6)
    assert torch.allclose(a1, a2, atol=1e-6)


def test_forward_content_positive_atoms_and_signed_activation():
    """dual-towers rev 2: atoms stay softplus-positive; SIGNED activations
    survive both for rung winners and the signed order-0 pass-through."""
    cs = _cs(nS=16, order=1)
    B = 1
    a_0 = torch.rand(8, B) + 0.1                       # snap presences > 0...
    a_0[1, 0] = -0.3                                   # ...plus a SIGNED entry
    what = torch.randn(16, _D)                         # has negative entries
    r = _mint_row(cs, 1, 101)                          # selected (only alloc)
    cs.add_concept_edge(r, 0, weight=-1.0)             # NEGATIVE weight
    content, a = cs.cs_forward_content(a_0, what)
    # the winner's activation is negative (neg weight * pos presence)
    assert a[r, 0] < 0
    # the atom used is softplus(what[r]) -> strictly positive
    import torch.nn.functional as F
    atom = F.softplus(what[r])
    assert (atom > 0).all()
    # code row = a * positive_atom -> points opposite the atom (anti-present)
    assert torch.allclose(content[0, r], a[r, 0] * atom, atol=1e-5)
    # the negative order-0 entry passes 1:1: content sign follows activation
    assert torch.allclose(content[0, 1], a_0[1, 0] * F.softplus(what[1]),
                          atol=1e-5)
    assert (content[0, 1] < 0).all()


def test_forward_content_differentiable():
    """dual-towers rev 2: grads reach the store values ONLY through
    allocated+selected rungs; a_0 and the dictionary always."""
    cs = _cs(nS=16, order=2)                           # taper (8, 4, 2)
    r1 = _mint_row(cs, 1, 101)
    r2 = _mint_row(cs, 2, 102)
    cs.add_concept_edge(r1, 0, weight=0.5)
    cs.add_concept_edge(r2, r1, weight=0.5)
    a_0 = torch.rand(8, 2, requires_grad=True)
    what = torch.randn(16, _D, requires_grad=True)
    content, _ = cs.cs_forward_content(a_0, what)
    content.sum().backward()
    assert what.grad is not None and what.grad.abs().sum() > 0
    assert a_0.grad is not None and a_0.grad.abs().sum() > 0
    ly = Spaces._concept_alloc_of(cs).layer(0)
    assert ly.values.grad is not None and ly.values.grad.abs().sum() > 0
