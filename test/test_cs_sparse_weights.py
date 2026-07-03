"""The shared untyped square concept store (v3) via AttentionLayer.

A concept is a stored ConceptDim atom. The snap block [0, n_snap) holds
order-0 concepts as codebook rows read by normalized-sum presence (no
edges); the relation pool [n_snap, N) holds everything else on ONE shared
untyped [N x N+1] store whose trailing column N is the EVERYTHING bias.
Direction/order live in the record store and the nesting, never in typed
columns. (doc/plans/2026-07-02-iterated-symbolic-loop.md)
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


def test_order_slice_two_block_contract():
    cs = _cs(nS=64, order=3)                         # two-block [32 | 32]
    assert cs.order_slice(0) == (0, 32)              # the snap block
    assert cs.order_slice(1) == (32, 64)             # the relation pool
    assert cs.order_slice(2) == cs.order_slice(1)    # every order >= 1: pool
    assert cs.order_slice(3) == cs.order_slice(1)


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
    """v3: every order shares ONE untyped [N x N+1] AttentionLayer -- no
    role blocks; the bias is the trailing column N; self-edges raise (the
    Quine atom); the learnable values register EXACTLY ONCE."""
    cs = _cs(nS=16, order=2)                         # two-block [8 | 8]
    _p, s1 = cs._sparse_families(1)
    assert _p is None                                # percept family RETIRED
    assert s1 is cs._sparse_families(2)[1]           # ONE shared store
    assert (s1.nOutput, s1.nInput) == (16, 17)       # square + bias col
    assert s1.roles is None                          # untyped: no role blocks
    cs.add_concept_edge(8, 2)                        # pool row <- snap col
    cs.add_concept_edge(8, 16)                       # the bias col N
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
    """The order-0 SNAP readout (P2): normalized-sum presence of the settled
    field against the ORDER-0 codebook block -- slot-mean projection onto the
    unit atom direction in hypercube-diagonal units. Bounded, input-dependent,
    and (per Alec) MAGNITUDE-SENSITIVE: objects in the unit hypercube
    differentiate by magnitude, so events are NOT unit-normalized (no
    cosine). Raw dot-SUM presences would saturate tanh to an input-blind
    constant (the sO=1 mean-collapse root cause)."""
    cs = _cs(nS=16, order=1)                           # caps [8, 8]
    torch.manual_seed(0)
    D_dict = int(cs.similarity_codebook.getW().shape[-1])
    e1 = torch.rand(2, 3, D_dict)                      # unit-hypercube events
    e2 = torch.rand(2, 3, D_dict)
    p1 = cs.cs_snap_order0(e1)
    p2 = cs.cs_snap_order0(e2)
    assert p1.shape == (8, 2)                          # [caps[0], B]
    assert p1.min() >= 0.0 and p1.max() < 1.0
    assert float((p1 > 0.99).float().mean()) == 0.0    # never saturates
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
    cs = _cs(nS=64, order=3)                           # two-block [32 | 32]
    what = torch.randn(64, _D)
    # the relation pool occupies rows [32:64); pick its first 2 active
    n_o = 32
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


# -- forward-content assembly (v3 iterated wave) -------------------------------
# a^{i+1} = tanh(W [a^i | 1] + s), K = symbolicOrder steps; s is the snap
# presence padded to the inventory (snap rows carry no in-edges -> tanh(a_0)).

def test_forward_content_shape_and_stacking():
    cs = _cs(nS=16, order=1)                           # two-block [8 | 8], K=1
    B = 2
    a_0 = torch.rand(8, B).tanh()                      # snap presences
    what = torch.randn(16, _D)
    # pool concept row 8 fires off a_0[0] (snap col 0).
    cs.add_concept_edge(8, 0, weight=2.0)
    content, a = cs.cs_forward_content(a_0, what)
    assert content.shape == (B, 16, _D)                # stacked [B, N, CDim]
    assert a.shape == (16, B)                          # full-inventory acts
    assert torch.allclose(a[:8], torch.tanh(a_0))      # snap rows read tanh(s)
    assert torch.allclose(a[8], torch.tanh(2.0 * a_0[0]), atol=1e-5)


def test_forward_content_legs_and_bias_sum_pre_tanh():
    cs = _cs(nS=16, order=1)
    B = 1
    a_0 = torch.zeros(8, B)
    a_0[3, 0] = 0.5
    a_0[5, 0] = 0.25
    what = torch.randn(16, _D)
    cs.add_concept_edge(10, 3, weight=3.0)             # constituent leg 1
    cs.add_concept_edge(10, 5, weight=-0.5)            # constituent leg 2
    cs.add_concept_edge(10, 16, weight=0.25)           # everything bias (col N)
    _content, a = cs.cs_forward_content(a_0, what)
    # a^1[10] = tanh(w1*a0[3] + w2*a0[5] + w_bias*1 + s[10]), s[10] == 0.
    want = torch.tanh(torch.tensor(3.0 * 0.5 + (-0.5) * 0.25 + 0.25))
    assert abs(float(a[10, 0]) - float(want)) < 1e-5


def test_forward_content_activation_is_bounded_tanh():
    cs = _cs(nS=16, order=2)                           # K=2: iterate the wave
    cs.add_concept_edge(8, 0, weight=3.0)              # big weight saturates
    cs.add_concept_edge(9, 8, weight=4.0)              # second-hop pile-on
    a_0 = torch.tanh(torch.full((8, 2), 5.0))
    what = torch.randn(16, _D)
    content, a = cs.cs_forward_content(a_0, what)
    assert a.min() >= -1.0 and a.max() <= 1.0          # tanh-bounded
    assert 0.9 < float(a[8, 0]) <= 1.0


def test_forward_content_empty_store_is_tanh_source():
    cs = _cs(nS=16, order=1)                           # no edges anywhere
    a_0 = torch.rand(8, 2)
    content, a = cs.cs_forward_content(a_0, torch.randn(16, _D))
    assert torch.allclose(a[:8], torch.tanh(a_0))      # snap: tanh(0 + s)
    assert torch.equal(a[8:], torch.zeros(8, 2))       # pool: tanh(0) == 0


def test_wave_depth2_vine_completes_at_k2():
    """A depth-d vine completes at iteration d (tail links first): the
    head link reads the REST link's activation only once the wave has
    propagated one hop -- K=1 leaves it dark, K=2 lights it."""
    a_0 = torch.zeros(8, 1)
    a_0[0, 0] = 0.5
    what = torch.randn(16, _D)
    t = torch.tanh
    tail1 = t(torch.tensor(2.0 * 0.5))                 # a^1[9] = tanh(2 s[0])
    for order, want10 in ((1, torch.tensor(0.0)),      # K=1: head still dark
                          (2, t(2.0 * tail1))):        # K=2: head reads a^1[9]
        cs = _cs(nS=16, order=order)
        cs.add_concept_edge(9, 0, weight=2.0)          # tail: pool <- snap
        cs.add_concept_edge(10, 9, weight=2.0)         # head: pool <- pool
        _c, a = cs.cs_forward_content(a_0, what)
        # the tail rides the CURRENT source: a^K[9] = tanh(2 a^{K-1}[0]).
        src = torch.tensor(0.5) if order == 1 else t(torch.tensor(0.5))
        assert torch.allclose(a[9, 0], t(2.0 * src), atol=1e-5)
        assert torch.allclose(a[10, 0], want10, atol=1e-5)


def test_wave_qe_is_report_only():
    cs = _cs(nS=16, order=2)
    cs.add_concept_edge(8, 0, weight=1.0)
    a_0 = torch.rand(8, 2, requires_grad=True)
    cs.cs_forward_content(a_0, torch.randn(16, _D))
    qe = cs._cs_wave_qe
    assert qe.shape == (2,) and not qe.requires_grad   # K entries, no grad
    assert torch.isfinite(qe).all()


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
    cs = _cs(nS=16, order=1)
    B = 1
    a_0 = torch.rand(8, B) + 0.1                       # snap presences > 0
    what = torch.randn(16, _D)                         # has negative entries
    cs.add_concept_edge(8, 0, weight=-1.0)             # NEGATIVE weight
    content, a = cs.cs_forward_content(a_0, what)
    # pool row 8's activation is negative (neg weight * pos presence)
    assert a[8, 0] < 0
    # the atom used is softplus(what[8]) -> strictly positive
    import torch.nn.functional as F
    atom = F.softplus(what[8])
    assert (atom > 0).all()
    # code row = a * positive_atom -> points opposite the atom (anti-present)
    assert torch.allclose(content[0, 8], a[8, 0] * atom, atol=1e-5)


def test_forward_content_differentiable():
    cs = _cs(nS=16, order=2)                           # grads through K=2 steps
    cs.add_concept_edge(8, 0, weight=0.5)
    cs.add_concept_edge(9, 8, weight=0.5)
    a_0 = torch.rand(8, 2, requires_grad=True)
    what = torch.randn(16, _D, requires_grad=True)
    content, _ = cs.cs_forward_content(a_0, what)
    content.sum().backward()
    assert what.grad is not None and what.grad.abs().sum() > 0
    assert a_0.grad is not None and a_0.grad.abs().sum() > 0
    ly = Spaces._concept_alloc_of(cs).layer(0)
    assert ly.values.grad is not None and ly.values.grad.abs().sum() > 0
