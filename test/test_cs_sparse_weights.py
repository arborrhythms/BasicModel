"""Ramsified per-order sparse weight tables (symbolic-only, P2) via SparseLayer.

A concept is a stored ConceptDim atom. Order 0 is the SNAP: concepts are
codebook rows read by normalized-sum presence (no edges). Order k >= 1
composes LOWER-ORDER SYMBOL ACTIVATIONS ONLY through role-split columns
[whole | part | bias] on ONE symbol-family SparseLayer per order, summed
pre-tanh so every activation is signed and bounded in (-1, 1). Capacity is
dyadic by order, stacked in the single inventory.
(doc/plans/2026-07-02-two-phase-loops-sparse-relation.md P2)
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


# -- dyadic capacity allocation -----------------------------------------------

def test_order_capacities_dyadic_sums_to_total():
    assert Spaces.ConceptualSpace.order_capacities(8, 3) == [4, 2, 1, 1]
    assert Spaces.ConceptualSpace.order_capacities(16, 3) == [8, 4, 2, 2]
    assert Spaces.ConceptualSpace.order_capacities(64, 3) == [32, 16, 8, 8]
    for N, S in [(8, 3), (16, 3), (64, 3), (100, 2), (12, 1)]:
        caps = Spaces.ConceptualSpace.order_capacities(N, S)
        assert len(caps) == S + 1
        assert sum(caps) == N                       # exact partition
        assert all(c >= 1 for c in caps)


def test_order_capacities_order0_is_all():
    assert Spaces.ConceptualSpace.order_capacities(64, 0) == [64]


def test_order_slice_contiguous_stack():
    cs = _cs(nS=64, order=3)                         # caps [32,16,8,8]
    assert cs.order_slice(0) == (0, 32)
    assert cs.order_slice(1) == (32, 48)
    assert cs.order_slice(2) == (48, 56)
    assert cs.order_slice(3) == (56, 64)


# -- per-order role-split symbol store (P2: symbolic-only) ---------------------

def test_add_concept_edge_dedup_and_query():
    cs = _cs()
    r1 = cs.add_concept_edge(1, 2, "part", 5, weight=1.5)
    r2 = cs.add_concept_edge(1, 2, "part", 5, weight=9.0)   # repeat -> no-op
    assert r1 == r2
    cs.add_concept_edge(1, 2, "whole", 1, weight=0.5)
    got = cs.concept_weights(1, 2)
    assert (("part", 5), 1.5) in got and (("whole", 1), 0.5) in got
    assert cs.concept_weights(1, 3) == []
    assert cs.concept_weights(2, 2) == []            # different order


def test_symbol_family_role_split_columns():
    cs = _cs(nS=16, order=2)                         # caps [8, 4, 4]
    _p, s1 = cs._sparse_families(1)
    assert _p is None                                # percept family RETIRED
    # order 1 reads a_0 (8 rows) through whole|part|bias role blocks.
    assert s1.role_slice("whole") == (0, 8)
    assert s1.role_slice("part") == (8, 16)
    assert s1.role_slice("bias") == (16, 17)
    cs.add_concept_edge(1, 0, "whole", 2)
    cs.add_concept_edge(1, 0, "part", 3)
    cs.add_concept_edge(1, 0, "bias", 0)
    assert s1.nnz == 3
    # The learnable values surface through getParameters (optimizer pickup).
    ids = {id(t) for t in cs.getParameters()}
    assert id(s1.values) in ids


def test_any_edge_at_order0_fails_loud():
    cs = _cs(nS=16, order=1)
    for role, col in (("part", 0), ("whole", 0), ("bias", 0)):
        try:
            cs.add_concept_edge(0, 0, role, col)
            assert False, "order-0 edges must raise (snap rows, not edges)"
        except IndexError:
            pass
    _p, s0 = cs._sparse_families(0)
    assert s0.nnz == 0


def test_family_values_grow_tail_preserving():
    cs = _cs()
    cs.add_concept_edge(1, 0, "part", 0)
    cs.add_concept_edge(1, 0, "part", 1)
    _p, s = cs._sparse_families(1)
    with torch.no_grad():
        s.values.copy_(torch.tensor([3.0, 4.0]))
    cs.add_concept_edge(1, 0, "whole", 2, weight=9.0)        # grow
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
    cs = _cs(nS=64, order=3)                           # caps [32,16,8,8]
    what = torch.randn(64, _D)
    # order 1 occupies rows [32:48); pick its first 2 concepts active
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


# -- ramsified per-order forward-content assembly (v2: from the snap) ----------

def test_cs_source_layout_ss_blocks_only():
    cs = _cs(nS=64, order=3)                            # caps [32,16,8,8]
    off, total = cs.cs_source_layout(0)
    assert off == {} and total == 0                     # order 0: no sources
    off, total = cs.cs_source_layout(2)
    # order 2 reads SS_0 (cap 32) then SS_1 (cap 16), role-block-local.
    assert off == {0: 0, 1: 32} and total == 48


def test_forward_content_shape_and_stacking():
    cs = _cs(nS=16, order=1)                            # caps [8, 8]
    B = 2
    a_0 = torch.rand(8, B).tanh()                      # snap presences
    what = torch.randn(16, _D)
    # order 1: concept 0 fires off a_0[0] through the whole-role block.
    cs.add_concept_edge(1, 0, "whole", 0, weight=2.0)
    content, a = cs.cs_forward_content(a_0, what)
    assert content.shape == (B, 16, _D)                # stacked [B, N, CDim]
    assert len(a) == 2 and a[0].shape == (8, B) and a[1].shape == (8, B)
    assert torch.equal(a[0], a_0)                      # order 0 IS the snap
    assert torch.allclose(a[1][0], torch.tanh(2.0 * a_0[0]), atol=1e-5)


def test_forward_content_whole_part_bias_sum_pre_tanh():
    cs = _cs(nS=16, order=1)
    B = 1
    a_0 = torch.zeros(8, B)
    a_0[3, 0] = 0.5
    what = torch.randn(16, _D)
    cs.add_concept_edge(1, 2, "whole", 3, weight=1.0)   # whole leg
    cs.add_concept_edge(1, 2, "part", 3, weight=2.0)    # part leg (same source)
    cs.add_concept_edge(1, 2, "bias", 0, weight=0.25)   # everything bias
    _content, a = cs.cs_forward_content(a_0, what)
    want = torch.tanh(torch.tensor(1.0 * 0.5 + 2.0 * 0.5 + 0.25))
    assert abs(float(a[1][2, 0]) - float(want)) < 1e-5


def test_forward_content_activation_is_bounded_tanh():
    cs = _cs(nS=16, order=1)
    cs.add_concept_edge(1, 0, "whole", 0, weight=3.0)   # big weight saturates
    a_0 = torch.tanh(torch.full((8, 2), 5.0))
    what = torch.randn(16, _D)
    content, a_list = cs.cs_forward_content(a_0, what)
    for a in a_list:
        assert a.min() >= -1.0 and a.max() <= 1.0      # tanh-bounded
    assert 0.9 < float(a_list[1][0, 0]) <= 1.0


def test_forward_content_empty_order_is_zero():
    cs = _cs(nS=16, order=1)                            # order 1 stays empty
    a_0 = torch.rand(8, 2)
    content, a = cs.cs_forward_content(a_0, torch.randn(16, _D))
    assert torch.equal(a[1], torch.zeros(8, 2))        # tanh(0) == 0


def test_forward_content_kernels_agree():
    cs = _cs(nS=16, order=1)
    cs.add_concept_edge(1, 2, "whole", 0, weight=-0.4)
    cs.add_concept_edge(1, 2, "part", 1, weight=0.7)
    a_0 = torch.rand(8, 2).tanh()
    d = torch.randn(16, _D)
    c1, a1 = cs.cs_forward_content(a_0, d)
    for (p, s) in cs._sparse_fam.values():             # flip kernels
        for ly in (p, s):
            if ly is not None:
                ly.kernel = "spmm"
    c2, a2 = cs.cs_forward_content(a_0, d)
    assert torch.allclose(c1, c2, atol=1e-6)


def test_forward_content_positive_atoms_and_signed_activation():
    cs = _cs(nS=16, order=1)
    B = 1
    a_0 = torch.rand(8, B)                             # snap presences >= 0
    what = torch.randn(16, _D)                         # has negative entries
    cs.add_concept_edge(1, 0, "whole", 0, weight=-1.0)  # NEGATIVE weight
    content, a = cs.cs_forward_content(a_0, what)
    # order-1 concept 0 activation is negative (neg weight * pos presence)
    assert a[1][0, 0] <= 0
    # the atom used is softplus(what[order-1 row 0]) -> strictly positive
    import torch.nn.functional as F
    start, _end = cs.order_slice(1)
    atom = F.softplus(what[start])
    assert (atom > 0).all()
    # code row = a_1 * positive_atom -> points opposite the atom (anti-present)
    assert torch.allclose(content[0, start], a[1][0, 0] * atom, atol=1e-5)


def test_forward_content_differentiable():
    cs = _cs(nS=16, order=1)
    cs.add_concept_edge(1, 0, "whole", 0, weight=0.5)
    a_0 = torch.rand(8, 2, requires_grad=True)
    what = torch.randn(16, _D, requires_grad=True)
    content, _ = cs.cs_forward_content(a_0, what)
    content.sum().backward()
    assert what.grad is not None and what.grad.abs().sum() > 0
    assert a_0.grad is not None and a_0.grad.abs().sum() > 0
    _p1, s1 = cs._sparse_families(1)
    assert s1.values.grad is not None and s1.values.grad.abs().sum() > 0
