"""Ramsified per-order sparse weight tables (PS/WS/SS -> CS), torch.sparse.

A concept is a stored ConceptDim atom; the per-order sparse weight matrix maps
source ACTIVATIONS (PS, WS, SS_0..SS_{k-1}) to concept activations via
torch.sparse.mm. Capacity is dyadic by order, stacked in the single inventory.
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


# -- per-order sparse weight store --------------------------------------------

def test_add_concept_weight_dedup_and_query():
    cs = _cs()
    r1 = cs.add_concept_weight(0, 2, 5, weight=1.5)
    r2 = cs.add_concept_weight(0, 2, 5, weight=9.0)   # repeat -> no-op
    assert r1 == r2
    cs.add_concept_weight(0, 2, 1, weight=0.5)
    assert cs.concept_weights(0, 2) == [(1, 0.5), (5, 1.5)]
    assert cs.concept_weights(0, 3) == []
    assert cs.concept_weights(1, 2) == []            # different order


# -- torch.sparse.mm encoder --------------------------------------------------

def test_sparse_encode_matches_dense():
    cs = _cs()
    # order 0: 3 concepts, 4 sources
    cs.add_concept_weight(0, 0, 0, weight=1.0)
    cs.add_concept_weight(0, 0, 3, weight=2.0)
    cs.add_concept_weight(0, 1, 2, weight=1.0)
    cs.add_concept_weight(0, 2, 1, weight=-0.5)
    act = torch.randn(4, 5)                          # [n_sources, B]
    out = cs.cs_sparse_encode(0, 3, 4, act)
    W = torch.zeros(3, 4)
    W[0, 0] = 1.0; W[0, 3] = 2.0; W[1, 2] = 1.0; W[2, 1] = -0.5
    assert torch.allclose(out, W @ act, atol=1e-5)


def test_sparse_encode_differentiable_in_weights_and_activation():
    cs = _cs()
    cs.add_concept_weight(0, 0, 0, weight=0.7)
    cs.add_concept_weight(0, 1, 1, weight=0.3)
    act = torch.randn(3, 2, requires_grad=True)
    out = cs.cs_sparse_encode(0, 2, 3, act)
    vals = cs._csw_vals[0]
    assert vals.requires_grad
    out.sum().backward()
    assert act.grad is not None and act.grad.abs().sum() > 0
    assert vals.grad is not None and vals.grad.abs().sum() > 0


def test_sparse_encode_empty_order_is_zero():
    cs = _cs()
    act = torch.randn(4, 2)
    out = cs.cs_sparse_encode(2, 5, 4, act)          # order 2 has no weights
    assert torch.equal(out, torch.zeros(5, 2))


def test_csw_rebuild_preserves_trained_weights_on_grow():
    cs = _cs()
    cs.add_concept_weight(0, 0, 0)
    cs.add_concept_weight(0, 0, 1)
    act = torch.randn(3, 1)
    cs.cs_sparse_encode(0, 1, 3, act)
    with torch.no_grad():
        cs._csw_vals[0].copy_(torch.tensor([3.0, 4.0]))
    cs.add_concept_weight(0, 0, 2, weight=9.0)        # grow
    cs.cs_sparse_encode(0, 1, 3, act)
    assert torch.allclose(cs._csw_vals[0].detach(),
                          torch.tensor([3.0, 4.0, 9.0]))


def test_encoder_uses_torch_sparse():
    import inspect
    src = inspect.getsource(Spaces.ConceptualSpace.cs_sparse_encode)
    src += inspect.getsource(Spaces.ConceptualSpace._build_csw)
    assert "torch.sparse.mm" in src
    assert "to_sparse_csr" in src


# -- source activation (encoder input) + dictionary decoder -------------------

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


def test_encode_then_decode_is_differentiable_end_to_end():
    cs = _cs(nS=16, order=1)                           # caps [8,8]
    cs.add_concept_weight(0, 0, 0, weight=0.5)
    cs.add_concept_weight(0, 1, 2, weight=1.0)
    what = torch.randn(16, _D, requires_grad=True)
    act = torch.randn(4, 2, requires_grad=True)        # [n_sources=4, B=2]
    ca = cs.cs_sparse_encode(0, 8, 4, act)             # [8, 2]
    code = cs.cs_decode(0, ca, what)                   # [2, 8, _D]
    code.sum().backward()
    assert what.grad is not None and what.grad.abs().sum() > 0
    assert act.grad is not None and act.grad.abs().sum() > 0
    assert cs._csw_vals[0].grad is not None and cs._csw_vals[0].grad.abs().sum() > 0
