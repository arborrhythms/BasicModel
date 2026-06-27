"""Sparse-coding edge store + scatter-add kernel on ConceptualSpace.

A Concept is a sparse, weighted linear combination over a basis of *symbols*
(symbols are concepts from the previous ramsified order). The decomposition is
a three-column COO table ``(ConceptIndex, SymbolIndex, Weight)`` with a learnable
``Weight``; the transform is a differentiable scatter-add SpMM
``concept[c] = sum_e Weight[e] * basis[SymbolIndex[e]]`` (``index_select`` +
``index_add``, NO ``torch.sparse``), differentiable in both ``Weight`` and the
basis. Nothing here touches the forward path => byte-identical everywhere.
"""
import inspect
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


def _cs(nS=64):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    return Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])


# -- Phase 1: COO edge store --------------------------------------------------

def test_add_edge_dedup():
    cs = _cs()
    r1 = cs.add_edge(5, 2)
    r2 = cs.add_edge(5, 2)                  # repeat (concept,symbol) -> no-op
    assert r1 == r2
    assert cs.concept_edges(5) == [(2, 1.0)]   # exactly one edge, default weight


def test_add_edge_carries_weight():
    cs = _cs()
    cs.add_edge(0, 3, weight=2.5)
    cs.add_edge(0, 1, weight=0.5)
    # deterministically ordered by symbol
    assert cs.concept_edges(0) == [(1, 0.5), (3, 2.5)]
    assert cs.concept_edges(7) == []           # untouched concept -> empty


def test_retire_drops_edges():
    cs = _cs()
    cs.add_edge(4, 1)
    cs.add_edge(4, 2)
    cs.add_edge(9, 1)                          # a different concept survives
    cs.retire_concept(4)
    assert cs.concept_edges(4) == []
    assert cs.concept_edges(9) == [(1, 1.0)]


def test_drop_edges_is_noop_when_unbuilt():
    cs = _cs()
    # never touched the edge store -> dropping is a pure no-op (byte-identical)
    assert getattr(cs, "_edge_concept", None) is None
    cs._drop_concept_edges(3)
    assert getattr(cs, "_edge_concept", None) is None


def test_refine_synthesis_inherits_edges():
    cs = _cs()
    A = cs.new_concept()
    for p in range(6):                         # 6 parts > k_many -> synthesize
        cs.add_part(A, p)
    cs.add_whole(A, 100)
    cs.add_edge(A, 11, weight=0.5)
    cs.add_edge(A, 12, weight=1.5)
    reqs = cs.refine_over_collected(k_parts=4, k_wholes=4)
    H = reqs[0]["result"]
    # H inherits the union of A's sparse edges; A itself is retired (no edges)
    assert cs.concept_edges(H) == [(11, 0.5), (12, 1.5)]
    assert cs.concept_edges(A) == []


def test_rebuild_preserves_trained_weights_on_grow():
    cs = _cs()
    cs.add_edge(0, 1)
    cs.add_edge(0, 2)
    _, _, w = cs._rebuild_edge_pools()
    with torch.no_grad():                      # "train" the existing weights
        w.copy_(torch.tensor([3.0, 4.0]))
    cs.add_edge(0, 3, weight=9.0)              # grow: append a new edge
    _, _, w2 = cs._rebuild_edge_pools()
    # tail-preserving: the first two rows keep their trained values; the new
    # tail row takes its initial weight from the host list.
    assert torch.allclose(w2.detach(), torch.tensor([3.0, 4.0, 9.0]))


def test_rebuild_is_cached_until_dirty():
    cs = _cs()
    cs.add_edge(0, 1)
    _, _, w1 = cs._rebuild_edge_pools()
    _, _, w2 = cs._rebuild_edge_pools()        # no new edge -> cached pool
    assert w1 is w2
    cs.add_edge(0, 2)                          # dirties
    _, _, w3 = cs._rebuild_edge_pools()
    assert w3 is not w1


# -- Phase 2: scatter-add SpMM kernel -----------------------------------------

def test_scatter_add_matches_dense():
    cs = _cs()
    cs.add_edge(0, 1)
    cs.add_edge(0, 3, weight=2.0)
    cs.add_edge(1, 2)
    V, D = 5, _D
    basis = torch.randn(V, D)
    out = cs.scatter_concept_event(2, basis, normalize=False)
    M = torch.zeros(2, V)
    M[0, 1] = 1.0
    M[0, 3] = 2.0
    M[1, 2] = 1.0
    ref = M @ basis                            # dense einsum reference
    assert torch.allclose(out, ref, atol=1e-6)


def test_scatter_uniform_is_mean():
    cs = _cs()
    cs.add_edge(0, 1)                          # uniform default weights
    cs.add_edge(0, 2)
    cs.add_edge(0, 3)
    basis = torch.randn(5, _D)
    out = cs.scatter_concept_event(1, basis, normalize=True)
    mean = basis[[1, 2, 3]].mean(dim=0, keepdim=True)
    assert torch.allclose(out, mean, atol=1e-6)


def test_scatter_empty_concept_is_zero():
    cs = _cs()
    cs.add_edge(1, 0)
    basis = torch.randn(4, _D)
    out = cs.scatter_concept_event(3, basis, normalize=True)
    assert torch.equal(out[0], torch.zeros(_D))   # concept 0 has no edges
    assert torch.equal(out[2], torch.zeros(_D))


def test_scatter_is_differentiable_in_weights_and_basis():
    cs = _cs()
    cs.add_edge(0, 1, weight=0.7)
    cs.add_edge(0, 2, weight=0.3)
    basis = torch.randn(4, _D, requires_grad=True)
    out = cs.scatter_concept_event(1, basis, normalize=False)
    _, _, w = cs._rebuild_edge_pools()
    assert w.requires_grad
    out.sum().backward()
    assert basis.grad is not None and basis.grad.abs().sum() > 0
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_scatter_uses_no_torch_sparse():
    fn = Spaces.ConceptualSpace.scatter_concept_event
    src = inspect.getsource(fn)
    head, _, rest = src.partition('"""')           # strip the """...""" docstring
    code = head + rest.partition('"""')[2]
    assert "torch.sparse" not in code              # MLX/executorch safety
    assert "index_add" in code and "index_select" in code
