"""SparseLayer: generic COO sparse linear substrate (tanh forward / transpose
reverse). ConceptualAttentionLayer: the square wave store + the concept
relation record store (moved off the generic base, 2026-07-06)."""
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))
from Layers import SparseLayer, ConceptualAttentionLayer  # noqa: E402


def _dense(layer):
    # Dense [nOutput, nInput] equivalent of the layer's COO store.
    W = torch.zeros(layer.nOutput, layer.nInput)
    for (r, c), i in layer._index.items():
        W[r, c] = layer.values[i]
    return W


def test_add_edge_idempotent_and_grows_values():
    ly = SparseLayer(4, 3)
    i0 = ly.add_edge(0, 1, weight=0.5)
    i1 = ly.add_edge(0, 1, weight=9.9)      # dup: same slot, weight kept
    assert i0 == i1 and ly.nnz == 1
    assert float(ly.values[i0]) == 0.5
    ly.add_edge(2, 3, weight=-1.0)
    assert ly.nnz == 2 and ly.values.shape == (2,)


def test_forward_matches_dense_tanh():
    torch.manual_seed(0)
    ly = SparseLayer(5, 3)
    for (r, c, w) in [(0, 0, 0.3), (0, 4, -0.7), (1, 2, 1.2), (2, 1, -0.2)]:
        ly.add_edge(r, c, weight=w)
    x = torch.rand(5, 2)
    got = ly.forward(x)
    want = torch.tanh(_dense(ly) @ x)
    assert torch.allclose(got, want, atol=1e-6)
    assert got.min() > -1.0 and got.max() < 1.0


def test_forward_linear_no_tanh_and_1d_squeeze():
    ly = SparseLayer(3, 2, nonlinear=False)
    ly.add_edge(1, 0, weight=2.0)
    x = torch.tensor([3.0, 0.0, 0.0])
    out = ly.forward(x)                      # [2]
    assert out.shape == (2,) and float(out[1]) == 6.0


def test_empty_layer_returns_zeros():
    ly = SparseLayer(4, 3)
    out = ly.forward(torch.rand(4, 2))
    assert out.shape == (3, 2) and torch.all(out == 0)


def test_grow_preserves_trained_values():
    ly = SparseLayer(4, 3)
    ly.add_edge(0, 0, weight=1.0)
    with torch.no_grad():
        ly.values[0] = 0.125                 # simulate training (fp32-exact)
    ly.add_edge(1, 1, weight=1.0)            # growth must keep the tail
    assert float(ly.values[0]) == 0.125 and ly.nnz == 2


def test_forward_differentiable_in_values_and_input():
    ly = SparseLayer(3, 2)
    ly.add_edge(0, 0, weight=0.5)
    ly.add_edge(1, 2, weight=-0.5)
    x = torch.rand(3, 1, requires_grad=True)
    ly.forward(x).sum().backward()
    assert ly.values.grad is not None and x.grad is not None
    assert torch.any(ly.values.grad != 0) and torch.any(x.grad != 0)


def test_reverse_matches_dense_transpose_tanh():
    torch.manual_seed(1)
    ly = SparseLayer(5, 3)
    for (r, c, w) in [(0, 0, 0.3), (1, 2, 1.2), (2, 1, -0.2), (2, 4, 0.9)]:
        ly.add_edge(r, c, weight=w)
    y = torch.rand(3, 2)
    got = ly.reverse(y)
    want = torch.tanh(_dense(ly).t() @ y)
    assert torch.allclose(got, want, atol=1e-6)


def test_remove_edges_compacts_and_preserves_survivors():
    ly = SparseLayer(4, 3)
    ly.add_edge(0, 0, weight=0.1)
    ly.add_edge(1, 1, weight=0.2)
    ly.add_edge(2, 2, weight=0.3)
    with torch.no_grad():
        ly.values[1] = 0.875                 # train the survivor (fp32-exact)
    ly.remove_edges([(0, 0), (2, 2)])
    assert ly.nnz == 1 and (1, 1) in ly._index
    assert float(ly.values[ly._index[(1, 1)]]) == 0.875
    ly.remove_edges([(1, 1)])
    assert ly.nnz == 0
    assert torch.all(ly.forward(torch.rand(4, 2)) == 0)


def test_spmm_kernel_parity():
    torch.manual_seed(2)
    a = SparseLayer(6, 4, kernel="scatter")
    b = SparseLayer(6, 4, kernel="spmm")
    for (r, c, w) in [(0, 5, 0.4), (3, 0, -1.1), (2, 2, 0.7)]:
        a.add_edge(r, c, weight=w)
        b.add_edge(r, c, weight=w)
    x = torch.rand(6, 3)
    assert torch.allclose(a.forward(x), b.forward(x), atol=1e-6)
    assert torch.allclose(a.reverse(x[:4] * 0.5), b.reverse(x[:4] * 0.5),
                          atol=1e-6)


def test_self_edge_forbidden_when_flagged():
    ly = SparseLayer(5, 4, forbid_self_edges=True)
    with pytest.raises(ValueError, match="self-edge"):
        ly.add_edge(2, 2)
    ly.add_edge(2, 3)                      # off-diagonal still fine
    ly.add_edge(2, 4)                      # bias col (== nInput-1) fine


def test_self_edge_allowed_by_default():
    ly = SparseLayer(5, 4)
    ly.add_edge(2, 2)                      # rectangular role layers unaffected


def test_conceptual_attention_layer_is_square_and_forbids_self_edges():
    ly = ConceptualAttentionLayer.square(8)   # N=8 concepts -> [9 x 8], no roles
    assert (ly.nInput, ly.nOutput) == (9, 8)
    with pytest.raises(ValueError, match="self-edge"):
        ly.add_edge(3, 3)


def test_conceptual_attention_layer_wave_step_matches_formula():
    ly = ConceptualAttentionLayer.square(4)
    ly.add_edge(2, 0, weight=0.5)          # relation row 2 reads concept 0
    a = torch.zeros(4, 1); a[0, 0] = 1.0
    s = a.clone()
    out = ly.wave_step(a, s)
    assert torch.allclose(out[2], torch.tanh(torch.tensor([0.5])))
    assert torch.allclose(out[0], torch.tanh(torch.tensor([1.0])))  # tanh(s)
    ly.add_edge(3, 4, weight=0.25)         # bias col (== nInput-1) reads 1
    out = ly.wave_step(a, s)
    assert torch.allclose(out[3], torch.tanh(torch.tensor([0.25])))
    # bias=0.0 masks the EVERYTHING pole: the bias-col edge contributes 0,
    # every activation-driven edge is unchanged.
    out0 = ly.wave_step(a, s, bias=0.0)
    assert torch.allclose(out0[3], torch.tensor([0.0]))
    assert torch.allclose(out0[2], out[2])
    assert torch.allclose(ly.wave_step(a, s, bias=1.0), out)   # default == 1


def test_conceptual_attention_layer_wave_step_empty_layer_is_tanh_s():
    ly = ConceptualAttentionLayer.square(3)
    assert torch.all(ly.wave_step(torch.rand(3, 2), torch.zeros(3, 2)) == 0)
    s = torch.tensor([[0.5, -0.5], [0.0, 1.0], [-1.0, 0.25]])
    out = ly.wave_step(torch.rand(3, 2), s)
    assert torch.allclose(out, torch.tanh(s))


# -- concept relation store (moved off SparseLayer onto ConceptualAttentionLayer)

def test_concept_store_absent_from_generic_sparse_layer():
    # The generic substrate must NOT carry the concept record API.
    ly = SparseLayer(4, 3)
    for m in ("ensure_row_key", "embed_pair", "constituents",
              "row_is_identity", "assign_row", "hebbian_strengthen_row"):
        assert not hasattr(ly, m), f"SparseLayer should not expose {m}"


def test_concept_store_embed_pair_and_identity():
    ly = ConceptualAttentionLayer.square(6)
    ly.embed_pair(1, whole_ref=("sym", 2), part_ref=("sym", 3))
    assert ly.constituents(1) == [("whole", ("sym", 2)), ("part", ("sym", 3))]
    assert ly.row_is_identity(1)                    # one whole + one part
    assert ly.discretize_row(1) == (("sym", 2), ("sym", 3))
    ly.add_constituent(1, "part", ("sym", 4))       # second part -> not 1:1
    assert not ly.row_is_identity(1)
    assert ly.discretize_row(1) is None
    ly.add_constituent(1, "part", ("sym", 4))       # idempotent
    assert ly.count_role(1, "part") == 2


def test_concept_store_assign_row_and_hebbian():
    ly = ConceptualAttentionLayer.square(4)
    r0 = ly.assign_row(10)
    assert ly.assign_row(10) == r0                  # stable per key
    assert ly.assign_row(11) == r0 + 1              # next free row
    assert ly.row_of(10) == r0 and ly.row_of(99) is None
    ly.add_edge(2, 0, weight=0.5)
    ly.hebbian_strengthen_row(2, eta=0.25, cap=4.0)
    assert abs(float(ly.values[ly._index[(2, 0)]]) - 0.75) < 1e-6
