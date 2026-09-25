"""Adjacent sigma orders propagate in one pass and cannot self-reference."""
import pytest
import torch
import Spaces
from test_basicmodel import _populate_test_config
from test_cs_sparse_weights import _evidence

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


def _rowp(cs, cid):
    """Global row of relation cid across the per-order namespaces (rev 2)."""
    for k, r in _layer(cs)._tensor_rows.items():
        if (isinstance(k, tuple) and len(k) == 2 and k[0] != "snap"
                and int(k[1]) == int(cid)):
            return r
    return None


def test_depth_d_sigma_path_completes_in_one_pass():
    cs = _cs(nS=64, order=3)
    rows = [cs._csw_concept_row(order, cid)
            for order, cid in ((1, 100), (1, 101), (2, 102), (3, 103))]
    for row, sources in zip(rows, ((0, 1), (2, 3), rows[:2], (rows[2],))):
        for source in sources:
            cs.add_concept_edge(row, source, 1.)
    a0 = torch.zeros(cs._order_caps()[0], 1)
    a0[:4, 0] = torch.tensor([.25, .5, .75, 1.])
    what = torch.randn(64, _D)
    _, a = cs.cs_forward_content(_evidence(a0), what)
    torch.testing.assert_close(a[rows, 0, 0, 0], torch.tensor([.5, 1., 1., 1.]))
    # Removing both first-rung alternatives removes the only path upward.
    layer = _layer(cs)
    with torch.no_grad():
        for (row, _), index in layer._index.items():
            if row in rows[:2]:
                layer.values[index] = 0.
    _, absent = cs.cs_forward_content(_evidence(a0), what)
    assert absent[rows[-1]].count_nonzero() == 0


def test_no_self_edge_via_populate():
    cs = _cs(nS=16, order=2)
    alloc = Spaces._concept_alloc_of(cs)
    other = alloc.new_concept()                      # min-support filler
    C = alloc.new_concept()
    alloc.add(C, "part", ("sym", C))                 # x = {x}: the Quine atom
    alloc.add(C, "part", ("sym", other))             # >= 2 sym constituents
    with pytest.raises(ValueError, match='self-edge'):
        cs._populate_concept_weights(C)
    assert _layer(cs).nnz == 0


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
