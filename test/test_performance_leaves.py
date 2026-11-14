"""Fullgraph and derivative certificates for retained numerical leaves."""
import os
import pytest
import torch
from performance_leaves import CASE_NAMES, cases, certify


@pytest.mark.parametrize('name', CASE_NAMES)
def test_retained_leaf_is_fullgraph_equal_without_batch_recompilation(name):
    case = next(case for case in cases() if case.name == name)
    certify(case)


@pytest.mark.device('mps' if torch.backends.mps.is_available() else 'cpu')
@pytest.mark.skipif(not torch.backends.mps.is_available() or
                    os.environ.get('BASICMODEL_DEVICE') not in ('mps', 'gpu'),
                    reason='explicit MPS certificate; default sweep uses CPU')
@pytest.mark.parametrize('name', ('fold_unsorted', 'where_decode_2', 'where_decode_index_2'))
def test_mps_retains_range_pullbacks_and_large_single_rung_addresses(name):
    # The range pullback must copy each unique edge once; the decoder must
    # divide, rather than multiply by a rounded reciprocal before rounding.
    case = next(case for case in cases('mps') if case.name == name)
    certify(case)
    if name == 'where_decode_2':
        # atan2 promotes integral input. The staged divisor must follow its
        # floating result, not truncate to the original carrier's dtype.
        axes = torch.tensor([[0, 1], [1, 0], [0, -1]], device='mps')
        torch.testing.assert_close(case.function(axes), case.function(axes.float()), rtol=0, atol=0)
        from Spaces import WhereEncoding
        half = torch.tensor([[1., .3], [-.3, 1.], [1., 1.]], device='mps', dtype=torch.float16)
        # Captured from 3926005 on MPS, including its negative-angle rounding.
        legacy = torch.tensor([12.03125, 60.03125, 6.99609375], device='mps', dtype=torch.float16)
        for period in (64, 256, 8192):
            encoding = WhereEncoding(maxP=period, nWhere=2)
            torch.testing.assert_close(encoding.decode(half), legacy * (period / 64), rtol=0, atol=0)


def test_sparse_range_cache_follows_edits_and_keeps_all_pullbacks():
    from Layers import SparseLayer
    layer = SparseLayer(6, 4)
    for row, col, weight in ((0, 0, .5), (1, 1, 1.), (1, 2, 0.), (3, 5, 2.)):
        layer.add_edge(row, col, weight)
    source = torch.arange(18, dtype=torch.float64).reshape(6, 3).remainder(7).div(6).requires_grad_()
    layer.double()
    for conjunction in (False, True):
        for dual in (False, True):
            def check():
                whole = layer.fold_presence(source, conjunctive=conjunction, dual=dual)[1:3]
                part = layer.fold_presence(source, start=1, end=3, conjunctive=conjunction, dual=dual)
                torch.testing.assert_close(part, whole, rtol=0, atol=0)
                left = torch.autograd.grad(part.sum(), (source, layer.values), retain_graph=True)
                right = torch.autograd.grad(whole.sum(), (source, layer.values))
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            check()
            layer.add_edge(2, 4, .75)
            check()
            layer.remove_edges([(2, 4)])
            check()
