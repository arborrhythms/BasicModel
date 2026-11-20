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
@pytest.mark.parametrize('name', ('sigma_unsorted', 'where_decode_2', 'where_decode_index_2'))
def test_mps_retains_range_pullbacks_and_large_single_rung_addresses(name):
    # The range pullback must copy each unique edge once; the decoder must
    # divide, rather than multiply by a rounded reciprocal before rounding.
    case = next(case for case in cases('mps') if case.name == name)
    certify(case)
    if name == 'where_decode_2':
        # atan2 promotes integral input. The staged divisor must follow its
        # floating result, not truncate to the original carrier's dtype.
        axes = torch.tensor([[0, 1], [1, 0], [0, 0]], device='mps')
        torch.testing.assert_close(case.function(axes), case.function(axes.float()), rtol=0, atol=0)
        from Spaces import WhereEncoding, unsigned_phase, signed_phase
        half = unsigned_phase(torch.tensor([[1., .3], [-.3, 1.], [1., 1.]], device='mps', dtype=torch.float16))
        # The unsigned storage has half-precision quantization of its own.
        # Keep the exact divide/rounding contract on the stored coordinates.
        phase = signed_phase(half)
        for period in (64, 256, 8192):
            encoding = WhereEncoding(maxP=period, nWhere=2)
            expected = (torch.atan2(phase[:, 0], phase[:, 1]) % (2*torch.pi)) / encoding.div_lf
            torch.testing.assert_close(encoding.decode(half), expected, rtol=0, atol=0)


def test_sparse_range_cache_follows_edits_and_keeps_all_pullbacks():
    from Layers import SparseLayer, NotLayer3, sigma_read
    layer = SparseLayer(6, 4)
    for row, col, weight in ((0, 0, .5), (1, 1, 1.), (1, 2, 0.), (3, 5, -2.)):
        layer.add_edge(row, col, weight)
    lanes = torch.arange(36, dtype=torch.float64).reshape(6, 3, 2).remainder(7).div(6).requires_grad_()
    layer.double()

    def check():
        source = NotLayer3.options(lanes)
        whole = sigma_read(layer, source)[1:3]
        part = sigma_read(layer, source, start=1, end=3)
        torch.testing.assert_close(part, whole, rtol=0, atol=0)
        left = torch.autograd.grad(part.sum(), (lanes, layer.values), retain_graph=True)
        right = torch.autograd.grad(whole.sum(), (lanes, layer.values))
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    check()
    layer.add_edge(2, 4, .75)
    check()
    layer.remove_edges([(2, 4)])
    check()
