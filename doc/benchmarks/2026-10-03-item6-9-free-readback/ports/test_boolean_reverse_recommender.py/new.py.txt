"""Product/mean reverse searches their own compose kernel over the code bank."""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

from Language import ConjunctionLayer, DisjunctionLayer


class _Basis:
    def __init__(self, W):
        self._W = W

    def getW(self):
        return self._W


_V = 6


def test_conjunction_reverse_no_basis_fails_loud():
    # ADAPTED (2026-07-04 serial plan Task 1): the lossy stub is revoked;
    # no-basis reverse raises the Gate-S1 inventory error.
    import pytest
    lyr = ConjunctionLayer()
    parent = torch.rand(2, _V)
    with pytest.raises(NotImplementedError, match="conjunction"):
        lyr.reverse(parent)


def test_disjunction_reverse_no_basis_fails_loud():
    # ADAPTED (2026-07-04 serial plan Task 1): the lossy stub is revoked.
    import pytest
    lyr = DisjunctionLayer()
    parent = torch.rand(2, _V)
    with pytest.raises(NotImplementedError, match="disjunction"):
        lyr.reverse(parent)


def test_conjunction_reverse_with_basis_recovers_pair():
    torch.manual_seed(0)
    lyr = ConjunctionLayer()
    W = torch.rand(5, _V)
    # Product binding of two distinct known codebook rows.
    parent = (W[1].norm()*W[3].norm()*torch.nn.functional.normalize(W[1]*W[3],dim=-1)).unsqueeze(0)        # [1, V]
    x1, x2 = lyr.reverse(parent, basis=_Basis(W))
    assert tuple(x1.shape) == (1, _V) and tuple(x2.shape) == (1, _V)
    # the recommender recovers operands whose intersection ~= parent.
    recon = x1.norm(dim=-1,keepdim=True)*x2.norm(dim=-1,keepdim=True)*torch.nn.functional.normalize(x1*x2,dim=-1)
    assert torch.allclose(recon, parent, atol=1e-5)


def test_disjunction_reverse_with_basis_recovers_pair():
    torch.manual_seed(0)
    lyr = DisjunctionLayer()
    W = torch.rand(5, _V)
    parent = ((W[0]+W[2])/2).unsqueeze(0)        # Mean bundle
    x1, x2 = lyr.reverse(parent, basis=_Basis(W))
    assert tuple(x1.shape) == (1, _V) and tuple(x2.shape) == (1, _V)
    recon = (x1+x2)/2
    assert torch.allclose(recon, parent, atol=1e-5)
