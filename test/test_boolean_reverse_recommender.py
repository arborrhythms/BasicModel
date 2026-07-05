"""Stage B: Conjunction/Disjunction reverse uses the codebook recommender.

AND/OR are many-to-one, so their reverse cannot invert in general -- but with a
codebook ``basis`` the mereology recommender (Ops.conjunctionReverse /
disjunctionReverse) recovers the operand pair whose intersection/union matches
the parent (exact on a discrete vocabulary -- the XOR reconstruction path),
mirroring IntersectionLayer/JoinLayer.reverse. Without a basis the lossy
``(parent, parent)`` fallback is preserved (byte-identical to the prior stub).
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
    # parent = the AND (min) of two known codebook rows.
    parent = torch.minimum(W[1], W[3]).unsqueeze(0)        # [1, V]
    x1, x2 = lyr.reverse(parent, basis=_Basis(W))
    assert tuple(x1.shape) == (1, _V) and tuple(x2.shape) == (1, _V)
    # the recommender recovers operands whose intersection ~= parent.
    recon = torch.minimum(x1, x2)
    assert torch.allclose(recon, parent, atol=1e-5)


def test_disjunction_reverse_with_basis_recovers_pair():
    torch.manual_seed(0)
    lyr = DisjunctionLayer()
    W = torch.rand(5, _V)
    parent = torch.maximum(W[0], W[2]).unsqueeze(0)        # OR (max)
    x1, x2 = lyr.reverse(parent, basis=_Basis(W))
    assert tuple(x1.shape) == (1, _V) and tuple(x2.shape) == (1, _V)
    recon = torch.maximum(x1, x2)
    assert torch.allclose(recon, parent, atol=1e-5)
