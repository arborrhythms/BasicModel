"""Task 1 of the serial-derivation plan (2026-07-04): per-op reverse
identity stubs become LOUD ERRORS -- the error inventory is the Gate-S1
decision procedure (write a real reverse() or remove the rule).

The fifth fail-loud application. Converted sites: the lattice/assertive
pseudo-inverses (union/intersection/conjunction/disjunction no-basis
tails; isEqual/isPart/part/query; contextual bind), the ``unreduce``
identity-stub sanction (revoked), and ``SyntacticLayer.reverse``'s
silent non-invertible skip. Rules with REAL inverses are untouched
(tense, lower's recommender/balanced split, union/difference's
∅-decomposition, exist's identity -- whose forward IS the identity, so
``invertible`` flips True as the trivially-satisfied WRITE verdict).
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
import torch


def test_non_invertible_rule_reverse_raises_with_inventory_row():
    """(a)+(c): the stubbed reverse raises NotImplementedError whose
    message carries the rule name, arity, space_role, and the remedy --
    the Gate-S1 inventory row."""
    from Language import IsEqualLayer
    layer = IsEqualLayer()
    parent = torch.randn(2, 3, 8)
    with pytest.raises(NotImplementedError) as ei:
        layer.reverse(parent)
    msg = str(ei.value)
    assert "isEqual" in msg
    assert "arity 2" in msg
    assert "space_role SS" in msg
    assert "write a real reverse() or remove the rule" in msg


def test_all_converted_stubs_raise():
    """The full converted set fails loud (no fabricated splits left)."""
    from Language import (IsPartLayer, PartLayer, QueryPartLayer,
                          ContextualBindLayer, JoinLayer,
                          IntersectionLayer, ConjunctionLayer,
                          DisjunctionLayer)
    parent = torch.rand(1, 2, 6)
    for layer in (IsPartLayer(), PartLayer(), ContextualBindLayer(),
                  JoinLayer(), IntersectionLayer(),
                  ConjunctionLayer(), DisjunctionLayer()):
        with pytest.raises(NotImplementedError):
            layer.reverse(parent)


def test_recommender_path_still_runs_with_basis():
    """Union/Intersection WITH a codebook basis keep the mereology
    recommender (a real attempt, not a stub): no raise, real pair."""
    from Language import JoinLayer, IntersectionLayer

    class _Shim:
        def __init__(self, W):
            self._W = W

        def getW(self):
            return self._W

    torch.manual_seed(0)
    W = torch.rand(6, 6)
    parent = torch.rand(1, 1, 6)
    for layer in (JoinLayer(monotonic=True),
                  IntersectionLayer(monotonic=True)):
        out = layer.reverse(parent, basis=_Shim(W))
        assert isinstance(out, tuple) and len(out) == 2


def test_real_inverses_untouched():
    """(b): rules with faithful reverses keep working."""
    from Language import TenseLayer, UnionLayer, ExistLayer
    from Spaces import event_when_encoding
    enc = event_when_encoding(4)
    head = torch.randn(1, 1, 4)
    x = torch.cat([head, enc.encode(1000).expand(1, 1, -1)], dim=-1)
    t = TenseLayer(); t.set_op("PAST")
    assert torch.allclose(t.reverse(t.forward(x)), x, atol=1e-5)
    fu = UnionLayer()
    p = torch.randn(2, 3, 8)
    left, right = fu.reverse(p)
    assert torch.equal(fu.compose(left, right), p)
    ex = ExistLayer()
    assert ex.invertible is True
    assert torch.equal(ex.reverse(p), p)      # identity forward -> exact


def test_exist_identity_is_faithful():
    """The exist wrapper's forward is the identity, so its identity
    reverse is EXACT -- the flag records it (the trivially-satisfied
    WRITE verdict of the Gate-S1 inventory)."""
    from Language import ExistLayer
    ex = ExistLayer()
    x = torch.randn(3, 4)
    assert torch.equal(ex.forward(x), x)
    assert torch.equal(ex.reverse(ex.forward(x)), x)


def test_syntactic_reverse_raises_on_non_invertible_hosted_rule():
    """The dispatch-level conversion: a recorded rule that resolves to a
    hosted non-invertible layer raises instead of silently skipping."""
    from Language import SyntacticLayer, IsEqualLayer

    layer = SyntacticLayer.__new__(SyntacticLayer)
    object.__setattr__(layer, "_word_space", None)
    object.__setattr__(layer, "_by_name", {"isEqual": IsEqualLayer()})
    object.__setattr__(
        layer, "_next_rule_name",
        lambda direction=None: "isEqual")
    with pytest.raises(NotImplementedError, match="isEqual"):
        layer.reverse(subspace=None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
