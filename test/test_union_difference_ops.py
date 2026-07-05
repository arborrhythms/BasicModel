"""Union/Difference — the residual-bearing op pair on CONCEPTS (Alec
2026-07-04/05; plan doc/plans/2026-07-04-union-difference-concept-ops.md;
the lattice max vacated the `union` name -> `join`).

The pair is the union/difference class ANALOGOUS to the symbol layer's
conjunction/disjunction class (the lattice `join`/`intersection`/
`join_from_bottom` folds, which saturate and carry NO residual):

  union(a, b)       = a + b          (the mereological sum; no tanh/clamp)
  difference(w, a)  = w - a          (the EXACT residual)
  union.reverse(p)  = (p, 0)         (the ∅-decomposition; recomposes exactly)
  union.reverse(p, basis=W)          (the peel step: (row, p - row))
  union.peel(w, basis)               (greedy matching pursuit over the store)

THE CONTRAST test pins the hypothesis: the lattice pair provably destroys
the residual (two distinct operands, same join) while union/difference
recovers constituents exactly.
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


class _BasisShim:
    """Minimal getW() carrier (the sibling basis-reverse contract)."""

    def __init__(self, W):
        self._W = W

    def getW(self):
        return self._W


def test_fusion_compose_is_additive():
    """union(a, b) == a + b exactly (no tanh, no clamp, no normalize)."""
    from Language import UnionLayer
    torch.manual_seed(0)
    a = torch.randn(2, 3, 8)
    b = torch.randn(2, 3, 8)
    out = UnionLayer().compose(a, b)
    assert torch.equal(out, a + b)


def test_difference_is_exact_residual():
    """difference recovers the other operand: bit-exact on integer-valued
    content, float-rounding-only on random content."""
    from Language import UnionLayer, DifferenceLayer
    fu, di = UnionLayer(), DifferenceLayer()
    # bit-exact: integer-valued floats survive +/- exactly
    a_i = torch.tensor([[3.0, -7.0, 0.0, 12.0]])
    b_i = torch.tensor([[5.0, 2.0, -9.0, 1.0]])
    assert torch.equal(di.compose(fu.compose(a_i, b_i), a_i), b_i)
    # float32 random: exact to rounding
    torch.manual_seed(1)
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    rec = di.compose(fu.compose(a, b), a)
    assert torch.allclose(rec, b, atol=1e-6), (rec - b).abs().max()


def test_bare_reverse_is_null_decomposition():
    """reverse(parent) = (parent, 0): the mereologically honest
    w = w ⊔ ∅ split (NOT a partition-blind halving), and it recomposes
    EXACTLY (the generate-exactness bar)."""
    from Language import UnionLayer, DifferenceLayer
    torch.manual_seed(2)
    for layer in (UnionLayer(), DifferenceLayer()):
        parent = torch.randn(2, 5, 12)
        left, right = layer.reverse(parent)
        assert torch.equal(left, parent)
        assert torch.equal(right, torch.zeros_like(parent))
        assert torch.equal(layer.compose(left, right), parent)
        # generate() is the reverse alias (grammar dual contract)
        g_left, g_right = layer.generate(parent)
        assert torch.equal(g_left, parent)
        assert torch.equal(g_right, torch.zeros_like(parent))


def test_basis_reverse_peels_one_part():
    """reverse(parent, basis=W) is the PEEL step: (best row, parent - row),
    exact by construction (composing back == parent), with the true
    constituent chosen on near-orthogonal (signed) rows."""
    from Language import UnionLayer
    torch.manual_seed(3)
    W = torch.randn(6, 16)
    parent = W[1] + W[4]
    left, right = UnionLayer().reverse(parent, basis=_BasisShim(W))
    assert torch.equal(left + right, parent)      # exact recomposition
    hit = [i for i in (1, 4) if torch.allclose(left, W[i])]
    assert hit, "peel step must select a true constituent row"
    other = W[4] if hit[0] == 1 else W[1]
    assert torch.allclose(right, other, atol=1e-5)


def test_peel_recovers_multiset_signed():
    """Greedy matching pursuit over a signed store recovers the exact
    constituent multiset with residual ~0 (the hypothesis's YES case)."""
    from Language import UnionLayer
    torch.manual_seed(4)
    W = torch.randn(8, 32)
    whole = W[2] + W[5] + W[6]
    idx, residual = UnionLayer.peel(whole, _BasisShim(W), max_parts=8)
    assert sorted(idx) == [2, 5, 6], idx
    assert float(residual.norm()) < 1e-4 * (1 + float(whole.norm()))


def test_lattice_union_destroys_residual_fusion_does_not():
    """THE CONTRAST (the hypothesis pin): the lattice union (monotonic
    max — the conjunction/disjunction class) maps DISTINCT operand pairs
    to the SAME whole, so no function of (whole, a) can recover b; the
    fusion/difference pair recovers b exactly from the same operands."""
    from Language import JoinLayer, UnionLayer, DifferenceLayer
    a = torch.tensor([[1.0, 0.0, 0.5]])
    b = torch.tensor([[0.2, 0.5, 0.1]])
    b_prime = torch.tensor([[0.7, 0.5, 0.3]])
    assert not torch.equal(b, b_prime)
    lattice = JoinLayer(monotonic=True)
    j1 = lattice.forward(a, b)
    j2 = lattice.forward(a, b_prime)
    assert torch.equal(j1, j2), "premise: max-join collapses the pair"
    fu, di = UnionLayer(), DifferenceLayer()
    r1 = di.compose(fu.compose(a, b), a)
    r2 = di.compose(fu.compose(a, b_prime), a)
    assert torch.allclose(r1, b, atol=1e-6)
    assert torch.allclose(r2, b_prime, atol=1e-6)
    assert not torch.equal(r1, r2)


def test_registry_and_fixity():
    """Both rules registered; fusion parses as binary infix (T2),
    difference as binary directional (T3 — order-sensitive like part)."""
    import Language
    from Language import (GRAMMAR_LAYER_CLASSES, UnionLayer,
                          DifferenceLayer)
    assert GRAMMAR_LAYER_CLASSES["union"] is UnionLayer
    assert GRAMMAR_LAYER_CLASSES["difference"] is DifferenceLayer
    from Layers import T2_BINARY_INFIX, T3_BINARY_DIRECTIONAL
    assert UnionLayer.surface_schema is T2_BINARY_INFIX
    assert DifferenceLayer.surface_schema is T3_BINARY_DIRECTIONAL


def test_class_contract_pins():
    """Attribute pins: CS-space_role arity-2 concept ops; reverse is real
    (invertible generate-exactness), content-read (not activation)."""
    from Language import UnionLayer, DifferenceLayer
    for cls, name in ((UnionLayer, "union"),
                      (DifferenceLayer, "difference")):
        assert cls.rule_name == name
        assert cls.arity == 2
        assert cls.space_role == "CS"
        assert cls.invertible is True
        assert cls.lossy is False
        assert cls.reads_activation is False


def test_grammar_config_declares_fusion_difference(tmp_path):
    """Lights-on via config: a grammar block declaring the pair parses
    into RuleDefs (method_name fusion/difference, CS space_role) through
    the normal model build — no code-level special case."""
    import Language
    import Models
    from data import TheData
    from util import init_config
    _PROJECT = os.path.dirname(_BIN)
    fixture = os.path.join(_PROJECT, "data", "MM_xor_fixture.xml")
    with open(fixture) as f:
        xml = f.read()
    xml = xml.replace(
        "</grammar>",
        "        <C>union(C, C)</C>\n"
        "        <C>difference(C, C)</C>\n      </grammar>", 1)
    cfg = tmp_path / "fusion_grammar.xml"
    cfg.write_text(xml)
    init_config(path=str(cfg),
                defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    Models.BaseModel.from_config(str(cfg), data=TheData)
    rules = [r for r in Language.TheGrammar.rules
             if r.method_name in ("union", "difference")]
    assert {r.method_name for r in rules} == {"union", "difference"}, (
        Language.TheGrammar.rules)
    # Direct-tag grammar style: lhs category C, arity 2 (the rule-level
    # space_role tag follows the section default, as for MM_boolean's
    # <C>intersection(C, C)</C>; the LAYER's own space_role stays CS).
    assert all(r.lhs == "C" and r.arity == 2 for r in rules), rules
    assert all(r.rhs_symbols == ("C", "C") for r in rules), rules


def test_butterfly_cascade_parity():
    """butterfly=True builds the standard cascade (structural parity with
    the sibling binary ops) and runs forward/reverse without error."""
    from Language import UnionLayer
    torch.manual_seed(5)
    layer = UnionLayer(nInput=8, nOutput=8, butterfly=True, N=8)
    x = torch.randn(2, 4, 2)          # flattened M=8
    y = layer.forward(x)
    assert y.shape == x.shape
    back = layer.reverse(y)
    assert torch.is_tensor(back) and back.shape == x.shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
