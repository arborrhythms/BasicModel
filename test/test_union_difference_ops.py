"""Concept arithmetic / lattice / chunk ops (Alec 2026-07-05 naming pass).

The 2026-07-05 pass reshaped the additive/lattice family:

  union(a, b)    = RadMax / lattice-max  (saturating, lossy) — DUAL to
                   intersection (RadMin / lattice-min); ex-"join".
  chunk(a, b)    = a + b                 (additive, residual-bearing) — the
                   STRUCTURAL <PartSpace> sum (ex-additive-"union"); may
                   someday replace the radix trie's token chunking.
  sum(a, b)      = a + b                 (element-wise arithmetic sum over
                   CONCEPTS, CS-space_role).
  product(a, b)  = a * b                 (element-wise Hadamard product, the
                   multiplicative dual of sum; lossy).
  difference     RETIRED: it is just sum of a negated operand (sum(a, not b));
                 the exact residual survives as ``ChunkLayer.difference``.

THE CONTRAST test pins the residual hypothesis: the lattice ``union`` provably
destroys the residual (two distinct operands, same max) while the additive
``chunk`` recovers constituents exactly.
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


def test_chunk_and_sum_compose_are_additive():
    """chunk(a, b) == sum(a, b) == a + b exactly (no tanh/clamp/normalize)."""
    from Language import ChunkLayer, SumLayer
    torch.manual_seed(0)
    a = torch.randn(2, 3, 8)
    b = torch.randn(2, 3, 8)
    assert torch.equal(ChunkLayer().compose(a, b), a + b)
    assert torch.equal(SumLayer().compose(a, b), a + b)


def test_product_compose_is_hadamard():
    """product(a, b) == a * b (element-wise)."""
    from Language import ProductLayer
    torch.manual_seed(0)
    a = torch.randn(2, 3, 8)
    b = torch.randn(2, 3, 8)
    assert torch.equal(ProductLayer().compose(a, b), a * b)


def test_chunk_difference_is_exact_residual():
    """chunk's residual helper recovers the other operand: bit-exact on
    integer-valued content, float-rounding-only on random content. (This is
    the retired ``difference`` op, now a static helper on additive chunk.)"""
    from Language import ChunkLayer
    fu = ChunkLayer()
    a_i = torch.tensor([[3.0, -7.0, 0.0, 12.0]])
    b_i = torch.tensor([[5.0, 2.0, -9.0, 1.0]])
    assert torch.equal(ChunkLayer.difference(fu.compose(a_i, b_i), a_i), b_i)
    torch.manual_seed(1)
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    rec = ChunkLayer.difference(fu.compose(a, b), a)
    assert torch.allclose(rec, b, atol=1e-6), (rec - b).abs().max()


def test_bare_reverse_is_null_decomposition():
    """reverse(parent) = (parent, 0) for the additive ops (chunk, sum): the
    mereologically honest w = w ⊔ ∅ split, recomposing EXACTLY."""
    from Language import ChunkLayer, SumLayer
    torch.manual_seed(2)
    for layer in (ChunkLayer(), SumLayer()):
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
    """chunk.reverse(parent, basis=W) is the PEEL step: (best row, parent-row),
    exact by construction, with the true constituent chosen on signed rows."""
    from Language import ChunkLayer
    torch.manual_seed(3)
    W = torch.randn(6, 16)
    parent = W[1] + W[4]
    left, right = ChunkLayer().reverse(parent, basis=_BasisShim(W))
    assert torch.equal(left + right, parent)      # exact recomposition
    hit = [i for i in (1, 4) if torch.allclose(left, W[i])]
    assert hit, "peel step must select a true constituent row"
    other = W[4] if hit[0] == 1 else W[1]
    assert torch.allclose(right, other, atol=1e-5)


def test_peel_recovers_multiset_signed():
    """Greedy matching pursuit over a signed store recovers the exact
    constituent multiset with residual ~0 (the hypothesis's YES case)."""
    from Language import ChunkLayer
    torch.manual_seed(4)
    W = torch.randn(8, 32)
    whole = W[2] + W[5] + W[6]
    idx, residual = ChunkLayer.peel(whole, _BasisShim(W), max_parts=8)
    assert sorted(idx) == [2, 5, 6], idx
    assert float(residual.norm()) < 1e-4 * (1 + float(whole.norm()))


def test_lattice_union_destroys_residual_chunk_does_not():
    """THE CONTRAST (the hypothesis pin): the lattice ``union`` (monotonic
    max) maps DISTINCT operand pairs to the SAME whole, so no function of
    (whole, a) can recover b; the additive chunk recovers b exactly from the
    same operands."""
    from Language import UnionLayer, ChunkLayer
    a = torch.tensor([[1.0, 0.0, 0.5]])
    b = torch.tensor([[0.2, 0.5, 0.1]])
    b_prime = torch.tensor([[0.7, 0.5, 0.3]])
    assert not torch.equal(b, b_prime)
    lattice = UnionLayer(monotonic=True)
    j1 = lattice.forward(a, b)
    j2 = lattice.forward(a, b_prime)
    assert torch.equal(j1, j2), "premise: max-union collapses the pair"
    fu = ChunkLayer()
    r1 = ChunkLayer.difference(fu.compose(a, b), a)
    r2 = ChunkLayer.difference(fu.compose(a, b_prime), a)
    assert torch.allclose(r1, b, atol=1e-6)
    assert torch.allclose(r2, b_prime, atol=1e-6)
    assert not torch.equal(r1, r2)


def test_registry_and_fixity():
    """The reshaped family is registered; chunk/sum/product parse as binary
    infix (T2). ``join`` / ``difference`` are gone."""
    from Language import (GRAMMAR_LAYER_CLASSES, UnionLayer, ChunkLayer,
                          SumLayer, ProductLayer)
    assert GRAMMAR_LAYER_CLASSES["union"] is UnionLayer       # lattice max
    assert GRAMMAR_LAYER_CLASSES["chunk"] is ChunkLayer       # additive sum
    assert GRAMMAR_LAYER_CLASSES["sum"] is SumLayer
    assert GRAMMAR_LAYER_CLASSES["product"] is ProductLayer
    assert "join" not in GRAMMAR_LAYER_CLASSES
    assert "difference" not in GRAMMAR_LAYER_CLASSES
    from Layers import T2_BINARY_INFIX
    for cls in (ChunkLayer, SumLayer, ProductLayer):
        assert cls.surface_schema is T2_BINARY_INFIX


def test_class_contract_pins():
    """Attribute pins: chunk/sum are exact additive CS ops; product is the
    lossy Hadamard dual; the lattice union is lossy / non-invertible."""
    from Language import ChunkLayer, SumLayer, ProductLayer, UnionLayer
    for cls, name in ((ChunkLayer, "chunk"), (SumLayer, "sum")):
        assert cls.rule_name == name
        assert cls.arity == 2
        assert cls.space_role == "CS"
        assert cls.invertible is True
        assert cls.lossy is False
        assert cls.reads_activation is False
    assert ProductLayer.rule_name == "product"
    assert ProductLayer.invertible is False and ProductLayer.lossy is True
    assert UnionLayer.rule_name == "union"
    assert UnionLayer.invertible is False and UnionLayer.lossy is True


def test_grammar_config_declares_sum_product(tmp_path):
    """Lights-on via config: a grammar block declaring sum/product parses into
    RuleDefs (method_name sum/product, CS space_role) through the normal model
    build — no code-level special case."""
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
        "        <C>sum(C, C)</C>\n"
        "        <C>product(C, C)</C>\n      </grammar>", 1)
    cfg = tmp_path / "arith_grammar.xml"
    cfg.write_text(xml)
    init_config(path=str(cfg),
                defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    Models.BaseModel.from_config(str(cfg), data=TheData)
    rules = [r for r in Language.TheGrammar.rules
             if r.method_name in ("sum", "product")]
    assert {r.method_name for r in rules} == {"sum", "product"}, (
        Language.TheGrammar.rules)
    # Direct-tag grammar style: lhs category C, arity 2 (the rule-level
    # space_role tag follows the section default; the LAYER's own space_role
    # stays CS).
    assert all(r.lhs == "C" and r.arity == 2 for r in rules), rules
    assert all(r.rhs_symbols == ("C", "C") for r in rules), rules
