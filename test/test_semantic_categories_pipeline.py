"""End-to-end: shape the operator codebook from consequence, then recover
grammatical categories from it (Phase R4-sem 3).

Composes R4-sem 1 (``SymbolicSpace.shape_operators`` -- the live codebook
shaped by truth/consequence through the soft superposition) with R4-sem 2
(``recover_semantic_categories`` -- categories from the operators' semantic
vectors). Demonstrates the full "live-codebook" pipeline the design calls
for, and connects the recovery to the transitional grammar's participation.
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


def _ss_with_ops(*op_names, nDim=8):
    from Spaces import SymbolicSpace
    ss = SymbolicSpace.__new__(SymbolicSpace)
    ss.nDim = nDim
    ss._operation_positions = {}
    ss._operation_vectors = {}
    for name in op_names:
        ss._operation_positions[name] = len(ss._operation_positions) + 1
        ss._operation_vectors[name] = ss._seed_operator_vector(name)
    return ss


def test_shape_then_recover_categorizes_by_operator_effect():
    """Shape conjunction/disjunction from min/max consequence, then recover
    symbol categories from the SHAPED codebook: symbols sharing an operator
    (any position) cluster; symbols on the opposite-effect operator split."""
    from semantic_categories import recover_semantic_categories
    ss = _ss_with_ops("conjunction", "disjunction")
    a = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    b = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    q_and = torch.zeros(8); q_and[0] = 1.0
    q_or = torch.zeros(8); q_or[1] = 1.0
    ss.shape_operators(
        [(q_and, a, b, torch.minimum(a, b)),
         (q_or, a, b, torch.maximum(a, b))],
        ["conjunction", "disjunction"])
    participation = {
        "n1": {("conjunction", 0)},
        "n2": {("conjunction", 1)},   # same operator, other position
        "v1": {("disjunction", 0)},
    }
    cls = recover_semantic_categories(participation, ss._operation_vectors)
    assert cls["n1"] == cls["n2"], cls       # same operator effect
    assert cls["n1"] != cls["v1"], cls       # opposite effect


def test_recovers_on_transitional_grammar_participation():
    """Semantic recovery runs on the real transitional grammar's role
    participation (operator codebook seeded) and collapses the order-variant
    role categories (CONJ_L3/4/5 share one operator slot -> one category) --
    the role-collapse the structural learner also achieves, now via the
    operator-vector signature."""
    from Language import Grammar, GRAMMAR_LAYER_CLASSES
    from Spaces import SymbolicSpace
    from participation import role_participation
    from semantic_categories import recover_semantic_categories

    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    part = role_participation(g)
    # Seed an operator codebook for every semantic operator in the grammar.
    seed = SymbolicSpace.__new__(SymbolicSpace)
    seed.nDim = 16
    op_vectors = {m: seed._seed_operator_vector(m)
                  for sym in part for (m, _p) in part[sym]
                  if m in GRAMMAR_LAYER_CLASSES}
    cls = recover_semantic_categories(part, op_vectors, threshold=0.999)
    assert cls, "expected recovered categories"

    def cid(sym):
        return cls.get(sym)
    for fam in (("CONJ_L3", "CONJ_L4", "CONJ_L5"),
                ("DISJ_R3", "DISJ_R4", "DISJ_R5")):
        present = [s for s in fam if s in cls]
        assert present, fam
        assert len({cid(s) for s in present}) == 1, (fam, [cid(s) for s in present])
