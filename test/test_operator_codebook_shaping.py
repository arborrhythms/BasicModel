"""Live operator-codebook shaping from truth/consequence (Phase R4-sem 1).

User steering (2026-06-03): grammatical categories are distinguished by the
SEMANTIC EFFECT of their associated operators, where the semantic effect is
the operator's vector in the WholeSpace operator codebook
(``_operation_vectors``). Those vectors are shaped LIVE by the soft operator
superposition under truth/consequence supervision -- all operators
participate in a soft superposition per slot until one is chosen, so even a
small amount of correct-answer signal shapes the choice. This generalizes R5
(a standalone {conjunction, disjunction} fit) to the live codebook: after
shaping, the superposition selects the operator whose consequence matches.
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
    from Spaces import WholeSpace
    ss = WholeSpace.__new__(WholeSpace)
    ss.nDim = nDim
    ss._operation_positions = {}
    ss._operation_vectors = {}
    for name in op_names:
        ss._operation_positions[name] = len(ss._operation_positions) + 1
        ss._operation_vectors[name] = ss._seed_operator_vector(name)
    return ss


def _truth_table():
    a = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    b = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    return a, b, torch.minimum(a, b), torch.maximum(a, b)


def test_shape_operators_selects_correct_op_per_context():
    """Consequence supervision through the soft superposition shapes the
    operator codebook so an AND-context query selects conjunction and an
    OR-context query selects disjunction."""
    ss = _ss_with_ops("conjunction", "disjunction")
    a, b, y_and, y_or = _truth_table()
    q_and = torch.zeros(8); q_and[0] = 1.0
    q_or = torch.zeros(8); q_or[1] = 1.0
    examples = [(q_and, a, b, y_and), (q_or, a, b, y_or)]
    ss.shape_operators(examples, ["conjunction", "disjunction"])
    d_and = ss.operator_superposition(q_and)
    d_or = ss.operator_superposition(q_or)
    assert d_and["conjunction"] > d_and["disjunction"], d_and
    assert d_or["disjunction"] > d_or["conjunction"], d_or


def test_shape_operators_writes_back_to_codebook():
    """Shaping updates the live ``_operation_vectors`` (not a throwaway
    copy), so the live superposition reflects the learned semantic effect."""
    ss = _ss_with_ops("conjunction", "disjunction")
    before = ss._operation_vectors["conjunction"].clone()
    a, b, y_and, _ = _truth_table()
    q = torch.zeros(8); q[0] = 1.0
    ss.shape_operators([(q, a, b, y_and)], ["conjunction", "disjunction"])
    after = ss._operation_vectors["conjunction"]
    assert not torch.allclose(before, after), "codebook vector was not shaped"


def test_shape_operators_weak_signal_still_shapes():
    """Because the superposition is soft, even a SINGLE supervised example
    is enough to bias the choice toward the matching operator."""
    ss = _ss_with_ops("conjunction", "disjunction")
    a, b, y_and, _ = _truth_table()
    q = torch.zeros(8); q[0] = 1.0
    ss.shape_operators([(q, a, b, y_and)], ["conjunction", "disjunction"],
                       steps=200)
    d = ss.operator_superposition(q)
    assert d["conjunction"] > d["disjunction"], d


def test_shape_operators_handles_unary_and_binary_ops():
    """The live operator-codebook shaper may see the whole grammar's op set;
    unary operators must not be called with a binary signature."""
    ss = _ss_with_ops("exist", "conjunction")
    a, b, y_and, _ = _truth_table()
    q = torch.zeros(8); q[0] = 1.0
    shaped = ss.shape_operators([(q, a, b, y_and)], steps=5)
    assert {"exist", "conjunction"} <= set(shaped)
