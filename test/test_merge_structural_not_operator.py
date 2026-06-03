"""``merge`` stays a structural action, not a learned operator (spec §9).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§9 "Carry-forward concerns": ``merge`` (bare-sequence concatenation) should
remain a structural action completed by the trie / role matcher. It must
NOT be inserted as a learned operator in the operator trie or the
operator-superposition table unless a concrete semantic layer is added for
it. ``insert_operations`` therefore registers only method names that have a
concrete ``GrammarLayer`` (a semantic operator), excluding structural
actions like ``merge``.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def _ss_and_grammar(grammar_file="complete.grammar"):
    from Spaces import SymbolicSpace
    from Language import Grammar
    ss = SymbolicSpace.__new__(SymbolicSpace)   # bypass full-model build
    ss.nDim = 8
    g = Grammar()
    g.load_from_grammar_file(grammar_file)
    return ss, g


def test_merge_not_inserted_as_operator():
    """complete.grammar declares structural ``merge`` rules, but merge is not
    registered in the operator codebook / superposition table."""
    ss, g = _ss_and_grammar()
    assert any(r.method_name == "merge" for r in g.rules), (
        "fixture sanity: complete.grammar should contain bare-sequence "
        "'merge' rules")
    pos = ss.insert_operations(g)
    assert "merge" not in pos
    assert ss.operation_position("merge") is None
    assert ss.operation_vector("merge") is None


def test_semantic_operators_still_inserted():
    """Real semantic operators (those with a GrammarLayer) ARE registered."""
    ss, g = _ss_and_grammar()
    ss.insert_operations(g)
    for op in ("conjunction", "disjunction", "isEqual", "lift", "exist"):
        assert ss.operation_position(op) is not None, op


def test_only_semantic_operators_in_codebook():
    """Every inserted operator has a concrete GrammarLayer."""
    from Language import GRAMMAR_LAYER_CLASSES
    ss, g = _ss_and_grammar()
    pos = ss.insert_operations(g)
    assert set(pos) <= set(GRAMMAR_LAYER_CLASSES), set(pos) - set(GRAMMAR_LAYER_CLASSES)
