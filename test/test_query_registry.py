"""Structural grammar declarations are the one checked thought catalogue."""
import pytest

from Language import Grammar, load_grammar
from Queries import GrammaticalThoughtRegistry
from AccessibleMind import Subsystem as Mind
from test_cs_symbol_table import _cs


@pytest.mark.parametrize("declaration", [
    "unimplementedTool(X)", "isPart(X)", "isPart(X, Y, Z)",
    "isTrue()", "query(X)", "tense(X)", "isPart(X, Y) trailing",
])
def test_grammar_rejects_retired_query_catalogue(declaration):
    grammar = Grammar()
    with pytest.raises(ValueError, match="Queries|retired"):
        grammar.configure({"Queries": {"query": [declaration]}})


def test_complete_grammar_declares_the_general_what_subgoal():
    grammar = Grammar()
    grammar.configure(load_grammar("complete.grammar"))
    operations = {operation.semantic_id for operation in grammar.thought_operations}
    assert {'what', 'lookup'}.issubset(operations)


def test_checked_operations_share_grammar_owned_identity_and_explicit_roles():
    grammar = Grammar()
    grammar.configure(load_grammar("complete.grammar"))
    registry = GrammaticalThoughtRegistry.install(_cs(), grammar)
    part = registry.operation_spec('part')
    whole = next(form for form in part.forms if form.structural_id == 'whole')
    assert part.semantic_id == 'part'
    assert part.operand_roles == ('I1', 'I2')
    assert whole.permutation == ('I2', 'I1')
    assert registry.descriptors['part'].domain == 'conceptual-taxonomy'
    assert registry.descriptors['part'].result_kind == 'concept'
    assert registry.descriptors['part'].write_target is Mind.SERIAL
    assert set(registry.descriptors['part'].write_scope) == {
        Mind.SERIAL, Mind.SYMBOLIC, Mind.KNOWING}
    assert registry.descriptors['arma'].evidence_kind == 'estimate'
    assert registry.descriptors['what'].result_kind == 'subgoal'
    assert 'isPart' not in registry.executable_operation_ids
    for descriptor in registry.descriptors.values():
        assert descriptor.read_scope
        assert callable(descriptor.executor)
