"""Grammar-owned thought metadata preserves the existing grammar copy path."""
import copy
from Language import Grammar
from Queries import GrammaticalThoughtRegistry
from test_cs_symbol_table import _cs


def test_copied_grammar_keeps_its_immutable_structural_thought_catalogue():
    source = Grammar()
    source.load_from_grammar_file('complete.grammar')
    cloned = copy.deepcopy(source)
    assert cloned.thought_operations == source.thought_operations
    part = next(operation for operation in cloned.thought_operations
                if operation.semantic_id == 'part')
    assert part.operand_roles == ('I1', 'I2')
    registry = GrammaticalThoughtRegistry.install(_cs(), cloned)
    assert registry.descriptors['part'].semantic_id == 'part'
    assert callable(registry.descriptors['what'].executor)
