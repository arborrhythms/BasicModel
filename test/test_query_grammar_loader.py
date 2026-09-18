"""Structural thought forms survive the file-loader's normalization."""
import pytest

from Language import Grammar, _expand_compact_order_sets


def test_order_expansion_preserves_non_rule_lists_and_expands_only_rule_bodies():
    anchors = ['partOf', 'part-of']
    cfg = {'Anchors': {'part': anchors},
           'compose': {'rule': ['S45 = not.forward(S45)']}}
    result = _expand_compact_order_sets(cfg)
    assert result['Anchors']['part'] == anchors
    assert result['compose']['rule'] == ['S4 = not.forward(S4)', 'S5 = not.forward(S5)']
    assert cfg['compose']['rule'] == ['S45 = not.forward(S45)']


@pytest.mark.parametrize('filename', ['complete.grammar', 'ladder.grammar', 'default.grammar', 'shamatha.grammar'])
def test_full_file_loading_preserves_structural_thought_contracts(filename):
    grammar = Grammar()
    grammar.load_from_grammar_file(filename)
    operations = {operation.semantic_id: operation
                  for operation in grammar.thought_operations}
    assert {'part', 'equal'}.issubset(operations)
    assert operations['part'].operand_roles == ('I1', 'I2')
    assert operations['equal'].result_role == 'O1'
