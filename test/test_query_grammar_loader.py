"""Query declarations survive the actual file-loader's grammar normalization."""
import pytest

from Language import Grammar, _expand_compact_order_sets


def test_order_expansion_preserves_non_rule_lists_and_expands_only_rule_bodies():
    query = ['isTrue(X)', 'isPart(X, Y)', 'what(Q)']
    anchors = ['partOf', 'part-of']
    cfg = {'Queries': {'query': query}, 'Anchors': {'part': anchors},
           'compose': {'rule': ['S45 = not.forward(S45)']}}
    result = _expand_compact_order_sets(cfg)
    assert result['Queries']['query'] == query
    assert result['Anchors']['part'] == anchors
    assert result['compose']['rule'] == ['S4 = not.forward(S4)', 'S5 = not.forward(S5)']
    assert cfg['compose']['rule'] == ['S45 = not.forward(S45)']


@pytest.mark.parametrize('filename', ['complete.grammar', 'ladder.grammar', 'default.grammar', 'shamatha.grammar'])
def test_full_file_loading_preserves_checked_predicate_signatures(filename):
    grammar = Grammar()
    grammar.load_from_grammar_file(filename)
    assert 'isTrue(X)' in grammar.query_ops
    assert 'isPart(X, Y)' in grammar.query_ops
    assert callable(grammar.query_signatures['isPart'].executor)
