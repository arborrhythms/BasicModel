"""Adding checked query metadata must preserve the existing grammar copy path."""
import copy
from Language import Grammar


def test_copied_grammar_keeps_owned_query_catalog_and_checked_contracts():
    source = Grammar()
    source.load_from_grammar_file('complete.grammar')
    cloned = copy.deepcopy(source)
    assert cloned.query_ops == source.query_ops
    assert cloned.query_signatures == source.query_signatures
    assert cloned.query_signatures is not source.query_signatures
    assert cloned.query_signatures['isPart'].semantic_id == 'part'
    assert callable(cloned.query_signatures['what'].executor)
