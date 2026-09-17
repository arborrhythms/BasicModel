"""Grammar declarations must have checked query contracts, not string menus."""
import pytest

from Language import Grammar, load_grammar


@pytest.mark.parametrize("declaration", [
    "unimplementedTool(X)", "isPart(X)", "isPart(X, Y, Z)",
    "isTrue()", "query(X)", "tense(X)", "isPart(X, Y) trailing",
])
def test_grammar_rejects_missing_or_malformed_query_executors(declaration):
    grammar = Grammar()
    with pytest.raises(ValueError, match="query|signature|executor|deferred"):
        grammar.configure({"Queries": {"query": [declaration]}})


def test_complete_grammar_declares_the_general_what_subgoal():
    grammar = Grammar()
    grammar.configure(load_grammar("complete.grammar"))
    assert "what(Q)" in grammar.query_ops
    assert "query(X, Y)" in grammar.query_ops


def test_checked_interfaces_share_semantic_relation_and_explicit_roles():
    grammar = Grammar()
    grammar.configure(load_grammar("complete.grammar"))
    signatures = grammar.query_signatures
    part, whole = signatures["isPart"], signatures["isWhole"]
    assert part.semantic_id == whole.semantic_id
    assert part.domain == whole.domain == "conceptual-taxonomy"
    assert part.argument_roles == (0, 2)
    assert whole.argument_roles == (2, 0)
    assert part.occupied_roles == (0, 1, 2)
    assert part.result_kind == "truth"
    assert part.write_scope == ()
    assert signatures["parts"].semantic_id == part.semantic_id
    assert signatures["parts"].argument_roles == (2,)
    assert signatures["parts"].open_roles == (0,)
    assert signatures["wholes"].argument_roles == (0,)
    assert signatures["wholes"].open_roles == (2,)
    assert signatures["arma"].evidence_kind == "estimate"
    assert signatures["what"].result_kind == "subgoal"
    assert signatures["query"].semantic_id != signatures["what"].semantic_id
    for signature in signatures.values():
        assert signature.read_scope
        assert callable(signature.executor)
