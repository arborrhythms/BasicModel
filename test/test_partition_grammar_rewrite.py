"""Golden tests that lock in grammar semantics after the S-only rewrite.

The rewrite (Task 1.2, 2026-04-19) collapsed the C tier into S: all
compositional operations (not, part, intersection, union, lift, lower)
are now S-tier productions.  This file records the post-rewrite snapshot.

API used: ``Grammar().configure(dict)`` — the real loader/parser entry point.
We bypass the XML-loading path and configure a fresh ``Grammar`` instance
directly from the hard-coded post-rewrite production dict.  This avoids any
dependency on ``TheXMLConfig`` or ``MentalModel.xml`` while still exercising
the real ``Grammar._parse_rule`` / ``configure`` path.

The golden values below record the state of MentalModel.xml's ``<grammar>``
block as of 2026-04-19 (post-rewrite, Task 1.2 complete).  This snapshot
is the lock-in point for Task 1.3.
"""
import pytest
from Language import Grammar

# ---------------------------------------------------------------------------
# Post-rewrite production dict (mirrors MentalModel.xml <grammar> after
# Task 1.2 S-only rewrite).  No <C> tags, no ternary lift.
# ---------------------------------------------------------------------------

_POST_REWRITE_GRAMMAR = {
    'S': [
        'true(S)',
        'false(S)',
        'non(S)',
        'conjunction(S, S)',
        'disjunction(S, S)',
        'what(S)',
        'where(S)',
        'when(S)',
        'query(S, S)',
        'swap(S, S)',
        'equals(S, S)',
        'not(S)',
        'part(S, S)',
        'intersection(S, S)',
        'union(S, S)',
        'lower(S, S)',
        'lift(S, S)',
    ],
    # Phase-A note: the previous P-tier entries have been removed from this
    # fixture. They were originally included to verify the old "silently
    # ignore non-S keys" rejection policy. After 2026-04-20 multi-LHS
    # configure, any non-S key is parsed as a typed production, so the
    # fixture now ships only the 17 S-tier productions it's locking in.
}


def _make_grammar():
    """Return a fresh Grammar configured with the post-rewrite production dict."""
    g = Grammar()
    g.configure(_POST_REWRITE_GRAMMAR)
    return g


@pytest.fixture(scope="module")
def grammar():
    return _make_grammar()


# ---------------------------------------------------------------------------
# Golden cases: canonical production string for each rule, in order.
#
# 17 S-tier rules. P-tier was deleted (2026-04-19) and Grammar now parses
# S-tier productions only.
# ---------------------------------------------------------------------------

GOLDEN_CANONICALS = [
    'S -> true(S)',
    'S -> false(S)',
    'S -> non(S)',
    'S -> conjunction(S, S)',
    'S -> disjunction(S, S)',
    'S -> what(S)',
    'S -> where(S)',
    'S -> when(S)',
    'S -> query(S, S)',
    'S -> swap(S, S)',
    'S -> equals(S, S)',
    'S -> not(S)',
    'S -> part(S, S)',
    'S -> intersection(S, S)',
    'S -> union(S, S)',
    'S -> lower(S, S)',
    'S -> lift(S, S)',
]

# ---------------------------------------------------------------------------
# Test: canonical strings are byte-exact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_id,expected_canonical", list(enumerate(GOLDEN_CANONICALS)))
def test_grammar_golden(grammar, rule_id, expected_canonical):
    """Post-rewrite: canonical production string for each rule index.

    Locks in the S-only production set after the Task 1.2 XML rewrite
    and the P-tier deletion (2026-04-19). 17 S-tier rules total.
    """
    assert len(grammar.rules) == len(GOLDEN_CANONICALS), (
        f"Rule count changed: got {len(grammar.rules)}, expected {len(GOLDEN_CANONICALS)}"
    )
    actual = grammar.rules[rule_id].canonical
    assert actual == expected_canonical, (
        f"Rule {rule_id}: got {actual!r}, expected {expected_canonical!r}"
    )


def test_grammar_rule_table_roundtrip(grammar):
    """Grammar exposes a rule_table: rule_id -> production string."""
    for rule_id, production in grammar.rule_table.items():
        assert isinstance(rule_id, int)
        assert isinstance(production, str)
        assert grammar.rule_by_id(rule_id) == production
