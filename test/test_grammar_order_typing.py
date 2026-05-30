"""Tests for order-typed grammar category parsing.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Order-Typed Grammar — Grammar syntax.

`Grammar._parse_category(token)` splits a category token like `NP3` /
`VP1` / `DET` / `NP*` / `NP*+1` into a `ParsedCategory(name, order)`
where `order` is an `OrderExpr(kind, delta)`.

2026-05-20 Kleene restoration (path-to-complete §2): polymorphic
``*`` order forms are accepted alongside explicit constants. At
REDUCE time the rule-local ``*`` binds from the operand's order and
propagates consistently across the rule's other variable slots.

Bare categories (no order annotation) bind to constant order 0 per the
plan's "bare category is syntactic sugar for order 0" decision.

Example fully-explicit rules:
    S4 = lift(NP3, VP1)        # event = lift contiguous noun + verb aspect
    NP3 = lower(DET, NP4)      # specify an abstract count noun

Example polymorphic rules:
    S = lift(NP*, VP1)         # any NP-order; LHS is variable too
    NP = conjunction(DET, N*)  # bind * to N's order
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def test_bare_category_is_constant_order_zero():
    """A category like 'DET' (no annotation) parses as constant 0."""
    from Language import Grammar
    parsed = Grammar._parse_category("DET")
    assert parsed.name == "DET"
    assert parsed.order.kind == "constant"
    assert parsed.order.delta == 0


def test_explicit_constant_order_zero():
    """'NP0' is constant 0."""
    from Language import Grammar
    parsed = Grammar._parse_category("NP0")
    assert parsed.name == "NP"
    assert parsed.order.kind == "constant"
    assert parsed.order.delta == 0


def test_explicit_constant_order_one():
    """'VP1' is constant 1."""
    from Language import Grammar
    parsed = Grammar._parse_category("VP1")
    assert parsed.name == "VP"
    assert parsed.order.kind == "constant"
    assert parsed.order.delta == 1


def test_explicit_constant_higher_order():
    """'X42' is constant 42 — multi-digit constants supported."""
    from Language import Grammar
    parsed = Grammar._parse_category("X42")
    assert parsed.name == "X"
    assert parsed.order.kind == "constant"
    assert parsed.order.delta == 42


def test_whitespace_tolerant():
    """Surrounding whitespace is stripped."""
    from Language import Grammar
    parsed = Grammar._parse_category("  VP1  ")
    assert parsed.name == "VP"
    assert parsed.order.kind == "constant"
    assert parsed.order.delta == 1


def test_kleene_variable_zero_offset():
    """``NP*`` parses as variable +0."""
    from Language import Grammar
    parsed = Grammar._parse_category("NP*")
    assert parsed.name == "NP"
    assert parsed.order.kind == "variable"
    assert parsed.order.delta == 0


def test_kleene_variable_positive_offset():
    """``NP*+1`` parses as variable +1."""
    from Language import Grammar
    parsed = Grammar._parse_category("NP*+1")
    assert parsed.name == "NP"
    assert parsed.order.kind == "variable"
    assert parsed.order.delta == 1


def test_kleene_variable_negative_offset():
    """``NP*-1`` parses as variable -1."""
    from Language import Grammar
    parsed = Grammar._parse_category("NP*-1")
    assert parsed.name == "NP"
    assert parsed.order.kind == "variable"
    assert parsed.order.delta == -1


def test_invalid_raises():
    """Malformed tokens raise ValueError."""
    from Language import Grammar
    import pytest
    with pytest.raises(ValueError):
        Grammar._parse_category("**")
    with pytest.raises(ValueError):
        Grammar._parse_category("")
    with pytest.raises(ValueError):
        Grammar._parse_category("NP*+")


def test_compact_order_set_rule_expands_pairwise():
    """In .grammar source, ``S45`` means concrete orders ``S4`` and
    ``S5``. Repeated use of the same suffix is correlated pairwise."""
    from Language import _expand_compact_order_sets_in_rule
    assert _expand_compact_order_sets_in_rule(
        "S45 = not.forward(NOT_S45)"
    ) == [
        "S4 = not.forward(NOT_S4)",
        "S5 = not.forward(NOT_S5)",
    ]


def test_parse_category_still_accepts_multi_digit_constant():
    """The expansion is grammar-file sugar; direct category parsing
    still treats multi-digit constants literally."""
    from Language import Grammar
    parsed = Grammar._parse_category("S45")
    assert parsed.name == "S"
    assert parsed.order.kind == "constant"
    assert parsed.order.delta == 45


# -- RuleOrderSignature tests -----------------------------------------
# Each parsed rule has an order signature capturing LHS / RHS categories,
# their OrderExprs, the op name, and the order_delta (+1 / -1 / 0).


def test_lift_rule_has_order_delta_plus_one():
    """``S4 = lift(NP3, VP1)`` parses with ``order_delta == +1`` and fully
    explicit constant order annotations on every category."""
    from Language import Grammar
    g = Grammar()
    rule = g._parse_rule("S4", "lift(NP3, VP1)", tier='S')
    sig = g._rule_order_signature(rule)
    assert sig.order_delta == 1
    assert sig.lhs_category == "S"
    assert sig.lhs_order_expr.kind == "constant"
    assert sig.lhs_order_expr.delta == 4
    assert sig.rhs_categories == ("NP", "VP")
    assert sig.rhs_order_exprs[0].kind == "constant"
    assert sig.rhs_order_exprs[0].delta == 3
    assert sig.rhs_order_exprs[1].kind == "constant"
    assert sig.rhs_order_exprs[1].delta == 1
    assert sig.op_name == "lift"


def test_lower_rule_has_order_delta_minus_one():
    """``NP3 = lower(DET, NP4)`` parses with ``order_delta == -1``. NP4
    (abstract count noun) is specified by DET into NP3 (contiguous
    individual). NP3 is the contiguous floor — there is no NP2."""
    from Language import Grammar
    g = Grammar()
    rule = g._parse_rule("NP3", "lower(DET, NP4)", tier='S')
    sig = g._rule_order_signature(rule)
    assert sig.order_delta == -1
    assert sig.lhs_category == "NP"
    assert sig.lhs_order_expr.kind == "constant"
    assert sig.lhs_order_expr.delta == 3
    assert sig.rhs_categories == ("DET", "NP")
    # DET is bare -> constant 0
    assert sig.rhs_order_exprs[0].kind == "constant"
    assert sig.rhs_order_exprs[0].delta == 0
    assert sig.rhs_order_exprs[1].kind == "constant"
    assert sig.rhs_order_exprs[1].delta == 4
    assert sig.op_name == "lower"


def test_ordinary_rule_is_order_preserving():
    """Non-lift / non-lower rules have ``order_delta == 0``."""
    from Language import Grammar
    g = Grammar()
    rule = g._parse_rule("NP", "conjunction(DET, N)", tier='S')
    sig = g._rule_order_signature(rule)
    assert sig.order_delta == 0
    assert sig.op_name == "conjunction"


def test_unary_rule_order_signature():
    """Unary rule (e.g. ``S = not(S)``) has one RHS slot."""
    from Language import Grammar
    g = Grammar()
    rule = g._parse_rule("S", "not(S)", tier='S')
    sig = g._rule_order_signature(rule)
    assert sig.order_delta == 0
    assert sig.lhs_category == "S"
    assert sig.rhs_categories == ("S",)
    assert len(sig.rhs_order_exprs) == 1
    assert sig.op_name == "not"


def test_lhs_order_captured():
    """LHS order annotations are captured (e.g. ``S0 = ...``)."""
    from Language import Grammar
    g = Grammar()
    rule = g._parse_rule("S0", "not(S)", tier='S')
    sig = g._rule_order_signature(rule)
    assert sig.lhs_category == "S"
    assert sig.lhs_order_expr.kind == "constant"
    assert sig.lhs_order_expr.delta == 0


# -- No static validation ----------------------------------------------
# Order admissibility lives at runtime (Phase 2 STM REDUCE), not at
# grammar load. Words are mapped to the category codebook by soft
# assignment that participates in the parser's superposition state —
# whether a word fills an NP3 vs NP4 slot is a runtime question, not
# something the grammar loader can pre-empt. The rule's signature is
# extracted statically; matching against operand orders is dynamic.


def test_modal_sentence_lift():
    """``S5 = lift(S4, MP1)`` — modal phrase (MP) raises a sentence into
    a 5D modal-probability space. Confirms the lift pattern works for
    sentence-level modal embedding, not just NP/VP construction."""
    from Language import Grammar
    g = Grammar()
    rule = g._parse_rule("S5", "lift(S4, MP1)", tier='S')
    sig = g._rule_order_signature(rule)
    assert sig.lhs_category == "S"
    assert sig.lhs_order_expr.delta == 5
    assert sig.rhs_categories == ("S", "MP")
    assert sig.rhs_order_exprs[0].delta == 4
    assert sig.rhs_order_exprs[1].delta == 1
    assert sig.op_name == "lift"
    assert sig.order_delta == 1


def test_complete_grammar_relation_ops_registered_and_mirrored():
    """The relation-truth rewrite uses productive op names, including
    queryPart as the interrogative counterpart to assertPart."""
    from Language import Grammar, GRAMMAR_LAYER_CLASSES

    def cats(text):
        return tuple(part.strip() for part in text.split(',')
                     if part.strip())

    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    methods = {r.method_name for r in g.rules_upward if r.method_name}
    assert "queryPart" in methods
    assert "assertPart" in methods
    assert "isEqual" in methods
    assert "queryEqual" not in methods
    assert "assertEqual" not in methods
    assert any(
        rule.method_name == "isEqual" and rule.query
        for rule in g.rules_upward)
    assert any(
        rule.method_name == "isEqual" and not rule.query
        for rule in g.rules_upward)
    assert any(
        rule.method_name == "isEqual" and rule.query
        for rule in g.rules_downward)

    unknown = {
        method for method in methods
        if method != "merge" and method not in GRAMMAR_LAYER_CLASSES
    }
    assert unknown == set()

    up = {
        (r.method_name, tuple(r.rhs_symbols or ()), cats(r.lhs))
        for r in g.rules_upward
    }
    down = {
        (r.method_name, tuple(r.rhs_symbols or ()), cats(r.lhs))
        for r in g.rules_downward
    }
    missing = []
    for rule in g.rules_upward:
        if rule.method_name is None:
            continue
        reverse = (
            rule.method_name,
            cats(rule.lhs),
            tuple(rule.rhs_symbols or ()),
        )
        if reverse not in down:
            missing.append(rule.canonical)
    assert missing == []


def test_signatures_capture_explicit_orders_for_runtime_use():
    """The signature carries the rule's declared orders verbatim; the
    runtime (STM REDUCE) decides admissibility against operand soft
    distributions. No grammar-load rejection."""
    from Language import Grammar
    g = Grammar()
    # Even a rule with "inconsistent-looking" orders is parseable and
    # signature-extractable. Whether it fires at runtime depends on
    # the operand soft category distributions, not on the loader.
    sig = g._rule_order_signature(
        g._parse_rule("NP3", "conjunction(DET, N3)", tier='S'))
    assert sig.lhs_category == "NP"
    assert sig.lhs_order_expr.delta == 3
    assert sig.rhs_categories == ("DET", "N")
    assert sig.rhs_order_exprs[0].delta == 0
    assert sig.rhs_order_exprs[1].delta == 3
    assert sig.order_delta == 0
