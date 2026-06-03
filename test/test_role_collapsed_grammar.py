"""Role-collapsed grammar variant validation (Phase R2).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§4 + §10. ``data/role_collapsed.grammar`` declares ONLY operator roles
(``op_I<n>`` inputs, ``op_O1`` output) -- no parts of speech, no surface
markers, no category-rename projection rules. The D1 gate is met (the
participation learner drives a parser-recovering collapse, R4 /
``test_d1_pos_recovery_gate.py``), so this is now the DEFAULT mental-model
grammar (``MentalModel.xml``); the transitional ``complete.grammar`` is
retained as the compatibility baseline (and is what the D1 collapse is
measured on). These tests validate the role-collapsed contract.
"""

import os
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_ROOT, "bin")
_DATA = os.path.join(_ROOT, "data")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_GRAMMAR_FILE = os.path.join(_DATA, "role_collapsed.grammar")

# POS / category / transitional-role state names that MUST NOT survive
# the role collapse (Section 4.2).
_FORBIDDEN_STATE_TOKENS = {
    "NP", "NP3", "NP4", "NP34", "NP345", "AP", "AP4", "VP", "VP1",
    "N3", "DET", "ADV", "ADJ", "P", "V1", "S3", "S4", "S5", "S34",
    "S45", "S345", "MP1", "PP", "REL_T", "ABS_T",
    "CONJ_L45", "CONJ_R45", "DISJ_L45", "DISJ_R45", "CONJ_L3", "CONJ_R3",
    "DISJ_L3", "DISJ_R3", "QLEFT_NP3", "QRIGHT_AP", "QRIGHT_NP3",
    "QRIGHT_S34", "NP_EQ3", "NP_EQ4", "NP_EQ345", "S_PART34",
    "QLEFT_PART34",
}

_REQUIRED_OPS = {
    "exist", "isEqual", "isPart", "not", "non",
    "conjunction", "disjunction", "intersection", "union",
    "lift", "lower",
}


def _load():
    from Language import Grammar
    g = Grammar()
    g.load_from_grammar_file("role_collapsed.grammar")
    return g


def _rule_tokens(rule):
    from Language import Grammar
    toks = [c.strip() for c in str(rule.lhs).split(",")]
    toks += list(rule.rhs_symbols or ())
    return [t for t in toks if t]


def test_grammar_loads():
    """The variant loads through the standard grammar path."""
    g = _load()
    assert len(g.rules) > 0
    assert len(g.ps_rules) > 0


def test_perceptualspace_has_everything_start():
    """PerceptualSpace carries the analyzer root ``<start>U</start>``."""
    g = _load()
    assert g.ps_start_symbol == "U"
    root = ET.parse(_GRAMMAR_FILE).getroot()
    ps = root.find("PerceptualSpace")
    assert ps is not None
    starts = [(s.get("name"), (s.text or "").strip()) for s in ps.findall("start")]
    assert ("everything", "U") in starts, starts


def test_symbolicspace_owns_its_starts():
    """SymbolicSpace owns the operator-output starts, split by name."""
    g = _load()
    ss_syms = {sym for pat in g.ss_start_patterns for sym in pat}
    assert {"isEqual_O1", "isPart_O1", "exist_O1"} <= ss_syms, ss_syms
    assert g.ss_relative_starts == frozenset({"isEqual_O1", "isPart_O1"})
    assert "exist_O1" in g.ss_absolute_starts


def test_no_top_level_start():
    """No grammar-wide top-level ``<start>`` (decision 7): starts are
    nested under PerceptualSpace / SymbolicSpace only."""
    root = ET.parse(_GRAMMAR_FILE).getroot()
    assert root.find("start") is None, (
        "role-collapsed grammar must not declare a top-level <start>")


def test_no_query_part_or_assert_part():
    """``queryPart`` / ``assertPart`` are folded into ``isPart`` + query."""
    g = _load()
    methods = {r.method_name for r in g.rules if r.method_name}
    assert "queryPart" not in methods
    assert "assertPart" not in methods
    assert "part" not in methods


def test_no_pos_or_category_state_names():
    """No POS / category / transitional-role state names survive."""
    g = _load()
    for r in g.rules:
        for tok in _rule_tokens(r):
            assert tok not in _FORBIDDEN_STATE_TOKENS, (
                f"forbidden state name {tok!r} in rule {r.canonical!r}")


def test_no_category_rename_projection_rules():
    """No bare ``X = Y`` POS-rename rules: every method-less rule is the
    injected identity no-op (``X = X``)."""
    g = _load()
    for r in g.rules:
        if r.method_name is None:
            assert r.rhs_symbols == (r.lhs,), (
                f"non-identity projection rule survives: {r.canonical!r}")


def test_isequal_and_ispart_use_role_names():
    """``isEqual`` / ``isPart`` operands are the collapsed I/O roles."""
    g = _load()
    for op in ("isEqual", "isPart"):
        fwd = [r for r in g.rules_upward if r.method_name == op]
        assert fwd, f"no forward {op} rule"
        r = fwd[0]
        assert r.lhs == f"{op}_O1"
        assert r.rhs_symbols == (f"{op}_I1", f"{op}_I2")


def test_every_relation_rule_carries_explicit_query():
    """Every ``isEqual`` / ``isPart`` ``<rule>`` declares ``query`` (decision 6)."""
    root = ET.parse(_GRAMMAR_FILE).getroot()
    for rule in root.iter("rule"):
        body = (rule.text or "")
        if "isEqual." in body or "isPart." in body:
            assert rule.get("query") is not None, (
                f"relation rule missing explicit query: {body.strip()!r}")


def test_operator_coverage():
    """All required operators are present (Section 4.3)."""
    g = _load()
    methods = {r.method_name for r in g.rules if r.method_name}
    missing = _REQUIRED_OPS - methods
    assert not missing, f"role-collapsed grammar missing ops: {missing}"


def test_forward_reverse_pairing():
    """Every compose operator has a matching generate (reverse) operator."""
    g = _load()
    up = {r.method_name for r in g.rules_upward if r.method_name}
    dn = {r.method_name for r in g.rules_downward if r.method_name}
    assert _REQUIRED_OPS <= up
    assert _REQUIRED_OPS <= dn


def test_transitional_grammar_unchanged():
    """The transitional ``complete.grammar`` is retained UNCHANGED as the
    compatibility baseline (still POS-categoried, still spells
    assertPart/queryPart) even though role-collapse is now the default."""
    from Language import Grammar
    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    methods = {r.method_name for r in g.rules if r.method_name}
    assert "assertPart" in methods or "queryPart" in methods, (
        "complete.grammar must remain unchanged as the compatibility baseline")
