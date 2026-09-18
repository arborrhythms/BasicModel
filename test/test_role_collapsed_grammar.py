"""Role-only grammar format validation (Phase R2).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§4 + §10. Live grammars declare ONLY operator roles (``op_I<n>`` inputs,
``op_O1`` output) -- no parts of speech, no surface markers, no
category-rename projection rules. The retired standalone role-collapse file
was absorbed into ``complete.grammar``; these tests validate that contract on
the broad live grammar and sweep the remaining data grammars.
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

_GRAMMAR_NAME = "complete.grammar"
_GRAMMAR_FILE = os.path.join(_DATA, _GRAMMAR_NAME)

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

# The relation family is structural: ``part`` / ``whole`` / ``equal`` own
# compose/generate faces and join their checked thought descriptors only at a
# completed boundary.
_REQUIRED_OPS = {
    "part", "whole", "equal", "exist", "lookup", "quantize", "arma", "what",
    "not", "non",
    "conjunction", "disjunction", "intersection", "union",
    "lift", "verb", "adverb", "lower",
    "preposition", "bind", "tense", "morphology",
}


def _load():
    from Language import Grammar
    g = Grammar()
    g.load_from_grammar_file(_GRAMMAR_NAME)
    return g


def _rule_tokens(rule):
    from Language import Grammar
    toks = [c.strip() for c in str(rule.lhs).split(",")]
    toks += list(rule.rhs_symbols or ())
    return [t for t in toks if t]


def test_grammar_loads():
    """The broad role-only grammar loads through the standard grammar path."""
    g = _load()
    assert len(g.rules) > 0
    assert len(g.ps_rules) > 0


def test_perceptualspace_has_everything_start():
    """PartSpace carries the analyzer root ``<start>U</start>``."""
    g = _load()
    assert g.ps_start_symbol == "U"
    root = ET.parse(_GRAMMAR_FILE).getroot()
    ps = root.find("PartSpace")
    assert ps is not None
    starts = [(s.get("name"), (s.text or "").strip()) for s in ps.findall("start")]
    assert ("everything", "U") in starts, starts


def test_symbolicspace_owns_its_starts():
    """WholeSpace owns the operator-output starts, split by name."""
    g = _load()
    ws_syms = {sym for pat in g.ws_start_patterns for sym in pat}
    # part / whole / equal are compositional relation forms heading the
    # relative-truth start; exist_O1 remains the absolute-truth start.
    assert "exist_O1" in ws_syms, ws_syms
    assert g.ws_relative_starts == frozenset({"part_O1", "whole_O1", "equal_O1"})
    assert "exist_O1" in g.ws_absolute_starts


def test_no_top_level_start():
    """No grammar-wide top-level ``<start>`` (decision 7): starts are
    nested under PartSpace / WholeSpace only."""
    root = ET.parse(_GRAMMAR_FILE).getroot()
    assert root.find("start") is None, (
        "role-only grammars must not declare a top-level <start>")


def test_no_retired_relation_or_query_aliases():
    """One structural family replaces all legacy relation/query spellings."""
    g = _load()
    methods = {r.method_name for r in g.rules if r.method_name}
    assert not (methods & {"queryPart", "assertPart", "isPart", "isEqual",
                           "queryEqual", "query"})
    assert {"part", "whole", "equal"} <= methods


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


def test_canonical_relation_forms_use_role_names():
    """The grammar uses canonical role-labelled relation forms."""
    g = _load()
    for op in ("part", "whole", "equal"):
        rules = [r for r in g.rules if r.method_name == op]
        assert rules, f"{op} must be a structural grammar form"
        assert all(op + "_" in r.canonical for r in rules)


def test_every_relation_rule_has_no_query_attribute():
    """Interrogative mode is on the completed meaning, never the rule."""
    root = ET.parse(_GRAMMAR_FILE).getroot()
    for rule in root.iter("rule"):
        body = (rule.text or "")
        if any(name + "." in body for name in ("part", "whole", "equal")):
            assert rule.get("query") is None, (
                f"relation rule retains retired query attribute: {body.strip()!r}")


def test_operator_coverage():
    """All required operators are present (Section 4.3)."""
    g = _load()
    methods = {r.method_name for r in g.rules if r.method_name}
    missing = _REQUIRED_OPS - methods
    assert not missing, f"{_GRAMMAR_NAME} missing ops: {missing}"


def test_forward_reverse_pairing():
    """Every reversible compose form has its matching generate face."""
    g = _load()
    up = {r.method_name for r in g.rules_upward if r.method_name}
    dn = {r.method_name for r in g.rules_downward if r.method_name}
    assert _REQUIRED_OPS <= up
    # ``lookup`` is intentionally compose-only: retrieval has no faithful
    # structural inverse.
    assert (_REQUIRED_OPS - {"lookup"}) <= dn


def test_transitional_baseline_uses_the_same_structural_relation_contract():
    """The POS baseline remains useful without reviving a query catalogue."""
    from Language import Grammar
    fixture = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures",
        "transitional_pos.grammar")
    g = Grammar()
    g.load_from_grammar_file(fixture)
    methods = {r.method_name for r in g.rules if r.method_name}
    assert {"part", "equal"} <= methods
    assert not (methods & {"assertPart", "queryPart", "isPart", "isEqual"})


# -- Per-grammar-file format conformance (GrammarOpsPass §1) --------------
#
# Every LIVE grammar under data/ conforms to the role-only format.
# The transitional POS-categoried baseline lives under test/fixtures/
# (test data for the D1 / participation measurements), not under data/.

import glob

import pytest

_ALL_GRAMMAR_FILES = sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(_DATA, "*.grammar")))

# Canonical structural forms only; query aliases are not a grammar API.
_RETIRED_METHOD_NAMES = {"queryPart", "assertPart", "isPart", "isEqual",
                         "queryEqual", "query"}

# Relation families whose rules must carry no retired query attribute.
_RELATION_DOTTED = ("isEqual.", "isPart.", "part.", "whole.", "equal.")


def test_sweep_covers_data_grammars():
    """The sweep sees the live grammar set (complete.grammar included)."""
    assert "complete.grammar" in _ALL_GRAMMAR_FILES
    assert "role_collapsed.grammar" not in _ALL_GRAMMAR_FILES
    assert len(_ALL_GRAMMAR_FILES) >= 4, _ALL_GRAMMAR_FILES


@pytest.mark.parametrize("fname", _ALL_GRAMMAR_FILES)
def test_grammar_file_conforms_to_role_collapsed_format(fname):
    """GrammarOpsPass §1 conformance, per grammar file: space-scoped
    starts only (no top-level <start>), PS analyzer root ``U``, explicit
    no ``query`` on relation rules, no retired relation/query method
    spellings, no POS / category / transitional-role state names, and no
    category-rename projection rules (method-less rules are identities)."""
    from Language import Grammar
    path = os.path.join(_DATA, fname)
    root = ET.parse(path).getroot()

    # Space-scoped starts only (decision 7).
    assert root.find("start") is None, (
        f"{fname}: top-level <start> (must be space-scoped)")

    # A PS section declares the analyzer root U.
    ps = root.find("PartSpace")
    if ps is not None:
        starts = [(s.get("name"), (s.text or "").strip())
                  for s in ps.findall("start")]
        assert ("everything", "U") in starts, (fname, starts)

    # Relation rules are structural; their completed ideas own mode.
    for rule in root.iter("rule"):
        body = (rule.text or "")
        if any(tag in body for tag in _RELATION_DOTTED):
            assert rule.get("query") is None, (
                f"{fname}: relation rule retains retired query attribute: "
                f"{body.strip()!r}")

    g = Grammar()
    g.load_from_grammar_file(fname)
    methods = {r.method_name for r in g.rules if r.method_name}
    assert not (methods & _RETIRED_METHOD_NAMES), (
        f"{fname}: retired method spellings {methods & _RETIRED_METHOD_NAMES}")
    for r in g.rules:
        for tok in _rule_tokens(r):
            assert tok not in _FORBIDDEN_STATE_TOKENS, (
                f"{fname}: forbidden state name {tok!r} in {r.canonical!r}")
        if r.method_name is None:
            assert r.rhs_symbols == (r.lhs,), (
                f"{fname}: non-identity projection rule {r.canonical!r}")
