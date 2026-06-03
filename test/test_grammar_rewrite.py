"""Grammar-file rewrite: marker-helper removal + per-operator-position
categories under <SymbolicSpace>.

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md (Phase 8b,
"Grammar file rewrite"): the ``*_MARK`` categories and the copy/swap
MARKER helper rules are deleted; surface markers become learned and owned
by the operator (absorb/emit). Each operator-argument position becomes its
own category in the operator's namespace (e.g. CONJ_L45 / CONJ_R45).
Existing modification rules (lower(VP, PP), ...) are kept, restated under
<SymbolicSpace> with per-position categories.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def _complete_grammar():
    from Language import Grammar
    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    return g


def test_grammar_has_no_marker_helper_rules():
    """complete.grammar carries no ``*_MARK`` categories and no copy/swap
    MARKER-helper rules after the rewrite."""
    from Language import Grammar
    g = _complete_grammar()

    # (1) No surface-marker categories survive anywhere in the SS table.
    for r in g.rules:
        tokens = [c.strip() for c in str(r.lhs).split(',')]
        tokens += list(r.rhs_symbols or ())
        for tok in tokens:
            assert not tok.endswith("_MARK"), (
                f"marker category {tok!r} survives in rule {r.canonical!r}")

    # (2) copy/swap were the marker idiom; they are retired from the
    #     symbolic grammar (markers are now learned and owned by the op).
    methods = {r.method_name for r in g.rules if r.method_name}
    assert "copy" not in methods, "copy MARKER-helper rules must be deleted"
    assert "swap" not in methods, "swap MARKER-helper rules must be deleted"


def test_pp_modifies_vp_via_lower():
    """The PP-modifies-VP modification rule survives the rewrite: a lower
    rule that takes a VP and a PP and yields a VP (PP modifies VP)."""
    from Language import Grammar
    g = _complete_grammar()

    def names(syms):
        return tuple(Grammar._parse_category(s).name for s in syms)

    found = False
    for r in g.rules_upward:
        if r.method_name != "lower":
            continue
        lhs_name = Grammar._parse_category(r.lhs).name
        rhs_names = names(r.rhs_symbols or ())
        if lhs_name == "VP" and rhs_names == ("VP", "PP"):
            found = True
            break
    assert found, (
        "expected a 'VP = lower(VP, PP)' modification rule "
        "(PP modifies VP) in the rewritten complete.grammar")


def test_per_operator_position_categories_present():
    """Conjunction's operands are per-position categories (CONJ_L*/CONJ_R*),
    not the deleted ``S_CONJ*`` MARKER-helper category."""
    from Language import Grammar
    g = _complete_grammar()
    rhs_tokens = set()
    for r in g.rules:
        for s in (r.rhs_symbols or ()):
            rhs_tokens.add(s)
    # The conjunction operator names its argument positions.
    assert any(t.startswith("CONJ_L") for t in rhs_tokens), sorted(rhs_tokens)
    assert any(t.startswith("CONJ_R") for t in rhs_tokens), sorted(rhs_tokens)
    # The old MARKER-helper category is gone.
    assert not any(t.startswith("S_CONJ") for t in rhs_tokens)
