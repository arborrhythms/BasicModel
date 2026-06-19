"""Relative-rule detection over the role-collapsed relation set (Phase R1.3).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§6: relative-rule detection replaces the transitional
``{isEqual, queryPart, assertPart, part, REL_T}`` set with
``{isEqual, isPart}`` plus the WholeSpace relative-start role states
(the ``<start name="relative_truth">`` outputs). A relative truth is a
binary predicate end-state (the isEqual / isPart family); its serial
sentence-boundary reduce stops at the depth-3 ``[predicate, idea1, idea2]``
state rather than collapsing to a single idea.
"""

import os
import sys
import textwrap

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


_COLLAPSED_GRAMMAR = textwrap.dedent("""\
    <?xml version="1.0"?>
    <grammar name="rel_collapse_probe">
      <WholeSpace>
        <start name="relative_truth">isEqual_O1</start>
        <start name="relative_truth">isPart_O1</start>
        <start name="absolute_truth">exist_O1</start>
        <compose>
          <rule query="true">isEqual_O1 = isEqual.forward(isEqual_I1, isEqual_I2)</rule>
          <rule query="false">isPart_O1 = isPart.forward(isPart_I1, isPart_I2)</rule>
          <rule>exist_O1 = exist.forward(exist_I1)</rule>
        </compose>
        <generate>
          <rule query="true">isEqual_I1, isEqual_I2 = isEqual.reverse(isEqual_O1)</rule>
          <rule query="false">isPart_I1, isPart_I2 = isPart.reverse(isPart_O1)</rule>
          <rule>exist_I1 = exist.reverse(exist_O1)</rule>
        </generate>
      </WholeSpace>
    </grammar>
""")

# A back-compat grammar whose relative start is the bare ``REL_T`` symbol
# with NO name attribute (the literal-fallback path).
_REL_T_NO_NAME_GRAMMAR = textwrap.dedent("""\
    <?xml version="1.0"?>
    <grammar name="rel_t_probe">
      <WholeSpace>
        <start>REL_T</start>
        <compose>
          <rule>REL_T = isEqual.forward(NP_A, NP_B)</rule>
        </compose>
        <generate>
          <rule>NP_A, NP_B = isEqual.reverse(REL_T)</rule>
        </generate>
      </WholeSpace>
    </grammar>
""")


def _load(text, monkeypatch, tmp_path):
    import Language
    path = tmp_path / "probe.grammar"
    path.write_text(text)
    monkeypatch.setattr(Language, "_GRAMMAR_DIR", tmp_path)
    g = Language.Grammar()
    g.load_from_grammar_file("probe.grammar")
    return g


def test_relative_op_names_are_isequal_ispart():
    """The op-name set collapses to ``{isEqual, isPart}``; the retired
    ``queryPart`` / ``assertPart`` / ``part`` names are gone."""
    from Language import Grammar
    assert Grammar._RELATIVE_OP_NAMES == frozenset({"isEqual", "isPart"})


def test_relative_start_categories_from_named_starts(monkeypatch, tmp_path):
    """The ``relative_truth``-named SS starts are the relative start set;
    the ``absolute_truth`` start is excluded."""
    g = _load(_COLLAPSED_GRAMMAR, monkeypatch, tmp_path)
    assert g._relative_start_categories() == {"isEqual_O1", "isPart_O1"}


def test_isequal_and_ispart_forward_rules_are_relative(monkeypatch, tmp_path):
    """Both relation families' forward rules flag relative; the absolute
    ``exist`` rule does not."""
    g = _load(_COLLAPSED_GRAMMAR, monkeypatch, tmp_path)
    by_lhs = {r.lhs: i for i, r in enumerate(g.rules_upward)}
    assert g.is_relative_rule(by_lhs["isEqual_O1"])
    assert g.is_relative_rule(by_lhs["isPart_O1"])
    assert not g.is_relative_rule(by_lhs["exist_O1"])


def test_ispart_rule_relative_by_op_name(monkeypatch, tmp_path):
    """A rule whose lhs is not a relative start is still flagged when its
    method is ``isPart`` (the op-name signal)."""
    g = _load(_COLLAPSED_GRAMMAR, monkeypatch, tmp_path)
    rel = g._relative_rule_id_set()
    ispart_reverse = [
        i for i, r in enumerate(g.rules)
        if r.method_name == "isPart" and r.lhs not in g.ws_relative_starts]
    assert ispart_reverse, "expected an isPart rule with a non-start lhs"
    for i in ispart_reverse:
        assert i in rel, (
            f"isPart rule {g.rules[i].canonical!r} not flagged relative")


def test_rel_t_back_compat_fallback(monkeypatch, tmp_path):
    """A bare ``<start>REL_T</start>`` (no name attribute) still yields the
    relative start via the literal-``REL_T`` fallback."""
    g = _load(_REL_T_NO_NAME_GRAMMAR, monkeypatch, tmp_path)
    assert g._relative_start_categories() == {"REL_T"}
    by_lhs = {r.lhs: i for i, r in enumerate(g.rules_upward)}
    assert g.is_relative_rule(by_lhs["REL_T"])
