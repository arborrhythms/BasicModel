"""Relative-rule detection over the role-collapsed relation set (Phase R1.3).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§6: relative-rule detection replaces the transitional
``{isEqual, queryPart, assertPart, part, REL_T}`` set with the canonical
``{equal, part, whole}`` structural family plus the WholeSpace relative-start
role states (the ``<start name="relative_truth">`` outputs). A relative
truth is a binary predicate end-state (the equal / part family); its serial
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
      <Symbolic>
        <start name="relative_truth">equal_O1</start>
        <start name="relative_truth">part_O1</start>
        <start name="absolute_truth">exist_O1</start>
        <compose>
          <rule>equal_O1 = equal.forward(equal_I1, equal_I2)</rule>
          <rule>part_O1 = part.forward(part_I1, part_I2)</rule>
          <rule>exist_O1 = exist.forward(exist_I1)</rule>
        </compose>
        <generate>
          <rule>equal_I1, equal_I2 = equal.reverse(equal_O1)</rule>
          <rule>part_I1, part_I2 = part.reverse(part_O1)</rule>
          <rule>exist_I1 = exist.reverse(exist_O1)</rule>
        </generate>
      </Symbolic>
    </grammar>
""")

# A back-compat grammar whose relative start is the bare ``REL_T`` symbol
# with NO name attribute (the literal-fallback path).
_REL_T_NO_NAME_GRAMMAR = textwrap.dedent("""\
    <?xml version="1.0"?>
    <grammar name="rel_t_probe">
      <Symbolic>
        <start>REL_T</start>
        <compose>
          <rule>REL_T = equal.forward(NP_A, NP_B)</rule>
        </compose>
        <generate>
          <rule>NP_A, NP_B = equal.reverse(REL_T)</rule>
        </generate>
      </Symbolic>
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


def test_relative_op_names_are_canonical_structural_forms():
    """The relative family is structural, not a legacy query alias table."""
    from Language import Grammar
    assert Grammar._RELATIVE_OP_NAMES == frozenset({"equal", "part", "whole"})


def test_relative_start_categories_from_named_starts(monkeypatch, tmp_path):
    """The ``relative_truth``-named SS starts are the relative start set;
    the ``absolute_truth`` start is excluded."""
    g = _load(_COLLAPSED_GRAMMAR, monkeypatch, tmp_path)
    assert g._relative_start_categories() == {"equal_O1", "part_O1"}


def test_equal_and_part_forward_rules_are_relative(monkeypatch, tmp_path):
    """Both relation families' forward rules flag relative; the absolute
    ``exist`` rule does not."""
    g = _load(_COLLAPSED_GRAMMAR, monkeypatch, tmp_path)
    by_lhs = {r.lhs: i for i, r in enumerate(g.rules_upward)}
    assert g.is_relative_rule(by_lhs["equal_O1"])
    assert g.is_relative_rule(by_lhs["part_O1"])
    assert not g.is_relative_rule(by_lhs["exist_O1"])


def test_part_rule_relative_by_op_name(monkeypatch, tmp_path):
    """A rule whose lhs is not a relative start is still flagged when its
    method is ``part`` (the op-name signal)."""
    g = _load(_COLLAPSED_GRAMMAR, monkeypatch, tmp_path)
    rel = g._relative_rule_id_set()
    part_reverse = [
        i for i, r in enumerate(g.rules)
        if r.method_name == "part" and r.lhs not in g.ws_relative_starts]
    assert part_reverse, "expected a part rule with a non-start lhs"
    for i in part_reverse:
        assert i in rel, (
            f"part rule {g.rules[i].canonical!r} not flagged relative")


def test_rel_t_back_compat_fallback(monkeypatch, tmp_path):
    """A bare ``<start>REL_T</start>`` (no name attribute) still yields the
    relative start via the literal-``REL_T`` fallback."""
    g = _load(_REL_T_NO_NAME_GRAMMAR, monkeypatch, tmp_path)
    assert g._relative_start_categories() == {"REL_T"}
    by_lhs = {r.lhs: i for i, r in enumerate(g.rules_upward)}
    assert g.is_relative_rule(by_lhs["REL_T"])
