"""Space-scoped grammar starts (Phase R1.1 / R1.4).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
decision 7 + §4.4 + §6: ``PartSpace.start`` configures the PS starts
(the analyzer root ``U``) and ``WholeSpace.start`` configures the SS
starts (the operator outputs that count as completed expressions). There
is no grammar-wide top-level ``<start>`` in the role-collapsed grammar;
the global ``start_symbol`` / ``start_patterns`` alias the *symbolic*
starts (the symbolic parse is what id_SS / is_start_pattern / relative
detection operate on). The ``<start name=...>`` attribute is retained so
``relative_truth`` / ``absolute_truth`` starts can be told apart.
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


_SCOPED_STARTS_GRAMMAR = textwrap.dedent("""\
    <?xml version="1.0"?>
    <grammar name="scoped_starts_probe">
      <PartSpace>
        <start name="everything">U</start>
        <compose>
          <rule>U = boundary.forward(U, U)</rule>
        </compose>
        <generate>
          <rule>U, U = boundary.reverse(U)</rule>
        </generate>
      </PartSpace>
      <WholeSpace>
        <start name="relative_truth">isEqual_O1</start>
        <start name="absolute_truth">exist_O1</start>
        <compose>
          <rule>isEqual_O1 = isEqual.forward(isEqual_I1, isEqual_I2)</rule>
          <rule>exist_O1 = exist.forward(exist_I1)</rule>
        </compose>
        <generate>
          <rule>isEqual_I1, isEqual_I2 = isEqual.reverse(isEqual_O1)</rule>
          <rule>exist_I1 = exist.reverse(exist_O1)</rule>
        </generate>
      </WholeSpace>
    </grammar>
""")

# A legacy grammar with only a top-level <start> (no PS/SS-nested start).
_TOP_LEVEL_START_GRAMMAR = textwrap.dedent("""\
    <?xml version="1.0"?>
    <grammar name="top_start_probe">
      <start>S</start>
      <WholeSpace>
        <compose>
          <rule>S = conjunction.forward(S, S)</rule>
        </compose>
        <generate>
          <rule>S, S = conjunction.reverse(S)</rule>
        </generate>
      </WholeSpace>
    </grammar>
""")


def _load_grammar_text(text, monkeypatch, tmp_path):
    import Language
    path = tmp_path / "probe.grammar"
    path.write_text(text)
    monkeypatch.setattr(Language, "_GRAMMAR_DIR", tmp_path)
    g = Language.Grammar()
    g.load_from_grammar_file("probe.grammar")
    return g


def test_ps_start_scoped_to_perceptual_space(monkeypatch, tmp_path):
    """``<start>U</start>`` under ``<PartSpace>`` configures PS starts."""
    g = _load_grammar_text(_SCOPED_STARTS_GRAMMAR, monkeypatch, tmp_path)
    assert g.ps_start_symbol == "U"
    assert ("U",) in g.ps_start_patterns


def test_ss_start_scoped_to_symbolic_space(monkeypatch, tmp_path):
    """``<start>`` under ``<WholeSpace>`` configures SS starts; the PS
    root ``U`` does NOT leak into the symbolic start set."""
    g = _load_grammar_text(_SCOPED_STARTS_GRAMMAR, monkeypatch, tmp_path)
    assert g.ss_start_symbol == "isEqual_O1"
    assert ("isEqual_O1",) in g.ss_start_patterns
    assert ("exist_O1",) in g.ss_start_patterns
    assert ("U",) not in g.ss_start_patterns


def test_global_start_aliases_symbolic_space(monkeypatch, tmp_path):
    """The back-compat global ``start_symbol`` / ``start_patterns`` mirror
    the WholeSpace starts (the symbolic parse), not the PS root."""
    g = _load_grammar_text(_SCOPED_STARTS_GRAMMAR, monkeypatch, tmp_path)
    assert g.start_symbol == g.ss_start_symbol == "isEqual_O1"
    assert tuple(g.start_patterns) == tuple(g.ss_start_patterns)
    flat = {sym for pat in g.start_patterns for sym in pat}
    assert "U" not in flat


def test_relative_and_absolute_starts_from_name_attribute(monkeypatch, tmp_path):
    """The ``name`` attribute distinguishes relative-truth from
    absolute-truth SS starts (used by relative-rule detection, R1.3)."""
    g = _load_grammar_text(_SCOPED_STARTS_GRAMMAR, monkeypatch, tmp_path)
    assert g.ss_relative_starts == frozenset({"isEqual_O1"})
    assert g.ss_absolute_starts == frozenset({"exist_O1"})


def test_identity_rule_uses_symbolic_start(monkeypatch, tmp_path):
    """R1.4: the injected identity no-op is on the WholeSpace start, not
    the PartSpace root ``U``."""
    g = _load_grammar_text(_SCOPED_STARTS_GRAMMAR, monkeypatch, tmp_path)
    assert g.id_SS is not None
    rd = g.rules[g.id_SS]
    assert rd.lhs == "isEqual_O1"
    assert rd.rhs_symbols == ("isEqual_O1",)
    assert rd.arity == 1
    assert rd.method_name is None


def test_top_level_start_back_compat(monkeypatch, tmp_path):
    """A grammar with only a top-level ``<start>`` still configures the SS
    (and global alias) start; PS starts stay empty."""
    g = _load_grammar_text(_TOP_LEVEL_START_GRAMMAR, monkeypatch, tmp_path)
    assert g.ss_start_symbol == "S"
    assert g.start_symbol == "S"
    assert g.ps_start_symbol is None
    assert tuple(g.ps_start_patterns) == ()
