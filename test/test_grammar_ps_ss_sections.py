"""Loader support for <PartSpace>/<WholeSpace> grammar sections.

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md
(Phase 8b / execution-manifest step 3): ``Grammar.load_from_grammar_file``
parses the two sections into separate PS/SS rule tables. Backward-compat:
a file with bare ``<compose>``/``<generate>`` loads as ``<WholeSpace>``.
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


_PS_SS_GRAMMAR = textwrap.dedent("""\
    <?xml version="1.0"?>
    <grammar name="ps_ws_probe">
      <start>S</start>
      <PartSpace>
        <compose>
          <rule>WORD = boundary.forward(WORD, WORD)</rule>
        </compose>
        <generate>
          <rule>WORD, WORD = boundary.reverse(WORD)</rule>
        </generate>
      </PartSpace>
      <WholeSpace>
        <compose>
          <rule>S = conjunction.forward(CONJ_L, CONJ_R)</rule>
        </compose>
        <generate>
          <rule>CONJ_L, CONJ_R = conjunction.reverse(S)</rule>
        </generate>
      </WholeSpace>
    </grammar>
""")

_BARE_GRAMMAR = textwrap.dedent("""\
    <?xml version="1.0"?>
    <grammar name="bare_probe">
      <start>S</start>
      <compose>
        <rule>S = conjunction.forward(S, S)</rule>
      </compose>
      <generate>
        <rule>S, S = conjunction.reverse(S)</rule>
      </generate>
    </grammar>
""")


def _load_grammar_text(text, monkeypatch, tmp_path):
    """Write ``text`` to a temp ``.grammar`` file and load it through the
    public ``load_from_grammar_file`` path (monkeypatching ``_GRAMMAR_DIR``)."""
    import Language
    path = tmp_path / "probe.grammar"
    path.write_text(text)
    monkeypatch.setattr(Language, "_GRAMMAR_DIR", tmp_path)
    g = Language.Grammar()
    g.load_from_grammar_file("probe.grammar")
    return g


def test_grammar_loads_ps_and_ws_sections(monkeypatch, tmp_path):
    """PS and SS sections parse into separate, disjoint rule tables."""
    g = _load_grammar_text(_PS_SS_GRAMMAR, monkeypatch, tmp_path)

    # SS section -> the parser's rule table (g.rules / g.ws_rules).
    ws_methods = {r.method_name for r in g.ws_rules if r.method_name}
    assert "conjunction" in ws_methods

    # PS section -> a separate ps_rules table, tagged tier 'P'.
    assert len(g.ps_rules) > 0
    ps_methods = {r.method_name for r in g.ps_rules if r.method_name}
    assert "boundary" in ps_methods
    assert all(r.tier == 'P' for r in g.ps_rules), (
        [(r.method_name, r.tier) for r in g.ps_rules])

    # The two tables are disjoint: no PS rule leaks into the SS table.
    assert "boundary" not in ws_methods
    assert all(r.method_name != "boundary" for r in g.rules)

    # Both directions of the PS section land (compose + generate).
    assert len(g.ps_rules_upward) > 0
    assert len(g.ps_rules_downward) > 0


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def test_cfg_loader_removed():
    """The legacy ``.cfg`` grammar path is gone: loader code + files.

    doc/plans/2026-05-30-...-terminal-emitter.md "Remove legacy .cfg":
    ``load_from_cfg`` / ``_parse_cfg_lines`` and the grammarCfg branch are
    deleted from bin/Language.py, and data/grammar2.cfg /
    data/grammar_legacy.cfg are removed.
    """
    import Language
    assert not hasattr(Language.Grammar, "load_from_cfg")
    assert not hasattr(Language.Grammar, "_parse_cfg_lines")
    assert not hasattr(Language, "_parse_cfg_lines")
    assert not os.path.exists(os.path.join(_DATA_DIR, "grammar2.cfg"))
    assert not os.path.exists(os.path.join(_DATA_DIR, "grammar_legacy.cfg"))


def test_bare_compose_generate_loads_as_symbolic_space(monkeypatch, tmp_path):
    """Backward-compat: a bare <compose>/<generate> file == WholeSpace."""
    g = _load_grammar_text(_BARE_GRAMMAR, monkeypatch, tmp_path)
    assert len(g.ps_rules) == 0
    ws_methods = {r.method_name for r in g.ws_rules if r.method_name}
    assert "conjunction" in ws_methods
    # ws_rules is the same table the existing parser reads.
    assert list(g.ws_rules) == list(g.rules)
