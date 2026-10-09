"""Tests confirming Grammar parses ``<start>S</start>`` from XML and
falls back sensibly when unset.

Plan reference: doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md §Verification
"""
import sys
import textwrap
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest


@pytest.fixture
def reset_grammar():
    """Snapshot/restore TheGrammar around tests that mutate config."""
    import Language
    import util
    saved_configured = Language.TheGrammar._configured
    saved_start = Language.TheGrammar.start_symbol
    saved_patterns = getattr(Language.TheGrammar, "start_patterns", (("S",),))
    saved_xml_root = getattr(util.TheXMLConfig, "_root", None)
    yield
    Language.TheGrammar._configured = False
    Language.TheGrammar.start_symbol = saved_start
    Language.TheGrammar.start_patterns = saved_patterns
    if saved_xml_root is not None:
        util.TheXMLConfig._root = saved_xml_root


def _load_inline_xml(xml_text):
    """Parse an inline model XML through TheXMLConfig and reset Grammar."""
    import tempfile
    import util
    import Language
    with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", delete=False) as f:
        f.write(xml_text)
        path = f.name
    util.TheXMLConfig.load(path)
    Language.TheGrammar._configured = False
    Language.TheGrammar._ensure_configured()
    return Language.TheGrammar


def test_start_symbol_parsed_from_inline_xml(reset_grammar):
    """``<start>S</start>`` populates ``Grammar.start_symbol``."""
    xml = textwrap.dedent("""\
        <?xml version="1.0" ?>
        <model>
          <SymbolSpace>
            <language>
              <start>S</start>
              <grammar>
                <S>not(S)</S>
                <S>join(S, S)</S>
              </grammar>
            </language>
          </SymbolSpace>
        </model>
    """)
    grammar = _load_inline_xml(xml)
    assert grammar.start_symbol == "S"


def test_start_symbol_default_without_xml_tag(reset_grammar):
    """No ``<start>`` tag → start_symbol defaults to ``"S"``."""
    xml = textwrap.dedent("""\
        <?xml version="1.0" ?>
        <model>
          <SymbolSpace>
            <language>
              <grammar>
                <S>not(S)</S>
              </grammar>
            </language>
          </SymbolSpace>
        </model>
    """)
    grammar = _load_inline_xml(xml)
    assert grammar.start_symbol == "S", (
        "missing <start> must fall back to the historical default 'S'"
    )


def test_start_symbol_alternate_root(reset_grammar):
    """An alternate start nonterminal name is preserved verbatim."""
    xml = textwrap.dedent("""\
        <?xml version="1.0" ?>
        <model>
          <SymbolSpace>
            <language>
              <start>ROOT</start>
              <grammar>
                <ROOT>not(ROOT)</ROOT>
              </grammar>
            </language>
          </SymbolSpace>
        </model>
    """)
    grammar = _load_inline_xml(xml)
    assert grammar.start_symbol == "ROOT"


def test_multiple_start_patterns_with_compact_order_set(reset_grammar):
    """Repeated ``<start>`` tags can define accepted unreduced forms."""
    xml = textwrap.dedent("""\
        <?xml version="1.0" ?>
        <model>
          <SymbolSpace>
            <language>
              <start>S45</start>
              <start>S45 REL S45</start>
              <grammar>
                <S4>not(S4)</S4>
                <S5>not(S5)</S5>
              </grammar>
            </language>
          </SymbolSpace>
        </model>
    """)
    grammar = _load_inline_xml(xml)
    assert grammar.start_symbol == "S4"
    assert grammar.start_patterns == (
        ("S4",),
        ("S5",),
        ("S4", "REL", "S4"),
        ("S5", "REL", "S5"),
    )
    assert grammar.is_start_pattern(("S4", "REL", "S4"))
    assert grammar.is_start_pattern(("S5", "REL", "S5"))


def test_start_symbol_persists_across_ensure_configured(reset_grammar):
    """Repeated ``_ensure_configured`` calls don't blow away start_symbol."""
    import Language
    xml = textwrap.dedent("""\
        <?xml version="1.0" ?>
        <model>
          <SymbolSpace>
            <language>
              <start>S</start>
              <grammar>
                <S>not(S)</S>
              </grammar>
            </language>
          </SymbolSpace>
        </model>
    """)
    grammar = _load_inline_xml(xml)
    grammar._ensure_configured()  # idempotent re-entry
    assert grammar.start_symbol == "S"
