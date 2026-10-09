"""Grammar identity no-op (S -> S) used only as the runtime grammatical
transition fired at padding columns of the static per-word loop.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §0.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ.setdefault("MODEL_COMPILE", "eager")

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "bin"))

import torch
import pytest


def _load_mm_grammar():
    """Load MM_grammar.xml and force a fresh Grammar configuration so
    ``TheGrammar.id_SS`` reflects the configured runtime no-op."""
    from util import init_config
    init_config(path=str(_root / "data" / "MM_grammar.xml"),
                defaults_path=str(_root / "data" / "model.xml"))
    from Language import TheGrammar
    TheGrammar._configured = False
    TheGrammar._ensure_configured()
    return TheGrammar


def test_id_SS_resolves_to_identity_rule():
    g = _load_mm_grammar()
    assert g.id_SS is not None, (
        "TheGrammar.id_SS must resolve after configure() so the static "
        "per-word loop has a padding no-op")
    rd = g.rules[g.id_SS]
    assert rd.lhs == g.start_symbol
    assert rd.rhs_symbols == (g.start_symbol,)
    assert rd.arity == 1
    assert rd.method_name is None


def test_syntactic_layer_execute_short_circuits_identity():
    g = _load_mm_grammar()
    from Language import SyntacticLayer
    id_SS = g.id_SS
    assert id_SS is not None
    layer = SyntacticLayer.__new__(SyntacticLayer)
    layer._by_name = {}
    layer.space_role = "SS"
    operand = torch.randn(2, 4)
    out = layer.execute(id_SS, operand)
    assert torch.equal(out, operand), (
        "execute(id_SS) must return the operand unchanged "
        "(method_name=None short-circuit, no layer lookup).")


def test_reverse_rule_derived_for_identity():
    g = _load_mm_grammar()
    assert g.id_SS is not None
    matched = [r for r in g.reverse_rules
               if r and r[1] == 'projectReverse'
               and g.start_symbol in r[0]]
    assert matched, (
        "The runtime identity no-op must auto-derive a projectReverse "
        "entry in TheGrammar.reverse_rules.")
