"""Regression guard: legacy symbols must not resurface in live modules."""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest
import Models
import Spaces


def _file_text(module):
    with open(module.__file__, "r") as f:
        return f.read()


_FORBIDDEN = (
    ("Models._forward_sequential", lambda: "_forward_sequential" in _file_text(Models)),
    ("Models._forward_legacy",     lambda: "_forward_legacy"     in _file_text(Models)),
    ("Models._reverse_legacy",     lambda: "_reverse_legacy"     in _file_text(Models)),
    ("Models._run_subsymbolic_order",  lambda: "_run_subsymbolic_order"  in _file_text(Models)),
    ("Models._run_forward_pipeline",  lambda: "_run_forward_pipeline"  in _file_text(Models)),
    ("Models._start_ar_forward",   lambda: "_start_ar_forward"   in _file_text(Models)),
    ("Spaces._forward_legacy",    lambda: "_forward_legacy"    in _file_text(Spaces)),
    ("Spaces._reverse_legacy",    lambda: "_reverse_legacy"    in _file_text(Spaces)),
    ("Spaces._forward_dispatch",  lambda: "_forward_dispatch"  in _file_text(Spaces)),
    # Phase 2: hierarchical level-views + additive feedback glue removed.
    ("Spaces._CSLevelView",         lambda: "_CSLevelView"         in _file_text(Spaces)),
    ("Spaces._SSLevelView",         lambda: "_SSLevelView"         in _file_text(Spaces)),
    ("Spaces._symbol_objective_terms", lambda: "_symbol_objective_terms" in _file_text(Spaces)),
    ("Spaces.symbol_objective_terms",  lambda: "symbol_objective_terms"  in _file_text(Spaces)),
    ("Spaces.reset_symbol_objective",  lambda: "reset_symbol_objective"  in _file_text(Spaces)),
    ("Spaces.accumulate_symbol_objective", lambda: "accumulate_symbol_objective" in _file_text(Spaces)),
    ("Spaces.SentencePrimingLayer",  lambda: "class SentencePrimingLayer" in _file_text(Spaces)),
)


@pytest.mark.parametrize("name,check", _FORBIDDEN, ids=[n for n, _ in _FORBIDDEN])
def test_legacy_symbol_absent(name, check):
    assert not check(), (
        f"{name} has resurfaced. If you genuinely need to re-add it, "
        "delete this guard with a note in the commit message.")
