"""Tests for the ``parallel`` parser backend.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 5.

In ``parser_backend='parallel'`` mode WordSpace constructs the STM
driver alongside running the chart. The chart is authoritative — its
output is the returned ``current_rules`` — while the STM side is
present for inspection / future verification / training-stats
collection. STM errors propagate as well; this is intentional for
catching driver bugs early.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _bare_word_space():
    from Language import WordSpace
    import torch.nn as nn
    ws = object.__new__(WordSpace)
    nn.Module.__init__(ws)
    return ws


def _bare_conceptual_space(stm_dim=4, max_depth=8):
    from Spaces import ConceptualSpace
    import torch.nn as nn
    cs = object.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs._init_typed_stm(batch=1, max_depth=max_depth, dim=stm_dim)
    return cs


def _tiny_view():
    from Language import Grammar
    from embed import build_knowledge_section, KnowledgeView
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", tier='S'),
        g._parse_rule("NP3", "lower(DET, NP4)", tier='S'),
    ]
    g._configured = True
    return KnowledgeView(build_knowledge_section(g))


def test_parallel_constructs_stm_driver_before_chart_runs():
    """In parallel mode, the STM driver is constructed before chart
    execution. On a bare WordSubSpace the chart will fail (missing
    cursor / per-sentence state); we expect that failure, but the STM
    driver must already exist when the chart error surfaces."""
    ws = _bare_word_space()
    ws.parser_backend = 'parallel'
    ws.attach_knowledge(_tiny_view())
    cs = _bare_conceptual_space(stm_dim=4, max_depth=8)
    object.__setattr__(ws, 'conceptualSpace', cs)
    # Chart will fail on the bare instance — catch and verify stm_driver
    # was set up first.
    try:
        ws.compose(input_vectors=None)
    except Exception:
        pass
    assert ws.stm_driver is not None
    from stm_driver import STMDriver
    assert isinstance(ws.stm_driver, STMDriver)


def test_parallel_requires_knowledge_for_stm_side():
    """Without attached knowledge, parallel mode raises (STM side
    can't initialize)."""
    ws = _bare_word_space()
    ws.parser_backend = 'parallel'
    import pytest
    with pytest.raises(RuntimeError, match='knowledge'):
        ws.compose(input_vectors=None)


def test_parallel_generate_also_constructs_driver():
    """``generate()`` under parallel mode follows the same pattern."""
    ws = _bare_word_space()
    ws.parser_backend = 'parallel'
    ws.attach_knowledge(_tiny_view())
    cs = _bare_conceptual_space(stm_dim=4, max_depth=8)
    object.__setattr__(ws, 'conceptualSpace', cs)
    try:
        ws.generate(target_vectors=None)
    except Exception:
        pass
    assert ws.stm_driver is not None
