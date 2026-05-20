"""Tests for the STM backend's wiring into WordSpace.compose().

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 4.

Scope: under ``parser_backend='stm'``, compose() lazily constructs an
``STMDriver`` from the attached ``KnowledgeView``'s rule signatures
and the attached ConceptualSpace's ``stm_typed``. Returns an empty
rules dict for now (the full SHIFT / REDUCE loop on input_vectors is
deferred — this step ships the wiring and constructor path).
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _bare_word_space():
    """Bare WordSpace with the minimum nn.Module init."""
    from Language import WordSpace
    import torch.nn as nn
    ws = object.__new__(WordSpace)
    nn.Module.__init__(ws)
    return ws


def _bare_conceptual_space(stm_dim=4, max_depth=8):
    """Bare ConceptualSpace with an allocated stm_typed."""
    from Spaces import ConceptualSpace
    import torch.nn as nn
    cs = object.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs._init_typed_stm(batch=1, max_depth=max_depth, dim=stm_dim)
    return cs


def _tiny_view():
    """KnowledgeView from a fixture grammar."""
    from Language import Grammar
    from embed import build_knowledge_section, KnowledgeView
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", tier='S'),
        g._parse_rule("NP3", "lower(DET, NP4)", tier='S'),
    ]
    g._configured = True
    return KnowledgeView(build_knowledge_section(g))


def test_stm_backend_compose_requires_knowledge_attached():
    """Without ``attach_knowledge`` first, compose under stm raises."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    import pytest
    with pytest.raises(RuntimeError, match='knowledge'):
        ws.compose(input_vectors=None)


def test_stm_backend_compose_requires_conceptual_stm_typed():
    """Knowledge attached but no conceptualSpace.stm_typed → clear error."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    # No conceptualSpace at all → error
    import pytest
    with pytest.raises(RuntimeError, match='stm_typed'):
        ws.compose(input_vectors=None)


def test_stm_backend_compose_constructs_driver():
    """Knowledge + conceptualSpace.stm_typed → compose creates stm_driver."""
    from stm_driver import STMDriver
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    # Wire conceptualSpace with stm_typed
    cs = _bare_conceptual_space(stm_dim=4, max_depth=8)
    import torch.nn as nn
    # Use object.__setattr__ to avoid nn.Module submodule registration
    # (mirrors WordSpace's real __init__).
    object.__setattr__(ws, 'conceptualSpace', cs)
    rules = ws.compose(input_vectors=None)
    assert isinstance(rules, dict)
    assert hasattr(ws, 'stm_driver')
    assert isinstance(ws.stm_driver, STMDriver)


def test_stm_driver_uses_knowledge_rule_signatures():
    """The driver's rule_signatures come from the attached view."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    view = _tiny_view()
    ws.attach_knowledge(view)
    cs = _bare_conceptual_space(stm_dim=4, max_depth=8)
    object.__setattr__(ws, 'conceptualSpace', cs)
    ws.compose(input_vectors=None)
    assert ws.stm_driver.rule_signatures == view.rule_order_signatures


def test_stm_driver_scorer_payload_dim_matches_stm_typed():
    """Scorer is sized to match the ConceptualSpace's stm_typed dim."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    cs = _bare_conceptual_space(stm_dim=6, max_depth=8)
    object.__setattr__(ws, 'conceptualSpace', cs)
    ws.compose(input_vectors=None)
    assert ws.stm_driver.scorer.payload_dim == 6


def test_stm_driver_constructed_once_idempotent():
    """Repeated compose() under stm backend reuses the same driver."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    cs = _bare_conceptual_space(stm_dim=4, max_depth=8)
    object.__setattr__(ws, 'conceptualSpace', cs)
    ws.compose(input_vectors=None)
    driver_1 = ws.stm_driver
    ws.compose(input_vectors=None)
    assert ws.stm_driver is driver_1


def test_stm_backend_generate_constructs_driver():
    """generate() under stm backend also wires the driver (same path)."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    cs = _bare_conceptual_space(stm_dim=4, max_depth=8)
    object.__setattr__(ws, 'conceptualSpace', cs)
    rules = ws.generate(target_vectors=None)
    assert isinstance(rules, dict)
    assert hasattr(ws, 'stm_driver')
