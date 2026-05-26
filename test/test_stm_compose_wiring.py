"""Tests for the STM backend's wiring into WordSpace.compose().

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 4. Updated 2026-05-21 (Phase D refactor): the typed
STM stack data now lives directly on ``WordSubSpace`` instead of on
``ConceptualSpace.stm_typed``; the driver still constructs lazily
on first ``compose()`` / ``generate()`` call under
``parser_backend='stm'``.

Scope: under ``parser_backend='stm'``, compose() lazily constructs an
``STMDriver`` from the attached ``KnowledgeView``'s rule signatures
and operates on WordSubSpace's own typed-STM buffers. Returns an
empty rules dict for now (the full SHIFT / REDUCE loop on
input_vectors is deferred — this step ships the wiring and
constructor path).
"""
import sys
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _bare_word_subspace(stm_dim=4, max_depth=8):
    """Bare WordSubSpace with the minimum nn.Module init, its own
    typed-STM buffers, and a stub conceptualSpace exposing a real
    ``ShortTermMemory`` Layer.

    Bypasses the real ``__init__`` (chart / layers / grammar) since
    these tests only exercise the STM-driver wiring, but post-Phase-E
    the driver lives on ``conceptualSpace.stm`` so we wire a minimal
    one up.
    """
    from Language import WordSubSpace
    from Layers import ShortTermMemory
    import torch.nn as nn
    ws = object.__new__(WordSubSpace)
    nn.Module.__init__(ws)
    ws.batch = 1
    ws._stm_capacity = int(max_depth)
    ws._stm_payload_dim = int(stm_dim)
    ws.max_depth = ws._stm_capacity
    ws.dim = ws._stm_payload_dim
    ws.register_buffer(
        '_buffer', torch.zeros(ws.batch, max_depth, stm_dim),
        persistent=False)
    ws.register_buffer(
        '_category',
        torch.full((ws.batch, max_depth), -1, dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_order',
        torch.zeros((ws.batch, max_depth), dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_ref_id',
        torch.full((ws.batch, max_depth), -1, dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_depth', torch.zeros(ws.batch, dtype=torch.long),
        persistent=False)
    ws._category_names = [
        [None] * max_depth for _ in range(ws.batch)]
    # Idea-stack buffers (Phase E completion of the 2026-05-21 refactor).
    ws._idea_capacity = max_depth
    ws._idea_max_depth_host = 0
    ws.register_buffer(
        '_idea_buffer',
        torch.zeros(ws.batch, max_depth, stm_dim),
        persistent=False)
    ws.register_buffer(
        '_idea_depth',
        torch.zeros(ws.batch, dtype=torch.long),
        persistent=False)
    # Stub conceptualSpace exposing a ShortTermMemory Layer.
    class _StubCS:
        pass
    cs = _StubCS()
    cs.stm = ShortTermMemory(batch=1, capacity=max_depth,
                             concept_dim=stm_dim)
    object.__setattr__(ws, 'conceptualSpace', cs)
    return ws


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
    ws = _bare_word_subspace()
    ws.parser_backend = 'stm'
    import pytest
    with pytest.raises(RuntimeError, match='knowledge'):
        ws.compose(input_vectors=None)


def test_stm_backend_compose_constructs_driver():
    """Knowledge + WordSubSpace's typed STM buffers → compose
    initialises the rule scorer on the conceptualSpace's ShortTermMemory
    Layer. Post-Phase-E ``ws.stm_driver`` is the ShortTermMemory itself
    (compatibility accessor); ``stm.scorer`` is the lazily-allocated
    rule MLP."""
    from Layers import ShortTermMemory
    ws = _bare_word_subspace(stm_dim=4, max_depth=8)
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    rules = ws.compose(input_vectors=None)
    assert isinstance(rules, dict)
    assert hasattr(ws, 'stm_driver')
    assert isinstance(ws.stm_driver, ShortTermMemory)
    assert ws.stm_driver.scorer is not None


def test_stm_driver_uses_knowledge_rule_signatures():
    """The driver's rule_signatures come from the attached view."""
    ws = _bare_word_subspace(stm_dim=4, max_depth=8)
    ws.parser_backend = 'stm'
    view = _tiny_view()
    ws.attach_knowledge(view)
    ws.compose(input_vectors=None)
    # rule_signatures lives on the ShortTermMemory Layer (via
    # ``init_scorer``); list equality is sufficient since both come
    # from the same view.
    assert list(ws.stm_driver.rule_signatures) == list(
        view.rule_order_signatures)


def test_stm_driver_scorer_payload_dim_matches_wordsubspace():
    """Scorer is sized to match WordSubSpace's typed-STM payload dim."""
    ws = _bare_word_subspace(stm_dim=6, max_depth=8)
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    ws.compose(input_vectors=None)
    assert ws.stm_driver.scorer.payload_dim == 6


def test_stm_driver_constructed_once_idempotent():
    """Repeated compose() under stm backend reuses the same scorer."""
    ws = _bare_word_subspace(stm_dim=4, max_depth=8)
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    ws.compose(input_vectors=None)
    scorer_1 = ws.stm_driver.scorer
    ws.compose(input_vectors=None)
    assert ws.stm_driver.scorer is scorer_1


def test_stm_backend_generate_constructs_driver():
    """generate() under stm backend also wires the driver (same path)."""
    ws = _bare_word_subspace(stm_dim=4, max_depth=8)
    ws.parser_backend = 'stm'
    ws.attach_knowledge(_tiny_view())
    rules = ws.generate(target_vectors=None)
    assert isinstance(rules, dict)
    assert ws.stm_driver is not None
    assert ws.stm_driver.scorer is not None
