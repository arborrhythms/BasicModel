"""Tests for the typed-STM stack carried on ``WordSubSpace``.

Plan: doc/plans/2026-05-21-wordsubspace-stm-layer-refactor.md §Phase D.

The typed-metadata stack that formerly lived at
``ConceptualSpace._stm_typed`` (allocated via ``_init_typed_stm``) now
lives directly on ``WordSubSpace`` as registered buffers, with public
methods ``push`` / ``pop`` / ``top`` / ``reduce_admissibility`` /
``_ensure_stm_batch`` mirroring the former ``TypedStack`` surface.
"""
import sys
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _bare_word_subspace(max_depth=8, dim=4):
    """Construct a bare WordSubSpace with manually-allocated STM buffers.

    The real ``__init__`` constructs the full pipeline (chart, layers,
    etc.); these tests only exercise the typed-STM data surface, so we
    bypass it and stamp the buffers directly. Mirrors the bare-instance
    style of the rest of the STM substrate tests.
    """
    from Language import WordSubSpace
    import torch.nn as nn
    ws = object.__new__(WordSubSpace)
    nn.Module.__init__(ws)
    ws.batch = 1
    ws._stm_capacity = int(max_depth)
    ws._stm_payload_dim = int(dim)
    ws.max_depth = ws._stm_capacity
    ws.dim = ws._stm_payload_dim
    ws.register_buffer(
        '_buffer', torch.zeros(ws.batch, ws._stm_capacity, dim),
        persistent=False)
    ws.register_buffer(
        '_category',
        torch.full((ws.batch, ws._stm_capacity), -1, dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_order',
        torch.zeros((ws.batch, ws._stm_capacity), dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_ref_id',
        torch.full((ws.batch, ws._stm_capacity), -1, dtype=torch.long),
        persistent=False)
    ws.register_buffer(
        '_depth', torch.zeros(ws.batch, dtype=torch.long),
        persistent=False)
    ws._category_names = [
        [None] * ws._stm_capacity for _ in range(ws.batch)]
    # Idea-stack buffers so ``_ensure_stm_batch`` can grow them in lockstep
    # (Phase E completion of the 2026-05-21 refactor).
    ws._idea_capacity = ws._stm_capacity
    ws._idea_max_depth_host = 0
    ws.register_buffer(
        '_idea_buffer',
        torch.zeros(ws.batch, ws._idea_capacity, dim),
        persistent=False)
    ws.register_buffer(
        '_idea_depth',
        torch.zeros(ws.batch, dtype=torch.long),
        persistent=False)
    return ws


def test_typed_stm_buffers_allocated_on_wordsubspace():
    """The typed STM lives on WordSubSpace as registered buffers."""
    ws = _bare_word_subspace(max_depth=8, dim=4)
    assert ws._buffer.shape == (1, 8, 4)
    assert ws._category.shape == (1, 8)
    assert ws._order.shape == (1, 8)
    assert ws._ref_id.shape == (1, 8)
    assert ws._depth.shape == (1,)
    assert ws.max_depth == 8
    assert ws.dim == 4


def test_push_pop_roundtrip():
    """A pushed frame round-trips through pop with all metadata intact."""
    ws = _bare_word_subspace(max_depth=8, dim=4)
    vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ws.push(0, vec, category_id=7, order=2, ref_id=11)
    assert int(ws._depth[0].item()) == 1
    out = ws.pop(0)
    assert torch.allclose(out['payload'], vec)
    assert out['category'] == 7
    assert out['order'] == 2
    assert out['ref_id'] == 11
    assert int(ws._depth[0].item()) == 0


def test_top_peek_does_not_pop():
    """`top` peeks without modifying the stack depth."""
    ws = _bare_word_subspace(max_depth=8, dim=4)
    vec = torch.tensor([0.5, 0.5, 0.5, 0.5])
    ws.push(0, vec, category_id=3, order=0, ref_id=5)
    out = ws.top(0)
    assert out['category'] == 3
    assert int(ws._depth[0].item()) == 1  # unchanged


def test_ensure_stm_batch_grows_rows():
    """`_ensure_stm_batch` grows the row dim and preserves existing data."""
    ws = _bare_word_subspace(max_depth=8, dim=4)
    ws.push(0, torch.tensor([1.0, 1.0, 1.0, 1.0]),
            category_id=2, order=0, ref_id=4)
    ws._ensure_stm_batch(3)
    assert ws._buffer.shape == (3, 8, 4)
    # Row 0's pushed payload survives the resize.
    assert torch.allclose(ws._buffer[0, 0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert int(ws._depth[0].item()) == 1
    assert int(ws._depth[1].item()) == 0
    assert int(ws._depth[2].item()) == 0
