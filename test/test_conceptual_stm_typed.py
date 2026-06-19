"""Tests for the typed-STM stack carried on ``SymbolicSubSpace``.

Plan: doc/plans/2026-05-21-wordsubspace-stm-layer-refactor.md §Phase D.

The typed-metadata stack that formerly lived at
``ConceptualSpace._stm_typed`` (allocated via ``_init_typed_stm``) now
lives directly on ``SymbolicSubSpace`` as registered buffers, with public
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
    """Construct a bare SymbolicSubSpace with manually-allocated STM buffers.

    The real ``__init__`` constructs the full pipeline (chart, layers,
    etc.); these tests only exercise the typed-STM data surface, so we
    bypass it and stamp the buffers directly. Mirrors the bare-instance
    style of the rest of the STM substrate tests.
    """
    from Language import SymbolicSubSpace
    import torch.nn as nn
    ss = object.__new__(SymbolicSubSpace)
    nn.Module.__init__(ss)
    ss.batch = 1
    ss._stm_capacity = int(max_depth)
    ss._stm_payload_dim = int(dim)
    ss.max_depth = ss._stm_capacity
    ss.dim = ss._stm_payload_dim
    ss.register_buffer(
        '_buffer', torch.zeros(ss.batch, ss._stm_capacity, dim),
        persistent=False)
    ss.register_buffer(
        '_category',
        torch.full((ss.batch, ss._stm_capacity), -1, dtype=torch.long),
        persistent=False)
    ss.register_buffer(
        '_order',
        torch.zeros((ss.batch, ss._stm_capacity), dtype=torch.long),
        persistent=False)
    ss.register_buffer(
        '_ref_id',
        torch.full((ss.batch, ss._stm_capacity), -1, dtype=torch.long),
        persistent=False)
    ss.register_buffer(
        '_depth', torch.zeros(ss.batch, dtype=torch.long),
        persistent=False)
    ss._category_names = [
        [None] * ss._stm_capacity for _ in range(ss.batch)]
    # Idea-stack buffers so ``_ensure_stm_batch`` can grow them in lockstep
    # (Phase E completion of the 2026-05-21 refactor).
    ss._idea_capacity = ss._stm_capacity
    ss._idea_max_depth_host = 0
    ss.register_buffer(
        '_idea_buffer',
        torch.zeros(ss.batch, ss._idea_capacity, dim),
        persistent=False)
    ss.register_buffer(
        '_idea_depth',
        torch.zeros(ss.batch, dtype=torch.long),
        persistent=False)
    return ss


def test_typed_stm_buffers_allocated_on_wordsubspace():
    """The typed STM lives on SymbolicSubSpace as registered buffers."""
    ss = _bare_word_subspace(max_depth=8, dim=4)
    assert ss._buffer.shape == (1, 8, 4)
    assert ss._category.shape == (1, 8)
    assert ss._order.shape == (1, 8)
    assert ss._ref_id.shape == (1, 8)
    assert ss._depth.shape == (1,)
    assert ss.max_depth == 8
    assert ss.dim == 4


def test_push_pop_roundtrip():
    """A pushed frame round-trips through pop with all metadata intact."""
    ss = _bare_word_subspace(max_depth=8, dim=4)
    vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ss.push(0, vec, category_id=7, order=2, ref_id=11)
    assert int(ss._depth[0].item()) == 1
    out = ss.pop(0)
    assert torch.allclose(out['payload'], vec)
    assert out['category'] == 7
    assert out['order'] == 2
    assert out['ref_id'] == 11
    assert int(ss._depth[0].item()) == 0


def test_top_peek_does_not_pop():
    """`top` peeks without modifying the stack depth."""
    ss = _bare_word_subspace(max_depth=8, dim=4)
    vec = torch.tensor([0.5, 0.5, 0.5, 0.5])
    ss.push(0, vec, category_id=3, order=0, ref_id=5)
    out = ss.top(0)
    assert out['category'] == 3
    assert int(ss._depth[0].item()) == 1  # unchanged


def test_ensure_stm_batch_grows_rows():
    """`_ensure_stm_batch` grows the row dim and preserves existing data."""
    ss = _bare_word_subspace(max_depth=8, dim=4)
    ss.push(0, torch.tensor([1.0, 1.0, 1.0, 1.0]),
            category_id=2, order=0, ref_id=4)
    ss._ensure_stm_batch(3)
    assert ss._buffer.shape == (3, 8, 4)
    # Row 0's pushed payload survives the resize.
    assert torch.allclose(ss._buffer[0, 0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert int(ss._depth[0].item()) == 1
    assert int(ss._depth[1].item()) == 0
    assert int(ss._depth[2].item()) == 0
