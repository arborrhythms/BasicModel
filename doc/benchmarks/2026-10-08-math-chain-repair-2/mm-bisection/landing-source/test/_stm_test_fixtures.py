"""Shared fixtures for STM-related tests after the 2026-05-21 SymbolSubSpace
/ STM Layer refactor.

Provides factory functions that produce the post-refactor equivalents of
the retired `typed_stack.TypedStack` + `stm_driver.STMDriver` +
`stm_driver.RuleScorer` combinations:

  * ``make_typed_stack(batch, max_depth, dim)`` -> a bare ``SymbolSubSpace``
    with manually-allocated typed-STM buffers (same surface as the
    retired ``TypedStack``: ``_buffer`` / ``_category`` / ``_order`` /
    ``_ref_id`` / ``_depth``, plus ``push`` / ``pop`` / ``top`` /
    ``reduce_admissibility``).

The ``STMDriver`` / ``_RuleScorer`` shift-reduce scorer surface these
fixtures used to wrap was deleted in the 2026-07-17 cleanup (Tier-2 item 8;
it had zero production callers). Only the live typed-stack storage factory
remains.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def make_typed_stack(batch=1, max_depth=8, dim=4):
    """Bare SymbolSubSpace with manually-allocated typed-STM buffers."""
    from Language import SymbolSubSpace
    ss = object.__new__(SymbolSubSpace)
    nn.Module.__init__(ss)
    ss.batch = int(batch)
    ss._stm_capacity = int(max_depth)
    ss._stm_payload_dim = int(dim)
    ss.max_depth = ss._stm_capacity
    ss.dim = ss._stm_payload_dim
    ss.register_buffer(
        '_buffer',
        torch.zeros(ss.batch, ss._stm_capacity, dim),
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
        '_depth',
        torch.zeros(ss.batch, dtype=torch.long),
        persistent=False)
    ss._category_names = [
        [None] * ss._stm_capacity for _ in range(ss.batch)]
    # Idea-stack buffers (Phase E completion of doc/specs/
    # 2026-05-21-wordsubspace-stm-layer-refactor.md): the chart's
    # ``ShortTermMemory.push`` lives here too so ``_ensure_stm_batch``
    # can grow both in lockstep. Same shape as the typed buffer.
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


