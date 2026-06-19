"""Shared fixtures for STM-related tests after the 2026-05-21 SymbolicSubSpace
/ STM Layer refactor.

Provides factory functions that produce the post-refactor equivalents of
the retired `typed_stack.TypedStack` + `stm_driver.STMDriver` +
`stm_driver.RuleScorer` combinations:

  * ``make_typed_stack(batch, max_depth, dim)`` -> a bare ``SymbolicSubSpace``
    with manually-allocated typed-STM buffers (same surface as the
    retired ``TypedStack``: ``_buffer`` / ``_category`` / ``_order`` /
    ``_ref_id`` / ``_depth``, plus ``push`` / ``pop`` / ``top`` /
    ``reduce_admissibility``).

  * ``make_driver(typed_stack, rule_signatures, payload_dim)`` -> a
    ``ShortTermMemory`` Layer with the scorer initialised. Exposes
    ``shift`` / ``reduce_step`` / ``reduce_step_soft`` /
    ``train_scorer_step`` / ``_score_reduce`` taking a SymbolicSubSpace
    (the ``typed_stack`` arg). Mirrors the retired ``STMDriver`` API
    where the driver was constructed once and methods looked up the
    typed stack from ``self.typed_stack``; here we hold the
    ``word_subspace`` reference for backward-compat.

  * ``make_scorer(payload_dim, n_rules)`` -> the private
    ``_RuleScorer`` (`bin/Layers.py`) created standalone. Used by tests
    that exercise the scorer's forward shape directly.
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
    """Bare SymbolicSubSpace with manually-allocated typed-STM buffers."""
    from Language import SymbolicSubSpace
    ss = object.__new__(SymbolicSubSpace)
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


class _DriverWrapper:
    """Thin compatibility wrapper around ``ShortTermMemory`` that holds a
    reference to the ``word_subspace`` so legacy callers can do
    ``driver.shift(b, ...)`` / ``driver.reduce_step_soft(b)`` without
    threading the word_subspace through every call.

    The retired ``STMDriver`` held its ``typed_stack`` as an attribute;
    the post-refactor ``ShortTermMemory.shift`` takes ``word_subspace``
    as an explicit argument. This wrapper preserves the old call surface
    for tests while the underlying implementation lives on the Layer.
    """

    def __init__(self, stm_layer, typed_stack, rule_signatures):
        self._stm = stm_layer
        self.typed_stack = typed_stack
        # Mirror ``rule_signatures`` so consumers like
        # ``driver.rule_signatures`` and ``driver.scorer`` keep working.
        self.rule_signatures = list(rule_signatures)

    @property
    def scorer(self):
        return self._stm.scorer

    def shift(self, b, payload, *, category, order, ref_id):
        self._stm.shift(self.typed_stack, b, payload,
                         category=category, order=order, ref_id=ref_id)

    def reduce_step(self, b):
        return self._stm.reduce_step(self.typed_stack, b)

    def reduce_step_soft(self, b):
        return self._stm.reduce_step_soft(self.typed_stack, b)

    def _score_reduce(self, b):
        return self._stm._score_reduce(self.typed_stack, b)

    def parameters(self):
        return self._stm.scorer.parameters()


def make_driver(typed_stack, rule_signatures, payload_dim=None):
    """ShortTermMemory + initialised scorer wrapped in a backward-compat
    object that pins the SymbolicSubSpace reference (mirrors the retired
    ``STMDriver(typed_stack, rule_signatures, scorer)`` constructor).
    """
    from Layers import ShortTermMemory
    if payload_dim is None:
        payload_dim = int(getattr(typed_stack, 'dim', 0))
    stm = ShortTermMemory(
        batch=1, capacity=int(getattr(typed_stack, 'max_depth', 8)),
        concept_dim=int(payload_dim))
    stm.init_scorer(rule_signatures=rule_signatures,
                    payload_dim=int(payload_dim))
    return _DriverWrapper(stm, typed_stack, rule_signatures)


def make_scorer(payload_dim, n_rules, hidden_dim=None):
    """The private ``_RuleScorer`` (in bin/Layers.py) standalone."""
    from Layers import _RuleScorer
    return _RuleScorer(
        payload_dim=int(payload_dim), n_rules=int(n_rules),
        hidden_dim=hidden_dim)


def make_train_step():
    """Returns a ``train_step(driver, input_vectors, target_rule_ids,
    snap_fn, optimizer=None)`` callable matching the retired
    ``stm_trainer.train_step`` signature. Internally dispatches to
    ``ShortTermMemory.train_scorer_step``.
    """
    def _train(driver, input_vectors, target_rule_ids, *,
               snap_fn, optimizer=None):
        return driver._stm.train_scorer_step(
            driver.typed_stack, input_vectors, target_rule_ids,
            snap_fn=snap_fn, optimizer=optimizer)
    return _train
