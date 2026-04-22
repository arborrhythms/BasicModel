"""Phase 2 Sequential pipeline glue modules.

Each class here is an nn.Module with a single-arg forward(subspace) ->
subspace contract. These fit between Spaces in nn.Sequential and
encapsulate cross-stage machinery (butterfly merge, grammar merge,
additive feedback, output split) that used to be inlined in
MentalModel._run_conceptual_order.

Design: basicmodel/doc/specs/2026-04-21-model-sequential-refactor-design.md
"""
import torch
import torch.nn as nn


class ReverseAdapter(nn.Module):
    """Wrap a module so its reverse() method is called via .forward().

    Lets us drop a Space (or any module with .reverse) into nn.Sequential
    on the reverse path. When the wrapped object is itself an nn.Module,
    registering it as a submodule means its parameters are reachable via
    .parameters() — without duplicating them.
    """

    def __init__(self, wrapped):
        super().__init__()
        if isinstance(wrapped, nn.Module):
            self.wrapped = wrapped
        else:
            object.__setattr__(self, "_wrapped_nonmodule", wrapped)
            self.wrapped = None

    def forward(self, subspace):
        target = self.wrapped if self.wrapped is not None else self._wrapped_nonmodule
        return target.reverse(subspace)


class StageWrapper(nn.Module):
    """Wrap a non-nn.Module stage view so it fits in nn.Sequential.

    ConceptualSpace._CSLevelView and SymbolicSpace._SSLevelView are plain
    Python classes that proxy forward()/reverse() through a specific
    per-stage sigma/pi. nn.Sequential requires nn.Module children, so we
    wrap them. The wrapper holds the view as a plain attribute (not a
    submodule) so its parameters come from the underlying Space — not
    duplicated via the wrapper.

    Empty-sentinel short-circuits at this layer.
    """

    def __init__(self, stage):
        super().__init__()
        object.__setattr__(self, "_stage", stage)

    def forward(self, subspace):
        if hasattr(subspace, "is_empty") and subspace.is_empty():
            return subspace
        return self._stage.forward(subspace)

    def reverse(self, subspace):
        if hasattr(subspace, "is_empty") and subspace.is_empty():
            return subspace
        return self._stage.reverse(subspace)


class CachePoint(nn.Module):
    """Identity module that caches whatever subspace passed through last.

    Used as the midpoint anchor in the round-trip Sequential (Case B).
    The model reads cache_point.last after a pipeline_rt call to recover
    the forward output before the reverse half ran.
    """

    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, subspace):
        self.last = subspace
        return subspace


class ButterflyGlue(nn.Module):
    """N-halving glue between conceptual-order stages.

    Isolates the merge step from the old Layers.py ButterflyStage:
    average adjacent pairs along the N axis, caching the pairwise
    difference so reverse() can recover the full N tensor. When
    is_last=True both forward and reverse are identity (the last
    stage in a butterfly cascade does not merge).

    Layers.py ButterflyStage.forward still performs permute/pack/inner/
    unpack/merge as a single fused operation; in the Sequential path the
    inner (sigma/pi) computation stays inside the Space's forward and
    this glue supplies only the merge.
    """

    def __init__(self, stage_idx: int, initial_n: int, is_last: bool):
        super().__init__()
        self.stage_idx = int(stage_idx)
        self.initial_n = int(initial_n)
        self.is_last = bool(is_last)
        self._merge_diff = None

    def forward(self, subspace):
        if subspace.is_empty():
            return subspace
        if self.is_last:
            return subspace
        x = subspace.materialize()
        left = x[:, 0::2, :]
        right = x[:, 1::2, :]
        self._merge_diff = left - right
        subspace.set_event((left + right) / 2)
        return subspace

    def reverse(self, subspace):
        if subspace.is_empty():
            return subspace
        if self.is_last:
            return subspace
        diff = self._merge_diff
        assert diff is not None, (
            "ButterflyGlue.reverse called without prior forward; _merge_diff is None")
        y = subspace.materialize()
        left = y + diff / 2
        right = y - diff / 2
        B, N_half, D = left.shape
        expanded = torch.zeros(B, N_half * 2, D, device=y.device, dtype=y.dtype)
        expanded[:, 0::2, :] = left
        expanded[:, 1::2, :] = right
        subspace.set_event(expanded)
        self._merge_diff = None
        return subspace


class GrammarMergeGlue(nn.Module):
    """Progressive-bottleneck glue for useGrammar == 'all' mode.

    Equivalent to MentalModel._butterfly_merge: average-merge adjacent
    pairs along the N axis, halving N per stage. Keeps dim constant.
    Caches pairwise differences for exact reverse (not present in the
    legacy inline version, but required for Case A reconstruction).
    """

    def __init__(self, stage_idx: int, initial_n: int, is_last: bool):
        super().__init__()
        self.stage_idx = int(stage_idx)
        self.initial_n = int(initial_n)
        self.is_last = bool(is_last)
        self._merge_diff = None

    def forward(self, subspace):
        if subspace.is_empty():
            return subspace
        if self.is_last:
            return subspace
        x = subspace.materialize()
        left = x[:, 0::2, :]
        right = x[:, 1::2, :]
        self._merge_diff = left - right
        subspace.set_event((left + right) / 2)
        return subspace

    def reverse(self, subspace):
        if subspace.is_empty():
            return subspace
        if self.is_last:
            return subspace
        diff = self._merge_diff
        assert diff is not None, (
            "GrammarMergeGlue.reverse called without prior forward")
        y = subspace.materialize()
        left = y + diff / 2
        right = y - diff / 2
        B, N_half, D = left.shape
        expanded = torch.zeros(B, N_half * 2, D, device=y.device, dtype=y.dtype)
        expanded[:, 0::2, :] = left
        expanded[:, 1::2, :] = right
        subspace.set_event(expanded)
        self._merge_diff = None
        return subspace


class AdditiveFeedbackGlue(nn.Module):
    """Combine current concepts with previous-stage symbol feedback.

    feedback_source is any object exposing materialize() -> tensor|None
    and is_empty() -> bool. In practice, the SymbolicSpace's subspace
    from the previous stage of the unrolled pipeline.

    Returns a self-materializing wrapper rather than mutating the
    incoming subspace. This is critical: the incoming subspace is
    typically ConceptualSpace.subspace, whose event tensor is later
    consumed by reverse() — overwriting it with ``x + fb`` (potentially
    out of [-1, 1]) breaks the reverse-Sigma's atanh.

    Semantics:
    - Incoming subspace empty -> return empty (empty infects).
    - feedback_source empty/None -> passthrough (first stage).
    - Else: output materializes to ``input + feedback`` without
      modifying the underlying subspace.
    """

    def __init__(self, stage_idx: int, feedback_source):
        super().__init__()
        self.stage_idx = int(stage_idx)
        self.feedback_source = feedback_source
        self._last_feedback = None
        self._combined = None
        self._passthrough_subspace = None

    def materialize(self):
        return self._combined

    def is_empty(self):
        return self._combined is None

    @property
    def batch(self):
        if self._combined is None:
            return 0
        return int(self._combined.shape[0])

    def forward(self, subspace):
        if subspace.is_empty():
            self._combined = None
            return subspace
        fb = self.feedback_source.materialize()
        if fb is None or self.feedback_source.is_empty():
            self._last_feedback = None
            self._combined = subspace.materialize()
            self._passthrough_subspace = subspace
            return subspace
        x = subspace.materialize()
        self._last_feedback = fb
        self._combined = x + fb
        self._passthrough_subspace = subspace
        return self

    def reverse(self, subspace):
        if subspace.is_empty():
            return subspace
        fb = self._last_feedback
        if fb is None:
            return subspace
        x = subspace.materialize()
        subspace.set_event(x - fb)
        self._last_feedback = None
        return subspace
