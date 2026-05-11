"""Phase 2 Sequential pipeline glue modules.

Each class here is an nn.Module with a single-arg forward(subspace) ->
subspace contract. These fit between Spaces in nn.Sequential and
encapsulate cross-stage machinery (grammar-mode N-halving merge,
round-trip midpoint cache, reverse-path adapter).
"""
import logging
import traceback
import torch
import torch.nn as nn


# Process-global dedupe set for advisory pipeline-stage exceptions.
# Key: (exception_type_name, file, line). Value: count.
# First occurrence logs with full traceback (exc_info=True); subsequent
# occurrences increment the count silently. A periodic flush every
# _DEDUPE_FLUSH_EVERY hits emits a one-liner summary so a steady-state
# failure isn't fully invisible.
_PIPELINE_EXC_SEEN: dict = {}
_DEDUPE_FLUSH_EVERY = 1000


def _log_advisory_exception(stage: str, exc: BaseException) -> None:
    """Log an advisory (caught) pipeline-stage exception with traceback
    on first occurrence and a count on subsequent occurrences.

    A pipeline stage that crashes means the grammar / chart / generation
    path silently degraded to its fallback. That's a real correctness
    issue (you're not training what the config says), so we surface
    it loudly the first time and quietly track repeats.
    """
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    file = tb.tb_frame.f_code.co_filename if tb else "?"
    line = tb.tb_lineno if tb else 0
    key = (type(exc).__name__, file, line)
    log = logging.getLogger(__name__)
    seen = _PIPELINE_EXC_SEEN.get(key, 0)
    _PIPELINE_EXC_SEEN[key] = seen + 1
    if seen == 0:
        log.warning(
            "%s failed (%s: %s) at %s:%d -- "
            "the chart's contribution is now a no-op. "
            "Subsequent occurrences of this exact failure will be deduped.",
            stage, type(exc).__name__, exc, file, line,
            exc_info=True,
        )
    elif (seen + 1) % _DEDUPE_FLUSH_EVERY == 0:
        log.warning(
            "%s: %s at %s:%d has now fired %d times.",
            stage, type(exc).__name__, file, line, seen + 1,
        )


class ReverseAdapter(nn.Module):
    """Wrap a module so its reverse() method is called via .forward().

    Lets us drop a Space (or any module with .reverse) into nn.Sequential
    on the reverse path. When the wrapped object is itself an nn.Module,
    registering it as a submodule means its parameters are reachable via
    .parameters() — without duplicating them.
    """

    def __init__(self, wrapped):
        """Hold ``wrapped`` as a submodule when it is an nn.Module.

        Non-Module wrapped objects are stashed via object.__setattr__ to
        avoid nn.Module's auto-registration of attributes that aren't
        modules / tensors.
        """
        super().__init__()
        if isinstance(wrapped, nn.Module):
            self.wrapped = wrapped
        else:
            object.__setattr__(self, "_wrapped_nonmodule", wrapped)
            self.wrapped = None

    def forward(self, subspace):
        """Delegate to ``wrapped.reverse(subspace)``.

        Resolves to the nn.Module submodule when present, otherwise
        falls back to the non-Module attribute stashed at __init__.
        """
        target = self.wrapped if self.wrapped is not None else self._wrapped_nonmodule
        return target.reverse(subspace)


class SubsymbolicTee(nn.Module):
    """Pipeline step that runs ``subsymbolicSpace`` as a side effect on
    the conceptual subspace, then passes the subspace through unchanged.

    Inserted between ConceptualSpace and SymbolicSpace in the body
    pipeline when ``<architecture><subsymbolicEnabled>true``. The
    SubsymbolicSpace's event tensor is consumed at the *next*
    conceptual order's combined input, not within the producing
    order, so its output is not threaded into the downstream
    sequential -- it lands on ``subsymbolicSpace.subspace.event`` as
    a side effect and SymbolicSpace continues to receive the
    original concept_subspace.

    No-op when ``subsymbolicSpace`` is ``None`` (subsymbolicEnabled
    was false at construction).
    """

    def __init__(self, subsymbolicSpace):
        """Hold the optional subsymbolic space. None = pure pass-through."""
        super().__init__()
        self.subsymbolicSpace = subsymbolicSpace

    def forward(self, subspace):
        """Invoke subsymbolicSpace as a side effect; return subspace unchanged.

        No-op when the subsymbolic space wasn't constructed or the
        input subspace is empty. The subsymbolic event is consumed at
        the *next* conceptual order, not here.
        """
        if self.subsymbolicSpace is None or subspace is None:
            return subspace
        if hasattr(subspace, 'is_empty') and subspace.is_empty():
            return subspace
        self.subsymbolicSpace(subspace)
        return subspace

    def reverse(self, subspace):
        """Identity on the reverse path; subsymbolic is forward-only."""
        return subspace


class CachePoint(nn.Module):
    """Identity module that caches whatever subspace passed through last.

    Used as the midpoint anchor in the round-trip Sequential (Case B).
    The model reads cache_point.last after a pipeline_rt call to recover
    the forward output before the reverse half ran.
    """

    def __init__(self):
        """Initialize the cache slot to ``None``."""
        super().__init__()
        self.last = None

    def forward(self, subspace):
        """Cache the subspace in ``self.last`` and pass through unchanged."""
        self.last = subspace
        return subspace

    def reverse(self, subspace):
        """Identity on reverse: the cached value already lives in ``last``."""
        return subspace


class ChartCompose(nn.Module):
    """Pipeline step that runs ``wordSpace.compose(...)`` on the
    incoming subspace, then passes the subspace through unchanged.

    Inserted into the forward Sequential between InputSpace and
    PerceptualSpace (2026-05-01 syntactic-layer refactor §6 / Q10.2).
    The chart parses the input data and writes per-(tier, step) rule
    selections into ``wordSpace.current_rules``; downstream spaces'
    SyntacticLayers read the selections during their forward pass.

    No-op when ``word_space`` is ``None`` (non-AR / non-ARUS modes
    where ``BasicModel.wordSpace`` was never built).
    """

    def __init__(self, word_space):
        """Hold a non-owning reference to the word space.

        Bypasses nn.Module's auto-registration to avoid a wordSpace ->
        chart -> ... -> wordSpace ownership cycle. The word space lives
        at the model level.
        """
        super().__init__()
        # Stash as a non-Module attribute to avoid the wordSpace -> chart
        # -> ... -> wordSpace cycle that nn.Module ownership would
        # introduce. wordSpace is owned at the model level.
        object.__setattr__(self, '_word_space', word_space)

    def forward(self, subspace):
        """Run the chart's compose pass; write rule cursors as side effect.

        Materializes the subspace, flattens any K microbatch axis,
        invokes ``wordSpace.compose``, and (signal router mode) writes
        the router's transformed slab back into ``subspace.event`` so
        the differentiable signal reaches downstream spaces. Failures
        are logged advisory-style and do not break the forward pass.
        """
        ws = self._word_space
        if ws is None or subspace is None:
            return subspace
        if hasattr(subspace, 'is_empty') and subspace.is_empty():
            return subspace
        data = subspace.materialize() if hasattr(
            subspace, 'materialize') else None
        if data is None:
            return subspace
        try:
            # ARIR microbatch path: data may arrive as [B, K, N, D] when
            # the upstream subspace carries a K-axis. The chart's inside
            # pass expects [B, N, D]; flatten K into the batch dim before
            # the call. K-dim restoration isn't needed because the chart
            # writes side-effects into ws.current_rules (per-row rule
            # cursors), and the downstream FlattenKWrapper-wrapped body
            # consumes those cursors at the same B*K row indexing.
            if data.dim() == 4:
                B, K, N, D = data.shape
                flat = data.reshape(B * K, N, D)
                ws.compose(flat, subspace=subspace)
            else:
                ws.compose(data, subspace=subspace)
            # On the signal path, propagate the router's transformed slab
            # back into subspace.event so the differentiable signal flows
            # to downstream spaces (otherwise the router's gates / scorer
            # never receive gradient from the task loss). On the chart
            # path, last_composed is left in place but not written back
            # (the legacy contract is "current_rules drives downstream").
            if ws.chart.router_kind == "signal":
                router = ws.chart._signal_router
                if router is not None and router._last_output is not None:
                    out = router._last_output
                    if out.shape == data.shape:
                        subspace.set_event(out)
        except Exception as exc:
            # Chart compose is advisory: a failure mustn't break the
            # forward pass. The per-space SyntacticLayer falls back to
            # its default rule when current_rules is empty. First
            # occurrence is logged with full traceback so the real bug
            # is debuggable; identical repeats are deduped.
            _log_advisory_exception("ChartCompose.forward", exc)
        return subspace


class ChartGenerate(nn.Module):
    """Reverse-pipeline mirror of ``ChartCompose``: runs
    ``wordSpace.generate(...)`` on the incoming subspace before the
    spaces' reverse passes fire.

    No-op when ``word_space`` is ``None``.
    """

    def __init__(self, word_space):
        """Hold a non-owning reference to the word space (see ChartCompose)."""
        super().__init__()
        object.__setattr__(self, '_word_space', word_space)

    def forward(self, subspace):
        """Run the chart's outside / generation pass on the subspace.

        Symmetric to ChartCompose.forward, but invokes ``generate``
        and does not write back into subspace.event. Same K-axis
        flatten convention, same advisory-error handling.
        """
        ws = self._word_space
        if ws is None or subspace is None:
            return subspace
        if hasattr(subspace, 'is_empty') and subspace.is_empty():
            return subspace
        data = subspace.materialize() if hasattr(
            subspace, 'materialize') else None
        if data is None:
            return subspace
        try:
            # Same K-axis flatten as ChartCompose.
            if data.dim() == 4:
                B, K, N, D = data.shape
                flat = data.reshape(B * K, N, D)
                ws.generate(flat, subspace=subspace)
            else:
                ws.generate(data, subspace=subspace)
        except Exception as exc:
            _log_advisory_exception("ChartGenerate.forward", exc)
        return subspace


class GrammarMergeGlue(nn.Module):
    """Progressive-bottleneck glue for useGrammar == 'all' mode.

    Equivalent to BasicModel._butterfly_merge: average-merge adjacent
    pairs along the N axis, halving N per stage. Keeps dim constant.
    Caches pairwise differences for exact reverse (not present in the
    legacy inline version, but required for Case A reconstruction).
    """

    def __init__(self, stage_idx: int, initial_n: int, is_last: bool):
        """Record stage index, initial N, and the last-stage flag.

        ``is_last`` makes this glue a transparent pass-through (the
        deepest conceptual stage shouldn't halve again).
        """
        super().__init__()
        self.stage_idx = int(stage_idx)
        self.initial_n = int(initial_n)
        self.is_last = bool(is_last)
        self._merge_diff = None

    def forward(self, subspace):
        """Average adjacent pairs along the N axis; cache the difference.

        Halves N from 2K to K and writes the cached ``left - right``
        difference to ``self._merge_diff`` so ``reverse`` can rebuild
        the unmerged pair. No-op on empty / last-stage subspaces.
        """
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
        """Invert the average-merge using the cached difference.

        Reconstructs left/right from ``mid +/- diff/2`` and re-interleaves
        them along N. Clears the cache after use. Asserts the matching
        forward already populated ``_merge_diff``.
        """
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


class FlattenKWrapper(nn.Module):
    """Reshape [B, K, N, D] -> [B*K, N, D] for the inner body, then back.

    Lets the body process all K microbatch windows in parallel. The body
    sees a flat batch dim and operates as if there were no K axis. On
    return we reshape to [B, K, N, Dout] (Dout may differ from D).
    Autograd handles the back-view automatically.

    Contract: when ``subspace.k_axis`` is True the input event has shape
    [B, K, N, D]; we flatten to [B*K, N, D], invoke the body, then
    restore [B, K, N, Dout] on the way out. When ``k_axis`` is False
    (non-AR / legacy path) the wrapper is a transparent pass-through.
    """

    def __init__(self, body):
        """Wrap the inner body that operates on [B*K, N, D] tensors."""
        super().__init__()
        self.body = body

    def forward(self, subspace):
        """Flatten K -> batch, run the body, restore the K axis.

        Pass-through when the subspace has no K axis. Otherwise reshapes
        [B, K, N, D] -> [B*K, N, D] before the body call and reshapes
        the body's [B*K, N, Dout] back to [B, K, N, Dout]. Mutates
        ``subspace.k_axis`` across the call to keep the body honest.
        """
        # Pass-through for non-microbatch paths (k_axis=False subspaces).
        if not subspace.k_axis:
            return self.body(subspace)
        x = subspace.materialize()
        assert x.dim() == 4, (
            f"FlattenKWrapper expects [B,K,N,D], got shape {tuple(x.shape)}"
        )
        B, K, N, D = x.shape
        flat = x.reshape(B * K, N, D)
        subspace.set_event(flat)
        subspace.k_axis = False
        out = self.body(subspace)
        y = out.materialize()
        # Body may change N (e.g., butterfly N-halving) and D (e.g., head
        # projection), but B*K must round-trip — the K axis is what we
        # restore on the way out.
        BK, Nout, Dout = y.shape
        assert BK == B * K, (
            f"FlattenKWrapper body changed batch dim: "
            f"in [B*K,N,D]={(B * K, N, D)}, out shape {tuple(y.shape)}"
        )
        out.set_event(y.view(B, K, Nout, Dout))
        out.k_axis = True
        return out


