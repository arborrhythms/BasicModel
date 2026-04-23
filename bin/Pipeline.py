"""Phase 2 Sequential pipeline glue modules.

Each class here is an nn.Module with a single-arg forward(subspace) ->
subspace contract. These fit between Spaces in nn.Sequential and
encapsulate cross-stage machinery (grammar-mode N-halving merge,
round-trip midpoint cache, reverse-path adapter).
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

    def reverse(self, subspace):
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
        super().__init__()
        self.body = body

    def forward(self, subspace):
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


