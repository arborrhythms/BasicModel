"""Core layer primitives used by BasicModel.

This module mixes conventional neural-network utilities with a set of
custom reversible, ergodic, and memory-style layers.  Most higher-level
model construction happens in ``BasicModel.py``; this file provides the
building blocks and the update rules they share.
"""
from __future__ import annotations  # allow X | Y union syntax on Python 3.9

import os
import warnings
import numpy as np
import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import torch.optim as optim
import time
from typing import Optional, Tuple
from collections import namedtuple

epsilon = 1e-7  # to avoid log(0)

# Device used by all layers.
import util
from util import TheXMLConfig, TheDevice
from util import TheMessage


class Lexicon(nn.Embedding):
    """nn.Embedding on the **projective unit ball** ``B^D / (x ~ -x)``.

    Default geometry is the **negation quotient** of the closed unit
    ball, realized as real projective space ``RP^D``. Each vocabulary
    row ``w_i`` lives in ``B^D = {x : ||x||_2 <= 1}`` and ``w_i`` is
    identified with its negation ``-w_i`` -- there is no edge to the
    embedding space; the boundary sphere wraps onto itself by the
    ``+/-`` quotient. This preserves the pode / wrapped-pode pair
    structure SBOW training depends on while giving an exact
    matmul-form lookup.

    Terminology pin (deliberate; see ``doc/Spaces.md``):

      * **Pode** of ``(a, b)``: midpoint ``(a + b)/2``. Attractor for
        SBOW positive-pair updates.
      * **Wrapped pode**: midpoint via the negation of ``b``,
        ``(a - b)/2``. Used to find the shorter arc.
      * **Antipode** of a single point ``p``: the *furthest* point from
        ``p`` in the manifold's topology -- SBOW's balancing repulsion
        target. Unique on the flat torus (``wrap(p + 1)`` per axis);
        on ``RP^D`` it is the orthogonal hyperplane ``{w : <p, w> =
        0}``, *not* a unique point.

    So ``-w`` is the **negation** of ``w`` (the quotient partner under
    ``RP^D``), *not* the antipode of ``w``. The ``rp_closer_rep`` and
    ``rp_wrapped_pode`` helpers operate on negation representatives;
    they do not produce antipodes.

    Distance:

        d_RP^2(a, b) = min(||a - b||^2, ||a + b||^2)
                     = ||a||^2 + ||b||^2 - 2 |<a, b>|.

    Lookup score (one matmul + abs + bias subtract):

        score(x, w_i) = |<x, w_i>| - 0.5 * ||w_i||^2.

    Sorting by largest score is exactly equivalent to sorting by smallest
    projective squared distance for fixed ``x``. See ``doc/Lexicon.md``
    for the full derivation and SBOW gradient consequences.

    Legacy: ``ball=False`` (or the deprecated ``torus=True`` alias)
    selects the prior flat-n-torus geometry ``T^D = [-1, 1)^D`` with
    wrapped-MSE distance. Kept for backward compatibility; the torus
    primitives (``wrap``, ``delta``, ``distance_sq``, ``similarity``,
    ``inner_distance``, ``outer_distance``, ``antipode``, ``random``)
    remain on the class as static methods so call sites that still
    operate in torus coordinates keep working unchanged. The
    backward-compat property ``self.torus`` mirrors ``not self.ball``.

    Geometry-aware methods that should be called from training and
    lookup paths (geometry-dispatching where applicable):
      - ``normalize()`` re-projects weights onto the active manifold
        (projective unit ball or torus) after an optimizer step.
      - ``project_unit_ball_()`` clips row norms to ``<= 1``.
      - ``lookup_index()`` returns the ``(W_index, W_norm2)`` pair the
        matmul lookup wants.
      - ``rp_scores`` / ``topk_rp`` compute the projective matmul-form
        lookup; ``l2_scores`` / ``topk_l2`` compute plain (non-projective)
        L2 lookup for callers that genuinely want each row treated
        independently.
      - ``rp_distance_sq`` / ``rp_similarity`` are the pairwise primitives
        SBOW negative-sampling can use directly.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 *, ball: bool = True, torus: Optional[bool] = None,
                 init: str = "uniform_ball",
                 padding_idx=None, **kwargs):
        """Initialize Lexicon; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(num_embeddings, embedding_dim,
                         padding_idx=padding_idx, **kwargs)
        # ``ball=True`` (default) selects the projective unit-ball
        # geometry. ``ball=False`` selects the LEGACY flat-torus
        # geometry. ``torus=`` is a legacy alias kwarg kept so call
        # sites that still pass ``torus=True`` keep working; remove it
        # once no in-tree caller uses it.
        if torus is not None:
            ball = not bool(torus)
        self.ball = bool(ball)
        if not self.ball:
            # LEGACY (torus): uniform draw on ``T^D = [-1, 1)^D``.
            with torch.no_grad():
                self.weight.copy_(
                    Lexicon.random(self.weight.shape,
                                   device=self.weight.device,
                                   dtype=self.weight.dtype))
        else:
            self.reset_unit_ball_parameters(init=init)

    # Backward-compat property: ``self.torus`` is the negation of
    # ``self.ball`` so legacy reads (``if self.torus: ...``) keep
    # working unchanged. Prefer ``self.ball`` in new code.
    @property
    def torus(self) -> bool:
        """Torus.
        
        See class docstring for the operation contract.
        """
        return not self.ball

    @torus.setter
    def torus(self, value: bool) -> None:
        """Torus.
        
        See class docstring for the operation contract.
        """
        self.ball = not bool(value)

    # -- Unit-ball geometry --------------------------------------------------

    @torch.no_grad()
    def reset_unit_ball_parameters(self, init: str = "uniform_ball") -> None:
        """Initialize weights on the unit ball.

        ``init="uniform_ball"`` (default) draws directions uniformly on
        the sphere and radii with the volume-correct ``r ~ U(0,1)^(1/D)``
        so the resulting distribution is uniform on the closed unit ball.
        ``init="small_normal"`` uses a small-stddev Gaussian projected
        into the ball -- useful when downstream layers expect tight
        clusters near the origin at the start of training.
        """
        V, D = self.weight.shape
        if init == "uniform_ball":
            W = torch.randn_like(self.weight)
            W = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            radius = torch.rand(
                V, 1,
                device=self.weight.device,
                dtype=self.weight.dtype,
            ).pow(1.0 / max(D, 1))
            self.weight.copy_(radius * W)
        elif init == "small_normal":
            self.weight.normal_(mean=0.0, std=0.02)
            self.project_unit_ball_()
        else:
            raise ValueError(f"unknown init: {init}")

    @staticmethod
    def project_unit_ball(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Return ``x`` projected rowwise into the closed unit ball.

        Rows with ``||x|| <= 1`` are unchanged; rows outside are scaled
        to lie on the unit sphere.
        """
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = torch.clamp(1.0 / norm, max=1.0)
        return x * scale

    @torch.no_grad()
    def project_unit_ball_(self, eps: float = 1e-12) -> None:
        """In-place rowwise projection of the embedding matrix into the
        unit ball. Call after ``optimizer.step()``.
        """
        norm = self.weight.norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = torch.clamp(1.0 / norm, max=1.0)
        self.weight.mul_(scale)

    @torch.no_grad()
    def lookup_index(self):
        """Return ``(W_index, W_norm2)`` for unit-ball matmul lookup.

        ``W_norm2`` should be refreshed after every optimizer step that
        changes the lexicon. The lookup is

            score = x @ W_index.T - 0.5 * W_norm2

        See ``l2_scores`` and ``topk_l2``.
        """
        W_index = self.weight.contiguous()
        W_norm2 = W_index.square().sum(dim=-1).contiguous()
        return W_index, W_norm2

    @staticmethod
    def l2_scores(x: torch.Tensor, W_index: torch.Tensor,
                  W_norm2: torch.Tensor) -> torch.Tensor:
        """Compute L2-equivalent lookup scores ``x @ W.T - 0.5 * ||W||^2``.

        Args:
            x: ``[..., D]`` query.
            W_index: ``[V, D]`` lexicon matrix.
            W_norm2: ``[V]`` precomputed squared row norms of ``W_index``.

        Returns:
            ``[..., V]`` similarity scores; larger is closer under L2.
        """
        if x.shape[-1] != W_index.shape[-1]:
            raise ValueError(
                f"x last dim {x.shape[-1]} must equal "
                f"W_index dim {W_index.shape[-1]}")
        return x @ W_index.T - 0.5 * W_norm2

    @staticmethod
    def topk_l2(x: torch.Tensor, W_index: torch.Tensor,
                W_norm2: torch.Tensor, k: int):
        """Exact top-k nearest rows under Euclidean L2 in the unit ball.

        Plain L2 (non-projective): ``w_i`` and ``-w_i`` are treated as
        distinct points. Use ``topk_rp`` for the projective form where
        antipodes are identified.

        Returns ``(indices, dist_sq, scores)`` each shape ``(..., k)``.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        V, D = W_index.shape
        k = min(k, V)
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, D)

        x_norm2 = x_flat.square().sum(dim=-1, keepdim=True)
        scores_all = x_flat @ W_index.T - 0.5 * W_norm2.unsqueeze(0)
        scores, indices = torch.topk(scores_all, k=k, dim=-1, largest=True)

        selected_w_norm2 = W_norm2[indices]
        selected_ip = scores + 0.5 * selected_w_norm2
        dist_sq = x_norm2 + selected_w_norm2 - 2.0 * selected_ip

        out_shape = original_shape + (k,)
        return (
            indices.reshape(out_shape),
            dist_sq.reshape(out_shape),
            scores.reshape(out_shape),
        )

    # -- Projective unit-ball geometry (RP^D, default) -----------------------
    #
    # Distance:  d_RP^2(a, b) = min(||a-b||^2, ||a+b||^2)
    #                        = ||a||^2 + ||b||^2 - 2 |<a, b>|
    # Lookup:    score(x, w_i) = |<x, w_i>| - 0.5 * ||w_i||^2
    # Pode:      pode(a, b)         = (a + b) / 2     (midpoint, attractor)
    # W. pode:   wrapped_pode(a, b) = (a - b) / 2     (mid via -b, the
    #                                                  negation rep of b)
    # The projective distance is twice the smaller of |a - pode| and
    # |a - wrapped_pode|, which is the user-facing pode/wrapped-pode
    # framing.
    #
    # NOTE: the **antipode** of a single point p is the *furthest* point
    # from p in the manifold's topology -- SBOW's balancing repulsion
    # target. Unique on the flat torus (wrap(p + 1) per axis); on RP^D
    # not a unique point (orthogonal hyperplane). The wrapped pode is
    # the midpoint via -b, NOT the antipode of the regular pode.

    @staticmethod
    def rp_distance_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Projective squared distance ``min(||a-b||^2, ||a+b||^2)``.

        Reduces the last dim. Range ``[0, max(||a||, ||b||)^2 * 4]``;
        for unit-norm rows bounded above by ``2``.
        """
        inner_abs = (a * b).sum(dim=-1).abs()
        return (a.square().sum(dim=-1) + b.square().sum(dim=-1)
                - 2.0 * inner_abs)

    @staticmethod
    def rp_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Projective similarity, larger is closer. Negation of
        ``rp_distance_sq``; bounded above by 0 (when ``a = +/- b`` and
        ``||a|| == ||b||``).
        """
        return -Lexicon.rp_distance_sq(a, b)

    @staticmethod
    def rp_pode(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pode of a pair: midpoint ``(a + b) / 2``.

        Equidistant from ``a`` and ``b`` along the regular Euclidean
        line segment. The projective distance from ``a`` to ``b`` is
        ``2 * min(||a - pode||, ||a - wrapped_pode||)``.
        """
        return 0.5 * (a + b)

    @staticmethod
    def rp_wrapped_pode(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Wrapped pode: midpoint via the **negation** of ``b``,
        ``(a - b) / 2``.

        On the projective unit ball this is the "go through the
        ``+/-`` quotient" alternative midpoint. For the SBOW pair-
        attraction gradient, comparing ``||a - pode||`` and
        ``||a - wrapped_pode||`` selects which negation representative
        of ``b`` the pair should converge through. (Despite the name,
        the wrapped pode is *not* the antipode of the regular pode --
        the antipode is the furthest point in the manifold and on
        ``RP^D`` is not a unique point.)
        """
        return 0.5 * (a - b)

    @staticmethod
    def rp_closer_rep(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Closer of ``b`` and ``-b`` to ``a``, picked per row.

        Returns ``sign(<a, b>) * b``; ties (``<a, b> == 0``) resolve to
        ``+b``. The "active negation representative" of ``b`` for
        projective gradient calculations: positive-pair attraction
        targets it, negative-pair repulsion pushes ``a`` away from it.
        Distinct from the **antipode** (the manifold's furthest point
        from ``a`` -- on ``RP^D`` an orthogonal hyperplane, not a
        single point).
        """
        sign = torch.sign((a * b).sum(dim=-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return sign * b

    @staticmethod
    def rp_scores(x: torch.Tensor, W_index: torch.Tensor,
                  W_norm2: torch.Tensor) -> torch.Tensor:
        """Projective lookup scores ``|<x, w>| - 0.5 * ||w||^2``.

        Argmax over V picks the same row as projective-distance
        argmin. One matmul + elementwise abs + bias subtract.
        """
        if x.shape[-1] != W_index.shape[-1]:
            raise ValueError(
                f"x last dim {x.shape[-1]} must equal "
                f"W_index dim {W_index.shape[-1]}")
        return (x @ W_index.T).abs() - 0.5 * W_norm2

    @staticmethod
    def topk_rp(x: torch.Tensor, W_index: torch.Tensor,
                W_norm2: torch.Tensor, k: int):
        """Exact top-k nearest rows under projective distance on the
        unit ball.

        Returns ``(indices, dist_sq, scores)`` each shape ``(..., k)``.
        ``dist_sq`` is the exact projective squared distance
        ``||x||^2 + ||w||^2 - 2|<x, w>|``.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        V, D = W_index.shape
        k = min(k, V)
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, D)

        x_norm2 = x_flat.square().sum(dim=-1, keepdim=True)
        inner = x_flat @ W_index.T
        inner_abs_all = inner.abs()
        scores_all = inner_abs_all - 0.5 * W_norm2.unsqueeze(0)
        scores, indices = torch.topk(scores_all, k=k, dim=-1, largest=True)

        selected_w_norm2 = W_norm2[indices]
        selected_inner_abs = scores + 0.5 * selected_w_norm2
        dist_sq = x_norm2 + selected_w_norm2 - 2.0 * selected_inner_abs

        out_shape = original_shape + (k,)
        return (
            indices.reshape(out_shape),
            dist_sq.reshape(out_shape),
            scores.reshape(out_shape),
        )

    # -- Geometry-dispatching maintenance ------------------------------------

    @torch.no_grad()
    def normalize(self) -> None:
        """Re-project weights onto the active manifold.

        Ball geometry (``self.ball=True``, default): rowwise projection
        so ``||w_i|| <= 1``. LEGACY torus geometry (``self.ball=False``):
        wrap into ``[-1, 1)^D``. Idempotent in both cases.
        """
        if self.ball:
            self.project_unit_ball_()
        else:
            # LEGACY (torus): coordinatewise wrap into the canonical cell.
            self.weight.copy_(Lexicon.wrap(self.weight))

    # ========================================================================
    # LEGACY: torus geometry primitives ``T^D = [-1, 1)^D`` with wrapped MSE.
    # ========================================================================
    # The block below is the prior default geometry and is preserved only so
    # call sites that still reach for ``Lexicon.wrap`` / ``Lexicon.delta`` /
    # ``Lexicon.similarity`` / ``Lexicon.distance_sq`` / ``Lexicon.antipode``
    # / ``Lexicon.inner_distance`` / ``Lexicon.outer_distance`` /
    # ``Lexicon.random`` keep working unchanged during the transition to the
    # projective unit ball. New code MUST use the projective primitives
    # above (``rp_*``) or the geometry-dispatching ``normalize()``.
    #
    # Removal plan: when no in-tree call site references any of these names,
    # delete this entire block and the ``torus`` branch in ``__init__`` /
    # ``normalize``. There are no other entry points; the block is
    # self-contained.
    # ========================================================================

    @staticmethod
    def wrap(x: torch.Tensor) -> torch.Tensor:
        """LEGACY (torus). Bring ``x`` into ``[-1, 1)^D``. Idempotent.

        Differentiable almost everywhere; gradients flow as if wrap
        were the identity.
        """
        return torch.remainder(x + 1.0, 2.0) - 1.0

    @staticmethod
    def delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """LEGACY (torus). Signed shortest-path per-axis displacement
        from ``a`` to ``b``; result in ``[-1, 1)``.
        """
        return Lexicon.wrap(b - a)

    @staticmethod
    def distance_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """LEGACY (torus). Mean squared per-axis wrap distance,
        reduces the last dim. Range ``[0, 1]``.
        """
        return Lexicon.delta(a, b).square().mean(dim=-1)

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """LEGACY (torus). Wrapped-MSE similarity ``-distance_sq``;
        range ``[-1, 0]``, larger is closer.
        """
        return -Lexicon.distance_sq(a, b)

    @staticmethod
    def inner_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """LEGACY (torus). Per-axis short-way distance, range ``[0, 1]``."""
        return Lexicon.delta(a, b).abs()

    @staticmethod
    def outer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """LEGACY (torus). Per-axis long-way distance, range ``[1, 2]``.
        ``inner + outer == 2`` per axis.
        """
        return 2.0 - Lexicon.inner_distance(a, b)

    @staticmethod
    def antipode(p: torch.Tensor) -> torch.Tensor:
        """LEGACY (torus). Per-axis ``+1 (mod 2)`` shift; the unique
        point at maximum torus distance from ``p``.
        """
        return Lexicon.wrap(p + 1.0)

    @staticmethod
    def random(shape, *, device=None, dtype=torch.float32) -> torch.Tensor:
        """LEGACY (torus). Uniform sample on ``T^D = [-1, 1)^D``.
        Initialization for a ``torus=True`` Lexicon.
        """
        return torch.empty(shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)


@torch.no_grad()
def topk_rp_chunked(
    x: torch.Tensor,
    W_index: torch.Tensor,
    W_norm2: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 131_072,
):
    """Exact chunked top-k nearest lookup under projective L2 in the unit ball.

    Avoids materializing a full ``(B, V)`` score matrix; scores in chunks of
    ``chunk_size`` lexicon rows and merges per-chunk top-k results. Score
    is ``|<x, w>| - 0.5 * ||w||^2`` (projective lookup); returned
    ``dist_sq`` is the exact projective squared distance
    ``||x||^2 + ||w||^2 - 2 |<x, w>|``.

    Args:
        x: ``[..., D]`` query.
        W_index: ``[V, D]`` lexicon matrix.
        W_norm2: ``[V]`` precomputed squared row norms of ``W_index``.
        k: number of nearest rows to return.
        chunk_size: rows of ``W_index`` to score per pass.

    Returns:
        ``(indices, dist_sq, scores)`` each shape ``(..., k)``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    V, D = W_index.shape
    k = min(k, V)
    original_shape = x.shape[:-1]
    x_flat = x.reshape(-1, D)

    best_scores = None
    best_indices = None

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        Wc = W_index[start:end]
        Nc = W_norm2[start:end]

        scores_c = (x_flat @ Wc.T).abs() - 0.5 * Nc.unsqueeze(0)

        local_k = min(k, end - start)
        scores_k, indices_k = torch.topk(
            scores_c, k=local_k, dim=-1, largest=True)
        indices_k = indices_k + start

        if best_scores is None:
            best_scores = scores_k
            best_indices = indices_k
        else:
            merged_scores = torch.cat([best_scores, scores_k], dim=-1)
            merged_indices = torch.cat([best_indices, indices_k], dim=-1)
            best_scores, pos = torch.topk(
                merged_scores, k=k, dim=-1, largest=True)
            best_indices = merged_indices.gather(-1, pos)

    x_norm2 = x_flat.square().sum(dim=-1, keepdim=True)
    Wk_norm2 = W_norm2[best_indices]
    selected_inner_abs = best_scores + 0.5 * Wk_norm2
    dist_sq = x_norm2 + Wk_norm2 - 2.0 * selected_inner_abs

    out_shape = original_shape + (k,)
    return (
        best_indices.reshape(out_shape),
        dist_sq.reshape(out_shape),
        best_scores.reshape(out_shape),
    )


@torch.no_grad()
def topk_l2_chunked(
    x: torch.Tensor,
    W_index: torch.Tensor,
    W_norm2: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 131_072,
):
    """Exact chunked top-k nearest lookup under plain (non-projective)
    Euclidean L2 in the unit ball.

    Avoids materializing a full ``(B, V)`` score matrix; instead scores
    in chunks of ``chunk_size`` lexicon rows and merges per-chunk
    top-k results. The final ``dist_sq`` is exact.

    Args:
        x: ``[..., D]`` query.
        W_index: ``[V, D]`` lexicon matrix.
        W_norm2: ``[V]`` precomputed squared row norms of ``W_index``.
        k: number of nearest rows to return.
        chunk_size: rows of ``W_index`` to score per pass.

    Returns:
        ``(indices, dist_sq, scores)`` each shape ``(..., k)``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    V, D = W_index.shape
    k = min(k, V)
    original_shape = x.shape[:-1]
    x_flat = x.reshape(-1, D)

    best_scores = None
    best_indices = None

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        Wc = W_index[start:end]
        Nc = W_norm2[start:end]

        scores_c = x_flat @ Wc.T - 0.5 * Nc.unsqueeze(0)

        local_k = min(k, end - start)
        scores_k, indices_k = torch.topk(
            scores_c, k=local_k, dim=-1, largest=True)
        indices_k = indices_k + start

        if best_scores is None:
            best_scores = scores_k
            best_indices = indices_k
        else:
            merged_scores = torch.cat([best_scores, scores_k], dim=-1)
            merged_indices = torch.cat([best_indices, indices_k], dim=-1)
            best_scores, pos = torch.topk(
                merged_scores, k=k, dim=-1, largest=True)
            best_indices = merged_indices.gather(-1, pos)

    x_norm2 = x_flat.square().sum(dim=-1, keepdim=True)
    Wk_norm2 = W_norm2[best_indices]
    selected_ip = best_scores + 0.5 * Wk_norm2
    dist_sq = x_norm2 + Wk_norm2 - 2.0 * selected_ip

    out_shape = original_shape + (k,)
    return (
        best_indices.reshape(out_shape),
        dist_sq.reshape(out_shape),
        best_scores.reshape(out_shape),
    )


#region Layers


class Layer(nn.Module):
    """Base class for custom layers with optional symbol/object axis swapping.

    Composite layers should register their child layers in ``self.layers``
    so that the ergodic interface (set_sigma, observe_sigma, etc.) and
    paramUpdate are automatically forwarded to them.
    """
    def __init__(self, nInput, nOutput):
        """Initialize Layer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super(Layer, self).__init__()
        self.nInput       = nInput
        self.nOutput      = nOutput
        self.batch        = 0
        self.layers       = []  # child layers; populate in subclass __init__

    def freeze(self, learn=False):
        """Freeze all params (learn=False) or unfreeze them (learn=True)."""
        for param in self.parameters():
            param.requires_grad = not learn
    def getParameters(self):
        """Return optimizable parameters owned by this module."""
        params = [p for n, p in self.named_parameters()]
        return params

    # --- Ergodic interface: dispatched to self.layers automatically ---
    def paramUpdate(self):
        """In-place parameter update hook called once per training step."""
        for layer in self.layers:
            layer.paramUpdate()
    def set_sigma(self, sigma):
        """Set the exploration sigma scale on this module."""
        for layer in self.layers:
            layer.set_sigma(sigma)
    def observe_sigma(self):
        """Observe sigma.
        
        See class docstring for the operation contract.
        """
        for layer in self.layers:
            layer.observe_sigma()
    def sigma_to_ergodic(self):
        """Sigma to ergodic.
        
        See class docstring for the operation contract.
        """
        for layer in self.layers:
            layer.sigma_to_ergodic()

    def Start(self):
        """Per-sentence state reset. Cascades to child layers.

        Layers with per-call state (e.g. accumulating regularizer
        buffers) override this to clear that state. The default walks
        self.layers.
        """
        for layer in self.layers:
            if hasattr(layer, 'Start'):
                layer.Start()

    def End(self):
        """Per-batch teardown. Counterpart to Start().

        Cascades End() to child layers so any per-call scratch state
        cleared in Start() is also released at batch completion. The
        default walks self.layers; subclasses with additional per-call
        caches override to drop them.
        """
        for layer in self.layers:
            if hasattr(layer, 'End'):
                layer.End()

    def forward(self, x):
        """Identity pass-through (subclasses override)."""
        batch = x.shape[0]
        return x
    def reverse(self, y):
        """Identity pass-through (subclasses override)."""
        batch = y.shape[0]
        # For 3D tensors, nOutput matches the last dim (embedding), not dim 1 (sequence)
        if y.ndim == 3:
            assert y.shape[2] == self.nOutput
        else:
            assert y.shape[1] == self.nOutput
        return y
class ErgodicLayer(Layer):
    """Layer base class that adapts its explore/exploit balance over training.

    Tracks **sigma** -- the running variance of the layer's gradient energy.
    Sigma drives scalar bias and var that broadcast over any weight shape:

        var  = sigma / (sigma + kappa)      exploration noise
        bias = 1 - var                      weight trust

    Low sigma (consistent gradient, found a minimum) -> high bias, low var.
    High sigma (unstable gradient) -> low bias, high var.

    Subclasses mix learned weights (scaled by ``bias``) with random noise
    (scaled by ``var``) in their forward passes:
        effective_weight = bias * W + var * noise

    External control via ``set_sigma(sigma)``:
        sigma=1: responsive exploration (low kappa)
        sigma=0: suppress exploration (high kappa, var ~= 0)

    ``ergodic=False`` (default): sigma tracking is still wired but
    sigma_to_ergodic / paramUpdate are no-ops, keeping bias=1, var=0.
    """
    def __init__(self, nInput, nOutput, ergodic=False):
        """Initialize ErgodicLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nInput, nOutput)
        self.ergodic = ergodic

        # --- Scalar explore/exploit (broadcasts over any weight shape) -----
        # Start in pure-exploit mode; sigma_to_ergodic drives these after
        # the first gradient step.
        self.register_buffer('bias', torch.ones(1))
        self.register_buffer('var',  torch.zeros(1))

        # --- Scalar sigma: running gradient variance ----------------------
        self.register_buffer('sigma',      torch.zeros(1))
        self.register_buffer('sigma_mean', torch.zeros(1))
        self.register_buffer('sigma_step', torch.tensor(0))
        self.sigma_beta = 0.99
        self.sigma_kappa = 0.01

        self.dropoutRate = 0.0
        # Initialize in moderate exploration mode
        self.set_sigma(0.5)

    def getParameters(self):
        """Return learnable params (var/bias are buffers, not parameters)."""
        return list(self.parameters())

    def set_sigma(self, sigma):
        """Control exploration meta-parameters.

        sigma=1: encourage exploration (low kappa, responsive to gradient variance)
        sigma=0: suppress exploration (high kappa, var ~= 0)
        """
        if sigma == 0:
            self.sigma_kappa = 1e6
        else:
            self.sigma_kappa = 0.01 / sigma
            self.sigma_beta = 0.99

    @torch.no_grad()
    def observe_sigma(self):
        """Track gradient variance via Welford's algorithm.

        Aggregates mean squared gradient energy across all weight parameters
        into a single scalar, then updates the running mean and variance.
        """
        grad_energy = None
        for name, param in self.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            if "noise" in name.lower():
                continue
            energy = param.grad.detach().pow(2).mean()
            grad_energy = energy if grad_energy is None else grad_energy + energy
        if grad_energy is None:
            return
        self.sigma_step += 1
        beta = self.sigma_beta
        delta = grad_energy - self.sigma_mean
        self.sigma_mean.add_((1 - beta) * delta)
        self.sigma.mul_(beta).add_((1 - beta) * delta * (grad_energy - self.sigma_mean))

    @torch.no_grad()
    def sigma_to_ergodic(self):
        """Update scalar bias and var from sigma (gradient variance)."""
        if not self.ergodic:
            return
        # One .item() call instead of two -- the value is reused below.
        # Each .item() is a GPU->CPU sync and runs once per backward step.
        step = self.sigma_step.item()
        if step == 0:
            return
        # Bias-corrected sigma estimate
        s = self.sigma / (1 - self.sigma_beta ** step)
        s = s.clamp(min=0)
        self.var.copy_((s / (s + self.sigma_kappa)).clamp(0, 0.95))
        self.bias.copy_((1.0 - self.var).clamp(min=0.05))

    def paramUpdate(self):
        """In-place parameter update hook called once per training step."""
        if not self.ergodic:
            return
        self.observe_sigma()
        self.sigma_to_ergodic()
class LinearLayer(ErgodicLayer):
    """Standard linear (affine) layer.

    Forward (ergodic):     y = x @ (bias*W + var*noise) [+ bias*biasWeight + var*biasNoise]
    Forward (non-ergodic): y = x @ W [+ b]

    ``bias`` and ``var`` come from the ErgodicLayer explore/exploit schedule.
    When ergodic=False (default), noise buffers are not allocated and forward
    uses the learned weights directly.
    """
    def __init__(self, nInput, nOutput, hasBias=True, naive=False, stable=False, ergodic=False):
        """Initialize LinearLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super(LinearLayer, self).__init__(nInput, nOutput, ergodic=ergodic)
        self.stable  = stable
        self.hasBias = hasBias
        if ergodic:
            self.W = nn.Parameter(torch.eye(self.nInput, self.nOutput))
        else:
            self.W = nn.Parameter((2 * torch.rand(self.nInput, self.nOutput) - 1) / self.nInput)
        if self.hasBias:
            self.biasWeight = nn.Parameter(torch.zeros(1, nOutput))
        if ergodic:
            self.register_buffer('noise', torch.randn(self.nInput, self.nOutput))
            if self.hasBias:
                self.register_buffer('biasNoise', torch.randn(1, nOutput))
        else:
            self.set_sigma(0)

    def resample_noise(self):
        """Draw fresh Gaussian noise matching W shape/device. No-op if not ergodic."""
        if self.ergodic:
            self.noise = torch.randn(self.W.shape, device=TheDevice.get(), dtype=self.W.dtype)
            if self.hasBias:
                self.biasNoise = torch.randn(
                    self.biasWeight.shape,
                    device=TheDevice.get(),
                    dtype=self.biasWeight.dtype,
                )

    def compute_W_current(self):
        """Effective W for outer-product use (respects ergodic noise if active)."""
        if self.ergodic:
            return self.bias * self.W + self.var * self.noise
        return self.W

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        if self.ergodic:
            self.resample_noise()
            W = self.bias * self.W + self.var * self.noise
        else:
            W = self.W
        output = x @ W
        return output
    def forwardBias(self, x):
        """Forward bias.
        
        See class docstring for the operation contract.
        """
        if self.hasBias:
            if self.ergodic:
                x = x + self.bias * self.biasWeight + self.var * self.biasNoise
            else:
                x = x + self.biasWeight
        return x

    def _effective_bias(self):
        """Bias for log-space use (unconstrained)."""
        if not self.hasBias:
            return 0
        if self.ergodic:
            return self.bias * self.biasWeight + self.var * self.biasNoise
        return self.biasWeight

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        nInput, nOutput = 3, 4
        layer = LinearLayer(nInput=nInput, nOutput=nOutput)
        input = torch.rand((1, nInput), device=TheDevice.get())
        output = layer(input)

        print(f"Input: {input}")
        print(f"After forward linear: {output}")
class NonNegativeLinearLayer(LinearLayer):
    """LinearLayer with entry-wise non-negative W via softplus.

    Applies softplus to the raw weight matrix so that all entries of the
    effective W are non-negative.  This preserves the monotonicity property
    of NonNegativeInvertibleLinearLayer without the LDU factorisation or
    invertibility machinery.

    Parameters are initialised so that softplus(raw) ~= 1.0 (near-identity
    scaling), matching the NonNegativeInvertibleLinearLayer convention.
    """

    def __init__(self, nInput, nOutput, hasBias=True, naive=False, stable=False, ergodic=False):
        """Initialize NonNegativeLinearLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nInput, nOutput, hasBias=hasBias, naive=naive, stable=stable, ergodic=ergodic)
        with torch.no_grad():
            self.W.fill_(math.log(math.e - 1))  # softplus_inverse(1.0)

    def _get_W(self):
        """Effective weight matrix, non-negative by construction."""
        return nn.functional.softplus(self.W)

    def compute_W_current(self):
        """Compute w current.
        
        See class docstring for the operation contract.
        """
        if self.ergodic:
            return nn.functional.softplus(self.bias * self.W + self.var * self.noise)
        return self._get_W()

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        if self.ergodic:
            self.resample_noise()
            W = nn.functional.softplus(self.bias * self.W + self.var * self.noise)
        else:
            W = self._get_W()
        return x @ W

    def _effective_bias(self):
        """Bias constrained to non-negative via softplus, matching NonNegativeInvertibleLinearLayer."""
        if not self.hasBias:
            return 0
        if self.ergodic:
            raw = self.bias * self.biasWeight + self.var * self.biasNoise
            return F.softplus(raw)
        return F.softplus(self.biasWeight)

    def forwardBias(self, x):
        """Forward bias.
        
        See class docstring for the operation contract.
        """
        if self.hasBias:
            x = x + self._effective_bias()
        return x
class InvertibleLinearLayer(ErgodicLayer):
    """Exactly-invertible linear layer factored as W = L @ D_embed @ U.

    L is unit-lower-triangular [nInput, nInput], D is diagonal [rank],
    U is unit-upper-triangular [nOutput, nOutput].  D_embed zero-pads D
    into [nInput, nOutput] for rectangular cases.

    Non-ergodic inverse is exact via triangular solves: W^-^1 = U^-^1 D^-^1 L^-^1.

    Ergodic mode injects noise into each factor before extracting the
    triangular structure, preserving the LDU form so the exact inverse is
    always available -- no approximation or SVD required:
        L_eff = I + strict_lower(raw_L + t * noise_raw_L)
        U_eff = I + strict_upper(raw_U + t * noise_raw_U)
        d_eff = b * d_effective + t * noise_d
        W_eff = L_eff @ D_eff_embed @ U_eff

    stable=True clamps d to [eps, 1] magnitude (sign preserved) via
    _d_effective(), keeping W well-conditioned and invertible.

    naive=False (default): sequential triangular solves, no W materialisation.
    naive=True: materialise W_eff as a dense matrix; reverse materialises the
    dense LDU inverse.
    """
    def __init__(self, nInput, nOutput, naive=False, ergodic=False,
                 hasBias=True, stable=False):
        """Initialize InvertibleLinearLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nInput, nOutput, ergodic=ergodic)
        self.naive   = naive
        self.hasBias = hasBias
        self.stable  = stable
        self.rank    = min(nInput, nOutput)

        # LDU learned parameters
        self.raw_L = nn.Parameter(torch.zeros(nInput, nInput))
        self.d     = nn.Parameter(torch.ones(self.rank))
        self.raw_U = nn.Parameter(torch.zeros(nOutput, nOutput))
        if hasBias:
            self.biasWeight = nn.Parameter(torch.zeros(1, nOutput))
        if ergodic:
            # Factor-level noise: perturb each LDU factor independently
            self.register_buffer('noise_raw_L', torch.randn(nInput, nInput))
            self.register_buffer('noise_raw_U', torch.randn(nOutput, nOutput))
            self.register_buffer('noise_d',     torch.ones(self.rank))
            if hasBias:
                self.register_buffer('biasNoise', torch.randn(1, nOutput))

    # --- Factor helpers ---
    def _L(self):
        """Unit-lower-triangular: strict lower of raw_L + I."""
        return torch.tril(self.raw_L, diagonal=-1) + torch.eye(
            self.nInput, device=self.raw_L.device, dtype=self.raw_L.dtype)

    def _U(self):
        """Unit-upper-triangular: strict upper of raw_U + I."""
        return torch.triu(self.raw_U, diagonal=1) + torch.eye(
            self.nOutput, device=self.raw_U.device, dtype=self.raw_U.dtype)

    def _d_effective(self):
        """Return d clamped to [eps, 1] magnitude with sign preserved when
        stable=True, else raw self.d.  Stability constraint lives here only.

        When a caller has stashed a per-call ``_current_gate`` (a ``[rank]``
        tensor of bounded multipliers, by convention in ``[-1, 1]`` via a
        tanh reparam owned by the caller -- see LiftLayer / LowerLayer),
        d is multiplied by it before the stable-clamp so the existing
        clamp bounds the gated value. With ``gate=None`` (default), this
        is a no-op and the layer behaves identically to its un-gated
        baseline.
        """
        d = self.d
        gate = getattr(self, '_current_gate', None)
        if gate is not None:
            d = d * gate
        if self.stable:
            return d.sign() * d.abs().clamp(epsilon, 1.0)
        return d

    def _D_embed(self):
        """Embed _d_effective() into [nInput, nOutput] rectangular diagonal."""
        d = self._d_effective()
        D = torch.zeros(self.nInput, self.nOutput, device=d.device, dtype=d.dtype)
        for i in range(self.rank):
            D[i, i] = d[i]
        return D

    # --- Noise resampling ---
    def resample_noise(self):
        """Resample noise.
        
        See class docstring for the operation contract.
        """
        if not self.ergodic:
            return
        # Mask to the entries that survive tril/triu, then unit-normalise so
        # that var is the exact Frobenius norm of the perturbation regardless
        # of matrix size.
        n_L = torch.tril(torch.randn_like(self.noise_raw_L), diagonal=-1)
        self.noise_raw_L = n_L / (n_L.norm() + 1e-8)

        n_U = torch.triu(torch.randn_like(self.noise_raw_U), diagonal=1)
        self.noise_raw_U = n_U / (n_U.norm() + 1e-8)

        n_d = torch.randn_like(self.noise_d)
        self.noise_d = n_d / (n_d.norm() + 1e-8)

        if self.hasBias:
            n_b = torch.randn_like(self.biasNoise)
            self.biasNoise = n_b / (n_b.norm() + 1e-8)

    # --- Effective ergodic factors ---
    def _L_eff(self):
        """L_eff = I + strict_lower(raw_L + var * noise_raw_L).
        noise_raw_L is pre-masked to strict lower triangular and unit-normalised."""
        raw = self.raw_L + self.var * self.noise_raw_L
        return (torch.tril(raw, diagonal=-1)
                + torch.eye(self.nInput, device=raw.device, dtype=raw.dtype))

    def _U_eff(self):
        """U_eff = I + strict_upper(raw_U + var * noise_raw_U).
        noise_raw_U is pre-masked to strict upper triangular and unit-normalised."""
        raw = self.raw_U + self.var * self.noise_raw_U
        return (torch.triu(raw, diagonal=1)
                + torch.eye(self.nOutput, device=raw.device, dtype=raw.dtype))

    def _d_eff(self):
        """d_eff = bias * d_effective + var * noise_d.
        noise_d is unit-normalised so var is the exact L2 norm of the perturbation.
        When stable=True, clamp magnitude to [eps, 1] so W_inv never blows up."""
        d = self.bias * self._d_effective() + self.var * self.noise_d
        if self.stable:
            d = d.sign() * d.abs().clamp(epsilon, 1.0)
        return d

    # --- W materialisation ---
    def compute_W(self):
        """W = L @ D_embed @ U.  Shape [nInput, nOutput]."""
        return self._L() @ self._D_embed() @ self._U()

    def compute_Winverse(self):
        """W^-^1 = U^-^1 @ D_embed^-^1 @ L^-^1.  Shape [nOutput, nInput].
        Uses _d_effective() so stable=True is automatically respected."""
        L = self._L()
        U = self._U()
        I_in  = torch.eye(self.nInput,  device=L.device, dtype=L.dtype)
        I_out = torch.eye(self.nOutput, device=U.device, dtype=U.dtype)
        L_inv = torch.linalg.solve_triangular(L, I_in,  upper=False, unitriangular=True)
        U_inv = torch.linalg.solve_triangular(U, I_out, upper=True,  unitriangular=True)
        d = self._d_effective()
        D_inv = torch.zeros(self.nOutput, self.nInput, device=d.device, dtype=d.dtype)
        for i in range(self.rank):
            D_inv[i, i] = 1.0 / d[i]
        return U_inv @ D_inv @ L_inv

    def compute_W_current(self):
        """Materialise W using current ergodic factors if ergodic, else clean W."""
        if not self.ergodic:
            return self.compute_W()
        L = self._L_eff()
        d = self._d_eff()
        U = self._U_eff()
        D = torch.zeros(self.nInput, self.nOutput, device=d.device, dtype=d.dtype)
        for i in range(self.rank):
            D[i, i] = d[i]
        return L @ D @ U

    def compute_Winverse_current(self):
        """Exact inverse of compute_W_current(). Shape [nOutput, nInput]."""
        if not self.ergodic:
            return self.compute_Winverse()
        L = self._L_eff()
        d = self._d_eff()
        U = self._U_eff()
        I_in  = torch.eye(self.nInput,  device=L.device, dtype=L.dtype)
        I_out = torch.eye(self.nOutput, device=U.device, dtype=U.dtype)
        L_inv = torch.linalg.solve_triangular(L, I_in,  upper=False, unitriangular=True)
        U_inv = torch.linalg.solve_triangular(U, I_out, upper=True,  unitriangular=True)
        D_inv = torch.zeros(self.nOutput, self.nInput, device=d.device, dtype=d.dtype)
        for i in range(self.rank):
            D_inv[i, i] = 1.0 / d[i]
        return U_inv @ D_inv @ L_inv

    # --- Parameterised sequential apply / solve ---
    def _apply_ldu(self, x, L, d, U):
        """Apply x @ L @ D_embed(d) @ U sequentially.  d is a [rank] vector.
        Supports arbitrary leading batch dimensions."""
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        x = x @ L
        if self.nInput <= self.nOutput:
            scaled = x * d
            if self.nOutput > self.nInput:
                pad = torch.zeros(x.shape[0], self.nOutput - self.nInput,
                                  device=x.device, dtype=x.dtype)
                x = torch.cat([scaled, pad], dim=-1)
            else:
                x = scaled
        else:
            # rank == nOutput here (min(nInput,nOutput)), so no zero-pad needed
            x = x[..., :self.rank] * d
        x = x @ U
        out_shape = list(orig_shape); out_shape[-1] = self.nOutput
        return x.reshape(out_shape)

    def _solve_ldu(self, y, L, d, U):
        """Solve y @ (L D U) = x exactly via triangular solves.  d is [rank].
        Supports arbitrary leading batch dimensions."""
        orig_shape = y.shape
        y = y.reshape(-1, orig_shape[-1])
        y = torch.linalg.solve_triangular(U.T, y.T, upper=False, unitriangular=True).T
        if self.nInput <= self.nOutput:
            # rank == nInput here (min(nInput,nOutput)), so no zero-pad needed
            y = y[..., :self.rank] / d
        else:
            y = y / d
            pad = torch.zeros(y.shape[0], self.nInput - self.rank,
                              device=y.device, dtype=y.dtype)
            y = torch.cat([y, pad], dim=-1)
        y = torch.linalg.solve_triangular(L.T, y.T, upper=True, unitriangular=True).T
        out_shape = list(orig_shape); out_shape[-1] = self.nInput
        return y.reshape(out_shape)

    def _apply_forward(self, x):
        """Non-ergodic sequential forward using clean L, d, U."""
        return self._apply_ldu(x, self._L(), self._d_effective(), self._U())

    def _solve_reverse(self, y):
        """Non-ergodic sequential reverse using clean L, d_effective, U."""
        return self._solve_ldu(y, self._L(), self._d_effective(), self._U())

    # --- Public matmul-free apply / solve --------------------------
    # ``compute_W_current`` / ``compute_Winverse_current`` materialise
    # the full ``[D, N]`` / ``[N, D]`` matrix; callers that only need
    # ``x @ W`` or ``y @ W^-1`` against a batched RHS pay an O(n^3)
    # build + an O(b*n^2) matmul.  These wrappers skip the build:
    # apply_W_current runs ``x @ L @ D_embed @ U`` directly via three
    # cheap matmuls; apply_Winverse_current runs two triangular solves
    # plus a diagonal divide -- total cost O(b*n^2).  No dense W or
    # W^-1 is ever materialised in either path, so transient memory
    # stays at O(b*n) for the intermediate states.

    def apply_W_current(self, x, gate=None):
        """Return ``x @ W_current`` without materialising W.

        Ergodic-aware: uses ``L_eff / d_eff / U_eff`` when
        ``self.ergodic`` is True, else the clean factors.  Matches
        the math of ``compute_W_current()`` but routes through the
        per-factor matmul chain so transient memory is ``O(b*n)``
        rather than ``O(n^2)``.
        """
        self._current_gate = gate
        try:
            if self.ergodic:
                return self._apply_ldu(
                    x, self._L_eff(), self._d_eff(), self._U_eff())
            return self._apply_forward(x)
        finally:
            self._current_gate = None

    def apply_Winverse_current(self, y, gate=None):
        """Return ``y @ Winverse_current`` without materialising W^-1.

        Ergodic-aware mirror of ``apply_W_current``: uses two
        triangular solves (``solve_triangular(L, ...)`` and
        ``solve_triangular(U, ...)``) on the RHS plus a diagonal
        divide.  Cost is ``O(b * n^2)`` and transient memory is
        ``O(b * n)``; the legacy ``y @ compute_Winverse_current()``
        path materialises ``L^-1`` / ``U^-1`` / ``D^-1`` (3 × ``O(n^2)``
        scratch) and chains two ``O(n^3)`` matmuls.
        """
        self._current_gate = gate
        try:
            if self.ergodic:
                return self._solve_ldu(
                    y, self._L_eff(), self._d_eff(), self._U_eff())
            return self._solve_reverse(y)
        finally:
            self._current_gate = None

    # --- Forward / Reverse ---
    def forward(self, x, gate=None):
        """Apply the LDU transform with optional ergodic noise injection.

        Ergodic (self.ergodic=True):
          Resamples noise at the start, then builds effective factors:
            L_eff = I + strict_lower(raw_L + t * noise_raw_L)
            U_eff = I + strict_upper(raw_U + t * noise_raw_U)
            d_eff = b * d_effective + t * noise_d
          naive=True: materialise W_eff = L_eff D_eff U_eff, dense matmul.
          naive=False: sequential triangular solves via _apply_ldu.
          Stored noise buffers remain unchanged after this call so that
          reverse() can reconstruct the identical factors exactly.

        Non-ergodic (self.ergodic=False):
          naive=True: materialise W = L D U, dense matmul.
          naive=False: sequential triangular solves.

        ``gate``: optional ``[rank]`` tensor multiplied into the diagonal
        ``d`` before the stable-clamp. Used by rule layers (LiftLayer /
        LowerLayer) to select a low-rank slice of the operator. With
        ``gate=None`` (default), behavior is unchanged from the un-gated
        baseline.
        """
        self._current_gate = gate
        try:
            if self.ergodic:
                self.resample_noise()
            if self.naive:
                W = self.compute_W_current()
                orig_shape = x.shape
                out_shape = list(orig_shape); out_shape[-1] = self.nOutput
                y = (x.reshape(-1, self.nInput) @ W).reshape(out_shape)
            else:
                if self.ergodic:
                    y = self._apply_ldu(x, self._L_eff(), self._d_eff(), self._U_eff())
                else:
                    y = self._apply_forward(x)
            y = self.forwardBias(y)
            return y
        finally:
            self._current_gate = None
    def _effective_bias(self):
        """Bias value for external use (e.g. PiLayer logit mode)."""
        if not self.hasBias:
            return 0
        if self.ergodic:
            return self.bias * self.biasWeight + self.var * self.biasNoise
        return self.biasWeight

    def forwardBias(self, x):
        """Forward bias.
        
        See class docstring for the operation contract.
        """
        if self.hasBias:
            if self.ergodic:
                x = x + self.bias * self.biasWeight + self.var * self.biasNoise
            else:
                x = x + self.biasWeight
        return x
    def forwardBiasInterleaved(self, x):
        """Forward bias interleaved.
        
        See class docstring for the operation contract.
        """
        if self.hasBias:
            # [..., 2*S, nOut]: pairs along dim=-2, alternate +b/-b every row
            signs = x.new_ones(x.shape[-2], 1)
            signs[1::2, 0] = -1
            bWeight = signs * self.biasWeight
            if self.ergodic:
                bNoise = signs * self.biasNoise
                x = x + self.bias * bWeight + self.var * bNoise
            else:
                x = x + bWeight
        return x

    def reverseBias(self, y):
        """Reverse bias.
        
        See class docstring for the operation contract.
        """
        if self.hasBias:
            if self.ergodic:
                y = y - (self.bias * self.biasWeight + self.var * self.biasNoise)
            else:
                y = y - self.biasWeight
        return y

    def reverseBiasInterleaved(self, y):
        """Reverse bias interleaved.
        
        See class docstring for the operation contract.
        """
        if self.hasBias:
            # [..., 2*S, nOut]: pairs along dim=-2, alternate +b/-b every row
            signs = y.new_ones(y.shape[-2], 1)
            signs[1::2, 0] = -1
            bWeight = signs * self.biasWeight
            if self.ergodic:
                bNoise = signs * self.biasNoise
                y = y - (self.bias * bWeight + self.var * bNoise)
            else:
                y = y - bWeight
        return y
    def reverse(self, y, gate=None):
        """Invert the LDU transform.

        Ergodic (self.ergodic=True):
          Reconstructs L_eff, d_eff, U_eff from the noise buffers set during
          the preceding forward() call -- no new resampling is done until after
          the result is computed.
          naive=False: triangular solves (exact, no materialisation).
          naive=True: materialise the dense LDU inverse.
          Resamples noise at end so the layer is ready for the next forward().

        Non-ergodic (self.ergodic=False):
          naive=True: materialise W_inv = U_inv D_inv L_inv, dense matmul.
          naive=False: sequential triangular solves.

        ``gate``: same gate tensor that was passed to ``forward``. The
        gate enters via ``_d_effective`` (and the ergodic ``_d_eff``),
        which in turn drives both ``compute_W_current`` (forward) and
        ``compute_Winverse_current`` (reverse), so the inverse uses
        ``1/(d * gate)`` automatically -- no separate gate-aware-reverse
        code path. With ``gate=None``, behavior is unchanged.
        """
        self._current_gate = gate
        try:
            y = self.reverseBias(y)
            if self.naive:
                W_inv = self.compute_Winverse_current()
                orig_shape = y.shape
                out_shape = list(orig_shape); out_shape[-1] = self.nInput
                result = (y.reshape(-1, self.nOutput) @ W_inv).reshape(out_shape)
            else:
                if self.ergodic:
                    result = self._solve_ldu(y, self._L_eff(), self._d_eff(), self._U_eff())
                else:
                    result = self._solve_reverse(y)
            if self.ergodic:
                self.resample_noise()
            return result
        finally:
            self._current_gate = None

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        torch.manual_seed(42)
        device = TheDevice.get()
        nInput, nOutput = 7, 11

        # Non-ergodic roundtrip (expand)
        layer = InvertibleLinearLayer(nInput=nInput, nOutput=nOutput, hasBias=False)
        layer.set_sigma(0)
        x = torch.randn(5, nInput, device=device)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = torch.norm(x - x_rec) / torch.norm(x)
        assert err < 1e-4, f"roundtrip error: {err:.2e}"
        print(f"InvertibleLinearLayer roundtrip: err={err:.2e} OK")

        # With bias
        layer2 = InvertibleLinearLayer(nInput=nInput, nOutput=nOutput, hasBias=True)
        layer2.set_sigma(0)
        y2 = layer2.forward(x)
        x_rec2 = layer2.reverse(y2)
        err2 = torch.norm(x - x_rec2) / torch.norm(x)
        assert err2 < 1e-4, f"bias roundtrip error: {err2:.2e}"
        print(f"InvertibleLinearLayer bias roundtrip: err={err2:.2e} OK")

        # Square W @ W_inv = I
        layer3 = InvertibleLinearLayer(nInput=5, nOutput=5, hasBias=False)
        layer3.set_sigma(0)
        W = layer3.compute_W(); W_inv = layer3.compute_Winverse()
        id_err = torch.norm(W @ W_inv - torch.eye(5, device=W.device))
        assert id_err < 1e-4, f"W@W_inv identity err: {id_err:.2e}"
        print(f"InvertibleLinearLayer W@W_inv identity: err={id_err:.2e} OK")

        # Ergodic roundtrip (factor-level noise -> exact inverse)
        elayer = InvertibleLinearLayer(nInput=5, nOutput=5, ergodic=True, stable=True)
        with torch.no_grad():
            elayer.var.fill_(0.2)
            elayer.bias.fill_(0.8)
        x3 = torch.randn(4, 5, device=device)
        y3 = elayer.forward(x3)
        x_rec3 = elayer.reverse(y3)
        err3 = torch.norm(x3 - x_rec3) / torch.norm(x3)
        assert err3 < 1e-4, f"ergodic roundtrip error: {err3:.2e}"
        print(f"InvertibleLinearLayer ergodic roundtrip: err={err3:.2e} OK")

        print("All InvertibleLinearLayer tests passed!")
class NonNegativeInvertibleLinearLayer(InvertibleLinearLayer):
    """InvertibleLinearLayer whose W = L @ D @ U is entry-wise non-negative.

    Applies softplus to off-diagonal entries of L and U, and to the
    diagonal d, so that all factors are non-negative.  Since the product
    of non-negative matrices is non-negative, W >= 0 by construction.

    The exact LDU inverse (triangular solves) is still available -- the
    inverse of a non-negative W is not itself non-negative, but that's
    fine since only the forward W needs the constraint.

    Parameters are initialised so that softplus(raw) starts near the
    identity: raw_L, raw_U off-diagonals at -5 (softplus(-5) ~ 0.007),
    d at softplus_inverse(1) ~ 0.541.
    """

    def __init__(self, nInput, nOutput, naive=False, ergodic=False,
                 hasBias=True, stable=False):
        """Initialize NonNegativeInvertibleLinearLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nInput, nOutput, naive=naive, ergodic=ergodic,
                         hasBias=hasBias, stable=stable)
        # Re-initialise so softplus(raw) gives near-identity W.
        with torch.no_grad():
            # softplus(-5) ~ 0.007 ~= 0 for off-diagonals
            self.raw_L.fill_(-5.0)
            self.raw_U.fill_(-5.0)
            # softplus(0.5414) ~ 1.0 for diagonal
            self.d.fill_(math.log(math.e - 1))  # softplus_inverse(1.0)

    # --- Non-negative factor helpers ---
    def _L(self):
        """Unit-lower-triangular with non-negative off-diagonals."""
        off = torch.tril(nn.functional.softplus(self.raw_L), diagonal=-1)
        return off + torch.eye(self.nInput, device=self.raw_L.device, dtype=self.raw_L.dtype)

    def _U(self):
        """Unit-upper-triangular with non-negative off-diagonals."""
        off = torch.triu(nn.functional.softplus(self.raw_U), diagonal=1)
        return off + torch.eye(self.nOutput, device=self.raw_U.device, dtype=self.raw_U.dtype)

    def _d_effective(self):
        """Strictly positive diagonal via softplus."""
        d = nn.functional.softplus(self.d)
        gate = getattr(self, '_current_gate', None)
        if gate is not None:
            d = d * gate
        if self.stable:
            d = d.clamp(epsilon, 1.0)
        return d

    # --- Ergodic overrides ---
    def _L_eff(self):
        """Ergodic L with non-negative off-diagonals."""
        raw = self.raw_L + self.var * self.noise_raw_L
        off = torch.tril(nn.functional.softplus(raw), diagonal=-1)
        return off + torch.eye(self.nInput, device=raw.device, dtype=raw.dtype)

    def _U_eff(self):
        """Ergodic U with non-negative off-diagonals."""
        raw = self.raw_U + self.var * self.noise_raw_U
        off = torch.triu(nn.functional.softplus(raw), diagonal=1)
        return off + torch.eye(self.nOutput, device=raw.device, dtype=raw.dtype)

    def _d_eff(self):
        """Ergodic d, strictly positive via softplus."""
        d_base = nn.functional.softplus(self.d)
        d = self.bias * d_base + self.var * self.noise_d
        d = nn.functional.softplus(d)  # ensure positive after noise
        gate = getattr(self, '_current_gate', None)
        if gate is not None:
            d = d * gate
        if self.stable:
            d = d.clamp(epsilon, 1.0)
        return d


    # _effective_bias, forwardBias, reverseBias inherited from
    # InvertibleLinearLayer -- no constraint needed with symmetric
    # log domain (-inf, +inf).

class GrammarLayer(Layer):
    """Base class for layers that implement grammar rule operators.

    A GrammarLayer wraps a tensor kernel as a proper Layer with a
    known arity, invertibility flag, and canonical rule_name. The
    Grammar's ``rule_probability`` lookup uses ``rule_name`` to gate
    the layer's contribution in the bottom-up forward path; the
    SyntacticLayer dispatches the layer top-down by the same name.

    Note (2026-04-30): ``PiLayer`` / ``SigmaLayer`` inherit directly
    from GrammarLayer so the parameterized fold layers ARE Grammar
    layers, not wrapped by separate IntersectionLayer / UnionLayer
    adapters. The wrappers remain for
    back-compat but are no longer the only path to attach a rule_name
    to a parameterized fold.

    ``Ops`` itself is in the process of being demoted to a private
    math-kernel namespace; new operators should derive from
    GrammarLayer rather than adding methods to Ops. See module-level
    docstring for the migration direction.

    Class attributes (override in subclasses):
        rule_name:   canonical operator name as it appears in grammar
                     bodies (e.g. "not", "intersection", "union").
        arity:       1 for unary, 2 for binary.
        invertible:  True if reverse() recovers the input exactly.
        lossy:       True if forward() loses information (no exact
                     reverse possible).
        tier:        which space the operator runs in:
                       'S' -- symbols (default; most ops live here)
                       'C' -- concepts (intersection / union)
                       'P' -- percepts (rare)
                     Used by the chart and per-space SyntacticLayer
                     to gate which rules can fire in which space.
        reads_activation: True iff the op consumes the bivector
                     activation ``[..., 2]`` from
                     ``subspace.materialize(mode='activation')``
                     rather than the muxed event tensor. The
                     per-space dispatcher consults this flag to
                     pick the right read source. Default False
                     (muxed event read via ``materialize()``).
    """
    rule_name        = ""
    arity            = 1
    invertible       = False
    lossy            = False
    tier             = 'S'
    reads_activation = False

    # Chart authority registration -- Surface-3 wiring (see
    # doc/research/three-surfaces.md). When a SyntacticLayer registers
    # itself as the class-level authority, every GrammarLayer instance
    # constructed afterwards adds itself to the authority's roster, and
    # its `gated_run(x, fn, *args)` helper consults the authority
    # before running `fn`. The chart can then gate Pi / Sigma /
    # Not / etc. via `rule_probability(rule_name)`. Default: no
    # authority -> layers run unconditionally (backward-compat).
    _chart_authority = None

    def __init__(self, nInput=0, nOutput=0):
        """Initialize GrammarLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nInput, nOutput)
        # Auto-register with the chart authority (if any). Only
        # layers that carry a non-empty rule_name participate; the
        # base GrammarLayer with rule_name="" stays anonymous.
        if (GrammarLayer._chart_authority is not None
                and self.rule_name):
            try:
                GrammarLayer._chart_authority.register_grammar_layer(self)
            except AttributeError:
                # Authority object is not a SyntacticLayer-shaped
                # registrar; ignore so we don't trip back-compat
                # construction paths.
                pass

    @classmethod
    def set_chart_authority(cls, syntactic_layer):
        """Install ``syntactic_layer`` as the class-level chart
        authority. Future GrammarLayer instances will auto-register;
        existing instances can be registered manually via the
        authority's ``register_grammar_layer`` method.

        Pass ``None`` to clear the authority (e.g. for tests that
        want unconditional forward behavior).
        """
        cls._chart_authority = syntactic_layer

    # -- Sentence / sugar absorption marker -----------------------------
    #
    # ``absorb(left, right)`` is a base-class method available on every
    # GrammarLayer subclass. Forward semantics: return ``left`` -- a
    # binary left-pass that discards ``right``. The grammar fires it
    # to consume syntactic sugar (whitespace, punctuation, low-
    # semantic-weight tokens, sentence boundary markers) into a
    # surrounding constituent without contributing to it.
    #
    # Lossy by contract: the right operand is unrecoverable; the
    # absorption is a one-way structural marker indicating "a sentence
    # boundary or sugar token was here, and it has been consumed".
    # No standalone AbsorbLayer subclass -- the marker semantics live
    # on the base so any GrammarLayer instance can fire it.
    def absorb(self, left, right):
        """Sentence / sugar absorption marker. Returns ``left``.

        Available on every GrammarLayer. Subclasses can override to
        record richer sentence-boundary side-effects (e.g. write
        to a buffer, increment a counter); the default is a pure
        binary left-pass.
        """
        return left

    def gated_run(self, x, fn, *fn_args, **fn_kwargs):
        """Soft-gate the layer's forward through the chart authority.

        If no authority is registered, this is a passthrough: returns
        ``fn(x, *fn_args, **fn_kwargs)``.

        If an authority is registered, queries
        ``authority.should_run_rule(rule_name)`` -> probability ``p``
        in [0, 1], then returns the soft mixture
        ``p * fn(x, ...) + (1 - p) * x`` so gradient flows through
        both branches. ``p == 1.0`` short-circuits to the un-mixed
        forward (the structural fast path the existing space-forward
        code already uses for `union(C, C)` / `Contiguous(S)` /
        `not(S)` gating).
        """
        auth = GrammarLayer._chart_authority
        if auth is None or not self.rule_name:
            return fn(x, *fn_args, **fn_kwargs)
        try:
            p = float(auth.should_run_rule(self.rule_name))
        except Exception:
            return fn(x, *fn_args, **fn_kwargs)
        if p >= 1.0 - 1e-9:
            return fn(x, *fn_args, **fn_kwargs)
        if p <= 1e-9:
            return x
        return p * fn(x, *fn_args, **fn_kwargs) + (1.0 - p) * x

    # -- Binary tensor op contract for the chart parser --------------
    #
    # ``compose(*operands)`` is the arity-aware binary tensor op the
    # CKY chart driver calls to combine child span vectors into a
    # parent span vector. ``decompose(parent)`` is the matching inverse
    # used by the outside pass / reconstruction loss to push gradients
    # back into child cells.
    #
    # Shape contract:
    #   arity == 1: compose(x: [..., D]) -> [..., D'];
    #               decompose(y: [..., D']) -> [..., D]
    #   arity == 2: compose(left: [..., D], right: [..., D]) -> [..., D'];
    #               decompose(y: [..., D']) -> (left, right)
    #
    # The base class provides arity-aware default implementations:
    # arity-1 layers route through ``forward`` / ``reverse``; arity-2
    # layers raise NotImplementedError until a subclass plugs in the
    # binary parameterization. Lossy layers can override decompose
    # with a pseudo-inverse (e.g. identity).
    def _check_bivector_shape(self, x):
        """Validate that ``x`` carries at least the bivector pole pair
        in its last dim.

        Every grammar op that consumes per-position activations expects
        a tensor whose trailing dim begins with the ``[pos, neg]``
        bivector. The materialized event tensor passed in by the
        dispatcher has shape ``[B, V, nWhat + nWhere + nWhen]`` with
        the bivector at ``[..., :2]`` (nWhat == 2 by convention);
        callers that pass a narrower tensor are working on degenerate
        input (e.g. an empty subspace) and the op cannot proceed.

        Centralised here so subclasses don't each re-implement the
        check; raises a uniform ``ValueError`` carrying the offending
        shape and the calling subclass name.
        """
        if not torch.is_tensor(x):
            raise ValueError(
                f"{type(self).__name__}: expected a Tensor, got "
                f"{type(x).__name__}")
        if x.shape[-1] < 2:
            raise ValueError(
                f"{type(self).__name__}: expected last dim >= 2 "
                f"(the .what bivector poles); got shape "
                f"{tuple(x.shape)}")

    def compose(self, *operands):
        """Combine child operand(s) into a parent vector.

        Default routes arity-1 through ``forward``. Arity-2 subclasses
        must override.
        """
        if self.arity == 1:
            if len(operands) != 1:
                raise ValueError(
                    f"{type(self).__name__}.compose: arity 1 expects "
                    f"1 operand, got {len(operands)}")
            return self.forward(operands[0])
        raise NotImplementedError(
            f"{type(self).__name__}.compose: arity-{self.arity} "
            f"binary tensor op not implemented")

    def generate(self, parent):
        """Inverse of ``compose``: split a parent into its operand(s).

        Default routes arity-1 through ``reverse`` when invertible,
        otherwise returns the parent unchanged (lossy identity
        recovery). Arity-2 subclasses must override.

        Renamed from ``decompose`` (2026-05-01) to align with the
        ``<compose>`` / ``<generate>`` grammar XML sections and the
        chart's parse / generate vocabulary. ``decompose`` is kept
        as a back-compat alias on every GrammarLayer.
        """
        if self.arity == 1:
            if self.invertible:
                return self.reverse(parent)
            return parent
        raise NotImplementedError(
            f"{type(self).__name__}.generate: arity-{self.arity} "
            f"binary tensor op not implemented")

    def decompose(self, *args, **kwargs):
        """Back-compat alias for ``generate``. Renamed 2026-05-01;
        existing call sites that invoke ``layer.decompose(...)``
        keep working unchanged."""
        return self.generate(*args, **kwargs)


class SigmaLayer(Layer):
    """Additive (summation) fold feature of the subsymbolic loop.

    Substrate, not a grammar operation. Instantiated directly by spaces
    (``ConceptualSpace.sigma_percept``) and used by the subsymbolic
    loop; not chart-dispatched.

    With ``nonlinear=True`` (legacy behavior), ``forward`` returns
    ``tanh(W @ x + b)``. With ``nonlinear=False``, ``forward`` returns the
    raw linear result. ``reverse`` mirrors the same choice.

    When ``invertible=True``, uses an invertible linear layer so
    ``reverse()`` is available via the exact LDU inverse.  When
    ``invertible=False`` (default), uses a plain LinearLayer.

    Weight initialization (non-ergodic) is scaled by 1/nInput so that
    the output stays in [-1, 1] at init when input is in [-1, 1].

    All ergodic machinery lives in the inner layer; SigmaLayer dispatches
    the ergodic interface (set_sigma, observe_sigma, etc.) there.

    ``monotonic`` selects the weight constraint (only meaningful with
    ``invertible=True``):
        monotonic=True:  W >= 0 (NonNegativeInvertibleLinearLayer) -- ordering preserved
        monotonic=False: W unrestricted (InvertibleLinearLayer)    -- bitonic response
    """

    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, nonlinear=True, stable=False,
                 monotonic=False):
        """Initialize SigmaLayer; allocate state for the class contract.

        See class docstring for invariants.
        """
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.ergodic    = ergodic
        self.nonlinear  = nonlinear
        self.stable     = stable
        self.monotonic  = monotonic
        self.activation = torch.zeros(1, nOutput, 1)
        if invertible:
            if monotonic:
                self.layer = NonNegativeInvertibleLinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic, stable=stable)
            else:
                self.layer = InvertibleLinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic, stable=stable)
        else:
            self.layer = LinearLayer(nInput, nOutput, hasBias=True, naive=naive, ergodic=ergodic)
        self.layers.append(self.layer)

    @property
    def bias(self): return self.layer.bias
    @property
    def var(self):  return self.layer.var

    # -- Inner Sigma transform on packed features --
    def _sigma_inner_forward(self, packed, binary=False, gate=None):
        """Apply atanh -> linear -> tanh on a packed [..., 2D] tensor."""
        if binary:
            packed = Ops.top2_select_ste(packed)
        if self.nonlinear:
            packed = torch.atanh(packed.clamp(-1 + epsilon, 1 - epsilon))
        out = (self.layer.forward(packed, gate=gate) if gate is not None
               else self.layer.forward(packed))
        if self.nonlinear:
            out = torch.tanh(out)
        self.activation = out.detach()
        return out

    def _sigma_inner_reverse(self, packed, gate=None):
        """Apply atanh -> linear.reverse -> tanh on a packed [..., 2D] tensor."""
        if self.nonlinear:
            packed = torch.atanh(packed.clamp(-1 + epsilon, 1 - epsilon))
        out = (self.layer.reverse(packed, gate=gate) if gate is not None
               else self.layer.reverse(packed))
        if self.nonlinear:
            out = torch.tanh(out)
        self.activation = out.detach()
        return out

    def forward(self, x, binary=False, gate=None):
        """Apply the additive OR fold.

        ``binary=True`` selects the top-2 input operands (by |x|) via
        Ops.top2_select_ste before the atanh entry transform; the rest
        drop to 0 in additive domain (silent in the OR).  Backward is
        STE -- gradient flows to every input as if no selection ran.

        ``gate`` (optional): per-call ``[rank]`` tensor passed through
        to the inner ``InvertibleLinearLayer`` as a multiplicative gate
        on its diagonal. Used by rule layers (LiftLayer / LowerLayer)
        to select a low-rank slice of the operator. The gating lives
        inside the LDU layer; SigmaLayer just passes it through.
        """
        if binary:
            x = Ops.top2_select_ste(x)
        if self.nonlinear:
            x = torch.atanh(x.clamp(-1 + epsilon, 1 - epsilon))
        y = self.layer.forward(x, gate=gate) if gate is not None else self.layer.forward(x)
        if self.nonlinear:
            y = torch.tanh(y)
        self.activation = y.detach()
        return y

    def reverse(self, y, gate=None):
        """Invert tanh then apply W^-1 then tanh. Requires invertible=True.

        ``gate`` (optional): same gate tensor passed to ``forward``;
        flows through to the inner LDU layer's reverse so the inverse
        uses ``1/(d * gate)``.
        """
        if self.nonlinear:
            y = torch.atanh(y.clamp(-1 + epsilon, 1 - epsilon))
        x = self.layer.reverse(y, gate=gate) if gate is not None else self.layer.reverse(y)
        if self.nonlinear:
            x = torch.tanh(x)
        self.activation = x.detach()
        return x

    # -- Binary tensor ops (chart parser) -----------------------------
    #
    # ``forward(x: [..., D])`` is the unary feature-fold form: it
    # applies the OR-fold linear transform within a single operand's
    # feature axis. The chart parser needs the *binary* form: combine
    # two operand vectors (one per child span) into one parent vector
    # via the same parameterization. For the OR-fold the natural
    # binary semantics is the additive (logit-domain) sum -- atanh
    # the operands, sum across operands, then apply the same linear
    # + tanh. ``compose`` and ``decompose`` close the loop with a
    # balanced inverse so the chart can also push parent gradients
    # back into child cells.
    def compose(self, left, right, gate=None):
        """Binary OR fold: combine two ``[..., D]`` operands into one.

        Args:
            left, right: ``[..., D]`` in [-1, 1].
            gate: optional per-call ``[rank]`` multiplier on the inner
                LDU's diagonal (used by LiftLayer for low-rank slicing).

        Returns:
            ``[..., nOutput]`` in [-1, 1].
        """
        if self.nonlinear:
            a_l = torch.atanh(left.clamp(-1 + epsilon, 1 - epsilon))
            a_r = torch.atanh(right.clamp(-1 + epsilon, 1 - epsilon))
        else:
            a_l = left
            a_r = right
        a_sum = a_l + a_r
        out = (self.layer.forward(a_sum, gate=gate) if gate is not None
               else self.layer.forward(a_sum))
        if self.nonlinear:
            out = torch.tanh(out)
        self.activation = out.detach()
        return out

    def generate(self, parent, gate=None):
        """Inverse of compose; balanced split.

        Given ``y = tanh((atanh(left) + atanh(right)) @ W + b)`` the
        reconstruction recovers the pre-transform sum
        ``s = atanh(left) + atanh(right) = layer.reverse(atanh(y))``
        and assigns ``left == right == tanh(s/2)``. Round-trip
        ``compose(*generate(y)) == y`` when the inner layer is
        invertible.

        Args:
            parent: ``[..., nOutput]`` in [-1, 1].
            gate: same gate passed to ``compose``.

        Returns:
            ``(left, right)``: each ``[..., nInput]`` in [-1, 1].

        Requires ``invertible=True`` on the inner LinearLayer.
        """
        if self.nonlinear:
            a_y = torch.atanh(parent.clamp(-1 + epsilon, 1 - epsilon))
        else:
            a_y = parent
        a_sum = (self.layer.reverse(a_y, gate=gate) if gate is not None
                 else self.layer.reverse(a_y))
        half = a_sum * 0.5
        if self.nonlinear:
            op = torch.tanh(half)
        else:
            op = half
        return op, op

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        nInput, nOutput = 3, 4
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, nonlinear=True)

        x = torch.randn((2, 5, nInput), device=TheDevice.get())
        layer.set_sigma(0.999)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(layer.getParameters(), lr=0.01)
        optimizer.zero_grad()
        y = layer(x)
        loss = criterion(y, y)
        loss.backward()
        print(f"Original input: {x}")
        print(f"After linear: {y}")

        nInput, nOutput = 5, 7
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, naive=False,
                           invertible=True, nonlinear=True)
        x = torch.randn((2, 5, nInput), device=TheDevice.get())
        layer.set_sigma(0.0)
        y = layer.forward(x)
        y_inv = layer.reverse(y)
        assert torch.norm(x - y_inv) < 0.00001
        layer = SigmaLayer(nInput=nInput, nOutput=nOutput, naive=True,
                           invertible=True, nonlinear=True)
        x = torch.randn((4, 8, nInput), device=TheDevice.get())
        layer.set_sigma(0.0)
        y = layer.forward(x)
        assert y.shape == (4, 8, nOutput), "Incorrect Size"
        y_inv = layer.reverse(y)
        assert torch.norm(x - y_inv) < 0.00001
        print("SigmaLayer tests passed.")

class NegationLayer(Layer):
    """Materialize bivalent or ternary literals for Sigma/Pi logic.

    Canonical DNF keeps negation at the literal level, then builds AND
    terms over those literals and ORs the terms.  The default bivalent
    layer is the signed pair:

        forward(x) = [x, -x]

    With ``ternary=True`` the layer inserts the non-affirming residual
    from the grammar's ternary partition:

        forward(x) = [x, Ops._non_kernel(x), -x]

    For crisp signed values {-1, 0, 1}, exactly one ternary channel is
    positive: affirmation, non-affirming negation, or not-negation.
    Soft values remain a differentiable relaxation of that partition.
    """

    def __init__(self, nInput, ternary=False):
        """Initialize NegationLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        ternary = bool(ternary)
        super().__init__(nInput, (3 if ternary else 2) * nInput)
        self.ternary = ternary

    def forward(self, x):
        """Return [x, -x] or [x, non(x), -x] along the final axis."""
        if x.shape[-1] != self.nInput:
            raise ValueError(
                f"NegationLayer expected last dim {self.nInput}, "
                f"got {x.shape[-1]}")
        if self.ternary:
            return torch.cat((x, Ops._non_kernel(x, monotonic=False), -x), dim=-1)
        return torch.cat((x, -x), dim=-1)

    def reverse(self, y):
        """Collapse a literal bank back to signed variable evidence."""
        if y.shape[-1] != self.nOutput:
            raise ValueError(
                f"NegationLayer.reverse expected last dim {self.nOutput}, "
                f"got {y.shape[-1]}")
        pos = y[..., :self.nInput]
        neg_start = 2 * self.nInput if self.ternary else self.nInput
        neg = y[..., neg_start:neg_start + self.nInput]
        return 0.5 * (pos - neg)

class NotLayer(GrammarLayer):
    """Parameter-free propositional negation on the bivalent symbol bivector.

    Implements the grammar rule ``S = not(S)``. Operates on the
    materialized muxed event tensor ``[B, V, nWhat + nWhere + nWhen]``;
    the ``.what`` bivector ``[pos, neg]`` lives at ``[..., :2]``
    (nWhat == 2 by convention) and any nWhere / nWhen channels follow.
    Negation swaps the leading 2 dims of the last axis to ``[neg, pos]``
    at every ``(B, V)`` position; nWhere / nWhen pass through unchanged.

    Contradictions are preserved: a position with both ``pos`` and
    ``neg`` high stays contradictory after the swap (new
    ``pos = old neg`` and new ``neg = old pos`` are still both high).
    Contrast with bitonic ``-x`` negation, which collapses
    contradictions onto opposite-sign components.

    The dispatcher hands NotLayer the materialized tensor (with the
    ``.active`` mask applied) -- never the codebook ``W``. Forward
    returns the muxed tensor with only the bivector channels swapped;
    the dispatcher writes back via ``set_event``.

    Shape-preserving and self-inverse.
    """
    rule_name  = "not"
    arity      = 1
    invertible = True
    tier       = 'C'

    def __init__(self):
        """Initialize NotLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self._check_bivector_shape(x)
        bivector = x[..., :2].flip(dims=(-1,))
        rest     = x[..., 2:]
        if rest.shape[-1] == 0:
            return bivector
        return torch.cat([bivector, rest], dim=-1)

    def reverse(self, y):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return self.forward(y)

class NonLayer(GrammarLayer):
    """Non-affirming negation (indeterminacy) on a bivector.

    For a ``[pos, neg]`` bivector at each axis, ``non`` returns
    ``[1 - pos, 1 - neg]`` per pole independently. This is the
    pole-wise complement: a position fully affirmed
    (``pos = 1, neg = 0``) becomes ``[0, 1]`` (pure negation);
    indeterminate (``pos = 0, neg = 0``) becomes ``[1, 1]`` (full
    contradiction); contradictory (``pos = 1, neg = 1``) becomes
    ``[0, 0]`` (full indeterminacy). The four corners of the
    tetralemma exchange via this map:

        affirm    [1,0] <-> [0,1] negate
        unknown   [0,0] <-> [1,1] contradict

    Self-inverse on each pole independently
    (``non(non(x)) = 1 - (1 - x) = x``), shape-preserving. Operates
    on the leading bivector slice ``[..., :2]`` of the muxed event;
    nWhere / nWhen channels at ``[..., 2:]`` pass through unchanged
    (same convention as ``NotLayer``).
    """
    rule_name  = "non"
    arity      = 1
    invertible = True
    tier       = 'C'

    def __init__(self):
        """Initialize NonLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self._check_bivector_shape(x)
        bivector = 1.0 - x[..., :2]
        rest     = x[..., 2:]
        if rest.shape[-1] == 0:
            return bivector
        return torch.cat([bivector, rest], dim=-1)

    def reverse(self, y):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return self.forward(y)


class IntersectionLayer(GrammarLayer):
    """``L -> intersection(L, L)`` -- per-pole "min toward zero" on
    a bivector activation tensor.

    Tiered ``L`` (logical) per the 2026-05-05 directive: the
    operator is a pure logical (lattice-min) primitive, neither
    purely conceptual nor purely symbolic. The dispatcher feeds
    it the bivector activation -- ``[B, V, 2]`` per position,
    ``[pos, neg]`` poles -- via ``reads_activation = True``. The
    operands' upstream tier (C vs S codebook activation) is
    determined by the chart binding, not by this layer.

    Math via ``Ops.intersection`` (a public alias of
    ``Ops._conjunction_kernel``):
        monotonic=False (default) -> RadMin: same-sign min
            magnitude, zero passthrough. The pole closer to
            zero wins per channel.
        monotonic=True            -> strict lattice min.

    Lossy: min collapses dominated operands. ``decompose`` returns
    ``(parent, parent)`` as the best-effort identity recovery
    without auxiliary structure.
    """
    rule_name        = "intersection"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = True

    def __init__(self, monotonic=False):
        """Initialize IntersectionLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)
        self.monotonic = bool(monotonic)

    def forward(self, left, right):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return Ops.intersection(left, right, monotonic=self.monotonic)

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class UnionLayer(GrammarLayer):
    """``L -> union(L, L)`` -- per-pole "max toward zero" (max
    magnitude, away from zero) on a bivector activation tensor.

    Tiered ``L`` (logical) -- counterpart to ``IntersectionLayer``.
    Same dispatch contract: feeds on bivector activation
    ``[B, V, 2]`` via the ``reads_activation = True`` flag. The
    operands' upstream tier is determined by the chart binding,
    not by this layer.

    Math via ``Ops.union`` (a public alias of
    ``Ops._disjunction_kernel``):
        monotonic=False (default) -> RadMax: same-sign max
            magnitude with zero passthrough.
        monotonic=True            -> strict lattice max.

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "union"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = True

    def __init__(self, monotonic=False):
        """Initialize UnionLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)
        self.monotonic = bool(monotonic)

    def forward(self, left, right):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return Ops.union(left, right, monotonic=self.monotonic)

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


# ===========================================================================
# Grammar-op GrammarLayer subclasses (Surface 3 facade, 2026-05-01).
#
# Each class below names one grammar operation (`rule_name`) and exposes
# the canonical GrammarLayer interface (forward / reverse / compose /
# decompose, plus `gated_run` from the base class). The math kernel
# delegates to the corresponding `SyntacticLayer.*Forward` /
# `*Reverse` method so semantics are byte-identical to the existing
# `_RULE_METHODS` dispatch path. The benefit is uniform surface:
# `isinstance(x, GrammarLayer)` and `x.rule_name` work for every op,
# the chart's per-rule per-cell dispatch can route through these
# subclasses without a separate `_RULE_METHODS` table, and
# `_chart_authority`'s gating applies uniformly via `gated_run`.
#
# The subclasses are stateless wrappers that look up the method by
# name on a SyntacticLayer instance passed in at call time -- they
# don't own a SyntacticLayer reference (which would be a circular
# ownership: SyntacticLayer constructs them via the chart's eager
# build, and they'd point back at SyntacticLayer). Pattern:
#
#     out = LiftLayer().forward(left, right, layer=syntactic_layer,
#                               subspace=subspace)
#
# Existing `_RULE_METHODS` dispatch in SyntacticLayer.project still
# works -- this is an addition, not a replacement. A follow-up can
# migrate the chart to dispatch through these classes.

# =====================================================================
# Grammar-op GrammarLayer subclasses (Step 8 of the 2026-05-01 syntactic-
# layer refactor). Each class is a self-contained GrammarLayer with
# direct Ops-based math. Replaces the prior `_GrammarOpFacade` /
# SyntacticLayer-dispatch pattern.
#
# `_GrammarOpFacade._registry` is retired: the chart now consults
# `wordSpace.host_layer(tier, rule_name)` first (Step 7) and falls
# back to a hardcoded class lookup `GRAMMAR_LAYER_CLASSES` declared
# at module scope below.
# =====================================================================
class LiftLayer(GrammarLayer):
    """``S -> lift(VP, NP)`` -- rule-id annotator over the unconditional
    subsymbolic loop.

    Post-2026-05-13 refactor.  LiftLayer no longer owns or borrows a
    substrate sigma; the composition between VP and NP happens in the
    always-on subsymbolic loop
    ``C = sigma_percept(pi_input(IS) + pi_concept(C_prev))``.
    LiftLayer's role at S is purely to **annotate** that the composed
    state is being framed as a predication (lift), distinguishing it
    from the same state framed as an attribution (lower).

    Mechanically the layer computes the S-tier lattice OR via
    ``Ops._lower_kernel`` (parameter-free), then the chart's parse
    tree records the rule_id ``lift``.  Downstream readers (truth
    layer, output decoder) consume the rule_id to discriminate
    predication from attribution; the substrate composition is shared.

    See ``doc/Language.md`` §"Lift / lower — rule-id annotators" for
    the cognitive rationale ("the boy runs" vs "the running boy" share
    the same neural composition; lift/lower is a labelling).
    """
    rule_name  = "lift"
    arity      = 2
    invertible = True
    tier       = 'P'

    def __init__(self, symbolicSpace=None, perceptualSpace=None,
                 conceptualSpace=None):
        """Initialize LiftLayer.

        ``symbolicSpace`` / ``perceptualSpace`` / ``conceptualSpace``
        are accepted for API compatibility with the legacy
        gated-substrate constructor signature; the new annotator path
        does not use them.  They are stored via ``object.__setattr__``
        to avoid nn.Module submodule tracking that would create
        cycles in the module tree.
        """
        super().__init__(0, 0)
        object.__setattr__(self, 'symbolicSpace', symbolicSpace)
        object.__setattr__(self, 'perceptualSpace', perceptualSpace)
        object.__setattr__(self, 'conceptualSpace', conceptualSpace)

    def forward(self, left, right):
        """Forward: S-tier lattice composition; rule-id ``lift`` is
        recorded by the chart on the surrounding parse cell.

        ``left`` is VP (the predicated operand), ``right`` is NP.
        """
        return Ops._lower_kernel(left, right, mode='AND', kind='smooth')

    def reverse(self, parent):
        """Reverse: lossy ``(parent, parent)`` pseudo-inverse.

        The lattice AND is not bijective in either operand.  Inherit
        the lossy reverse convention from the legacy LiftLayer; the
        chart's generate path uses the rule_id to backtrace rather
        than relying on a bijective inverse here.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class LowerLayer(GrammarLayer):
    """``S -> lower(VP, NP)`` -- rule-id annotator over the unconditional
    subsymbolic loop.

    Symmetric to ``LiftLayer`` but records rule_id ``lower`` instead
    of ``lift``.  The composition between VP and NP happens in the
    always-on subsymbolic loop (`sigma_percept(pi_input + pi_concept)`);
    LowerLayer's role at S is purely to annotate that the composed
    state is being framed as an attribution (lower), distinguishing
    it from the same state framed as a predication (lift).

    Mechanically the layer computes the S-tier lattice OR via
    ``Ops._lift_kernel`` (parameter-free); the chart's parse tree
    records the rule_id ``lower``.

    Post-2026-05-13: no internal substrate sigma/pi; LiftLayer and
    LowerLayer differ only in their static lattice kernel and the
    rule_id stamped on the parse cell.
    """
    rule_name  = "lower"
    arity      = 2
    invertible = True
    tier       = 'P'

    def __init__(self, symbolicSpace=None, conceptualSpace=None,
                 perceptualSpace=None):
        """Initialize LowerLayer.

        Constructor accepts ``symbolicSpace`` / ``conceptualSpace`` /
        ``perceptualSpace`` for API compatibility with the legacy
        gated-substrate signature; the new annotator path does not
        use them.
        """
        super().__init__(0, 0)
        object.__setattr__(self, 'symbolicSpace', symbolicSpace)
        object.__setattr__(self, 'conceptualSpace', conceptualSpace)
        object.__setattr__(self, 'perceptualSpace', perceptualSpace)

    def forward(self, left, right):
        """Forward: S-tier lattice composition; rule-id ``lower`` is
        recorded by the chart on the surrounding parse cell.

        ``left`` is VP (the attributed operand), ``right`` is NP.
        """
        return Ops._lift_kernel(left, right, mode='OR', kind='smooth')

    def reverse(self, parent):
        """Reverse: lossy ``(parent, parent)`` pseudo-inverse.

        See ``LiftLayer.reverse`` for the lattice-invertibility note.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class ConjunctionLayer(GrammarLayer):
    """``S -> conjunction(S, S)`` -- monotonic min on the
    post-codebook scalar activation.

    Symbolic-tier conjunction is the AND of two **codebook
    activation patterns**. Per the 2026-05-05 directive,
    SymbolicSpace's ``materialize(mode='activation')`` returns
    the **post-codebook** activation -- a ``[B, V]`` *scalar*
    strength per prototype (``effective_activation()``: the
    bivector ``[pos, neg]`` reduced via ``max(pos, neg)`` and
    gated by modal presence). Conjunction over two such patterns
    asks "which prototypes are active in *both* operands".

    Because the post-codebook activation is non-negative scalar,
    the natural composition kernel is the **monotonic** lattice
    min: ``torch.minimum(x, y)``. RadMin (the bivector kernel)
    would be wrong here -- there's no negative pole to manage.
    The class hard-codes ``monotonic=True`` and forwards to
    ``Ops.intersection`` so the kernel collapses to ``torch.min``
    via ``_lower_kernel(kind='strict')``.

    Distinct from ``IntersectionLayer`` (C-tier): IntersectionLayer
    operates on a bivector ``[..., 2]`` activation (concept-tier
    pre-codebook) and supports both RadMin and lattice-min;
    ConjunctionLayer operates on a *scalar* ``[B, V]`` post-
    codebook activation and is strictly monotonic.

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "conjunction"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'S'
    reads_activation = True

    def __init__(self):
        """Initialize ConjunctionLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        # Post-codebook activation is monotonic-only -- no negative
        # pole to manage, so RadMin would be wrong.
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return Ops.intersection(left, right, monotonic=True)

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class DisjunctionLayer(GrammarLayer):
    """``S -> disjunction(S, S)`` -- monotonic max on the
    post-codebook scalar activation.

    Symbolic-tier disjunction is the OR of two **codebook
    activation patterns**: ``[B, V]`` post-codebook scalar
    activation (see ``ConjunctionLayer`` for the activation-
    semantics rationale). The natural composition kernel is the
    monotonic lattice max ``torch.maximum(x, y)``; the class
    hard-codes ``monotonic=True`` and forwards to ``Ops.union``,
    which collapses to ``torch.max`` via
    ``_lift_kernel(kind='strict')``.

    Distinct from ``UnionLayer`` (C-tier): UnionLayer operates on
    a bivector ``[..., 2]`` activation and supports both RadMax
    and lattice-max; DisjunctionLayer operates on a scalar
    ``[B, V]`` post-codebook activation and is strictly monotonic.

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "disjunction"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'S'
    reads_activation = True

    def __init__(self):
        """Initialize DisjunctionLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return Ops.union(left, right, monotonic=True)

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


def _argmax_prototype(x):
    """Per-batch top-1 prototype index from a ``[B, V, D]`` muxed
    event tensor.

    Computes the L2 norm of the ``.what`` bivector slice
    ``[..., :2]`` only (not the full muxed event) so the ranking
    reflects symbol identity / presence, not the nWhere / nWhen
    positional channels. When the input has last_dim < 2 the full
    last dim is used (degenerate, single-channel fallback).

    Returns a ``[B]`` long tensor where each entry is the position
    (codebook prototype index) with the largest ``.what`` L2 norm
    in that batch row -- the most-active prototype for that
    operand. Used by ``PartLayer`` / ``IsEqualLayer`` / ``QueryLayer``
    to map continuous activations to discrete codebook indices for
    mereological-tree bookkeeping.
    """
    if not torch.is_tensor(x):
        return torch.zeros(0, dtype=torch.long)
    if x.dim() < 2:
        return torch.zeros(1, dtype=torch.long, device=x.device)
    # Slice .what bivector when available; else use whole last-dim.
    what = x[..., :2] if x.shape[-1] >= 2 else x
    norms = what.norm(dim=-1)            # [B, V]
    if norms.dim() < 2:
        norms = norms.unsqueeze(0)
    return norms.argmax(dim=-1)          # [B]


def _parthood_geometric(left, right):
    """Clipped cosine parthood on per-batch dominant bivector activations.

    Replaces the explicit ``MereologicalTree`` lookup for
    ``PartLayer`` / ``IsEqualLayer`` / ``QueryLayer`` after the
    "codebook IS the meronymic tree" unification: parthood is
    expressed by codebook geometry on the bivector cone (see
    ``Architecture.md`` §"Monotonicity of the bivector chain").

    For per-batch dominant slot bivectors ``a = left[b, argmax_left]``
    and ``b = right[b, argmax_right]``, returns the clipped cosine
    similarity

        part(a, b) = max(0, a · b) / (|a| * |b|)

    which is the canonical mereological projection on the
    non-negative paired-index cone. The return is a ``[B]`` tensor
    in ``[0, 1]`` -- 1 means "fully a part of", 0 means disjoint.
    """
    a_idx = _argmax_prototype(left)             # [B]
    b_idx = _argmax_prototype(right)            # [B]
    B = int(left.shape[0])
    a_vec = left[torch.arange(B, device=left.device), a_idx]    # [B, K]
    b_vec = right[torch.arange(B, device=right.device), b_idx]  # [B, K]
    # Restrict to bivector head [pos, neg]; if last_dim < 2 use full.
    K = min(2, a_vec.shape[-1])
    a_biv = a_vec[..., :K]
    b_biv = b_vec[..., :K]
    dot = (a_biv * b_biv).sum(dim=-1)
    na = a_biv.norm(dim=-1)
    nb = b_biv.norm(dim=-1)
    return (dot.clamp(min=0.0) / (na * nb + 1e-9))               # [B]


class IsEqualLayer(GrammarLayer):
    """``S -> isEqual(S, S)`` -- symbolic identity assertion.

    S-tier identity: ``isEqual(A, B)`` asserts that A and B name the
    same concept by producing a single parent symbol that represents
    the wholeness of its arguments — a higher-epistemic-level
    assertion that cannot be expressed at the subsymbolic level.
    Compare with the C-tier ``equal`` which performs the geometric
    identity check on concept bivectors directly.

    Post-MereologicalTree retirement: equality is expressed purely
    geometrically. The codebook is now the meronymic structure; an
    asserted equality between ``A`` and ``B`` shows up as
    bivector co-location on the cone (mutual parthood
    ``part(A, B) ≈ 1`` AND ``part(B, A) ≈ 1``), which the codebook
    learns through training. No explicit equivalence-class table
    is stored.

    The forward returns ``torch.maximum(left, right)`` -- the
    lattice join under the bivector cone's max-as-disjunction
    interpretation -- so the chart's CKY consumer sees a single
    parent vector (semantics unchanged from the tree-backed
    version; the difference is only the absence of the tree write).

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "isEqual"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'S'
    reads_activation = False

    def __init__(self, tree=None):
        """Initialize IsEqualLayer.

        ``tree`` is accepted but ignored for backward compatibility
        with the chart's lazy-build call sites; the
        ``MereologicalTree`` has been retired.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Lattice join on the bivector cone (max element-wise)."""
        return torch.maximum(left, right)

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.

        Lossy ``(parent, parent)`` pseudo-inverse -- the max-fold
        is not bijective.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class EqualLayer(GrammarLayer):
    """``C -> equal(C, C)`` -- geometric identity on concept bivectors.

    C-tier identity: ``equal(A, B)`` returns the mutual-parthood
    score ``part(A, B) * part(B, A)`` on bivector activations,
    yielding 1 where A and B are co-located on the bivector cone
    and 0 where they are disjoint. Acts directly on the concept-tier
    bivector activation; no codebook lookup is required.

    Compare with the S-tier ``isEqual`` which produces a single
    parent symbol asserting identity — a higher-epistemic-level
    wholeness operation that cannot be expressed at the subsymbolic
    level.

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "equal"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = True

    def __init__(self):
        """Initialize EqualLayer."""
        super().__init__(0, 0)

    def forward(self, left, right):
        """Mutual parthood: ``part(left, right) * part(right, left)``."""
        return Ops._equal_kernel(left, right)

    def reverse(self, parent):
        """Lossy ``(parent, parent)`` pseudo-inverse."""
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class PartLayer(GrammarLayer):
    """``S -> part(S, S)`` -- mereological part-of on the bivector
    codebook.

    Post-MereologicalTree retirement: parthood is expressed
    geometrically by codebook position on the non-negative
    paired-index bivector cone (see ``Architecture.md``
    §"Monotonicity of the bivector chain"). The codebook IS the
    meronymic tree: A is part of B iff the clipped cosine
    projection of A's prototype onto B's prototype is high. The
    codebook learns this geometry through training on the rule
    composition; no separate adjacency table is stored.

    The forward returns ``right`` -- the encompassing parent --
    so the chart's CKY consumer sees a single parent vector
    (semantics unchanged from the tree-backed version; the
    difference is only the absence of the tree write).

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "part"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = False

    def __init__(self, tree=None):
        """Initialize PartLayer.

        ``tree`` is accepted but ignored for backward compatibility
        with the chart's lazy-build call sites; the
        ``MereologicalTree`` has been retired.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Pass the encompassing parent ``right`` through to the
        CKY consumer. The parthood relationship between ``left``
        and ``right`` is captured by the codebook geometry that
        learns under training -- no explicit tree write.
        """
        return right

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.

        Lossy ``(parent, parent)`` pseudo-inverse -- ``part(A, B)``
        does not preserve A's identity in the parent vector.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class TrueLayer(GrammarLayer):
    """``S -> true(S)`` -- keep only the pos pole of the bivector
    activation; zero the neg pole.

    The dispatcher hands TrueLayer the muxed event tensor
    (``reads_activation=False``); the bivector ``[pos, neg]`` lives
    at ``[..., :2]`` (same convention as ``NotLayer`` / ``NonLayer``).
    Returns the same shape with ``neg`` replaced by zero -- the
    bivector now affirms only what was on the positive pole, with
    no negative evidence. nWhere / nWhen channels at ``[..., 2:]``
    pass through unchanged.

    Lossy: the neg pole is destroyed; reverse is a passthrough.
    """
    rule_name        = "true"
    arity            = 1
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = False

    def __init__(self):
        """Initialize TrueLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self._check_bivector_shape(x)
        pos    = x[..., 0:1]
        zeros  = torch.zeros_like(pos)
        rest   = x[..., 2:]
        bivec  = torch.cat([pos, zeros], dim=-1)
        if rest.shape[-1] == 0:
            return bivec
        return torch.cat([bivec, rest], dim=-1)

    def reverse(self, y):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return y


class FalseLayer(GrammarLayer):
    """``S -> false(S)`` -- keep only the neg pole of the bivector
    activation; zero the pos pole.

    Mirror of ``TrueLayer``: dispatcher hands the muxed event tensor
    (``reads_activation=False``); bivector ``[pos, neg]`` at
    ``[..., :2]`` becomes ``[0, neg]``. nWhere / nWhen channels at
    ``[..., 2:]`` pass through unchanged.

    Lossy: the pos pole is destroyed; reverse is a passthrough.
    """
    rule_name        = "false"
    arity            = 1
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = False

    def __init__(self):
        """Initialize FalseLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self._check_bivector_shape(x)
        neg    = x[..., 1:2]
        zeros  = torch.zeros_like(neg)
        rest   = x[..., 2:]
        bivec  = torch.cat([zeros, neg], dim=-1)
        if rest.shape[-1] == 0:
            return bivec
        return torch.cat([bivec, rest], dim=-1)

    def reverse(self, y):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return y


class SwapLayer(GrammarLayer):
    """``S -> swap(S, S)`` -- swap the left and right arguments.

    Forward returns ``right`` -- the right operand takes the
    canonical (parent) slot, simulating a left/right argument
    swap for downstream rules. Lossy (the original left is
    discarded); reverse is the symmetric ``(parent, parent)``
    pseudo-inverse.

    Parameter-free; no Sinkhorn / marker machinery (those were
    retired with the 2026-05-04 operator overhaul -- soft
    permutation belongs in the chart's CKY pair-selection logic,
    not in a per-cell GrammarLayer).
    """
    rule_name        = "swap"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = False

    def __init__(self):
        """Initialize SwapLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return right

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class CopyLayer(GrammarLayer):
    """``S -> copy(S, S)`` -- copy the left argument to the parent slot.

    Phase 1b dual of `SwapLayer`: forward returns ``left`` -- the
    left operand takes the canonical (parent) slot, discarding the
    right operand.  Lossy (the original right is unrecoverable);
    reverse is the symmetric ``(parent, parent)`` pseudo-inverse
    shared with `SwapLayer`.

    Parameter-free.  Like `swap`, the gradient signal that trains
    `copy` flows through ``Grammar.rule_probability('copy')`` and
    the chart's CKY pair-selection state — the standard mechanism
    already differentiable under the prediction loss.

    Naming and arity dual to `swap`:
        swap.forward(a, b) -> b      copy.forward(a, b) -> a
        swap.reverse(p)    -> (p, p) copy.reverse(p)    -> (p, p)
    """
    rule_name        = "copy"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = False

    def __init__(self):
        """Initialize CopyLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return left

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class QueryLayer(GrammarLayer):
    """``S -> query(S, S)`` -- mereological-truth query "is A part
    of B?" answered geometrically against the bivector codebook.

    Post-MereologicalTree retirement: the answer is the clipped
    cosine parthood between the per-batch dominant bivector
    activations of ``left`` and ``right`` (see
    ``_parthood_geometric``). Returns a continuous truth value
    in ``[0, 1]`` rather than the prior tree-lookup boolean --
    the codebook geometry IS the meronymic structure, so
    parthood is *always* defined for any two symbols (no
    "unknown" state). Returns a ``[B, V, 2]`` truth bivector
    broadcast across the V dimension:

        part(A, B) ≈ 1  -> [pos=1, neg=0] (full affirmation)
        part(A, B) ≈ 0  -> [pos=0, neg=0] (disjoint / no overlap)

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "query"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = False

    def __init__(self, tree=None):
        """Initialize QueryLayer.

        ``tree`` is accepted but ignored for backward compatibility
        with the chart's lazy-build call sites; the
        ``MereologicalTree`` has been retired in favour of
        codebook geometry.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Geometric parthood query: returns a per-batch ``[B, V, K]``
        truth bivector broadcast across V, with the bivector head
        carrying ``[pos=parthood, neg=0]``.
        """
        parthood = _parthood_geometric(left, right)    # [B] in [0, 1]
        pos = parthood.to(dtype=right.dtype)
        neg = torch.zeros_like(pos)
        truth = torch.stack([pos, neg], dim=-1)        # [B, 2]
        if right.dim() < 3:
            return truth
        V = right.shape[1]
        rest_dim = right.shape[-1] - 2
        if rest_dim > 0:
            # Match right's nWhere/nWhen tail by zero-padding so the
            # caller can still do bivector slicing on [..., :2].
            tail = torch.zeros(truth.shape[0], rest_dim,
                               dtype=right.dtype, device=right.device)
            truth = torch.cat([truth, tail], dim=-1)
        return truth.unsqueeze(1).expand(-1, V, -1)

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


# Hardcoded module-level lookup -- replaces the retired
# -- Conceptual introspection (2026-05-12) ----------------------------
#
# Per-stage introspective grammar operations that read mental content
# and produce scalar / vector annotations the network can condition on
# at subsequent conceptual orders. Design source:
# doc/plans/2026-05-04-conceptual-introspection-handoff.md
#
# Each is implemented as a fully differentiable function of the input
# activation (no learned parameters of its own). Plug them in by
# registering an op-class entry in `GRAMMAR_LAYER_CLASSES` and a
# `<rule>area(S)</rule>`-style entry in the model's grammar XML.

def area_op(x, sigma=None):
    """Normalised Gaussian region area: ``min(sigma**2, 1)``.

    Args:
        x: ``[..., D]`` activation (the proposition whose extent we
            measure). Used only for device / dtype routing -- the area
            depends solely on `sigma`.
        sigma: scalar or tensor (mean reduced). When None falls back to
            `_DEFAULT_SUBSYMBOLIC_SIGMA`.

    Returns:
        Scalar tensor on `x.device` / `x.dtype` in [0, 1].
    """
    if sigma is None:
        sigma = _DEFAULT_SUBSYMBOLIC_SIGMA
    if torch.is_tensor(sigma):
        s = sigma.float().mean()
    else:
        s = float(sigma)
        s = (x.new_tensor(s) if torch.is_tensor(x) else torch.tensor(s))
    return torch.clamp(s.pow(2), max=1.0)


def luminosity_op(x_a, x_b, sigma=None):
    """Pairwise luminosity ``area − overlapArea * |t_A − t_B|`` ∈ [-1, 1].

    Both inputs are bivector-tail activations ``[..., D]`` (D >= 2);
    the first two components carry the [pos, neg] poles. Returns a
    scalar consistent with `Mereology.Luminosity`'s pairwise term.
    """
    if sigma is None:
        sigma = _DEFAULT_SUBSYMBOLIC_SIGMA
    sigma_f = float(sigma) if not torch.is_tensor(sigma) else float(
        sigma.float().mean().item())
    a_flat = x_a.reshape(-1, x_a.shape[-1])
    b_flat = x_b.reshape(-1, x_b.shape[-1])
    overlap = _gaussian_kernel_overlap(
        a_flat, b_flat, sigma_f, sigma_f).mean()
    if x_a.shape[-1] >= 2 and x_b.shape[-1] >= 2:
        dot_a = (x_a[..., 0] - x_a[..., 1]).mean()
        dot_b = (x_b[..., 0] - x_b[..., 1]).mean()
    else:
        dot_a = x_a.mean()
        dot_b = x_b.mean()
    disagree = (dot_a - dot_b).abs()
    area = area_op(x_a, sigma_f).to(device=overlap.device, dtype=overlap.dtype)
    lum = area - overlap * disagree
    return torch.clamp(lum, min=-1.0, max=1.0)


def isa_part_op(child, parent, sigma=None):
    """One-step kernel overlap ``K(child, parent) ∈ (0, 1]`` per the
    plan's "is the child contained in the parent at this conceptual
    order" semantics.
    """
    if sigma is None:
        sigma = _DEFAULT_SUBSYMBOLIC_SIGMA
    sigma_f = float(sigma) if not torch.is_tensor(sigma) else float(
        sigma.float().mean().item())
    c_flat = child.reshape(-1, child.shape[-1])
    p_flat = parent.reshape(-1, parent.shape[-1])
    overlap = _gaussian_kernel_overlap(
        c_flat, p_flat, sigma_f, sigma_f)
    return overlap.mean()


class AreaLayer(GrammarLayer):
    """``S -> area(S)`` -- introspective scalar in [0, 1]."""
    rule_name        = "area"
    arity            = 1
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = False

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, x):
        return area_op(x)

    def reverse(self, parent):
        return parent

    def compose(self, x):
        return self.forward(x)

    def generate(self, parent):
        return self.reverse(parent)


class LuminosityLayer(GrammarLayer):
    """``S -> luminosity(S, S)`` -- introspective scalar in [-1, 1]."""
    rule_name        = "luminosity"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'C'
    reads_activation = False

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, left, right):
        return luminosity_op(left, right)

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)


class IsaPartLayer(GrammarLayer):
    """``S -> isaPart(S, S)`` -- one-step kernel overlap ∈ (0, 1]."""
    rule_name        = "isaPart"
    arity            = 2
    invertible       = False
    lossy            = True
    tier             = 'S'
    reads_activation = False

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, child, parent):
        return isa_part_op(child, parent)

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)


# `_GrammarOpFacade._registry` (Step 8). Maps rule method_name to a
# GrammarLayer class that implements the rule's math directly.
GRAMMAR_LAYER_CLASSES = {
    'not':          NotLayer,
    'non':          NonLayer,
    'intersection': IntersectionLayer,
    'union':        UnionLayer,
    # 'Fusion' / 'Contiguous' removed 2026-05-04: the S-tier
    # max-on-codebook-activation behavior is exactly DisjunctionLayer.
    # No back-compat alias -- grammars referencing ``Fusion(S, S)`` /
    # ``Contiguous(S)`` should migrate to ``disjunction(S, S)``.
    'lift':         LiftLayer,
    'lower':        LowerLayer,
    'conjunction':  ConjunctionLayer,
    'disjunction':  DisjunctionLayer,
    'isEqual':      IsEqualLayer,
    'equal':        EqualLayer,
    'part':         PartLayer,
    'true':         TrueLayer,
    'false':        FalseLayer,
    'swap':         SwapLayer,
    'copy':         CopyLayer,           # Phase 1b: S-tier dual of swap
    'query':        QueryLayer,
    # Conceptual introspection (2026-05-12):
    'area':         AreaLayer,
    'luminosity':   LuminosityLayer,
    'isaPart':      IsaPartLayer,
    # 'absorb' removed 2026-05-04: the absorb / sentence-marker
    # behavior is a base-class method on ``GrammarLayer`` itself, so
    # callers invoke ``layer.absorb(left, right)`` on any GrammarLayer
    # instance; no dedicated subclass.
}


# Ops whose .reverse / .generate is a continuous monotonic kernel and
# therefore preserves connectedness of the back-projected fiber. Used
# by Models.BaseModel.hoc_shape / Contiguous to decide whether a leaf
# at the bottom of the derivation tree carries trustworthy
# hyperrectangle geometry.
#
#   pi    -- multiplicative log-domain AND-fold; monotonic LDU.
#   sigma -- additive matmul; monotonic LDU.
#   lift  -- gated SigmaLayer.compose (per-rule rank gate; same LDU).
#   lower -- gated PiLayer.compose (per-rule rank gate; same LDU).
#   not   -- bivector pole swap; bijection (permutation of channels).
#   non   -- per-pole bivector complement [1-pos, 1-neg]; affine
#            self-inverse bijection on each pole.
#
# All others (intersection / union / conjunction / disjunction / equals
# / part / Contiguous / true / false / what / where / when / swap /
# query / absorb) have lossy reverse paths -- the pseudo-inverse
# `(parent, parent)` or a pole projection -- so a leaf whose path
# includes any of them carries degraded contiguity information.
CONTIGUITY_PRESERVING_OPS = frozenset({'pi', 'sigma', 'lift', 'lower', 'not', 'non'})


class PiLayer(Layer):
    r"""Multiplicative boundary fold feature of the subsymbolic loop:
    ``[-1, 1] -> [-1, 1]``.

    Substrate, not a grammar operation. Instantiated directly by spaces
    (``PerceptualSpace.pi_input`` / ``pi_concept``) and used by the
    subsymbolic loop; not chart-dispatched.

    Both modes share the symmetric log-domain embedding (1+x)/(1-x):

        Forward:  z = _from_mult(exp(W @ log(_to_mult(x)) + b))
        Reverse:  x = _from_mult(exp(W^-^1 @ (log(_to_mult(z)) - b)))

    Entry transform (1+x)/(1-x) = exp(2*atanh(x)):
        x = 0  ->  1  ->  log = 0   : absent = multiplicative identity
        x = +k and x = -k produce equal and opposite log-space contributions

    Exit transform (y-1)/(y+1) is the exact inverse.

    ``monotonic`` selects the weight constraint:
        monotonic=True:  W >= 0 (NonNegativeInvertibleLinearLayer) -- ordering preserved
        monotonic=False: W unrestricted (InvertibleLinearLayer) -- bitonic response
    """

    _eps = 1e-6

    def __init__(self, nInput, nOutput, ergodic=False, naive=True,
                 invertible=False, hasBias=True, stable=True,
                 monotonic=False, nonlinear=True):
        """Initialize PiLayer; allocate state for the class contract.

        See class docstring for invariants.
        """
        super().__init__(nInput, nOutput)
        self.invertible = invertible
        self.stable     = stable
        self.hasBias    = hasBias
        self.monotonic  = monotonic
        self.nonlinear  = nonlinear
        if invertible:
            if monotonic:
                self.layer = NonNegativeInvertibleLinearLayer(nInput, nOutput, hasBias=hasBias,
                                                              naive=naive, ergodic=ergodic,
                                                              stable=stable)
            else:
                self.layer = InvertibleLinearLayer(nInput, nOutput, hasBias=hasBias,
                                                   naive=naive, ergodic=ergodic,
                                                   stable=stable)
        else:
            self.layer = LinearLayer(nInput, nOutput, hasBias=hasBias)
        self.layers.append(self.layer)

    @property
    def bias(self): return self.layer.bias
    @property
    def var(self):  return self.layer.var

    def resample_noise(self):
        """Resample noise.
        
        See class docstring for the operation contract.
        """
        self.layer.resample_noise()

    # -- Symmetric domain transforms ----------------------------------

    def _to_mult(self, x):
        """Map [-1, 1] -> (0, inf), identity at 0 -> 1."""
        if self.nonlinear:
            x = x.clamp(-1 + self._eps, 1 - self._eps)
        return (1 + x) / (1 - x)

    def _from_mult(self, y):
        """Map (0, inf) -> (-1, 1), identity at 1 -> 0."""
        return (y - 1) / (y + 1)

    # -- forward / reverse --------------------------------------------

    def forward(self, x, binary=False, gate=None):
        """Apply the log-domain multiplicative AND fold.

        ``binary=True`` selects the top-2 input operands (by |x|) via
        Ops.top2_select_ste before the entry transform; the rest drop
        to 0 in raw domain -> 1 (mult-identity) in mult-domain ->
        factor 1 in the product, i.e. irrelevant to the AND.  Backward
        is STE -- gradient flows to every input as if no selection ran.

        ``gate`` (optional): per-call ``[rank]`` tensor stashed on
        ``self.layer._current_gate`` so that ``compute_W_current``
        -> ``_d_effective`` picks it up. Used by rule layers
        (LiftLayer / LowerLayer) for low-rank operator slicing.
        """
        # Inline inner-forward: log-domain multiplicative AND fold.
        self.layer._current_gate = gate
        try:
            if self.layer.ergodic:
                self.resample_noise()
            W = self.layer.compute_W_current()
            x = x.to(W.device)
            if binary:
                x = Ops.top2_select_ste(x)
            m = self._to_mult(x)
            l = torch.log(m)
            wl = l @ W
            b = self.layer._effective_bias()
            wl = wl + b
            if self.nonlinear:
                out = torch.tanh(wl / 2)
            else:
                out = torch.exp(wl)
        finally:
            self.layer._current_gate = None
        return out

    def reverse(self, y, gate=None):
        """Recover x from y.  Requires invertible=True.

        ``gate`` flows through ``compute_Winverse_current`` ->
        ``_d_effective`` -> ``1/(d * gate)``.
        """
        # Inline inner-reverse: log-domain multiplicative AND fold inverse.
        self.layer._current_gate = gate
        try:
            W_inv = self.layer.compute_Winverse_current()
            y = y.to(W_inv.device)
            if self.nonlinear:
                m = self._to_mult(y)
                l = torch.log(m)
            else:
                l = torch.log(y)
            b = self.layer._effective_bias()
            lx = (l - b) @ W_inv
            if self.nonlinear:
                out = torch.tanh(lx / 2)
            else:
                out = self._from_mult(torch.exp(lx))
            if self.layer.ergodic:
                self.resample_noise()
        finally:
            self.layer._current_gate = None
        return out

    # -- Binary tensor ops (chart parser) -----------------------------
    #
    # ``forward(x: [..., D])`` is the unary feature-fold form. The
    # chart parser needs the *binary* form: combine two operand
    # vectors (one per child span) into one parent vector via the
    # same parameterization. For the AND-fold, the natural binary
    # semantics is multiplication in mult-domain == addition in
    # log-mult domain: log_mult(left) + log_mult(right) is then
    # passed through the same W + b + tanh-half pipeline as the
    # unary form. ``decompose`` closes the loop with a balanced
    # split so the chart can also push parent gradients back into
    # child cells.
    def compose(self, left, right, gate=None):
        """Binary AND fold: combine two ``[..., D]`` operands into one.

        Args:
            left, right: ``[..., D]`` in [-1, 1].
            gate: optional per-call ``[rank]`` multiplier on the inner
                LDU's diagonal (used by LowerLayer for low-rank slicing).

        Returns:
            ``[..., nOutput]`` in [-1, 1].
        """
        self.layer._current_gate = gate
        try:
            if self.layer.ergodic:
                self.resample_noise()
            W = self.layer.compute_W_current()
            left = left.to(W.device)
            right = right.to(W.device)
            if self.nonlinear:
                l_l = torch.log(self._to_mult(left))
                l_r = torch.log(self._to_mult(right))
            else:
                l_l = torch.log(left)
                l_r = torch.log(right)
            l_sum = l_l + l_r
            wl = l_sum @ W
            b = self.layer._effective_bias()
            wl = wl + b
            if self.nonlinear:
                return torch.tanh(wl / 2)
            return torch.exp(wl)
        finally:
            self.layer._current_gate = None

    def generate(self, parent, gate=None):
        """Inverse of compose; balanced split.

        Given ``y = tanh((W @ (log_mult(left) + log_mult(right)) + b)
        / 2)`` recover the pre-transform sum
        ``s = log_mult(left) + log_mult(right)`` and assign
        ``left == right == _from_mult(exp(s/2))``. Round-trip
        ``compose(*generate(y)) == y`` when the inner layer is
        invertible.

        Args:
            parent: ``[..., nOutput]`` in [-1, 1].
            gate: same gate passed to ``compose``.

        Returns:
            ``(left, right)``: each ``[..., nInput]`` in [-1, 1].

        Requires ``invertible=True``.
        """
        self.layer._current_gate = gate
        try:
            W_inv = self.layer.compute_Winverse_current()
            parent = parent.to(W_inv.device)
            if self.nonlinear:
                log_mult_y = torch.log(self._to_mult(parent))
            else:
                log_mult_y = torch.log(parent)
            b = self.layer._effective_bias()
            s = (log_mult_y - b) @ W_inv
            half = s * 0.5
            if self.nonlinear:
                op = self._from_mult(torch.exp(half))
            else:
                op = torch.exp(half)
            if self.layer.ergodic:
                self.resample_noise()
            return op, op
        finally:
            self.layer._current_gate = None

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        nBatch, nInput, nOutput = 5, 3, 4
        layer = PiLayer(nInput=nInput, nOutput=nOutput, nonlinear=True)
        device = next(layer.parameters()).device
        # Inputs in [-1, 1]
        x = torch.rand((nBatch, 6, nInput), device=device) * 2 - 1
        layer.set_sigma(0.999)
        y = layer(x)
        assert y.shape == (nBatch, 6, nOutput), f"shape mismatch: {y.shape}"
        assert torch.isfinite(y).all(), "PiLayer forward produced non-finite values"
        assert torch.all(y >= -1) and torch.all(y <= 1), "PiLayer output must be in [-1, 1]"
        print(f"PiLayer forward: input {x.shape} -> output {y.shape}")

        def check_roundtrip(desc, **kwargs):
            """Check roundtrip.
            
            See class docstring for the operation contract.
            """
            kw = dict(nInput=3, nOutput=6, invertible=True, nonlinear=True)
            kw.update(kwargs)
            layer = PiLayer(**kw)
            device = next(layer.parameters()).device
            nI = kw['nInput']
            # Inputs in [-1, 1]
            inputs = [('3D [B,S,nIn]', torch.rand(16, 5, nI, device=device) * 2 - 1),
                      ('2D [B,nIn]',   torch.rand(16, nI,    device=device) * 2 - 1)]
            for tag, x in inputs:
                layer.set_sigma(0.0)
                y = layer.forward(x)
                x_recon = layer.reverse(y)
                error = torch.norm(x - x_recon) / torch.norm(x)
                assert error < 1e-4, f"{desc} {tag}: reconstruction error {error:.2e}"
            print(f"  {desc}: OK")

        def check_stability(desc, **kwargs):
            """Check stability.
            
            See class docstring for the operation contract.
            """
            kw = dict(nInput=3, nOutput=6, invertible=True, stable=True,
                      nonlinear=True)
            kw.update(kwargs)
            layer = PiLayer(**kw)
            device = next(layer.parameters()).device
            nI = kw['nInput']
            dtype = next(layer.parameters()).dtype
            layer.set_sigma(0.0)

            # Test boundary values in [-1, 1]
            x_edge = torch.tensor(
                [[-1.0] * nI,
                 [0.0] * nI,
                 [1.0] * nI,
                 [-0.5, 0.0, 0.5][:nI] if nI <= 3 else [-0.5, 0.0, 0.5] + [0.25] * (nI - 3)],
                device=device,
                dtype=dtype,
            )

            y = layer.forward(x_edge)
            assert torch.isfinite(y).all(), f"{desc}: forward produced non-finite values"
            assert torch.all(y >= -1) and torch.all(y <= 1), f"{desc}: output outside [-1, 1]"

            x_recon = layer.reverse(y)
            assert torch.isfinite(x_recon).all(), f"{desc}: reverse produced non-finite values"
            print(f"  {desc}: OK")

        print("Invertible PiLayer roundtrip variations:")
        check_roundtrip("naive=T hasBias=T", naive=True,  hasBias=True)
        check_roundtrip("naive=T hasBias=F", naive=True,  hasBias=False)
        check_roundtrip("naive=F hasBias=T", naive=False, hasBias=True)
        check_roundtrip("naive=F hasBias=F", naive=False, hasBias=False)
        check_roundtrip("square nIn=nOut=6", naive=True,  hasBias=False, nInput=6, nOutput=6)
        check_roundtrip("ergodic naive=T hasBias=F", naive=True,  hasBias=False, ergodic=True)
        check_roundtrip("ergodic naive=T hasBias=T", naive=True,  hasBias=True,  ergodic=True)
        check_roundtrip("ergodic naive=F hasBias=T", naive=False, hasBias=True,  ergodic=True)

        print("Stable PiLayer boundary variations:")
        check_stability("stable naive=T hasBias=T", naive=True, hasBias=True)
        check_stability("stable naive=T hasBias=F", naive=True, hasBias=False)
        check_stability("stable naive=F hasBias=T", naive=False, hasBias=True)
        check_stability("stable naive=F hasBias=F", naive=False, hasBias=False)
        check_stability("stable square nIn=nOut=6", naive=True, hasBias=False, nInput=6, nOutput=6)
        print("PiLayer tests passed.")

class MapppingLayer(InvertibleLinearLayer):
    """Bias-free, stable reversible linear layer for mapping between row/column spaces.

    Thin wrapper that pins ``hasBias=False`` and ``stable=True`` on
    the parent ``InvertibleLinearLayer`` so reversibility is exact.
    Used by spaces that need a clean row<->column rotation without an
    additive offset.
    """
    def __init__(self, nInput, nOutput, init='orthogonal'):
        """Initialize MapppingLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nInput, nOutput, naive=False, hasBias=False, stable=True)
class SortingLayer(Layer):
    """NeuralSort: differentiable O(1)-depth sorting (Grover et al. 2019).

    Learns a content-determined canonical ordering of vectors along dim 1.
    A direction vector ``w`` scores each vector: s_i = w^Tv_i + bias.
    Scores produce a soft permutation matrix via:

        P[i,j] = softmax((N+1-2i)*s / tau,  dim=j)

    Row i of P concentrates on the element with the i-th largest score.
    The sorted output is P @ V -- a single batched matmul, no loops.

    The scale of w controls effective temperature: small w -> soft
    (uniform P, near-identity), large w -> hard (crisp permutation).

    Forward caches the pre-sort tensor for exact restoration in reverse().
    """

    def __init__(self, symbol_dim, n_passes=None):
        """Initialize SortingLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(symbol_dim, symbol_dim)
        self.symbol_dim = symbol_dim
        # n_passes accepted for config compat but unused by NeuralSort
        self.w = nn.Parameter(torch.randn(symbol_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        self._pre_sort = None

    def forward(self, act):
        """NeuralSort on [B, N, D] along dim 1 -> [B, N, D]."""
        self._pre_sort = act.clone()
        B, N, D = act.shape
        if N <= 1:
            return act
        # Score each vector: [B, N]
        scores = (act * self.w).sum(dim=-1) + self.bias
        # NeuralSort coefficients (1-indexed ranks)
        rank = torch.arange(1, N + 1, device=act.device, dtype=scores.dtype)
        coeff = (N + 1 - 2 * rank)                        # [N]
        # logits[b,i,j] = coeff[i] * scores[b,j]
        logits = coeff.unsqueeze(0).unsqueeze(-1) * scores.unsqueeze(1)
        P = torch.softmax(logits, dim=-1)                  # [B, N, N]
        return torch.bmm(P, act)                           # [B, N, D]

    def reverse(self, act):
        """Restore pre-sort tensor cached during forward."""
        if self._pre_sort is not None:
            return self._pre_sort
        return act

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        nBatch, nSeq, nDim = 4, 8, 16
        layer = SortingLayer(symbol_dim=nDim, n_passes=None)
        device = next(layer.parameters()).device

        x = torch.randn(nBatch, nSeq, nDim, device=device)
        y = layer.forward(x)
        assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
        assert torch.isfinite(y).all(), "forward produced non-finite values"

        x_restored = layer.reverse(y)
        assert x_restored.shape == x.shape, f"reverse shape mismatch"
        err = (x_restored - x).abs().max().item()
        assert err < 1e-6, f"reverse restoration error: {err}"

        # Gradient flow through w and bias
        x2 = torch.randn(nBatch, nSeq, nDim, device=device, requires_grad=True)
        y2 = layer.forward(x2)
        loss = y2.sum()
        loss.backward()
        assert layer.w.grad is not None, "no gradient on w"
        assert layer.bias.grad is not None, "no gradient on bias"
        assert layer.w.grad.abs().sum() > 0, "zero gradient on w"

        # N=1 edge case
        x1 = torch.randn(nBatch, 1, nDim, device=device)
        y1 = layer.forward(x1)
        assert torch.allclose(x1, y1), "N=1 should be identity"

        # Soft permutation matrix is row-stochastic
        x3 = torch.randn(2, 5, nDim, device=device)
        scores = (x3 * layer.w).sum(dim=-1) + layer.bias
        rank = torch.arange(1, 6, device=device, dtype=scores.dtype)
        coeff = (6 - 2 * rank)
        logits = coeff.unsqueeze(0).unsqueeze(-1) * scores.unsqueeze(1)
        P = torch.softmax(logits, dim=-1)
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "P rows must sum to 1"

        print("SortingLayer tests passed.")

class DecisionBoundaryLayer(Layer):
    """Learns a hyperplane normal vector via online updates (not backprop).

    forward() returns +1/-1 on each side of the boundary.
    update() nudges the weight toward or away from an observation depending
    on which side it falls.
    """
    def __init__(self, nInput, nOutput, learning_rate=0.01):
        """Initialize DecisionBoundaryLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super(DecisionBoundaryLayer, self).__init__(nInput, nOutput)
        self.learning_rate = learning_rate
        self.weight        = nn.Parameter(torch.zeros(nInput, nOutput))
        self.register_buffer('noise', torch.randn(nInput, nOutput))

    def forward(self, x, t=0):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        if t != 0:
            self.noise = torch.randn(
                self.weight.shape,
                device=TheDevice.get(),
                dtype=self.weight.dtype,
            )

        W = self.weight + t*self.noise
        dot_product = torch.matmul(x, W)
        decision = torch.where(dot_product >= 0, torch.tensor(1.0, device=x.device),
                               torch.tensor(-1.0, device=x.device))
        return decision

    def update(self, x, t=0):
        """Update.
        
        See class docstring for the operation contract.
        """
        d1 = torch.norm(x - self.weight) ** 2
        d2 = torch.norm(x + self.weight) ** 2
        if d1 < d2:
            self.weight.data += self.learning_rate * (x.unsqueeze(1) - self.weight.data)
        else:
            self.weight.data -= self.learning_rate * (x.unsqueeze(1) + self.weight.data)

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        import matplotlib.pyplot as plt
        n_points = 100
        data = torch.randn(n_points, 2, device=TheDevice.get())
        data[:, 0] *= 1.5

        layer = DecisionBoundaryLayer(nInput=2, nOutput=1, learning_rate=0.01)
        for _ in range(1000):
            idx = torch.randint(0, n_points, (1,))
            x = data[idx].squeeze()
            layer.update(x)

        w = layer.weight.detach().cpu().numpy()
        w_neg = -w

        data_np = data.cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(data_np[:, 0], data_np[:, 1], label="Data", alpha=0.6)

        plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='r', linestyle='-',label='w')
        plt.quiver(0, 0, w_neg[0], w_neg[1], angles='xy', scale_units='xy', scale=1, color='b', linestyle='-',label='-w')

        a, b = w
        x_vals = np.linspace(np.min(data_np[:, 0]) - 1, np.max(data_np[:, 0]) + 1, 100)
        if np.abs(b) > 1e-5:
            y_vals = - (a / b) * x_vals
            plt.plot(x_vals, y_vals, color='g', linestyle='dashed', label='Hyperplane')
        else:
            plt.axvline(0, color='g', linestyle='dashed', label='Hyperplane')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Decision Boundary Learning')
        plt.legend()
        plt.grid(True)
        plt.xlim(np.min(data_np[:, 0]) - 2, np.max(data_np[:, 0]) + 2)
        plt.ylim(np.min(data_np[:, 1]) - 2, np.max(data_np[:, 1]) + 2)
        warnings.filterwarnings('ignore', message='.*line style')
        plt.show(block=False)
class AttentionLayer(Layer):
    """Unified attention layer with three modes.

    type="symmetric"   -- Hopfield-like: scores = A^T @ A (positive semi-definite).
                         Attends across feature channels.
    type="asymmetric"  -- Channel attention: scores = Q^T @ K.
                         Attends across feature channels.
    type="transformer" -- Standard multi-head attention over the object/token axis.
                         Q K^T / sqrt(d) with multi-head splitting.

    All modes require 3D input [batch, nObj, dim].
    """
    def __init__(self, nInput, nOutput, nHidden=None, type="asymmetric", nHeads=1):
        """Initialize AttentionLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super(AttentionLayer, self).__init__(nInput, nOutput)
        self.nHidden = nOutput if not nHidden else nHidden
        self.type = type
        self.mask = None
        self.beta = 10
        self.reversible = False
        self.nHeads = nHeads

        if self.type == "symmetric":
            self.A = LinearLayer(self.nInput, self.nHidden)
            self.V = LinearLayer(self.nInput, self.nHidden)
        else:
            self.Q = LinearLayer(self.nInput, self.nHidden)
            self.K = LinearLayer(self.nInput, self.nHidden)
            self.V = LinearLayer(self.nInput, self.nHidden)
        self.Out = LinearLayer(self.nHidden, self.nOutput)

        # Transformer-specific: multi-head geometry
        if self.type == "transformer":
            if nHeads < 1:
                raise ValueError(f"nHeads must be >= 1, got {nHeads}")
            if self.nHidden % self.nHeads != 0:
                raise ValueError(
                    f"nHidden ({self.nHidden}) must be divisible by nHeads ({self.nHeads})")
            self.headDim = self.nHidden // self.nHeads
            self.scale = self.headDim ** -0.5

    def set_mask(self, mask: Optional[torch.Tensor]):
        """Set an optional attention mask (used by all types)."""
        self.mask = mask

    # --- Transformer helpers (multi-head) ---

    def _reshape_heads(self, x):
        batch, n_obj, _ = x.shape
        x = x.view(batch, n_obj, self.nHeads, self.headDim)
        return x.transpose(1, 2)

    def _normalize_mask(self, mask, batch, n_obj):
        """Normalize mask.
        
        See class docstring for the operation contract.
        """
        if mask is None:
            return None
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.dim() == 2:
            if list(mask.shape) != [batch, n_obj]:
                raise ValueError(
                    f"2D mask must have shape {[batch, n_obj]}, got {list(mask.shape)}")
            mask = mask[:, None, None, :].expand(-1, self.nHeads, n_obj, -1)
        elif mask.dim() == 3:
            if list(mask.shape) != [batch, n_obj, n_obj]:
                raise ValueError(
                    f"3D mask must have shape {[batch, n_obj, n_obj]}, got {list(mask.shape)}")
            mask = mask[:, None, :, :].expand(-1, self.nHeads, -1, -1)
        elif mask.dim() == 4:
            if mask.shape[0] != batch or mask.shape[-2:] != (n_obj, n_obj):
                raise ValueError(
                    f"4D mask must end with {[n_obj, n_obj]} and batch {batch}, "
                    f"got {list(mask.shape)}")
            if mask.shape[1] == 1:
                mask = mask.expand(-1, self.nHeads, -1, -1)
            elif mask.shape[1] != self.nHeads:
                raise ValueError(
                    f"4D mask head dimension must be 1 or nHeads ({self.nHeads}), "
                    f"got {mask.shape[1]}")
        else:
            raise ValueError(f"Unsupported mask rank {mask.dim()}; expected 2, 3, or 4")
        return mask

    # --- Forward dispatch ---

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        assert x.ndim == 3, f"AttentionLayer expects 3D input [B, N, D], got {list(x.shape)}"
        if self.type == "transformer":
            return self._forward_transformer(x)
        elif self.type == "symmetric":
            return self._forward_symmetric(x)
        else:
            return self._forward_asymmetric(x)

    def _forward_symmetric(self, x):
        """Forward symmetric.
        
        See class docstring for the operation contract.
        """
        a2     = self.A(x)
        value  = x if self.nHidden == self.nInput else self.V(x)
        scores = torch.matmul(a2.transpose(-2, -1), a2) / (self.nInput ** 0.5)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float('-inf'))
        attn = F.softmax(self.beta * scores, dim=-1) if not self.reversible else scores
        output = value @ attn
        if self.nHidden != self.nOutput:
            output = self.Out(output)
        return output

    def _forward_asymmetric(self, x):
        """Forward asymmetric.
        
        See class docstring for the operation contract.
        """
        query  = self.Q(x)
        key    = self.K(x)
        value  = x if self.nHidden == self.nInput else self.V(x)
        scores = torch.matmul(query.transpose(-2, -1), key) / (self.nInput ** 0.5)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float('-inf'))
        attn = F.softmax(self.beta * scores, dim=-1) if not self.reversible else scores
        output = value @ attn
        if self.nHidden != self.nOutput:
            output = self.Out(output)
        return output

    def _forward_transformer(self, x):
        """Forward transformer.
        
        See class docstring for the operation contract.
        """
        batch, n_obj, _ = x.shape
        query = self._reshape_heads(self.Q(x))
        key   = self._reshape_heads(self.K(x))
        value = self._reshape_heads(self.V(x))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = self._normalize_mask(self.mask, batch, n_obj)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        # Cached for diagnostic readers (e.g. InterSentenceLayer uses
        # the last-position entropy as a confidence signal).  Detached
        # so holding it doesn't pin the graph.
        self.last_attn = attn.detach()
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(batch, n_obj, self.nHidden)
        output = self.Out(output)
        return output

    def reverse(self, y, bias=None, temp=None):
        """Attention is not analytically invertible; keep reverse as identity."""
        return super().reverse(y, bias=bias, temp=temp)

    @staticmethod
    def test():
        # Test all three types with 3D input
        """Self-test; verifies the round-trip / invariant."""
        for atype in ["symmetric", "asymmetric", "transformer"]:
            kwargs = {"nInput": 8, "nOutput": 8, "type": atype}
            if atype == "transformer":
                kwargs["nHeads"] = 2
            layer = AttentionLayer(**kwargs)
            x = torch.randn(4, 5, 8, device=TheDevice.get())
            y = layer(x)
            assert list(y.shape) == [4, 5, 8], f"type={atype}: expected [4,5,8], got {list(y.shape)}"
        # Test nInput != nOutput
        layer = AttentionLayer(nInput=6, nOutput=3, nHidden=7, type="asymmetric")
        x = torch.randn(4, 5, 6, device=TheDevice.get())
        y = layer(x)
        assert list(y.shape) == [4, 5, 3], f"asymmetric nIn!=nOut: expected [4,5,3], got {list(y.shape)}"
class AssociationLayer(Layer):
    """Cross-symbol associative memory for bidirectional pattern completion.

    Given a [B, N] activation vector over N symbol slots, computes an
    association matrix that measures how each symbol relates to every other.
    This implements the EQUALS rule: when symbol A is symbol B, they have a
    high cross-association score.

    type="symmetric"  -- Hopfield-like: scores = A(x)^T @ A(x), positive
                        semi-definite.  Associations are symmetric (A==B <-> B==A).
    type="hopfield"   -- Modern Hopfield / softmax retrieval: projects x into
                        queries and keys, softmax attention across symbols,
                        returns the retrieved pattern.

    Input:  [B, N] symbol activations.
    Output: [B, N] associated activations (cross-symbol pattern completion).
    """

    def __init__(self, nInput, nOutput=None, nHidden=None,
                 type="symmetric", beta=10.0):
        """Initialize AssociationLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        nOutput = nOutput or nInput
        super().__init__(nInput, nOutput)
        self.nHidden = nHidden or nInput
        self.type = type
        self.beta = beta

        if self.type == "symmetric":
            # Hopfield energy: E = -x^T W x, stored patterns in W = A^T A
            self.A = LinearLayer(self.nInput, self.nHidden)
        elif self.type == "hopfield":
            # Modern Hopfield: separate query/key projections
            self.Q = LinearLayer(self.nInput, self.nHidden)
            self.K = LinearLayer(self.nInput, self.nHidden)
        else:
            raise ValueError(f"AssociationLayer type must be 'symmetric' or 'hopfield', got '{type}'")

        if self.nHidden != self.nOutput:
            self.Out = LinearLayer(self.nHidden, self.nOutput)
        else:
            self.Out = None

        self.layers = [self.A] if self.type == "symmetric" else [self.Q, self.K]
        if self.Out is not None:
            self.layers.append(self.Out)

    def forward(self, x):
        """Compute cross-symbol associations.

        Args:
            x: [B, N] symbol activation vector.

        Returns:
            [B, N] associated activation -- pattern-completed via stored associations.
        """
        # Reshape to [B, N, 1] for matmul compatibility
        x3 = x.unsqueeze(-1)  # [B, N, 1]

        if self.type == "symmetric":
            # Project: [B, N, 1] -> [B, N, H] via broadcasting
            # A operates on the last dim, so reshape to [B, N, nInput]
            # For 1-dim symbols, we tile to nInput width
            a = self.A.forward(x)               # [B, nHidden]
            # Association scores: outer product in hidden space
            scores = torch.bmm(
                a.unsqueeze(-1),                # [B, nHidden, 1]
                a.unsqueeze(-2)                 # [B, 1, nHidden]
            )                                   # [B, nHidden, nHidden]
            # Apply to original activation via projection
            attn = F.softmax(self.beta * scores, dim=-1)  # [B, H, H]
            # Map back: use A to project x to hidden, attend, project back
            h = a.unsqueeze(-1)                 # [B, H, 1]
            out = torch.bmm(attn, h).squeeze(-1)  # [B, H]

        elif self.type == "hopfield":
            q = self.Q.forward(x)               # [B, nHidden]
            k = self.K.forward(x)               # [B, nHidden]
            # Similarity in hidden space -> association strength
            scores = q * k                      # [B, nHidden] element-wise
            out = F.softmax(self.beta * scores, dim=-1) * k  # [B, nHidden]

        if self.Out is not None:
            out = self.Out.forward(out)          # [B, nOutput]
        return out

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        for atype in ["symmetric", "hopfield"]:
            layer = AssociationLayer(nInput=8, type=atype)
            x = torch.randn(4, 8, device=TheDevice.get())
            y = layer(x)
            assert y.shape == (4, 8), f"type={atype}: expected (4,8), got {y.shape}"
        # nInput != nOutput
        layer = AssociationLayer(nInput=6, nOutput=4, nHidden=8, type="symmetric")
        x = torch.randn(2, 6, device=TheDevice.get())
        y = layer(x)
        assert y.shape == (2, 4), f"nIn!=nOut: expected (2,4), got {y.shape}"

class LiftingLayer(Layer):
    """Codebook of verb weight matrices for conceptual composition (lift).

    Each verb in the codebook is a learned [D, D] weight matrix that
    transforms concept vectors bidirectionally (invertible linear map).
    Verb selection is soft: cosine similarity between a query embedding
    and learned codebook keys produces blending weights.

    Transitive -- lift(C, C):
        VP_eff @ C1 -> attention added to C2 (forward)
        VP_eff^T @ C2 -> attention added to C1 (backward)

    Intransitive -- lift(C):
        VP_eff @ C1 -> self-attention added to C1
    """
    def __init__(self, nVerbs, nDim, ergodic=False):
        """Initialize LiftingLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nDim, nDim)
        self.nVerbs = nVerbs
        self.nDim = nDim
        # Codebook keys for soft VERB selection [nVerbs, nDim]
        self.keys = nn.Parameter(torch.randn(nVerbs, nDim))
        # VERB weight matrices -- stack of [nDim, nDim] (initialized near-identity)
        self.vp_weights = nn.ParameterList([
            nn.Parameter(torch.eye(nDim) + 0.01 * torch.randn(nDim, nDim))
            for _ in range(nVerbs)
        ])

    def _select_vp(self, query):
        """Soft-select a blended VERB matrix via cosine similarity.

        Args:
            query: [B, D] embedding vector for VERB selection.
        Returns:
            [B, D, D] blended VERB weight matrix.
        """
        sim = F.cosine_similarity(
            query.unsqueeze(1),           # [B, 1, D]
            self.keys.unsqueeze(0),       # [1, V, D]
            dim=-1
        )                                 # [B, V]
        weights = F.softmax(sim * 10.0, dim=-1)  # sharp selection
        vp_stack = torch.stack(list(self.vp_weights))  # [V, D, D]
        vp_eff = torch.einsum('bv, vij -> bij', weights, vp_stack)  # [B, D, D]
        return vp_eff

    def forward_transitive(self, C1, C2, vp_query):
        """Transitive VERB: C1 VERB C2.

        VP_eff @ C1 -> attention added to C2 (forward)
        VP_eff^T @ C2 -> attention added to C1 (backward)

        Args:
            C1: [B, N, D] subject concepts.
            C2: [B, N, D] object concepts.
            vp_query: [B, D] VERB embedding for codebook lookup.
        Returns:
            (C1', C2') updated concept tensors.
        """
        vp_eff = self._select_vp(vp_query)              # [B, D, D]
        fwd = torch.bmm(C1, vp_eff)                     # [B, N, D]
        bwd = torch.bmm(C2, vp_eff.transpose(1, 2))     # [B, N, D]
        return C1 + bwd, C2 + fwd

    def forward_reflexive(self, C1, vp_query):
        """Intransitive VERB: C1 VERB (self-application).

        Applies VERB as self-attention on C1, producing the post-VERB state.
        Replaces C1 in-place for this iteration.

        # NOTE: In a stateful model, this produces a state-of-affairs at T2
        # which is distinct from the T1 input. The temporal distinction is
        # significant for learning over sequences of events but is not
        # represented in the current single-timestamp encoding.

        Args:
            C1: [B, N, D] concept vectors.
            vp_query: [B, D] VERB embedding for codebook lookup.
        Returns:
            C1': [B, N, D] self-attended concept vectors.
        """
        vp_eff = self._select_vp(vp_query)              # [B, D, D]
        attended = torch.bmm(C1, vp_eff)                 # [B, N, D]
        return C1 + attended

    def forward_transitive_svo(self, subject, verb, obj, symbolic_space):
        """Ternary LIFT: S V O.

        The object restricts the verb's lifting operation by intersecting
        verb symbols with object symbols in symbolic space, then mapping
        the restricted action back to conceptual space and applying it
        to the subject.

        PiLayer maps on the nDim axis: [B, N, concept_dim] -> [B, N, symbol_dim].
        Full concept vectors pass through directly.

        Args:
            subject: [B, N, D] subject concepts (S).
            verb: [B, N, D] verb concepts (V).
            obj: [B, N, D] object concepts (O).
            symbolic_space: SymbolicSpace for concept<->symbol projection.
        Returns:
            [B, N, D] lifted subject concepts.
        """
        ss = symbolic_space

        # 1. Project concept vectors to symbol space via SymbolicSpace.forward()
        ss.subspace.set_event(verb)
        verb_syms = ss.forward(ss.subspace).materialize()     # [B, N, symbol_dim]
        ss.subspace.set_event(obj)
        obj_syms = ss.forward(ss.subspace).materialize()      # [B, N, symbol_dim]

        # 2. Intersect: restrict verb by object (monotonic -> min)
        restricted_syms = torch.min(verb_syms, obj_syms)      # [B, N, symbol_dim]

        # 3. Symbols are already in concept-space coordinates
        # (symbol_dim == concept_dim is enforced in SymbolicSpace).
        restricted = restricted_syms                          # [B, N, D]

        # 4. Weight verb by restricted concept norms -> query
        rw = restricted.norm(dim=-1, keepdim=True)            # [B, N, 1]
        rw = rw / (rw.max(dim=1, keepdim=True).values + 1e-6)
        query = (verb * rw).mean(dim=1)                       # [B, D]

        # 5. Select verb matrix and apply to subject
        vp_eff = self._select_vp(query)                       # [B, D, D]
        fwd = torch.bmm(subject, vp_eff)                      # [B, N, D]
        return subject + fwd

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        device = TheDevice.get()
        B, N, D, V = 4, 8, 16, 6
        layer = LiftingLayer(nVerbs=V, nDim=D)

        C1 = torch.randn(B, N, D, device=device)
        C2 = torch.randn(B, N, D, device=device)
        query = torch.randn(B, D, device=device)

        # Soft selection shape
        vp = layer._select_vp(query)
        assert vp.shape == (B, D, D), f"select shape: {vp.shape}"

        # Transitive
        C1_out, C2_out = layer.forward_transitive(C1, C2, query)
        assert C1_out.shape == (B, N, D), f"transitive C1: {C1_out.shape}"
        assert C2_out.shape == (B, N, D), f"transitive C2: {C2_out.shape}"

        # Reflexive
        C1_refl = layer.forward_reflexive(C1, query)
        assert C1_refl.shape == (B, N, D), f"reflexive: {C1_refl.shape}"

        # Gradient flow
        C1_grad = C1.clone().requires_grad_(True)
        q_grad = query.clone().requires_grad_(True)
        out = layer.forward_reflexive(C1_grad, q_grad)
        out.sum().backward()
        assert C1_grad.grad is not None, "no gradient on C1"
        assert q_grad.grad is not None, "no gradient on query"

        # -- Ternary SVO ------------------------------------------
        # Mock symbolic space: PiLayer maps on nDim axis [B, N, D] -> [B, N, D]
        class _MockSubspace:
            """Test double for SubSpace; just stores the event tensor.

            Implements the minimal subset of SubSpace's API that
            LiftingLayer.test exercises (set_event / materialize).
            """
            def __init__(self):
                """Start with no event tensor."""
                self._event = None
                self.batch = 0
            def set_event(self, t, compute_activation=False):
                """Stash the event tensor."""
                self._event = t
            def materialize(self):
                """Return the stashed event tensor."""
                return self._event
        class _MockSymSpace:
            """Test double for SymbolicSpace bound to a PiLayer.

            Implements just enough of SymbolicSpace's API for
            LiftingLayer.test to exercise the ternary SVO lift path.
            """
            def __init__(self, pi):
                """Hold the bound PiLayer and a fresh _MockSubspace."""
                self.layer = pi
                self.subspace = _MockSubspace()
            def forward(self, vspace):
                """Forward pass.
                
                See class docstring for the operation this layer applies.
                """
                act = vspace.materialize()
                act = self.layer.forward(act)
                vspace.set_event(act)
                return vspace
        mock_ss = _MockSymSpace(PiLayer(D, D, monotonic=True,
                                        invertible=True, nonlinear=True))
        S = torch.randn(B, N, D, device=device)
        V = torch.randn(B, N, D, device=device)
        O = torch.randn(B, N, D, device=device)
        result = layer.forward_transitive_svo(S, V, O, mock_ss)
        assert result.shape == (B, N, D), f"SVO shape: {result.shape}"

        # Gradient flow through SVO
        S_g = S.clone().requires_grad_(True)
        V_g = V.clone().requires_grad_(True)
        O_g = O.clone().requires_grad_(True)
        out_svo = layer.forward_transitive_svo(S_g, V_g, O_g, mock_ss)
        out_svo.sum().backward()
        assert S_g.grad is not None, "no gradient on subject"
        assert V_g.grad is not None, "no gradient on verb"
        assert O_g.grad is not None, "no gradient on object"

        print("LiftingLayer tests passed.")
class LoweringLayer(Layer):
    """Rank-reducing bottleneck for conceptual composition (lower).

    Compresses a concept vector through a smaller dimension then expands
    back: [D] -> [bottleneck] -> [D]. For binary lower(C, C), the second
    argument gates the bottleneck representation to select a specific
    instance from the set represented by the first argument.
    """
    def __init__(self, nDim, bottleneck=None):
        """Initialize LoweringLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        if bottleneck is None:
            bottleneck = max(4, nDim // 4)
        super().__init__(nDim, nDim)
        self.nDim = nDim
        self.bottleneck = bottleneck
        self.down = LinearLayer(nDim, bottleneck)
        self.up = LinearLayer(bottleneck, nDim)

    def forward(self, left, right=None):
        """Lower a concept through the bottleneck.

        Args:
            left: [B, N, D] or [B, D] concept vectors (the set/type).
            right: [B, N, D] or [B, D] optional selector concept.
                   When provided, gates the bottleneck to select an instance.
        Returns:
            Lowered concept, same shape as left.
        """
        compressed = self.down.forward(left)        # [..., bottleneck]
        if right is not None:
            gate = torch.sigmoid(self.down.forward(right))  # [..., bottleneck]
            compressed = compressed * gate
        return self.up.forward(compressed)           # [..., D]

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        device = TheDevice.get()
        B, N, D = 4, 8, 16
        layer = LoweringLayer(nDim=D, bottleneck=4)

        left = torch.randn(B, N, D, device=device)
        right = torch.randn(B, N, D, device=device)

        # Unary
        out = layer.forward(left)
        assert out.shape == (B, N, D), f"unary shape: {out.shape}"

        # Binary (with selector)
        out2 = layer.forward(left, right)
        assert out2.shape == (B, N, D), f"binary shape: {out2.shape}"

        # Gradient flow
        left_g = left.clone().requires_grad_(True)
        right_g = right.clone().requires_grad_(True)
        out3 = layer.forward(left_g, right_g)
        out3.sum().backward()
        assert left_g.grad is not None, "no gradient on left"
        assert right_g.grad is not None, "no gradient on right"

        print("LoweringLayer tests passed.")
        
class SparsityRegLayer(Layer):
    """Soft-threshold L1 proximal operator.

    Shared by PerceptualSpace, ConceptualSpace, and SymbolicSpace so a
    single sparsity implementation is reused across tiers. Extracted from
    ``SymbolicSpace.l1_proximal``.

    Acts as identity when disabled or when ``l1_lambda <= 0``. Otherwise
    applies ``sign(x) * max(|x| - l1_lambda, 0)``, which zeros activations
    below the threshold and shrinks survivors.
    """

    def __init__(self, l1_lambda: float = 0.0, enabled: bool = True):
        # nInput/nOutput are unused -- this is a pointwise op -- but the
        # Layer base contract requires both. Pass zeros; dim-agnostic.
        """Initialize SparsityRegLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)
        self.l1_lambda = float(l1_lambda)
        self.enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        if not self.enabled or self.l1_lambda <= 0.0:
            return x
        return torch.sign(x) * torch.clamp(
            torch.abs(x) - self.l1_lambda, min=0.0
        )
class SmoothingRegLayer(Layer):
    """Total-variation penalty on consecutive concepts of a symbol vector.

    Complements ``SparsityRegLayer`` by pressuring the symbol vector toward
    a piecewise-flat profile in addition to an L1 sparsity norm. Penalises
    ``|S[..., k+1] - S[..., k]|`` and returns a scalar.

    In bivector mode (even last dim) the operand is reshaped to
    ``[..., K, 2]`` and collapsed along the pole axis with ``amax`` before
    differencing. This respects the paired-index convention from the
    bivector encoding: indices ``2k`` and ``2k+1`` are poles of the same
    concept -- penalising their difference would fight the 4-valued
    (quaternary) truth encoding, so we measure discontinuity between
    *distinct* concepts only. See basicmodel/doc/BuddhistParallels.md
    for the tetralemma (catuskoti) mapping.

    Acts as identity (returns scalar zero) when disabled or ``lam <= 0``.
    """

    def __init__(self, lam: float = 0.0, enabled: bool = True):
        """Initialize SmoothingRegLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)
        self.lam = float(lam)
        self.enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        if not self.enabled or self.lam <= 0.0:
            return x.new_tensor(0.0) if torch.is_tensor(x) else torch.tensor(0.0)
        if x.shape[-1] >= 2 and x.shape[-1] % 2 == 0:
            pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
            collapsed = pair.amax(dim=-1)
        else:
            collapsed = x
        if collapsed.shape[-1] < 2:
            return x.new_tensor(0.0)
        diff = collapsed[..., 1:] - collapsed[..., :-1]
        return self.lam * diff.abs().mean()
class ImpenetrableLayer(Layer):
    """Mereological separation regularizer over a symbol codebook.

    Classifies each ordered pair (i, j) of codebook rows into one of five
    mereological relations using ``Basis.part`` (clipped-cosine scalar
    parthood, scalar=True) and penalises overlap between rows whose
    EMA usage frequencies disagree.

    Penalty: ``overlap_strength(i, j) * |trust(i) - trust(j)|``
    where ``overlap_strength = min(P[i,j], P[j,i]) * (1 - max(P[i,j], P[j,i])^k)``
    damps to zero as the pair approaches mutual identity (``equal``).

    Five relations (with thresholds τ and ε):
      disjoint:  P[i,j] < ε  and  P[j,i] < ε
      part_ij:   P[i,j] > τ  and  P[j,i] < ε
      part_ji:   P[i,j] < ε  and  P[j,i] > τ
      equal:     P[i,j] > τ  and  P[j,i] > τ
      overlap:   both partial (neither > τ nor < ε)

    Trust source: ``basis.vq.cluster_size`` EMA when a VectorQuantize is
    present; falls back to row norms when VQ is off.

    A separate variance floor guards against row-collapse (all rows
    converging to a single point).

    Returns a scalar. When ``enabled`` is false or all weights are zero,
    the layer short-circuits to zero without touching the codebook.

    See basicmodel/doc/BuddhistParallels.md for the tetralemma (catuskoti)
    mapping of the 4-valued truth logic that the separated codebook carries.
    """

    def __init__(self, overlap_weight: float = 0.1,
                 variance_floor: float = 0.01,
                 enabled: bool = True,
                 full_part_threshold: float = 0.9,
                 disjoint_threshold: float = 0.1,
                 equal_suppression: float = 4.0):
        """Initialize ImpenetrableLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)
        self.overlap_weight = float(overlap_weight)
        self.variance_floor = float(variance_floor)
        self.enabled = bool(enabled)
        self.tau = float(full_part_threshold)
        self.eps = float(disjoint_threshold)
        self.equal_k = float(equal_suppression)
        # Diagnostic slots populated on each forward pass.
        self.last_overlap_loss = None
        self.last_variance = None
        self.last_relation_counts = None

    def _pairwise_parthood(self, codebook: torch.Tensor,
                           basis) -> torch.Tensor:
        """Compute P[i, j] = part(cb[i], cb[j], scalar=True) for all K*K pairs."""
        K = codebook.shape[0]
        cb_i = codebook.unsqueeze(1).expand(K, K, -1)
        cb_j = codebook.unsqueeze(0).expand(K, K, -1)
        return basis.part(cb_i, cb_j, monotonic=True, scalar=True)

    def _trust(self, codebook: torch.Tensor, basis) -> torch.Tensor:
        """Trust per codebook row.

        Prefer VQ cluster_size EMA usage when the underlying basis has a
        live VectorQuantize; fall back to normalised row norms otherwise.
        Returns a [K] tensor on ``codebook.device``.
        """
        vq = getattr(basis, "vq", None)
        if vq is not None and hasattr(vq, "cluster_size"):
            counts = vq.cluster_size
            if torch.is_tensor(counts) and counts.numel() == codebook.shape[0]:
                counts = counts.to(codebook.device).float()
                return counts / counts.sum().clamp(min=1.0)
        n = codebook.norm(dim=-1).float()
        return n / n.max().clamp(min=epsilon)

    def _classify(self, P: torch.Tensor) -> dict:
        """Classify each ordered off-diagonal (i, j) pair into one of five
        mereological relations. The diagonal is masked out so diagnostic
        counts sum to ``K * (K - 1)``."""
        high = P > self.tau
        low = P < self.eps
        high_T = high.transpose(0, 1)
        low_T = low.transpose(0, 1)
        eye = torch.eye(P.shape[0], device=P.device, dtype=torch.bool)
        off = ~eye
        return {
            "disjoint": (low & low_T) & off,
            "part_ij":  (high & low_T) & off,
            "part_ji":  (low & high_T) & off,
            "equal":    (high & high_T) & off,
            "overlap":  (~(high | low) & ~(high_T | low_T)) & off,
        }

    def forward(self, codebook: torch.Tensor, basis=None) -> torch.Tensor:
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        zero = (codebook.new_tensor(0.0) if isinstance(codebook, torch.Tensor)
                else torch.tensor(0.0))
        self.last_overlap_loss = None
        self.last_variance = None
        self.last_relation_counts = None
        if not self.enabled:
            return zero
        if codebook is None or not isinstance(codebook, torch.Tensor):
            return zero
        if codebook.ndim != 2 or codebook.shape[0] < 2:
            return zero
        want_overlap = self.overlap_weight > 0.0 and basis is not None
        want_var = self.variance_floor > 0.0
        if not (want_overlap or want_var):
            return zero

        K = codebook.shape[0]
        total = zero

        if want_overlap:
            P = self._pairwise_parthood(codebook, basis)
            trust = self._trust(codebook, basis)
            trust_diff = (trust.unsqueeze(0) - trust.unsqueeze(1)).abs()
            P_T = P.transpose(0, 1)
            min_P = torch.minimum(P, P_T)
            max_P = torch.maximum(P, P_T)
            # Damp overlap as the pair approaches mutual identity so that
            # two rows meant to encode the same concept (``equal``) do
            # not contribute to the overlap penalty.
            damp = (1.0 - max_P.clamp(0.0, 1.0) ** self.equal_k).clamp(min=0.0)
            eye = torch.eye(K, device=codebook.device, dtype=torch.bool)
            keep = (~eye).float()
            denom = keep.sum().clamp(min=1.0)
            overlap_loss = (min_P * damp * trust_diff * keep).sum() / denom
            self.last_overlap_loss = overlap_loss.detach()
            total = total + self.overlap_weight * overlap_loss
            # `last_relation_counts` is diagnostic only; the 5
            # `v.sum().item()` calls are host syncs (cudaMemcpyDtoH)
            # that break CUDA-graph capture. Gate behind MODEL_DEBUG
            # (bucket-2/3 pattern; `last_relation_counts` is already
            # None-initialised so consumers tolerate it). `_classify`
            # is only needed for these counts, so skip it too.
            if util.MODEL_DEBUG:
                rels = self._classify(P)
                self.last_relation_counts = {
                    k: int(v.sum().item()) for k, v in rels.items()
                }

        if want_var:
            std = codebook.std(dim=0, unbiased=False).mean()
            var_pen = torch.relu(codebook.new_tensor(self.variance_floor) - std)
            self.last_variance = var_pen.detach()
            total = total + var_pen

        return total


# Default Gaussian region width for Phase 1b introspective grammar layers.
# `area` and `luminosity` use this when the slot has no calibrated extent
# attached.  Matches `Spaces._DEFAULT_SYMBOL_SIGMA`.
_DEFAULT_SUBSYMBOLIC_SIGMA = 0.1


def _gaussian_kernel_overlap(X, Y, sigma_x, sigma_y, eps=1e-8):
    """Gaussian kernel overlap: ``exp(-d^2 / 2(sigma_x^2 + sigma_y^2))``.

    Reference implementation for the contiguity / overlap kernel reused by
    `Mereology.Area`, `Mereology.Luminosity`, and the existing
    `Ops.corner_overlap` / `Ops.epsilon_delta` measures.

    Args:
        X: ``[N, D]`` left points.
        Y: ``[M, D]`` right points.
        sigma_x: scalar or ``[N]`` per-row sigma for X.
        sigma_y: scalar or ``[M]`` per-row sigma for Y.

    Returns:
        ``[N, M]`` overlap matrix with values in ``(0, 1]``.
    """
    d2 = torch.cdist(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0) ** 2  # [N, M]
    if torch.is_tensor(sigma_x):
        sx = sigma_x
    else:
        sx = torch.full((X.shape[0],), float(sigma_x),
                        device=X.device, dtype=X.dtype)
    if torch.is_tensor(sigma_y):
        sy = sigma_y
    else:
        sy = torch.full((Y.shape[0],), float(sigma_y),
                        device=Y.device, dtype=Y.dtype)
    denom = 2.0 * (sx.unsqueeze(1) ** 2 + sy.unsqueeze(0) ** 2) + eps
    return torch.exp(-d2 / denom)


def ste_answer(q, f):
    """Straight-through-estimator wrapper for non-differentiable lookups.

    Forward returns ``f`` (the discrete answer); backward routes the
    gradient through ``q`` (the differentiable query).  Used for
    Mereonomy / TruthLayer reads on questions so the question-formation
    path receives gradient.

    Args:
        q: differentiable query tensor.
        f: discrete answer tensor (same shape, no grad needed).

    Returns:
        Tensor that equals ``f`` in the forward pass and routes
        gradient to ``q`` on backward.
    """
    return q + (f - q).detach()


class TruthLayer(Layer):
    """Truth store on SymbolicSpace: encoded truth statements scaled by DoT.

    Each truth statement is processed through the model pipeline to produce
    a symbolic activation ``[nSymbols]``.  The activation is then scaled by
    the DegreeOfTruth before storage:

        stored = activation * degree

    This means the stored vector carries the DoT intrinsically:
      - degree = +1 -> full activation stored (attractor)
      - degree = -1 -> negated activation stored (disperser)
      - degree =  0 -> zero vector (inert, prunable)

    The ``field()`` method projects stored truths into ConceptualSpace
    via cosine similarity.  Because the degree is baked into the stored
    vectors, positive-DoT truths attract and negative-DoT truths repel
    without needing a separate degree buffer.

    Propositional structure is defined by the S-tier grammar:
      - ``part(S, S)`` -- parthood / containment
      - ``equals(S, S)`` -- identity / equivalence
    """

    def __init__(self, nDim: int, max_truths: int = 1024):
        """Initialize TruthLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nDim, nDim)
        self.nDim = nDim
        self.max_truths = max_truths

        # Storage buffer: activation * degree (DoT baked in)
        self.register_buffer(
            'truths',
            torch.zeros(max_truths, nDim),
        )
        self.register_buffer(
            'count',
            torch.tensor(0, dtype=torch.long),
        )
        # Parallel metadata used by suggest_clarifications(). Indexed
        # alongside truths[:count]; missing entries fall back to a
        # generic "(truth #i)" reference.
        self._sources = []
        self._trusts = []

        # Per-tick pending buffer for record_batch() / compact().
        # Sized 4x the persistent store so a single tick can stage many
        # candidates before pruning. compact() filters by trust and
        # promotes survivors to the persistent buffer; lives outside the
        # compute brick to keep `forward + backward + step` sync-free.
        self.register_buffer(
            '_pending_truths',
            torch.zeros(max_truths * 4, nDim),
        )
        self.register_buffer(
            '_pending_trust',
            torch.zeros(max_truths * 4),
        )
        # Python int (not a tensor) — host-side cursor; reading / writing
        # it never touches the GPU. Reset to 0 after each compact().
        self._pending_count = 0

        # Host-side emptiness flag. ``len(self)`` reads ``count.item()``
        # (a GPU->host sync); the only in-brick consumer
        # (WordSpace.truth_modulated_loss) needs *emptiness*, not the
        # count. This bool mirrors ``count > 0`` and is maintained
        # co-located with every ``count`` write from an already-host-side
        # value, so it never adds a sync and cannot drift. Read via
        # ``is_empty()``.
        self._nonempty = False

    # -- Record / Query ------------------------------------------------

    @torch.no_grad()
    def record(self, activation: torch.Tensor, degree: float,
               basis=None) -> int:
        """Store a truth: activation scaled by its DegreeOfTruth.

        Bivector path (``basis`` provided and ``basis.monotonic`` and
        ``activation.shape[-1]`` is even): indices 2k / 2k+1 encode the
        positive / negative poles of concept k. For ``degree >= 0``,
        store ``activation * degree`` (positive poles already hot). For
        ``degree < 0``, paired-index flip via ``basis.negation(...,
        monotonic=True)`` lands the mass on the negative poles, then
        scale by ``|degree|``. This preserves 4-valued (quaternary)
        truth semantics: asserting A and asserting not(A) are
        orthogonal, not cancelling. See
        basicmodel/doc/BuddhistParallels.md for the tetralemma mapping.

        Legacy path (no basis or odd last dim): the stored vector is
        ``activation * degree``, so the DoT is encoded in both the
        magnitude and (for negative degrees) the sign.

        Args:
            activation: (nDim,) symbolic activation from the model pipeline.
            degree: scalar in [-1, 1].  +1 = certainly true, -1 = certainly
                    false, 0 = unknown/inert.
            basis: optional Basis with negation(monotonic=True). When
                   provided and monotonic, enables bivector storage.

        Returns:
            Index of the stored entry.
        """
        if self.count >= self.max_truths:
            raise RuntimeError(
                f"Truth store full ({self.max_truths} entries). "
                "Increase max_truths or prune stale entries."
            )
        degree = max(-1.0, min(1.0, degree))
        idx = self.count.item()
        # Coerce to the persistent buffer's device. Callers (tests,
        # Models.py) sometimes construct activation tensors on the
        # default device while the truth_layer's buffers live on the
        # model's device; without this coercion the contradiction-check
        # matmul below trips a cross-device gather.
        vec = activation.detach().to(self.truths.device)

        bivector_mode = (
            basis is not None
            and getattr(basis, 'monotonic', False)
            and vec.shape[-1] % 2 == 0
        )

        if bivector_mode and degree < 0:
            vec = Ops._negation_kernel(vec, monotonic=True)
            stored_vec = vec * abs(degree)
        else:
            stored_vec = vec * degree
        self.truths[idx] = stored_vec
        self.count += 1
        self._nonempty = True  # record() only ever grows the store

        # Legacy contradiction warning: anti-parallel cosine only applies
        # to bitonic storage. Under bivector, A and not(A) land on
        # orthogonal paired indices (cosine 0, not -1), so this branch
        # is skipped.
        if not bivector_mode and idx > 0:
            existing = self.truths[:idx]
            s_norm = F.normalize(stored_vec.unsqueeze(0), dim=-1)
            e_norm = F.normalize(existing, dim=-1)
            sims = (s_norm @ e_norm.T).squeeze(0)
            worst = sims.min()
            if worst.item() < -0.7:
                j = sims.argmin().item()
                warnings.warn(
                    f"TruthLayer: new truth [{idx}] contradicts existing "
                    f"truth [{j}] (cosine similarity {worst.item():.3f})",
                    stacklevel=2,
                )

        return idx

    @torch.no_grad()
    def record_batch(self, activations: torch.Tensor, trust: torch.Tensor,
                     degree: float, basis=None) -> None:
        """Stage a batch of activations into the per-tick pending buffer.

        Unlike ``record()``, this method makes no per-cell storage decision
        inside the compute brick. Every entry in ``activations`` is staged
        with its associated ``trust`` value; ``compact()`` (called outside
        the brick) drops entries with trust below the threshold and
        promotes survivors to the persistent ``truths`` buffer.

        Args:
            activations: ``[N, nDim]`` activations to stage.
            trust: ``[N]`` per-entry trust scores in ``[0, 1]``. The caller
                computes these (typically magnitude-based, masked by
                ``valid_mask`` and scaled by ``accumulateTruth``).
            degree: scalar in ``[-1, 1]`` -- the DegreeOfTruth applied to
                every staged activation. Sign flips (negative degree)
                use the same bivector-aware path as ``record()``.
            basis: optional Basis with ``negation(monotonic=True)``. When
                provided and monotonic with even ``nDim``, enables the
                bivector-storage path for ``degree < 0``.

        No host syncs. The pending buffer's leading dim is bounded by
        ``4 * max_truths``; overflows are truncated (host-side bounds
        check on a Python int counter).
        """
        if activations.numel() == 0:
            return
        n = activations.shape[0]
        degree = max(-1.0, min(1.0, degree))
        # Pin to the buffer's device — same coercion as record() so
        # cross-device callers don't trip the in-place buffer write below.
        vec = activations.detach().to(self._pending_truths.device)

        bivector_mode = (
            basis is not None
            and getattr(basis, 'monotonic', False)
            and vec.shape[-1] % 2 == 0
        )

        if bivector_mode and degree < 0:
            vec = Ops._negation_kernel(vec, monotonic=True)
            stored = vec * abs(degree)
        else:
            stored = vec * degree

        # Bounds-check via Python int (no GPU sync).
        cap = self._pending_truths.shape[0] - self._pending_count
        n_take = min(n, cap)
        if n_take <= 0:
            return
        end = self._pending_count + n_take
        self._pending_truths[self._pending_count:end] = stored[:n_take]
        self._pending_trust[self._pending_count:end] = trust[:n_take].detach() \
            .to(self._pending_trust.device)
        self._pending_count = end

    @torch.no_grad()
    def compact(self, min_trust: float = 0.5) -> int:
        """Promote pending entries with trust >= ``min_trust`` to the
        persistent store. Resets the pending buffer.

        Intended call site: outer doc-streaming loop, AFTER the compute
        brick (forward + backward + optimizer.step). Lives outside the
        brick so the one host sync (``mask.sum().item()``) does not
        block CUDA-graph capture of the brick body.

        Returns the number of entries actually promoted.
        """
        if self._pending_count == 0:
            return 0
        pending = self._pending_truths[:self._pending_count]
        trust = self._pending_trust[:self._pending_count]
        mask = trust >= min_trust
        n_keep_t = mask.sum()
        n_keep = int(n_keep_t.item())  # one sync, outside the brick
        self._pending_count = 0
        if n_keep == 0:
            return 0
        survivors = pending[mask]
        cur = int(self.count.item())
        cap = self.max_truths - cur
        n_actual = min(n_keep, cap)
        if n_actual <= 0:
            return 0
        self.truths[cur:cur + n_actual] = survivors[:n_actual]
        self.count += n_actual
        self._nonempty = True  # n_actual >= 1 here (guarded above)
        return n_actual


    def query(self, activation: torch.Tensor, threshold: float = 0.9
              ) -> Optional[Tuple[int, float]]:
        """Find the closest stored truth to ``activation``.

        Compares against the *direction* of stored truths (normalised).
        The sign of the cosine similarity tells you consonance (+) vs
        dissonance (-) with the stored truth.

        Args:
            activation: (nDim,) or (B, nDim) query vector.
            threshold: minimum absolute cosine similarity to count as a match.

        Returns:
            (index, similarity) of the best match, or None if no match
            exceeds the threshold.  similarity > 0 means consonant with
            a positive truth or dissonant with a negative truth.
        """
        n = self.count.item()
        if n == 0:
            return None

        stored = self.truths[:n]                                 # (n, D)
        q = activation.detach()
        if q.ndim == 1:
            q = q.unsqueeze(0)                                   # (1, D)

        q_norm = torch.nn.functional.normalize(q, dim=-1)
        s_norm = torch.nn.functional.normalize(stored, dim=-1)
        sims = (q_norm @ s_norm.T).squeeze(0)                   # (n,)

        best_abs, best_idx = sims.abs().max(dim=0)
        if best_abs.item() < threshold:
            return None
        idx = best_idx.item()
        return (idx, sims[idx].item())

    # -- Truth Field ---------------------------------------------------

    def field(self, concepts: torch.Tensor, eps: float = 1e-8
              ) -> torch.Tensor:
        """Project stored truths into a scalar truth field over concepts.

        Because the DoT is baked into the stored vectors, the field
        naturally produces attractors for positive truths and dispersers
        for negative truths:

            field(c) = (1/n) Sigma_i  sim(c, truth_i)

        where ``truth_i = activation_i * degree_i``.

        Args:
            concepts: (B, N, D) concept vectors in ConceptualSpace.

        Returns:
            field: (B, N) scalar field in [-1, 1].
        """
        n = self.count.item()
        if n == 0:
            return torch.zeros(
                concepts.shape[0], concepts.shape[1],
                device=concepts.device, dtype=concepts.dtype,
            )

        stored = self.truths[:n]                                 # (n, D)

        c_norm = torch.nn.functional.normalize(
            concepts, dim=-1, eps=eps)                            # (B, N, D)
        # Don't normalize stored -- the magnitude carries the DoT
        # Use dot product: stronger DoT -> stronger field influence
        dots = torch.einsum('bnd,md->bnm', c_norm, stored)      # (B, N, n)
        truth_field = dots.sum(dim=-1) / (n + eps)               # (B, N)

        return truth_field.clamp(-1.0, 1.0)

    # -- Luminosity -----------------------------------------------------

    def _positive_poles(self, v: torch.Tensor) -> torch.Tensor:
        """Extract positive poles from a **paired-index** storage vector.

        Assumes ``v`` is laid out as repeated pairs
        ``[pos_0, neg_0, pos_1, neg_1, ...]`` over ``v.shape[-1] == 2K``.
        Even indices are positive poles; odd indices are negative poles.

        .. warning::
            This layout is **not** the current SymbolicSpace codebook layout,
            where each row is ``[pos_pole, neg_pole, where..., when...]``
            with a single leading bivector plus trailing positional data.
            Before calling ``luminosity``/``darkness``/
            ``tetralemma_balance_penalty`` with a symbol activation from the
            new codebook, **slice the leading bivector** via
            ``v[..., :2]`` at the call site. See basicmodel/doc/Spaces.md
            "Codebook shape" note for the mapping.
        """
        return v[..., 0::2]

    def _negative_poles(self, v: torch.Tensor) -> torch.Tensor:
        """Extract negative poles from a paired-index storage vector.

        See :meth:`_positive_poles` for the layout caveat regarding the
        new leading-bivector SymbolicSpace codebook.
        """
        return v[..., 1::2]

    # ``luminosity`` / ``area`` are now on the Mereology mixin
    # (bin/Mereology.py) -- callable as ``model.Luminosity()`` /
    # ``model.Area()`` on any concrete model.  TruthLayer no longer
    # carries its own luminosity surface; consumers either go through
    # the model or call ``Ops.hyperrectangle_volume`` directly.

    def darkness(self, pi_layer=None) -> torch.Tensor:
        """Diagnostic: ||relu(min(negative_poles(truths)))||.

        Mirror of ``luminosity`` on the negative-pole half of bivector
        storage. Elevated darkness means many truths have co-active
        negative poles (i.e. a shared "not-X" conjunction). Returns 0
        for non-bivector storage (odd last dim).
        """
        n = self.count.item()
        if n == 0:
            return torch.tensor(0.0, device=self.truths.device)

        stored = self.truths[:n]
        if pi_layer is not None:
            stored = pi_layer.reverse(stored)

        if stored.shape[-1] % 2 != 0:
            return torch.tensor(0.0, device=self.truths.device)

        neg = self._negative_poles(stored)
        conjunction = neg.min(dim=0).values
        return torch.relu(conjunction).norm()

    def tetralemma_balance_penalty(self, bivector_activation: torch.Tensor,
                               allow_excluded_middle: int = 1,
                               allow_contradiction: int = 0,
                               neither_threshold: float = 0.1) -> torch.Tensor:
        """Penalize forbidden corners of the quaternary truth lattice (tetralemma).

        ``bivector_activation`` is expected to be a `[..., 2K]` tensor in
        **paired-index** layout, where each consecutive pair ``(t+, t-)``
        encodes one concept's tetralemma corner:

            T = (1, 0)   F = (0, 1)
            N = (0, 0)   B = (1, 1)

        .. warning::
            The current SymbolicSpace codebook layout is
            ``[pos_pole, neg_pole, where..., when...]`` — a **single**
            leading bivector plus trailing positional template data, not
            repeated pairs. If passing a symbol activation from the new
            codebook, slice the leading bivector first
            (``sym_act[..., :2]``) to avoid applying tetralemma corner
            policy to positional-template dims. See
            basicmodel/doc/Spaces.md "Codebook shape" and
            basicmodel/doc/BuddhistParallels.md.

        Flags control which corners are penalized:
            allow_excluded_middle == -1  =>  penalize N (force classical LEM)
            allow_excluded_middle ==  1  =>  permit N (Kleene default)
            allow_contradiction   ==  0  =>  penalize B (non-contradiction)
            allow_contradiction   ==  1  =>  permit B (paraconsistent / LP)

        ``neither_threshold`` is the activation level below which a concept
        is considered "dark" (no commitment). Scaled linearly so the penalty
        is well-behaved near the threshold.

        Returns a scalar tensor. Odd last dim -> returns 0 (non-bivector).
        """
        act = bivector_activation
        zero = act.new_tensor(0.0) if torch.is_tensor(act) else torch.tensor(0.0)
        if act is None or not torch.is_tensor(act):
            return zero
        if act.shape[-1] % 2 != 0:
            return zero

        pair = act.reshape(*act.shape[:-1], act.shape[-1] // 2, 2)
        t_pos = pair[..., 0]
        t_neg = pair[..., 1]

        total = zero
        if int(allow_excluded_middle) == -1:
            hottest = torch.maximum(t_pos, t_neg)
            total = total + torch.relu(
                act.new_tensor(float(neither_threshold)) - hottest
            ).mean()
        if int(allow_contradiction) == 0:
            total = total + (t_pos * t_neg).mean()
        return total

    # -- Mereological fusion -------------------------------------------

    def fusion(self, indices: torch.Tensor = None) -> torch.Tensor:
        """Mereological fusion (least upper bound) of stored truths.

        Returns the elementwise ``max`` across the stored truth set
        (or the rows selected by ``indices``) — the axis-aligned bounding
        hyperrectangle in bivector space. Every individual truth is
        componentwise dominated by the fusion: ``t_i <= fusion`` per dim.

        Under paired-index storage ``[2K]``, pairs ``(2k, 2k+1)`` encode
        concept k's ``(pos, neg)`` poles, so fusion tightens each pair to
        its per-truth maximum — the hyperrectangle's "top right" corner
        in the 2D ``(pos, neg)`` plane of every concept.

        Under the current SymbolicSpace codebook layout
        ``[pos_pole, neg_pole, where..., when...]``, fusion on raw stored
        rows also maxes the positional trailers; slice ``[..., :2]``
        at the call site for a pure bivector fusion.

        Returns a ``(D,)`` tensor in the same layout as stored truths, or
        a zero vector when no truths are stored.
        """
        n = int(self.count.item())
        if n == 0:
            return torch.zeros(self.truths.shape[-1], device=self.truths.device,
                               dtype=self.truths.dtype)
        stored = self.truths[:n] if indices is None else self.truths[indices]
        return stored.max(dim=0).values

    def truth_conjunction(self, basis, pi_layer=None):
        """Conjunction of all stored truths via bitonic intersection.

        Folds stored truths with ``Basis.conjunction()`` (sign-aware
        element-wise min), optionally projecting from symbolic to
        conceptual space first via ``pi_layer.reverse()``.

        Returns:
            (D,) conjunction vector, or None if no truths stored.
        """
        n = self.count.item()
        if n == 0:
            return None

        stored = self.truths[:n]                          # (n, symbol_dim)

        # Project to conceptual space if needed
        if pi_layer is not None:
            stored = pi_layer.reverse(stored)              # (n, concept_dim)

        # Fold via bitonic conjunction
        conj = stored[0]
        for i in range(1, n):
            conj = Ops._conjunction_kernel(conj, stored[i])

        return conj

    # -- Universality (Golden Rule) ------------------------------------

    def universality(self, subject, verb, obj, lifting_layer, symbolic_space,
                     model=None):
        """Golden rule: measure luminosity change from K(X,Y) + K(Y,X).

        1. Compute luminosity_before (baseline truth brightness).
        2. Apply SVO: lift(S, V, O) -> project -> temporarily store.
        3. Apply OVS: lift(O, V, S) -> project -> temporarily store.
        4. Compute luminosity_after.
        5. Return luminosity_after - luminosity_before.

        Positive = action preserves/increases illumination (kind).
        Negative = action diminishes illumination (unkind).

        Args:
            subject: [B, N, D] subject concepts.
            verb: [B, N, D] verb concepts.
            obj: [B, N, D] object concepts.
            lifting_layer: LiftingLayer for verb application.
            symbolic_space: SymbolicSpace for projection.
            model: ``Mereology``-mixed model providing
                ``Luminosity(truth_layer=...)``.  Required for the
                measure; when omitted (legacy callers) the method
                returns ``0.0`` so universality drops out of the
                training signal cleanly.

        Returns:
            Scalar universality score (Python float).
        """
        if model is None or not hasattr(model, 'Luminosity'):
            return 0.0
        ss = symbolic_space
        luminosity_before = float(model.Luminosity(truth_layer=self))

        # K(X, Y): original action SVO
        result_svo = lifting_layer.forward_transitive_svo(
            subject, verb, obj, ss)

        # K(Y, X): dual action OVS
        result_ovs = lifting_layer.forward_transitive_svo(
            obj, verb, subject, ss)

        # Project results to symbol space via SymbolicSpace.forward()
        ss.subspace.set_event(result_svo)
        svo_syms = ss.forward(ss.subspace).materialize()  # [B, N, symbol_dim]
        ss.subspace.set_event(result_ovs)
        ovs_syms = ss.forward(ss.subspace).materialize()  # [B, N, symbol_dim]

        # Temporarily extend truth store (average over batch and vectors)
        saved_count = self.count.item()
        basis = getattr(getattr(ss, 'subspace', None), 'basis', None)
        self.record(svo_syms.mean(dim=(0, 1)).detach(), degree=1.0, basis=basis)
        self.record(ovs_syms.mean(dim=(0, 1)).detach(), degree=1.0, basis=basis)

        luminosity_after = float(model.Luminosity(truth_layer=self))

        # Restore truth store
        self.count.fill_(saved_count)
        self._nonempty = saved_count > 0  # saved_count is host-side
        self.truths[saved_count:] = 0

        return luminosity_after - luminosity_before

    # -- Implication Derivation ----------------------------------------

    @torch.no_grad()
    def derive(self, part_fn, threshold: float = 0.7,
               attenuation: float = 0.8, basis=None) -> int:
        """Derive implied truths via pairwise mereological inference.

        For each pair of stored truths, checks if one is contained in
        the other (via ``part_fn``).  When the parthood score exceeds
        *threshold*, a new implied truth is recorded with attenuated DoT.

        Args:
            part_fn: callable(left, right) -> parthood score tensor.
                     Typically a ``PartLayer().compose`` from
                     ``GRAMMAR_LAYER_CLASSES['part']`` (post-2026-05-01
                     refactor; was ``SyntacticLayer.partForward``).
            threshold: minimum parthood score to derive an implication.
            attenuation: DoT scaling to prevent runaway chains.
            basis: optional Basis forwarded to record() for bivector storage.

        Returns:
            Number of new derived truths.
        """
        n = self.count.item()
        if n < 2:
            return 0

        stored = self.truths[:n]
        norms = stored.norm(dim=-1)
        derived = 0

        for i in range(n):
            if norms[i] < 1e-6:
                continue
            for j in range(n):
                if i == j or norms[j] < 1e-6:
                    continue
                if self.count >= self.max_truths:
                    return derived

                score = part_fn(
                    stored[i].unsqueeze(0), stored[j].unsqueeze(0), None)
                if isinstance(score, torch.Tensor):
                    score = score.mean().item()
                if score > threshold:
                    # Direction of truth_j, degree attenuated by score
                    direction = F.normalize(stored[j].unsqueeze(0), dim=-1).squeeze(0)
                    # Sign of truth_i's DoT (positive or negative truth)
                    sign_i = 1.0 if stored[i].mean().item() >= 0 else -1.0
                    degree = attenuation * score * sign_i
                    self.record(direction, degree, basis=basis)
                    derived += 1

        return derived

    # -- Consistency Scoring -------------------------------------------

    @torch.no_grad()
    def consistency(self, sim_threshold: float = 0.7,
                    pair_threshold: float = 0.3,
                    basis=None,
                    part_threshold: float = 0.3,
                    return_report: bool = False):
        """Detect logical contradictions within stored truths.

        Default (``return_report=False``) returns a scalar score for
        back-compat. With ``return_report=True`` returns
        ``(score, contradictions)`` where ``contradictions`` is a list
        of ``(idx_i, idx_j, description)`` tuples describing cross-truth
        part-of relations with opposite sign polarity.

        Bivector path (even last dim):
        - Scalar (default): 1 - (fraction of (truth, concept) slots that
          are BOTH-hot within a single truth). Anti-parallel cosine is
          NOT a contradiction; under bivector encoding A and not(A)
          land on orthogonal paired indices.
        - Report path: ignores within-truth BOTH (valid catuṣkoṭi) and
          instead emits one entry per cross-truth pair (i, j) where
          ``max(basis.part(i, j), basis.part(j, i)) >= part_threshold``
          and the two truths have opposite sign polarity (one mostly
          positive-pole, the other mostly negative-pole). When
          ``basis`` is omitted, a structural proxy via positive-pole
          overlap ratio is used so the caller still gets a report.

        Legacy path (odd last dim): two truths pointing in opposite
        directions (anti-parallel cosine) represent a contradiction;
        the report path simply emits those anti-parallel pairs.

        Args:
            sim_threshold: legacy cosine threshold for anti-parallel
                truths to count as conflicting.
            pair_threshold: bivector threshold above which both poles
                of a concept are considered co-active (BOTH).
            basis: optional Basis with a ``part(a, b, monotonic=True)``
                method. Used only by the report path; omit for the
                scalar default.
            part_threshold: minimum part-of score to count (i, j) as a
                containment relation under the report path.
            return_report: when True, return ``(score, contradictions)``.

        Returns:
            Scalar in [0, 1] where 1 = fully consistent, OR a tuple
            ``(score, contradictions)`` when ``return_report=True``.
        """
        n = self.count.item()
        if n < 1:
            score = torch.tensor(1.0, device=self.truths.device)
            return (score, []) if return_report else score

        stored = self.truths[:n]
        bivector = stored.shape[-1] % 2 == 0

        if return_report:
            contradictions = self._detect_contradictions(
                stored, basis, part_threshold, sim_threshold
            )
            n_pairs = max(1, (n * (n - 1)) // 2)
            score = torch.tensor(
                float(max(0.0, 1.0 - len(contradictions) / n_pairs)),
                device=self.truths.device,
            )
            return score, contradictions

        # Scalar back-compat path.
        if bivector:
            pos = self._positive_poles(stored)
            neg = self._negative_poles(stored)
            both_hot = (pos > pair_threshold) & (neg > pair_threshold)
            total = both_hot.numel()
            if total == 0:
                return torch.tensor(1.0, device=self.truths.device)
            frac = both_hot.float().mean()
            return (1.0 - frac).clamp(0.0, 1.0).to(self.truths.device)

        # Legacy pairwise anti-parallel check.
        if n < 2:
            return torch.tensor(1.0, device=self.truths.device)

        norms = stored.norm(dim=-1)
        valid = norms > 1e-6
        if valid.sum() < 2:
            return torch.tensor(1.0, device=self.truths.device)

        directions = F.normalize(stored[valid], dim=-1)
        sim_matrix = directions @ directions.T
        m = directions.shape[0]

        n_conflicts = 0
        n_pairs = 0
        for i in range(m):
            for j in range(i + 1, m):
                n_pairs += 1
                if sim_matrix[i, j] < -sim_threshold:
                    n_conflicts += 1

        if n_pairs == 0:
            return torch.tensor(1.0, device=self.truths.device)
        return torch.tensor(1.0 - n_conflicts / n_pairs,
                            device=self.truths.device)

    def _detect_contradictions(self, stored, basis, part_threshold,
                               sim_threshold):
        """Return List[(i, j, description)] of cross-truth contradictions.

        Bivector: pair (i, j) is a contradiction when one is a
        part-of the other (``max(part(i, j), part(j, i)) >=
        part_threshold``) and the two have opposite sign polarity.
        Legacy (odd last dim): pair is a contradiction when cosine
        similarity is below ``-sim_threshold``.
        """
        n = stored.shape[0]
        contradictions = []
        if n < 2:
            return contradictions

        bivector = stored.shape[-1] % 2 == 0

        if bivector:
            for i in range(n):
                for j in range(i + 1, n):
                    p_ij = self._part_score(stored[i], stored[j], basis)
                    p_ji = self._part_score(stored[j], stored[i], basis)
                    score = max(p_ij, p_ji)
                    if score < part_threshold:
                        continue
                    if self._sign(stored[i]) == self._sign(stored[j]):
                        continue
                    contradictions.append(
                        (i, j,
                         f"part-of (score={score:.2f}) with opposite "
                         f"sign polarity")
                    )
            return contradictions

        # Legacy: anti-parallel pairs.
        norms = stored.norm(dim=-1)
        for i in range(n):
            if norms[i] < 1e-6:
                continue
            ni = F.normalize(stored[i].unsqueeze(0), dim=-1).squeeze(0)
            for j in range(i + 1, n):
                if norms[j] < 1e-6:
                    continue
                nj = F.normalize(stored[j].unsqueeze(0), dim=-1).squeeze(0)
                sim = float((ni * nj).sum().item())
                if sim < -sim_threshold:
                    contradictions.append(
                        (i, j, f"anti-parallel (cos={sim:.2f})")
                    )
        return contradictions

    def _part_score(self, a, b, basis):
        """Scalar part-of score for (a, b). Uses basis when provided,
        otherwise a positive-pole overlap proxy on bivector storage."""
        if basis is not None and hasattr(basis, "part"):
            score = basis.part(a, b, monotonic=True, scalar=True)
            if torch.is_tensor(score):
                return float(score.mean().item())
            return float(score)
        # Structural proxy: how much of b's positive-pole energy is
        # covered by a's positive-pole energy.
        if a.shape[-1] % 2 == 0:
            a_pos = self._positive_poles(a)
            b_pos = self._positive_poles(b)
            denom = float(b_pos.abs().sum().item()) + 1e-8
            return float(torch.minimum(a_pos, b_pos).sum().item()) / denom
        denom = float(b.abs().sum().item()) + 1e-8
        return float(torch.minimum(a, b).clamp_min(0).sum().item()) / denom

    def _sign(self, v):
        """Return +1 if v is predominantly positive-pole, -1 otherwise.

        Bivector: compare summed positive-pole mass to summed
        negative-pole mass. Legacy: use the sign of the mean activation.
        """
        if v.shape[-1] % 2 == 0:
            pos = self._positive_poles(v).abs().sum().item()
            neg = self._negative_poles(v).abs().sum().item()
            return 1 if pos >= neg else -1
        return 1 if v.mean().item() >= 0 else -1

    def suggest_clarifications(self, basis=None,
                               part_threshold: float = 0.3) -> list:
        """Generate one user-facing message per detected contradiction.

        The template is fixed (for test stability and translation
        ease): ::

            "'{source_i}' (trust={trust_i}) and '{source_j}' "
            "(trust={trust_j}) appear to contradict — please revise "
            "to enable more rational thought."

        Missing source falls back to ``"(truth #i)"`` and missing trust
        to ``"unknown"``.
        """
        _, contradictions = self.consistency(
            basis=basis, part_threshold=part_threshold, return_report=True
        )
        messages = []
        sources = getattr(self, "_sources", []) or []
        trusts = getattr(self, "_trusts", []) or []
        for i, j, _desc in contradictions:
            src_i = sources[i] if i < len(sources) and sources[i] else None
            src_j = sources[j] if j < len(sources) and sources[j] else None
            trust_i = trusts[i] if i < len(trusts) and trusts[i] is not None else None
            trust_j = trusts[j] if j < len(trusts) and trusts[j] is not None else None
            label_i = f"'{src_i}'" if src_i is not None else f"(truth #{i})"
            label_j = f"'{src_j}'" if src_j is not None else f"(truth #{j})"
            t_i = f"{trust_i:g}" if trust_i is not None else "unknown"
            t_j = f"{trust_j:g}" if trust_j is not None else "unknown"
            messages.append(
                f"{label_i} (trust={t_i}) and {label_j} (trust={t_j}) "
                f"appear to contradict — please revise to enable more "
                f"rational thought."
            )
        return messages

    # -- Client-facing paraconsistent assessment ---------------------

    @torch.no_grad()
    def assess(self, basis=None):
        """Terminal paraconsistent read of the truth accumulator.

        Returns per-region ``support`` / ``conflict`` / ``ignorance``
        in ``[0, 1]`` (the only legitimate terminal bivector surface --
        every inter-component wire is a single signed scalar; this
        client-facing assessment recovers the catuskoti corners from
        the accumulated TruthSet)::

            support   = aP * (1 - aN)             net affirmation
            conflict  = aP * R_N + aN * R_P       the set both affirms
                                                  AND denies (contested)
            ignorance = (1 - max(aP, aN))
                        * (1 - max(R_P, R_N))     the set is silent

        ``aP`` / ``aN`` are the affirming / denying poles of the
        TruthSet's mean signed Degree-of-Truth; ``R_P`` / ``R_N`` are
        its strongest affirming / denying evidence. Keeping ``conflict``
        (a TruthSet that splits on a proposition) distinct from
        ``ignorance`` (a TruthSet silent on it) is exactly the
        degeneracy a scalar ``aP - aN`` collapse loses -- the reason
        the accumulator stays the paraconsistent surface while the
        wire is scalar. ``basis`` is accepted for signature parity with
        ``consistency`` / ``suggest_clarifications``; unused here.
        """
        n = self.count.item()
        if n == 0:
            return {"support": 0.0, "conflict": 0.0, "ignorance": 1.0}
        stored = self.truths[:n]
        # Per-truth signed Degree-of-Truth (mean activation; sign = the
        # belief direction baked in by record()).
        s = stored.mean(dim=-1)                          # [n]
        pos = torch.relu(s)                              # affirming
        neg = torch.relu(-s)                             # denying
        aP = float(pos.mean().item())                    # net affirmation
        aN = float(neg.mean().item())                    # net denial
        R_P = float(pos.max().item())                    # strongest affirm
        R_N = float(neg.max().item())                    # strongest deny
        support = aP * (1.0 - aN)
        conflict = aP * R_N + aN * R_P
        ignorance = (1.0 - max(aP, aN)) * (1.0 - max(R_P, R_N))
        clamp = lambda x: max(0.0, min(1.0, float(x)))
        return {
            "support": clamp(support),
            "conflict": clamp(conflict),
            "ignorance": clamp(ignorance),
        }

    # -- TruthLoss: Union Norm Reduction -----------------------------

    def falsity_penalty(self, symbol_states, basis):
        """Compute additive truth loss via union norm reduction.

        For each proposition in ``symbol_states``, measure how much the
        TruthSet union norm drops when the proposition is included.
        Contradiction cancels dimensions (via ``Basis.disjunction``'s
        bitonic same-sign logic), reducing the norm -> positive penalty.

        Agreeing propositions preserve or extend the union -> no penalty.
        Unknown propositions (zero dims) pass through -> no penalty.
        DoT weighting is implicit: high-DoT truths contribute more energy
        to the union, so contradicting them causes a larger norm drop.

        Both sides of the disjunction live in symbol space by
        construction: stored truths are recorded from
        ``SymbolicSpace.forwardEnd``, and ``symbol_states`` should be
        the post-pi activations cached during the Sigma-Pi loop (the
        model's ``self.symbol_states[-1]``).  Using symbols rather
        than pre-pi concepts keeps both operands in the basis's
        native space.

        Args:
            symbol_states: (B, N, D) symbolic activations from the
                forward pass (post-pi, post-l1_proximal -- the
                committed beliefs).
            basis: Basis instance with disjunction() method.

        Returns:
            Scalar penalty >= 0 (differentiable).
        """
        n = self.count.item()
        if n == 0:
            return symbol_states.new_tensor(0.0)

        stored = self.truths[:n]  # (n, D)

        # Fold stored truths into union vector via successive disjunction
        truth_union = stored[0]
        for i in range(1, n):
            truth_union = Ops._disjunction_kernel(truth_union, stored[i])
        union_norm = truth_union.norm()

        # For each proposition, compute norm reduction
        B, N, D = symbol_states.shape
        propositions = symbol_states.reshape(-1, D)  # (B*N, D)

        penalties = []
        for p in range(propositions.shape[0]):
            extended = Ops._disjunction_kernel(truth_union, propositions[p])
            reduction = union_norm - extended.norm()
            penalties.append(torch.relu(reduction))

        return torch.stack(penalties).mean()

    # -- Maintenance ---------------------------------------------------

    @torch.no_grad()
    def prune(self, min_norm: float = 1e-6):
        """Remove near-zero entries (truths with DoT ~= 0).

        Compacts the store in-place.
        """
        n = self.count.item()
        if n == 0:
            return
        norms = self.truths[:n].norm(dim=-1)
        keep = norms > min_norm
        kept = self.truths[:n][keep]
        new_n = kept.shape[0]
        self.truths[:new_n] = kept
        self.truths[new_n:] = 0
        self.count.fill_(new_n)
        self._nonempty = new_n > 0  # new_n is a host-side tensor shape

    @torch.no_grad()
    def orthogonalize(self, sim_threshold: float = 0.85):
        """Remove redundant truths using luminosity as the quality measure.

        For each pair with |cosine similarity| > sim_threshold, the truth
        contributing less to luminosity is removed.  Luminosity measures
        the norm of the positive conjunction across all truths -- dropping
        a redundant entry barely changes it, while dropping a unique entry
        reduces it.

        Args:
            sim_threshold: similarity above which two truths are considered
                near-duplicates.  Default 0.85.

        Returns:
            Number of truths removed.
        """
        n = self.count.item()
        if n < 2:
            return 0

        stored = self.truths[:n]
        norms = stored.norm(dim=-1)
        valid = norms > 1e-6
        if valid.sum() < 2:
            return 0

        directions = F.normalize(stored, dim=-1)
        sim_matrix = directions @ directions.T           # (n, n)

        # Find redundant pairs (|sim| > threshold, excluding diagonal)
        remove = set()
        for i in range(n):
            if i in remove or not valid[i]:
                continue
            for j in range(i + 1, n):
                if j in remove or not valid[j]:
                    continue
                if sim_matrix[i, j].abs().item() > sim_threshold:
                    # Drop whichever contributes less to luminosity
                    lum_without_i = self._luminosity_without(i)
                    lum_without_j = self._luminosity_without(j)
                    # Keep the one whose removal hurts luminosity more
                    if lum_without_i >= lum_without_j:
                        remove.add(i)
                    else:
                        remove.add(j)

        if not remove:
            return 0

        # Compact: keep non-removed entries
        keep_mask = torch.ones(n, dtype=torch.bool, device=stored.device)
        for idx in remove:
            keep_mask[idx] = False
        kept = stored[keep_mask]
        new_n = kept.shape[0]
        self.truths[:new_n] = kept
        self.truths[new_n:] = 0
        self.count.fill_(new_n)
        self._nonempty = new_n > 0  # new_n is a host-side tensor shape
        return len(remove)

    def _luminosity_without(self, exclude_idx: int) -> float:
        """Luminosity with one truth excluded (for orthogonalization)."""
        n = self.count.item()
        indices = [i for i in range(n) if i != exclude_idx]
        if not indices:
            return 0.0
        subset = self.truths[indices]
        if subset.shape[-1] % 2 == 0:
            subset = self._positive_poles(subset)
        conjunction = subset.min(dim=0).values
        return torch.relu(conjunction).norm().item()

    # -- Consolidated truth computation (Phase 1: bivector retirement) --
    # These four methods own the truth-store computation that previously
    # lived on the model (Mereology._luminosity_truth_fold,
    # Models.isConsistent/ground/extrapolate). The bivector poles stay
    # internal to this accumulator; callers delegate here.

    def luminosity(self, sym=None) -> float:
        """Cumulative-vs-rest luminosity fold over stored truths.

        Consolidated from ``Mereology._luminosity_truth_fold``. Decode
        each stored truth to concept-space via ``sym.decode_to_concept``
        then fold pairwise; result in ``[-1, 1]``. The bivector poles
        (``[..., :2]``) are internal to this accumulator.
        """
        try:
            n = int(self.count.item())
        except Exception:
            return 0.0
        if n == 0:
            return 0.0
        if sym is None:
            return 0.0
        try:
            stored = self.truths[:n]
        except Exception:
            return 0.0

        decoded = []
        decoder = getattr(sym, 'decode_to_concept', None)
        for i in range(n):
            row = stored[i:i+1]
            if decoder is not None:
                try:
                    row_c = decoder(row)
                except Exception:
                    row_c = row
            else:
                row_c = row
            if not torch.is_tensor(row_c) or row_c.shape[-1] < 2:
                continue
            decoded.append(row_c)
        if not decoded:
            return 0.0
        if len(decoded) == 1:
            box = decoded[0][..., :2]
            box = box.reshape(*box.shape[:-1], 1, 2) if box.dim() < 3 else box.unsqueeze(-2)
            vol = float(Ops.hyperrectangle_volume(box).mean().item())
            return max(-1.0, min(1.0, vol))

        def _box(t):
            b = t[..., :2]
            return b.reshape(*b.shape[:-1], 1, 2) if b.dim() < 3 else b.unsqueeze(-2)

        def _dot(t):
            return float((t[..., 0] - t[..., 1]).mean().item())

        running = decoded[0]
        running_lum = float(Ops.hyperrectangle_volume(_box(running)).mean().item())
        running_lum = max(-1.0, min(1.0, running_lum))

        for i in range(1, len(decoded)):
            t = decoded[i]
            v_run = float(Ops.hyperrectangle_volume(_box(running)).mean().item())
            v_new = float(Ops.hyperrectangle_volume(_box(t)).mean().item())
            shared = float(Ops.hyperrectangle_overlap_volume(
                _box(running), _box(t)).mean().item())
            disagree = abs(_dot(running) - _dot(t))
            pair_lum = (v_run + v_new) - shared * disagree
            pair_lum = max(-1.0, min(1.0, pair_lum))
            running_lum = pair_lum
            try:
                running = Ops.union(running, t, monotonic=False)
            except Exception:
                running = torch.where(running.abs() >= t.abs(), running, t)

        return running_lum

    def isConsistent(self):
        """Fold stored truths via ``Ops.disjunction``; consistency summary.

        Consolidated from ``Models.BaseModel.isConsistent``. Conflicting
        +/- assertions cancel dimensions. Returns
        ``dict(consistent, score, sites, union_vector)``.
        """
        n = self.count.item()
        if n == 0:
            return {'consistent': True, 'score': 1.0,
                    'sites': torch.tensor([]), 'union_vector': torch.tensor([])}

        stored = self.truths[:n]

        union = stored[0].clone()
        for i in range(1, n):
            union = Ops.disjunction(union, stored[i])

        score = union.abs().mean().item()
        threshold = 0.1
        weak_dims = (union.abs() < threshold).nonzero(as_tuple=True)[0]
        consistent = len(weak_dims) == 0 or score > 0.5

        return {
            'consistent': consistent,
            'score': score,
            'sites': weak_dims,
            'union_vector': union,
        }

    @torch.no_grad()
    def ground(self, activation, threshold=0.6, model=None):
        """Minimal-subset entailment of a query against stored truths.

        Consolidated from ``Models.BaseModel.ground``. Partition-aware
        filtering uses ``model._partitions`` / ``model._activation_order``
        when a model handle is supplied; otherwise it degrades to a
        plain cosine match. Returns
        ``dict(grounded, basis, trace, confidence)``.
        """
        n = self.count.item()
        if n == 0:
            return {'grounded': False, 'basis': [], 'trace': [], 'confidence': 0.0}

        stored = self.truths[:n]
        partitions = getattr(model, '_partitions', None) if model is not None else None

        query_order = None
        if partitions is not None and len(partitions) > 1:
            query_order = model._activation_order(activation, partitions)

        activation = activation.to(stored.device)

        a_norm = F.normalize(activation.unsqueeze(0), dim=-1)
        s_norm = F.normalize(stored, dim=-1)
        sims = (a_norm @ s_norm.T).squeeze(0)

        if query_order is not None and partitions is not None:
            for i in range(n):
                truth_order = model._activation_order(stored[i], partitions)
                if truth_order != query_order:
                    sims[i] = 0.0

        mask = sims.abs() > threshold
        basis_indices = mask.nonzero(as_tuple=True)[0].tolist()

        if not basis_indices:
            part_cls = GRAMMAR_LAYER_CLASSES.get('part')
            part_inst = part_cls() if part_cls is not None else None
            part_fn = (lambda left, right, _inst=part_inst, **kw:
                       _inst.compose(left, right)) if part_inst is not None else None
            max_depth = 3
            for depth in range(max_depth):
                if part_fn is None:
                    break
                derived = self.derive(part_fn, threshold=threshold)
                if derived == 0:
                    break
                n_new = self.count.item()
                stored_new = self.truths[:n_new]
                s_norm_new = F.normalize(stored_new, dim=-1)
                sims_new = (a_norm @ s_norm_new.T).squeeze(0)
                mask_new = sims_new.abs() > threshold
                basis_indices = mask_new.nonzero(as_tuple=True)[0].tolist()
                if basis_indices:
                    break

        if not basis_indices:
            return {'grounded': False, 'basis': [], 'trace': [], 'confidence': 0.0}

        basis_sims = sims[:self.count.item()][basis_indices] if basis_indices else torch.tensor([0.0])
        basis_norms = stored[:self.count.item()][basis_indices].norm(dim=-1)
        confidence = (basis_sims * basis_norms).sum().item() / max(len(basis_indices), 1)
        confidence = max(-1.0, min(1.0, confidence))

        trace = [{'index': idx, 'similarity': sims[idx].item() if idx < len(sims) else 0.0}
                 for idx in basis_indices]

        return {
            'grounded': True,
            'basis': basis_indices,
            'trace': trace,
            'confidence': confidence,
        }

    @torch.no_grad()
    def extrapolate(self, model=None, seed_indices=None, max_new=64,
                    attenuation=0.8):
        """Generalize ``derive()`` to all two-argument grammar methods.

        Consolidated from ``Models.BaseModel.extrapolate``. Requires a
        model handle for the SymbolicSpace/ConceptualSpace context, the
        luminosity non-decrease gate, partition-aware ordering and the
        basis. Returns ``dict(added, rejected)``.
        """
        n = self.count.item()
        if n < 2:
            return {'added': [], 'rejected': []}

        stored = self.truths[:n]
        partitions = getattr(model, '_partitions', None) if model is not None else None

        two_arg_rules = ['union', 'intersection', 'isEqual', 'part']

        indices = seed_indices if seed_indices is not None else list(range(n))
        added = []
        rejected = []

        ss = getattr(model, 'symbolicSpace', None) if model is not None else None
        rule_kernels = {}
        for rule_name in two_arg_rules:
            cls = GRAMMAR_LAYER_CLASSES.get(rule_name)
            if cls is not None:
                try:
                    rule_kernels[rule_name] = cls()
                except TypeError:
                    pass

        basis = model._get_basis() if model is not None else None

        for i in indices:
            if stored[i].norm() < 1e-6:
                continue
            for j in indices:
                if i == j or stored[j].norm() < 1e-6:
                    continue
                if len(added) >= max_new:
                    return {'added': added, 'rejected': rejected}

                if partitions is not None and len(partitions) > 1:
                    order_i = model._activation_order(stored[i], partitions)
                    order_j = model._activation_order(stored[j], partitions)
                    if order_i != order_j:
                        continue

                for rule_name in two_arg_rules:
                    kernel = rule_kernels.get(rule_name)
                    if kernel is None:
                        continue
                    try:
                        candidate = kernel.compose(
                            stored[i].unsqueeze(0),
                            stored[j].unsqueeze(0))
                    except Exception:
                        continue
                    if candidate is None:
                        continue
                    candidate = candidate.squeeze(0)

                    if candidate.norm() < 1e-6:
                        continue

                    if ss is not None and model is not None:
                        lum_before = model.Luminosity(truth_layer=self)
                    else:
                        lum_before = 0.0
                    saved_count = self.count.item()

                    dot_i = stored[i].norm().item()
                    dot_j = stored[j].norm().item()
                    degree = attenuation * min(dot_i, dot_j)

                    direction = F.normalize(candidate.unsqueeze(0), dim=-1).squeeze(0)
                    self.record(direction, degree, basis=basis)
                    if ss is not None and model is not None:
                        lum_after = model.Luminosity(truth_layer=self)
                    else:
                        lum_after = lum_before

                    delta = float(lum_after) - float(lum_before)

                    if delta >= 0:
                        added.append(self.count.item() - 1)
                    else:
                        self.count.fill_(saved_count)
                        self.truths[saved_count:] = 0
                        rejected.append((i, j, rule_name, delta))

        return {'added': added, 'rejected': rejected}

    def is_empty(self) -> bool:
        """True iff no truths are stored -- sync-free.

        Mirror of ``len(self) == 0`` that reads the host-side
        ``_nonempty`` flag instead of ``count.item()``, so the in-brick
        consumer (``WordSpace.truth_modulated_loss``) does not force a
        GPU->host sync (CUDA-graph-capture contract; see
        doc/BrickHostSyncStatus.md residual F).
        """
        return not self._nonempty

    def clear(self) -> None:
        """Reset the store to empty (count, sources, trusts, flag).

        Single atomic reset so ``_nonempty`` cannot drift from
        ``count`` -- callers must not zero ``count`` directly.
        """
        self.count.zero_()
        self._sources = []
        self._trusts = []
        self._nonempty = False

    def __len__(self):
        """Number of stored elements."""
        return self.count.item()

    def __repr__(self):
        """Debug-friendly representation."""
        return (f"TruthLayer(nDim={self.nDim}, "
                f"truths={self.count.item()}/{self.max_truths})")

    # -- Test ----------------------------------------------------------

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        D = 32
        tl = TruthLayer(D, max_truths=64)
        assert len(tl) == 0

        # Record truths -- DoT is baked into the stored activation
        t1 = torch.randn(D)
        t2 = torch.randn(D)
        idx1 = tl.record(t1, degree=0.9)
        idx2 = tl.record(t2, degree=-0.7)
        assert len(tl) == 2

        # Stored vector = activation * degree
        assert torch.allclose(tl.truths[0], t1 * 0.9, atol=1e-6)
        assert torch.allclose(tl.truths[1], t2 * -0.7, atol=1e-6)

        # Query -- exact match (high similarity)
        result = tl.query(t1, threshold=0.8)
        assert result is not None
        assert result[0] == 0
        assert result[1] > 0  # consonant with positive truth

        # Query -- no match for random vector
        result = tl.query(torch.randn(D), threshold=0.99)
        assert result is None

        # Field projection
        concepts = torch.randn(2, 8, D)
        f = tl.field(concepts)
        assert f.shape == (2, 8)
        assert f.min() >= -1.0 and f.max() <= 1.0

        # Prune near-zero (DoT ~= 0 produces near-zero stored vector)
        tl.record(torch.randn(D), degree=0.0)
        assert len(tl) == 3
        tl.prune(min_norm=1e-6)
        assert len(tl) == 2

        # -- Luminosity --------------------------------------------
        # Luminosity is now a Mereology measure on the model itself
        # (see bin/Mereology.py); the standalone TruthLayer.luminosity
        # surface was removed when SymbolicSpace.luminosity migrated.
        # The new measure is exercised in test/test_mereology.py.

        # -- Consistency -------------------------------------------
        tl3 = TruthLayer(D, max_truths=64)
        # Two consistent truths (same direction, same sign)
        v = torch.randn(D)
        tl3.record(v, degree=0.9)
        tl3.record(v * 1.1, degree=0.8)
        assert tl3.consistency().item() == 1.0, "same-sign truths should be consistent"

        # Add contradictory truth (same direction, opposite sign)
        tl3.record(v, degree=-0.9)
        c = tl3.consistency().item()
        assert c < 1.0, f"contradictory truth should lower consistency: {c}"

        # Empty / single truth -> consistent
        tl4 = TruthLayer(D, max_truths=64)
        assert tl4.consistency().item() == 1.0
        tl4.record(torch.randn(D), degree=0.5)
        assert tl4.consistency().item() == 1.0

        # -- Universality -----------------------------------------
        # PiLayer maps activations [B, N] -> [B, nSym].
        # TruthLayer nDim must match nSym (symbol dimension).
        B, N = 2, 4
        nSym = 6
        tl5 = TruthLayer(nSym, max_truths=64)
        # Store a "positive" moral axiom
        axiom = torch.rand(nSym) * 0.5 + 0.25  # positive vector
        tl5.record(axiom, degree=1.0)

        S = torch.randn(B, N, D)
        V = torch.randn(B, N, D)
        O = torch.randn(B, N, D)

        # Mock symbolic space: expose ``sigma`` attribute (the canonical
        # surface after the C->S swap to SigmaLayer on SymbolicSpace).
        _sigma = SigmaLayer(N, nSym, monotonic=True, invertible=True,
                            nonlinear=True)
        class _MockSS:
            """Test double for SymbolicSpace exposing only a ``sigma`` attribute.

            TruthLayer's record / record_svo path reads
            ``symbolic_space.sigma`` to project SVO vectors into the
            symbolic activation space; this mock supplies a real
            ``SigmaLayer`` for those tests.
            """
            sigma = _sigma
        mock_ss = _MockSS()
        lifting = LiftingLayer(nVerbs=8, nDim=D)

        # Universality requires a Mereology-mixed model for the
        # luminosity measure; without one (this is a unit test that
        # builds a bare TruthLayer) the method returns 0.0 cleanly.
        u_score = tl5.universality(S, V, O, lifting, mock_ss)
        assert u_score == 0.0, (
            "universality without `model=` should return 0.0; "
            f"got {u_score}")
        # Truth store should be untouched on the no-model fast path.
        assert len(tl5) == 1, f"truth store not restored: {len(tl5)}"

        print("TruthLayer tests passed.")

class InterSentenceLayer(Layer):
    """Inter-sentence ARMA(p, q) next-sentence predictor.

    **What it is.** Sentence-level autoregression lives here, not in
    the within-sentence body (which is now IR-only / masked-LM).  Each
    sentence is summarised into a flat ``[B, sentence_dim]`` rep
    ``s_t`` (the body's S-tier root slot pooled per row); the
    predictor consumes the last ``p`` sentence reps plus the last
    ``q`` prediction errors and produces ``s_hat_{t+1}``.

    **Loss.** ``MSE(predictor(s_{t-1..t-p}, e_{t-1..t-q}), s_t)``.
    The residual ``e_t = s_t - s_hat_t`` is pushed into the MA ring
    so the next prediction can correct for systematic bias in the AR
    extrapolation.

    **Buffers (per row).**

    - ``_s_history``: ``[B, p, sentence_dim]`` ring of last ``p``
      sentence reps (most recent at index ``-1``).
    - ``_e_history``: ``[B, q, sentence_dim]`` ring of last ``q``
      residuals.
    - ``_s_count`` / ``_e_count``: ``[B]`` long, current fill levels.

    All buffers are non-persistent (transient state, not checkpoint
    material) and follow ``.to(device)`` through ``register_buffer``.

    **Reset.** ``Reset()`` clears both rings.  ARMA lags carry across
    document boundaries by default — discourse continuity is
    information the predictor wants to use; callers that need a hard
    boundary call ``Reset()`` explicitly.

    **Subclass.** ``Layer`` rather than ``Space``: no SubSpace, no
    forward/reverse tensor-map contract.  Lives inside
    ``WordSpace.layers`` so the Layer ergodic walk reaches the MLP
    predictor's parameters via ``WordSpace.params``.

    Replaces the pre-2026-05-14 contrastive cosine machinery
    (``context_window`` recent buffer, ``centroid_history`` repulsive
    ring, ``lam`` cosine push) and the AttentionLayer-based
    predictive head.  See ``doc/Architecture.md`` §"Sentence-level
    AR (InterSentenceLayer)" for the design rationale.
    """

    name = "Discourse"

    def __init__(self, n_symbols, max_depth, n_dim,
                 p=5, q=2, hidden_dim=None,
                 concept_dim=None, batch=1,
                 # Legacy-named kwargs accepted for back-compat; ignored.
                 context_window=None, centroid_history=None, lam=None):
        """Initialize the ARMA(p, q) predictor.

        ``n_symbols``, ``max_depth``, ``n_dim`` describe the legacy
        ``[S | W]`` snapshot layout the caller used to hand in; under
        ARMA we only need the flat ``sentence_dim`` (defaults to
        ``n_symbols * n_dim``) but keep the constructor signature for
        smooth migration of the WordSpace instantiation site.

        ``p`` = AR lag count (number of past sentence reps consumed
        by the predictor); ``q`` = MA lag count (past residuals).
        ``hidden_dim`` defaults to ``2 * sentence_dim``.
        ``concept_dim`` is the target dim for the optional priming
        cast that lifts ``s_hat_{t+1}`` into C-tier for the chat-
        loop's ``_c_prior`` injection.

        ``context_window`` / ``centroid_history`` / ``lam`` are
        accepted for signature back-compat with the retired
        contrastive constructor and ignored.
        """
        del context_window, centroid_history, lam  # retired knobs
        n_symbols = int(n_symbols)
        n_dim = int(n_dim)
        # Sentence-rep pooling: the per-sentence rep ``s_t`` is the
        # **root S-tier slot** (row 0 of the body's final-stage S-tier
        # event), not the full ``[n_symbols, n_dim]`` flatten.  The
        # root is what the start-symbol reduction wrote -- it carries
        # the sentence-level signal; the other slots are intermediate
        # composition state.  This keeps the predictor's input width
        # bounded at ``(p + q) * n_dim`` instead of
        # ``(p + q) * n_symbols * n_dim`` (which OOMs the allocator on
        # configs where ``n_symbols * n_dim`` is in the hundreds of
        # thousands -- the legacy contrastive snapshot avoided this
        # by never feeding the snapshot through a Linear).
        sentence_dim = n_dim
        super().__init__(sentence_dim, sentence_dim)

        self.n_symbols = n_symbols
        self.max_depth = int(max_depth)
        self.n_dim = n_dim
        self.sentence_dim = sentence_dim
        self.p = int(p)
        self.q = int(q)
        # Cap hidden_dim so the predictor's Linear stays under the
        # allocator's per-tensor budget on every config we ship --
        # 1024 is comfortably bigger than ``sentence_dim`` for every
        # MM_* training config (sentence_dim == n_dim, typically 2-100)
        # and small enough that the MLP fits in a few MB.
        _hidden_cap = 1024
        self.hidden_dim = int(hidden_dim if hidden_dim is not None
                              else min(_hidden_cap, 2 * sentence_dim))
        self.concept_dim = (int(concept_dim)
                            if concept_dim is not None else None)

        self._batch = int(batch)
        # Phase-3-style staging slot: when the compiled step is active,
        # ``stage_prediction()`` runs the (``@torch.compiler.disable``'d)
        # ARMA predictor EAGERLY before the traced forward and parks the
        # ``(pred, conf)`` tuple here.  ``predict()`` then returns the
        # staged tuple without touching ``predict_next`` (the disabled
        # function that forced a graph break) or the ``_s_count.item()``
        # host sync.  ``None`` == not staged (eager path, compute live).
        self._staged_prediction = None
        self.register_buffer(
            "_s_history",
            torch.zeros(self._batch, self.p, self.sentence_dim),
            persistent=False)
        self.register_buffer(
            "_s_count",
            torch.zeros(self._batch, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_e_history",
            torch.zeros(self._batch, max(1, self.q), self.sentence_dim),
            persistent=False)
        self.register_buffer(
            "_e_count",
            torch.zeros(self._batch, dtype=torch.long),
            persistent=False)

        # ``self.layers`` (parent Layer's ergodic-walk list) only holds
        # objects that implement ``set_sigma`` / ``observe_sigma`` etc.
        # The ARMA MLP predictor is an ``nn.Sequential`` (no ergodic
        # interface), so it's NOT added here -- its parameters still
        # register with the host module via plain attribute
        # assignment, so the optimizer sees them.  ``LinearLayer`` is
        # ergodic-compatible and gets added.
        self.layers = []
        in_dim = (self.p + self.q) * self.sentence_dim
        if in_dim > 0 and self.sentence_dim > 0:
            self.predictor = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.sentence_dim),
            )
        else:
            self.predictor = None

        # Optional priming cast for the chat-loop's c_prior injection
        # (Phase 4 of the IR-only refactor).  When concept_dim is set
        # the cast lifts s_hat_{t+1} into C-tier so it can be added
        # as a bias before the body's sigma-pi loop.
        self.cast = None
        if self.concept_dim is not None and self.sentence_dim > 0:
            self.cast = LinearLayer(self.sentence_dim, self.concept_dim)
            self.layers.append(self.cast)

    # -- per-batch resize ---------------------------------------------
    def ensure_batch(self, batch):
        """Resize per-row substrate to a new batch size.  Cascaded
        from ``WordSpace.ensure_batch`` so the body's per-B state is
        sized correctly.  Zeroes the rings.
        """
        batch = int(batch)
        if batch == self._batch:
            return
        device = self._s_history.device
        dtype = self._s_history.dtype
        self._batch = batch
        self._s_history = torch.zeros(
            batch, self.p, self.sentence_dim,
            dtype=dtype, device=device)
        self._s_count = torch.zeros(
            batch, dtype=torch.long, device=device)
        self._e_history = torch.zeros(
            batch, max(1, self.q), self.sentence_dim,
            dtype=dtype, device=device)
        self._e_count = torch.zeros(
            batch, dtype=torch.long, device=device)

    # -- sentence-rep pooling -----------------------------------------
    def _pool_sentence_rep(self, s_tensor):
        """Pool an S-tier event into a per-row ``[B, sentence_dim]``
        sentence rep.

        Accepts:
          * ``[B, N, D]`` (microbatch S-tier event) -- take row 0
            (the root slot the start-symbol reduction wrote into).
          * ``[N, D]`` (B=1 legacy) -- same: take row 0.
          * ``[B, D]`` (already-pooled).
        Right-pads / truncates to ``sentence_dim`` so dim mismatches
        between training configs and the layer's stored width stay
        well-defined.
        """
        if s_tensor is None:
            return None
        x = s_tensor
        if x.ndim == 2:
            if x.shape[-1] == self.sentence_dim:
                return x
            x = x.unsqueeze(0)                # [N, D] -> [1, N, D]
        if x.ndim != 3:
            return None
        # Take the root S-tier slot (row 0) per batch row.
        rooted = x[:, 0, :]                   # [B, D]
        cur = rooted.shape[-1]
        if cur == self.sentence_dim:
            return rooted
        if cur > self.sentence_dim:
            return rooted[:, :self.sentence_dim]
        return F.pad(rooted, (0, self.sentence_dim - cur))

    # -- prediction ---------------------------------------------------
    def _predictor_input(self):
        """Concatenate ``[s_history | e_history]`` per row.

        Returns ``[B, (p + q) * sentence_dim]``.  Empty slots in
        either ring are already zero-padded by ``ensure_batch`` /
        ``Reset``, so the predictor always sees a stable shape on
        cold-start.
        """
        s_flat = self._s_history.reshape(self._batch, -1)  # [B, p*D]
        if self.q > 0:
            e_flat = self._e_history.reshape(self._batch, -1)  # [B, q*D]
            return torch.cat([s_flat, e_flat], dim=-1)
        return s_flat

    @torch.compiler.disable
    def predict_next(self, b=None):
        """Predict the next sentence rep without committing to history.

        Returns ``[sentence_dim]`` for B=1 or with explicit ``b``;
        otherwise returns ``[B, sentence_dim]`` stacked across rows.
        Rows whose history is empty still get a (zero-input)
        prediction, so the call always produces a tensor of stable
        shape — callers gate on ``_s_count`` if they need to
        distinguish cold-start from a real prediction.
        """
        if self.predictor is None:
            return None
        x = self._predictor_input()
        out = self.predictor(x)                          # [B, sentence_dim]
        if b is not None:
            return out[int(b)]
        if self._batch == 1:
            return out[0]
        return out

    # -- observe (push + compute loss) --------------------------------
    @torch.compiler.disable
    def observe(self, s_tensor, mask=None):
        """Capture sentence rep ``s_t``, compute the ARMA MSE loss,
        and update the AR/MA rings.

        Returns the per-batch mean MSE between ``s_hat_t`` (predicted
        from history) and ``s_t``.  Returns ``None`` when no row has
        a primed predictor (all-empty ``_s_count``) — the first
        sentence per row has no AR signal to score against, but its
        rep still goes into ``_s_history`` so subsequent observes
        can predict.

        ``mask`` is an optional ``[B] bool`` selecting which rows
        ended a sentence this step (defaults to all True).
        """
        pooled = self._pool_sentence_rep(s_tensor)
        if pooled is None:
            return None
        B = pooled.shape[0]
        if B != self._batch:
            self.ensure_batch(B)
        s_hat = self.predict_next()
        if s_hat is None:
            return None
        if s_hat.ndim == 1:
            s_hat = s_hat.unsqueeze(0).expand(B, -1)
        # Only score rows whose history was primed. Fully masked: no
        # ``bool(primed.any())`` host gate and no boolean-mask indexing
        # ``s_hat[primed]`` (both force a cudaMemcpyDtoH). Zero the
        # non-primed rows, divide by the primed count. Equivalent to
        # the old ``diff[primed].pow(2).mean()`` when >=1 row is primed;
        # on an all-cold-start batch this returns a 0.0 tensor instead
        # of ``None`` -- training-neutral (the diff is exactly zero so
        # the term carries no gradient), it just logs ``arma=0`` rather
        # than skipping the log. Contract change recorded in the
        # discourse tests.
        primed = self._s_count > 0                        # [B] bool
        if mask is not None:
            primed = primed & mask.to(primed.device)
        pm = primed.to(s_hat.dtype).unsqueeze(-1)         # [B, 1]
        diff = (s_hat - pooled.detach()) * pm             # [B, D]
        n_primed = primed.sum().clamp(min=1).to(diff.dtype)
        loss = diff.pow(2).sum() / (n_primed * s_hat.shape[-1])
        # Update rings -- VECTORIZED. The prior per-row Python loop +
        # ``_push_row`` issued ``_s_count[b].item()`` /
        # ``bool(active[b])`` per row: B host syncs per call, the
        # dominant ``cudaMemcpyDtoH`` source (attribution probe on
        # metalbaby: 192/224 recorded syncs). This is numerically
        # identical -- same fill-then-shift layout, same per-row
        # gating -- expressed as masked tensor ops so no GPU->host
        # round-trip occurs. (Loss block above is untouched.)
        residual = (pooled - s_hat).detach()
        active = (mask if mask is not None
                  else torch.ones(B, dtype=torch.bool, device=pooled.device))
        s_all = pooled.detach()                              # [B, D]
        Bv, D = s_all.shape

        def _ring_push(hist, count, cap, new_rows):
            # hist [B, cap, D]; count [B] long (pre-push); cap int;
            # new_rows [B, D]. Mirrors ``_push_row``: rows in the
            # fill phase (count < cap) scatter the new row into slot
            # ``cap-count-1`` and increment; rows at capacity roll
            # left (newest at the tail); inactive rows untouched.
            cnt = count
            fill = active & (cnt < cap)                      # [B] bool
            shift = active & (cnt >= cap)                    # [B] bool
            rolled = torch.roll(hist, shifts=-1, dims=1).clone()
            rolled[:, -1, :] = new_rows
            idx = (cap - cnt - 1).clamp(0, cap - 1)          # [B] long
            scattered = hist.clone()
            scattered.scatter_(
                1, idx.view(Bv, 1, 1).expand(Bv, 1, D),
                new_rows.unsqueeze(1))
            new_hist = torch.where(
                shift.view(Bv, 1, 1), rolled,
                torch.where(fill.view(Bv, 1, 1), scattered, hist))
            new_count = cnt + fill.to(cnt.dtype)
            return new_hist, new_count

        self._s_history, self._s_count = _ring_push(
            self._s_history, self._s_count, self.p, s_all)
        if self.q > 0:
            self._e_history, self._e_count = _ring_push(
                self._e_history, self._e_count, self.q, residual)
        return loss

    # -- lifecycle ----------------------------------------------------
    def Reset(self, batch=None, hard=False):
        """Clear AR/MA rings on hard discourse boundary.

        ``batch`` selects a specific per-row index to clear; ``None``
        clears all rows.  ``hard`` is accepted for signature parity
        with the per-Space Reset contract and currently ignored —
        ARMA has only one reset semantic.
        """
        del hard
        if batch is None:
            self._s_history.zero_()
            self._s_count.zero_()
            self._e_history.zero_()
            self._e_count.zero_()
        else:
            bi = int(batch)
            self._s_history[bi].zero_()
            self._s_count[bi] = 0
            self._e_history[bi].zero_()
            self._e_count[bi] = 0
    reset = Reset

    def __len__(self):
        """Max ring fill level across rows."""
        if self._batch == 1:
            return int(self._s_count[0].item())
        return int(self._s_count.max().item())

    def latest(self, b=None):
        """Return the most recent sentence rep for row ``b`` (or row 0
        for B=1), or ``None`` if the row is empty.
        """
        if b is None:
            b = 0
        n = int(self._s_count[b].item())
        if n == 0:
            return None
        return self._s_history[b, -1]

    # -- back-compat shims --------------------------------------------
    # The pre-2026-05-14 contrastive API exposed predict/snapshot/
    # contrastive_loss/predictive_loss/prime.  Keep thin shims so the
    # runBatch + Language wiring transitions smoothly until callers
    # migrate to the explicit observe/predict_next/Reset API.

    def predict(self, b=None):
        """Legacy alias for ``predict_next``.

        Returns ``(prediction, confidence)`` where confidence is a
        scalar / per-row tensor fixed at 1.0 — the legacy attention-
        entropy confidence had no analog in the MLP predictor, but
        downstream callers (``WordSpace.stm_residual_microbatch``)
        gate the priming bias on ``conf is None`` so we emit a
        placeholder rather than break that wiring.  Real cold-start
        gating happens via ``_s_count``: rows whose ring is empty
        return ``(None, None)``.

        Compiled path: ``predict()`` is called from two traced sites
        (``BasicModel._forward_per_stage`` and
        ``WordSpace.stm_residual_microbatch``), always with ``b is
        None``.  When ``stage_prediction()`` has parked a tuple, return
        it directly — the live body calls ``predict_next`` (which is
        ``@torch.compiler.disable``'d → graph break) and reads
        ``_s_count.item()`` (host sync), both fatal for fullgraph
        capture.  Staged tuple == pure tensor read, no break, no sync.
        """
        if b is None:
            staged = self._staged_prediction
            if staged is not None:
                return staged
        return self._predict_live(b=b)

    def _predict_live(self, b=None):
        """Uncached ARMA prediction (the pre-staging ``predict()``
        body).  Used on the eager path and to populate the staging
        slot in ``stage_prediction()``.
        """
        if self.predictor is None:
            return None, None
        if (self._batch == 1
                and int(self._s_count[0].item()) == 0):
            return None, None
        pred = self.predict_next(b=b)
        if pred.ndim == 1:
            conf = pred.new_ones(())
        else:
            conf = pred.new_ones(pred.shape[0])
        return pred, conf

    def stage_prediction(self):
        """Eagerly compute and park ``(pred, conf)`` for the upcoming
        compiled forward.  Called from ``runBatch`` (eager region,
        before the traced step) alongside the Phase-3 lex+embed stem
        staging.  No-op-safe to call repeatedly; ``predict()`` returns
        the parked tuple until ``clear_staged_prediction()`` runs.
        """
        self._staged_prediction = self._predict_live(b=None)

    def clear_staged_prediction(self):
        """Drop the staged tuple so the eager path computes live
        again.  Called once after the compiled forward returns."""
        self._staged_prediction = None

    def snapshot(self, s_tensor, w_tensor=None, mask=None):
        """Legacy alias for ``observe`` (ignores ``w_tensor``).

        The pre-ARMA discourse layer pooled an ``[S | W]`` row; ARMA
        consumes only the S-tier sentence rep so ``w_tensor`` is
        dropped.  Returns the ARMA loss the same way ``observe``
        does so callers that already wrote ``layer.snapshot(...)`` for
        side-effects only still work.
        """
        del w_tensor
        return self.observe(s_tensor, mask=mask)

    def contrastive_loss(self, s_tensor, w_tensor=None):
        """Legacy alias: under ARMA the single loss term is the MSE
        from ``observe``; the contrastive cosine pair was retired
        2026-05-14.  Kept so the existing runBatch wiring still
        compiles.
        """
        del w_tensor
        return self.observe(s_tensor)

    def predictive_loss(self, s_tensor, w_tensor=None, predicted=None):
        """No-op under ARMA (the single MSE term in ``observe``
        replaces the separate predictive head)."""
        del s_tensor, w_tensor, predicted
        return None

    def prime(self, predicted, confidence, scale):
        """Lift a predicted sentence rep into C-tier for the chat-
        loop's ``_c_prior`` injection.

        Returns ``cast(predicted) * scale`` (``confidence`` accepted
        for signature back-compat but always treated as 1.0 under
        ARMA, which has no per-prediction confidence score).
        Returns ``None`` when inputs or the cast are missing.
        """
        del confidence  # not produced by the ARMA predictor
        if self.cast is None or predicted is None:
            return None
        if predicted.ndim == 1:
            cast_out = self.cast(predicted.unsqueeze(0)).squeeze(0)
            return cast_out * float(scale)
        return self.cast(predicted) * float(scale)
class ChunkLayer(Layer):
    """Learned BPE-style codebook for perceptual chunking.

    Each entry stores a merge prototype (what the pair looks like) and a
    split prototype (the two constituents).  Forward scores adjacent pairs
    against the codebook; reverse looks up the entry to reconstruct.

    The entropic gate merges a pair only when the codebook match score
    exceeds a learned threshold -- i.e. when encoding the pair as a single
    chunk saves more bits than it costs.

    Merging stops at word boundaries (whitespace characters) so that chunks
    never cross word edges.  In the byte stream the space character (0x20)
    and common whitespace bytes serve as boundary markers.
    """

    def __init__(self, nDim, bpe=False,
                 n_vectors=1024, word_learning=2,
                 cold_start_floor=300):
        """Initialize ChunkLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(nDim, nDim)
        self.nDim = nDim
        # -- BPE state (active only when ``bpe`` is True) ------------------
        # Cold-start: empty merges table, vocab seeded with 256 single-byte
        # ids so byte_value == chunk_id for the 0..255 range.  This mirrors
        # the Embedding.create() placement of bytes at codebook indices
        # 0..255 (see Spaces.py -- ``byte_value == codebook_index``).
        self.bpe = bool(bpe)
        self.n_vectors = int(n_vectors)
        # ``word_learning`` is now a frozen-vs-active gate:
        #   0 (or negative) -> codebook is held constant for the rest of
        #                       the run (used when loading a pre-trained
        #                       artifact -- avoids Inductor recompile
        #                       churn that vocab growth would trigger).
        #   >= 1            -> active learning mode. The integer value
        #                       no longer matters for the threshold;
        #                       promotion is governed by the two-gate
        #                       criterion below.
        self.word_learning = int(word_learning)
        # ``cold_start_floor`` lets the lift gate be skipped while the
        # vocab is still small (V close to 256). With V=256, V^2=65k,
        # and the gate would demand count > N/65k, which stalls the
        # very first merges. Until len(vocab) reaches this floor, we
        # take argmax unconditionally.
        self.cold_start_floor = int(cold_start_floor)
        self.merges = []            # list[tuple[int, int]] in insertion order
        self.vocab = {}             # dict[tuple[int,...], int]
        self.id_to_bytes = {}       # dict[int, tuple[int,...]]
        for i in range(256):
            key = (i,)
            self.vocab[key] = i
            self.id_to_bytes[i] = key
        self._next_id = 256
        self._max_merge_len = 1
        # Cumulative pair- and unigram-counts across batches. The two
        # gates of the promotion criterion read these directly:
        #   prior gate:     count(pair) * V^2 > N         (lift > 1)
        #   posterior gate: count(pair) > min unigram_count over vocab
        #                                  (only binds when codebook full)
        # ``_total_pairs`` is the running N (sum of all observed pair
        # counts). ``_pair_counts`` and ``_unigram_counts`` are
        # Python Counters keyed by chunk-id tuple / chunk-id.
        from collections import Counter as _Counter
        self._pair_counts = _Counter()
        self._unigram_counts = _Counter()
        self._total_pairs = 0

    # -- Persistence (unified vocab-artifact schema) -------------------
    #
    # ChunkLayer's BPE merge table is per-corpus state -- discovering it
    # from byte-pair statistics takes many training batches and triggers
    # Inductor recompiles each time the vocab grows. Saving / loading
    # via the vocab-artifact schema in :mod:`embed` lets a trained
    # merge table persist across runs and avoids the cold-start
    # churn. The artifact format is
    # shared with ``WordVectors`` so a single ``.kv`` file can carry
    # both a Lexicon and a BPE codebook side-by-side.

    def save(self, path: str, lexicon_section: dict = None,
             truth_data: dict = None, metadata: dict = None) -> None:
        """Save the BPE merge table to ``path`` via the unified format.

        Args:
            path: destination file path.
            lexicon_section: optional pre-built Lexicon section from
                ``embed.lexicon_section_from_word_vectors``;
                when supplied the artifact carries both the BPE
                codebook and the Lexicon (kind="both").
            truth_data: optional LTM snapshot (legacy passthrough).
            metadata: optional free-form metadata dict.
        """
        from embed import save_artifact, bpe_section_from_chunk_layer
        save_artifact(
            path,
            lexicon=lexicon_section,
            bpe=bpe_section_from_chunk_layer(self),
            truth_data=truth_data,
            metadata=metadata,
        )

    def load(self, path: str) -> "ChunkLayer":
        """Restore the BPE merge table in place from ``path``.

        Accepts unified-schema artifacts with ``kind`` in
        ``{"bpe", "both"}``. Loading a Lexicon-only artifact raises;
        use ``embed.lexicon_to_bpe_seed`` to convert if you
        want to cold-start BPE from a Lexicon's word list.

        Returns ``self`` so calls can chain (``layer.load(...).forward(...)``).
        """
        from embed import load_artifact, KIND_LEXICON
        payload = load_artifact(path)
        if payload.get("kind") == KIND_LEXICON:
            raise ValueError(
                f"ChunkLayer.load: {path!r} is a Lexicon-only artifact "
                f"(kind=lexicon); no BPE section to load. Use "
                f"WordVectors.load instead, or convert via "
                f"embed.lexicon_to_bpe_seed.")
        section = payload.get("bpe")
        if section is None:
            raise ValueError(
                f"ChunkLayer.load: {path!r} has no bpe section "
                f"(kind={payload.get('kind')!r})")
        self.merges = [tuple(p) for p in section.get("merges", [])]
        self.vocab = {tuple(k): int(v) for k, v in section["vocab"].items()}
        self.id_to_bytes = {
            int(k): tuple(v) for k, v in section["id_to_bytes"].items()}
        self._next_id = int(section.get(
            "next_id", max(self.id_to_bytes.keys()) + 1 if self.id_to_bytes else 256))
        self._max_merge_len = int(section.get("max_merge_len", 1))
        # Honor n_vectors / word_learning from the artifact if the
        # caller didn't override at construction.
        self.n_vectors = int(section.get("n_vectors", self.n_vectors))
        self.word_learning = int(section.get(
            "word_learning", self.word_learning))
        return self

    # -- Boundary detection --------------------------------------------

    BOUNDARY_BYTES = frozenset({0x00, 0x09, 0x0A, 0x0D, 0x20})

    def is_word_boundary(self, data, b, pos, subspace=None, byte_indices=None):
        """True if position is a word boundary.

        Byte mode: check byte_indices against BOUNDARY_BYTES (fast, exact).
        Word mode: cosine similarity against space embedding (learned, soft).
        """
        if byte_indices is not None:
            return byte_indices[b, pos].item() in self.BOUNDARY_BYTES
        if subspace is None:
            return False
        try:
            space_emb = subspace.vocabulary.get_space_embedding()
        except (AttributeError, RuntimeError):
            return False
        vec = data[b, pos]
        sim = F.cosine_similarity(vec.unsqueeze(0), space_emb.unsqueeze(0), dim=-1)
        return sim.item() > 0.9

    # -- BPE forward + training ----------------------------------------

    def forward(self, byte_indices):
        """Greedy longest-match BPE encoding of a batch of byte sequences.

        Active only in BPE mode (``self.bpe == True``); in legacy mode
        this is an identity pass-through so existing downstream callers
        keep their previous semantics.

        Args:
            byte_indices: ``[B, N]`` long tensor of byte values 0..255.
                A zero entry terminates the row (padding sentinel).
        Returns:
            When ``bpe`` is True:
                ``(chunks, spans)`` -- ``chunks`` is ``list[list[int]]`` of
                chunk ids per row; ``spans`` is ``list[list[(start, end,
                key)]]`` where ``key`` is the byte-tuple backing that
                chunk.  ``end`` is inclusive.
            When ``bpe`` is False:
                ``byte_indices`` unchanged.
        """
        if not self.bpe:
            return byte_indices
        if byte_indices.dim() != 2:
            raise ValueError(
                f"ChunkLayer.forward expects [B, N] byte indices, got "
                f"{tuple(byte_indices.shape)}")
        B, N = byte_indices.shape
        rows = byte_indices.tolist()
        # Trie-based longest-match: walks the vocab once per match in
        # O(actual_match_len) instead of the previous nested
        # ``for L in range(upper, 0, -1): tuple(row[i:i+L]) in vocab``
        # which paid O(_max_merge_len) tuple constructions and dict
        # lookups at every position even for short matches. The trie
        # is built lazily and rebuilt only when ``vocab`` grows
        # (frozen vocab = built once, then a no-op).
        self._ensure_trie()
        trie = self._trie
        id_to_bytes = self.id_to_bytes
        vocab = self.vocab
        all_chunks = []
        all_spans = []
        for b in range(B):
            row = rows[b]
            chunks = []
            spans = []
            i = 0
            while i < N:
                bval = row[i]
                if bval == 0:
                    break
                # Descend the trie one byte at a time. Each node is
                # ``[children_dict, terminal_id_or_None]``; track the
                # most recent terminal as the running longest match.
                node = trie
                matched_id = None
                matched_len = 0
                j = i
                while j < N:
                    child = node[0].get(row[j])
                    if child is None:
                        break
                    node = child
                    j += 1
                    if node[1] is not None:
                        matched_id = node[1]
                        matched_len = j - i
                if matched_id is None:
                    # 256 single-byte ids are seeded at __init__ so any
                    # byte 0..255 always has a single-byte match. This
                    # branch is defensive for unexpected vocab gaps.
                    matched_id = vocab.get((bval,), bval)
                    matched_len = 1
                chunks.append(matched_id)
                # ``id_to_bytes[matched_id]`` is the canonical byte tuple
                # for this id (kept in sync with ``vocab`` by
                # ``train_step`` and ``__init__``); the trie walk only
                # accepts paths that exist in the vocab, so the matched
                # row slice equals ``id_to_bytes[matched_id]`` and we
                # avoid re-slicing the row.
                spans.append((i, i + matched_len - 1, id_to_bytes[matched_id]))
                i += matched_len
            all_chunks.append(chunks)
            all_spans.append(spans)
        return all_chunks, all_spans

    def _ensure_trie(self):
        """Build / rebuild the longest-match trie when ``vocab`` changes.

        Each node is ``[children_dict, terminal_id_or_None]``. When
        ``terminal_id`` is not ``None`` the path from root to this node
        spells the byte tuple of vocab entry ``terminal_id``. Rebuild
        triggers off ``len(vocab)`` (vocab only grows under
        ``train_step``); when ``word_learning == 0`` the vocab is
        frozen and this is built once then no-op forever after.
        """
        cur_size = len(self.vocab)
        if getattr(self, '_trie_size', -1) == cur_size:
            return
        root = [{}, None]
        for byte_tuple, chunk_id in self.vocab.items():
            node = root
            for b in byte_tuple:
                children = node[0]
                child = children.get(b)
                if child is None:
                    child = [{}, None]
                    children[b] = child
                node = child
            node[1] = chunk_id
        self._trie = root
        self._trie_size = cur_size

    def train_step(self, byte_indices, k_merges=1):
        """Learn up to ``k_merges`` new BPE merges from pair frequencies.

        Encodes the batch with the current merge table, accumulates
        adjacent-pair and unigram counts into the cumulative tallies
        (``self._pair_counts``, ``self._unigram_counts``,
        ``self._total_pairs``), then promotes top-k pairs that pass
        the two-gate criterion. Returns the number of new merges added
        (0 when none qualify).

        Promotion criterion:
          1. **Prior gate (lift > 1):** ``count(pair) * V^2 > N``,
             where ``V = len(self.vocab)`` and ``N = self._total_pairs``.
             A pair must be more common than uniform-distribution
             chance over the current alphabet. Skipped while
             ``V < cold_start_floor`` (early in the run V^2 is so
             small the gate would stall the first merges).
          2. **Posterior gate (earns its slot):** when the codebook
             is full, ``count(pair)`` must exceed the minimum
             ``self._unigram_counts[c]`` across all current chunk
             ids. While ``len(self.vocab) < n_vectors`` this gate is
             vacuous. The first time it fails after the codebook
             fills, training is converged for the corpus.

        Idempotent w.r.t. vocab size: stops once ``len(vocab)`` reaches
        ``n_vectors`` AND no candidate clears Gate 2. No-op in legacy
        (``bpe=False``) mode.

        ``word_learning <= 0`` is the **frozen** marker -- the
        merge table is held constant for the rest of the run, no new
        entries are added regardless of byte-pair statistics. Use this
        after loading a pre-trained BPE artifact (via ``ChunkLayer.load``)
        so subsequent training runs don't trigger Inductor recompiles
        every time a new pair crosses a threshold.
        """
        if not self.bpe:
            return 0
        if self.word_learning <= 0:
            # Frozen: vocab stays as-is for the rest of the run.
            return 0
        all_chunks, _ = self.forward(byte_indices)
        # Update cumulative tallies from this batch.
        from collections import Counter
        batch_counts = Counter()
        batch_pairs = 0
        for chunks in all_chunks:
            for c in chunks:
                self._unigram_counts[c] += 1
            for a, b in zip(chunks, chunks[1:]):
                batch_counts[(a, b)] += 1
                batch_pairs += 1
        for pair, freq in batch_counts.items():
            self._pair_counts[pair] += freq
        self._total_pairs += batch_pairs
        if not self._pair_counts:
            return 0
        # Compute the two-gate threshold.
        V = len(self.vocab)
        N = self._total_pairs
        prior_threshold = (N // (V * V)) if V >= self.cold_start_floor else 0
        vocab_full = V >= self.n_vectors
        if vocab_full:
            min_unigram = min(self._unigram_counts.values()) \
                if self._unigram_counts else 0
            posterior_threshold = min_unigram
        else:
            posterior_threshold = 0
        threshold = max(prior_threshold, posterior_threshold)
        # Promote top-k pairs from cumulative counts that clear both gates.
        added = 0
        for pair, freq in self._pair_counts.most_common():
            if added >= k_merges:
                break
            if freq <= threshold:
                # Best remaining candidate is below threshold ->
                # nothing else will clear it either; converged.
                break
            if vocab_full:
                # Codebook full and no eviction -- Gate 2 has bound.
                break
            left_bytes = self.id_to_bytes.get(pair[0])
            right_bytes = self.id_to_bytes.get(pair[1])
            if left_bytes is None or right_bytes is None:
                continue
            new_key = left_bytes + right_bytes
            if new_key in self.vocab:
                continue
            new_id = self._next_id
            self._next_id += 1
            self.vocab[new_key] = new_id
            self.id_to_bytes[new_id] = new_key
            self.merges.append(pair)
            if len(new_key) > self._max_merge_len:
                self._max_merge_len = len(new_key)
            added += 1
            # Re-evaluate gates after a promotion (V grew by 1; if we
            # just filled the codebook, Gate 2 now binds).
            V = len(self.vocab)
            prior_threshold = (N // (V * V)) if V >= self.cold_start_floor else 0
            vocab_full = V >= self.n_vectors
            if vocab_full:
                min_unigram = min(self._unigram_counts.values()) \
                    if self._unigram_counts else 0
                posterior_threshold = min_unigram
            else:
                posterior_threshold = 0
            threshold = max(prior_threshold, posterior_threshold)
        return added

    # -- Byte-mode hard merge + compaction -----------------------------

    def hard_merge_spans(self, data, byte_indices):
        """Merge contiguous byte slots into spans via mean aggregation.

        Dispatches on ``self.bpe``:
          - ``bpe=False`` (legacy): whitespace-boundary spans via
            ``BOUNDARY_BYTES``.
          - ``bpe=True``: spans come from the learned BPE merge table
            (greedy longest-match via ``self.forward``).  When training,
            runs a single ``train_step`` first so the merge table grows
            from this batch's pair statistics.

        Args:
            data: ``[B, N, D]`` byte-level vectors.
            byte_indices: ``[B, N]`` long -- byte values 0..255.
        Returns:
            ``(data, span_meta)`` where ``data`` has span-start slots
            holding mean vectors (rest zeroed) and ``span_meta`` is
            ``list[list[(start, end, original_vectors)]]`` per batch row.
        """
        if self.bpe:
            if self.training:
                self.train_step(byte_indices)
            return self._hard_merge_spans_bpe(data, byte_indices)

        B, N, D = data.shape
        data = data.clone()
        span_meta = []
        for b in range(B):
            spans = []
            i = 0
            while i < N:
                bval = byte_indices[b, i].item()
                if bval == 0:
                    break
                if bval in self.BOUNDARY_BYTES:
                    data[b, i] = 0.0
                    i += 1
                    continue
                start = i
                while i < N and byte_indices[b, i].item() not in self.BOUNDARY_BYTES:
                    i += 1
                end = i - 1
                original = data[b, start:end + 1].clone()
                data[b, start] = data[b, start:end + 1].mean(dim=0)
                if end > start:
                    data[b, start + 1:end + 1] = 0.0
                spans.append((start, end, original))
            span_meta.append(spans)
        return data, span_meta

    def _hard_merge_spans_bpe(self, data, byte_indices):
        """BPE-mode span construction.

        Each BPE chunk produced by ``self.forward`` becomes one span;
        the span-start slot holds the mean of the span's byte vectors
        and the trailing slots are zeroed, matching the layout that
        ``compact`` / ``uncompact`` expect.  The original pre-merge
        byte vectors are stored in ``span_meta`` so reconstruction via
        ``uncompact`` is exact -- the merge table is used only for
        forward segmentation, never for decoding bytes.
        """
        B, N, D = data.shape
        data = data.clone()
        _, raw_spans = self.forward(byte_indices)
        span_meta = []
        for b in range(B):
            spans = []
            for start, end, _key in raw_spans[b]:
                if end >= N:
                    end = N - 1
                original = data[b, start:end + 1].clone()
                data[b, start] = data[b, start:end + 1].mean(dim=0)
                if end > start:
                    data[b, start + 1:end + 1] = 0.0
                spans.append((start, end, original))
            span_meta.append(spans)
        return data, span_meta

    def compact(self, data, nWordSlots, span_meta, where_encoding=None):
        """Pack active span-start positions into dense [B, nWordSlots, D].

        After packing, overwrites the where-encoding dims ([-4, -3]) with
        sinusoidal encoding of each span's start byte offset, so downstream
        layers see word-level positional encoding instead of byte-level.

        Args:
            data: [B, N, D] sparse byte-level vectors (span starts populated)
            nWordSlots: target dense width
            span_meta: list[list[(start, end)]] or
                list[list[(start, end, original_vectors)]] per batch
            where_encoding: WhereEncoding instance (for sin/cos rewrite)
        Returns:
            dense: [B, nWordSlots, D]
            compact_map: list[list[(dense_idx, start, end, original_vectors)]] -- for reverse
        """
        B, N, D = data.shape
        dense = torch.zeros(B, nWordSlots, D, device=data.device)
        compact_map = []
        for b in range(B):
            mapping = []
            for dense_idx, span in enumerate(span_meta[b]):
                if dense_idx >= nWordSlots:
                    break
                start, end = span[0], span[1]
                original = span[2] if len(span) > 2 else None
                dense[b, dense_idx] = data[b, start]
                mapping.append((dense_idx, start, end, original))
            compact_map.append(mapping)

        # Overwrite where-encoding dims with span start byte offset
        if where_encoding is not None and where_encoding.nDim > 0:
            where_idx = [D + i for i in where_encoding.index]  # e.g. [-4,-3] -> absolute
            for b in range(B):
                for dense_idx, start, _end, _original in compact_map[b]:
                    pos_enc = where_encoding.encode(float(start))  # [2] sin/cos
                    dense[b, dense_idx, where_idx] = pos_enc

        return dense, compact_map

    def uncompact(self, dense, compact_map, nByteSlots):
        """Scatter dense word vectors back to byte positions (span copy)."""
        B, _, D = dense.shape
        data = torch.zeros(B, nByteSlots, D, device=dense.device)
        for b in range(B):
            for item in compact_map[b]:
                dense_idx, start, end = item[:3]
                original = item[3] if len(item) > 3 else None
                span_len = end - start + 1
                if original is not None:
                    data[b, start:end + 1] = original.to(device=dense.device, dtype=dense.dtype)
                else:
                    data[b, start:end + 1] = dense[b, dense_idx].unsqueeze(0).expand(span_len, -1)
        return data
#endregion

#region Operations

# Sentinel for the unified lift / lower dispatcher: distinguishes the
# legacy positional form Ops._lift_kernel(left, right) (deprecated; routes to the
# old elementwise-product body) from the new keyword form
# Ops._lift_kernel(X1, X2, mode='OR', ...).
_NO_MODE = object()


class Ops:
    """[FROZEN — DEPRECATED FOR GRAMMAR OPS] Pure operations on
    activation vectors in [-1, +1].

    **Status (2026-04-28):** Ops is frozen. No new methods should be
    added here. Grammar-relevant operators (``not``, ``non``, ``lift``,
    ``lower``, ``intersection``, ``union``, ``absorb``, ``swap``,
    ``equals``, ``part``, ``whole``, ``project``, etc.) belong as
    ``GrammarLayer`` subclasses with ``rule_name``, ``arity``,
    ``invertible``, and ``lossy`` fields plus ``forward``/``reverse``
    methods. The Layer-shaped invocation surface lets the bottom-up
    forward path (rule-probability gating) and the SyntacticLayer's
    top-down dispatch use the same Layer instances.

    Existing methods below stay until callers migrate. New operators
    should derive from GrammarLayer instead. ``GrammarLayer``
    subclasses are added one at a time as concrete grammars start
    invoking the corresponding rule — not pre-emptively.

    Tensor utilities (sign, saturate, threshold, distance, what/where/
    when accessors, minMag/maxMag, ...) that are not grammar operators
    can stay on Ops indefinitely as a math-kernel namespace.

    Supersedes the legacy ``Activation`` class. Static namespace; ``Basis``
    (in Spaces.py) delegates its logic, mereology, and metric methods
    here so each formula has a single source of truth.

    Domain/range conventions:
        Values live in [-1, +1].
        monotonic=False (bitonic): sign = direction, magnitude = confidence.
            +1 = true, 0 = unknown, -1 = false.
        monotonic=True (unsigned): [0, 1] scale with a tetralemma pair
            representation for signed ops (negation flips the pair).

    Layout:
        - Tensor ops (conjunction, disjunction, negation, non, pos, norm,
          distance, part, whole, equal, overlap, underlap, boundary,
          copart) expect torch tensors with last-dim semantics.
        - Scalar/elementwise ops (positive, negative, neutral, sign,
          saturate, threshold, complement, convertSensation, minMag,
          maxMag, error, isActive, isEqual, isReducer, true, false,
          unknown) accept torch tensors or python scalars; non-tensor
          inputs are coerced via ``torch.as_tensor``.
    """

    # ---- Tetralemma constants --------------------------------------------

    @staticmethod
    def true():
        """True.
        
        See class docstring for the operation contract.
        """
        return 1

    @staticmethod
    def false():
        """False.
        
        See class docstring for the operation contract.
        """
        return -1

    @staticmethod
    def unknown():
        """Unknown.
        
        See class docstring for the operation contract.
        """
        return 0

    # ---- Scalar / elementwise primitives ---------------------------------

    @staticmethod
    def positive(x):
        """Positive.
        
        See class docstring for the operation contract.
        """
        t = torch.as_tensor(x)
        return torch.relu(t)

    @staticmethod
    def negative(x):
        """Negative.
        
        See class docstring for the operation contract.
        """
        t = torch.as_tensor(x)
        return -torch.relu(-t)

    @staticmethod
    def neutral(x):
        """Neutral.
        
        See class docstring for the operation contract.
        """
        return 1 - torch.as_tensor(x).abs()

    @staticmethod
    def sign(v):
        """Signum with sgn(0) = 1 (not 0)."""
        t = torch.as_tensor(v).float()
        y = torch.sign(t)
        return torch.where(y == 0, torch.ones_like(y), y)

    @staticmethod
    def saturate(x):
        """Clamp to [-1, +1]. NaN -> 0."""
        t = torch.as_tensor(x)
        return torch.nan_to_num(t.clamp(-1.0, 1.0))

    @staticmethod
    def threshold(x, activationThreshold=0.01):
        """Threshold.
        
        See class docstring for the operation contract.
        """
        t = torch.as_tensor(x)
        return torch.where(t.abs() < activationThreshold, torch.zeros_like(t), t)

    @staticmethod
    def complement(x):
        """Complement.
        
        See class docstring for the operation contract.
        """
        t = torch.as_tensor(x)
        return Ops.sign(t) - t

    @staticmethod
    def convertSensation(x):
        """Convert sensation.
        
        See class docstring for the operation contract.
        """
        return 2 * torch.as_tensor(x) - 1

    @staticmethod
    def minMag(x1, x2):
        """Min mag.
        
        See class docstring for the operation contract.
        """
        t1, t2 = torch.as_tensor(x1), torch.as_tensor(x2)
        return torch.where(t1.abs() <= t2.abs(), t1, t2)

    @staticmethod
    def maxMag(x1, x2):
        """Max mag.
        
        See class docstring for the operation contract.
        """
        t1, t2 = torch.as_tensor(x1), torch.as_tensor(x2)
        return torch.where(t1.abs() >= t2.abs(), t1, t2)

    @staticmethod
    def error(x1, x2):
        """Error.
        
        See class docstring for the operation contract.
        """
        t1, t2 = torch.as_tensor(x1).float(), torch.as_tensor(x2).float()
        return torch.linalg.norm(t1 - t2)

    @staticmethod
    def isActive(x, activationThreshold=0.01):
        """Is active.
        
        See class docstring for the operation contract.
        """
        return torch.as_tensor(x).abs() >= activationThreshold

    @staticmethod
    def isEqual(x1, x2):
        """Is equal.
        
        See class docstring for the operation contract.
        """
        return torch.equal(torch.as_tensor(x1), torch.as_tensor(x2))

    @staticmethod
    def isReducer(x1, x2):
        """Is reducer.
        
        See class docstring for the operation contract.
        """
        t1, t2 = torch.as_tensor(x1), torch.as_tensor(x2)
        return ((t2 - t1).abs().sum() < t2.abs().sum()).item()

    # ---- Tensor primitives -----------------------------------------------

    @staticmethod
    def pos(x):
        """Positive projection (ReLU). Domain [-1, 1], range [0, 1]."""
        return torch.relu(x)

    @staticmethod
    def norm(x):
        """Last-dim L2 norm."""
        return torch.norm(x, dim=-1)

    # ---- Binary fuzzy logic ----------------------------------------------

    # ---- RadMin / RadMax (bitonic same-sign magnitude pooling) -----------
    # Private helpers so kind='radial' on lift / lower can call them
    # directly without round-tripping through Ops.conjunction /
    # Ops.disjunction (which themselves now forward back to lift / lower).

    @staticmethod
    def _radmin(x, y):
        """Same-sign minimum magnitude (zero collapse).
        Pre-Step-2 bitonic conjunction body."""
        same_sign = (x * y > 0).float()
        min_mag = torch.min(torch.abs(x), torch.abs(y))
        return same_sign * torch.sign(x) * min_mag

    @staticmethod
    def _radmax(x, y):
        """Same-sign maximum magnitude with zero passthrough.
        Pre-Step-2 bitonic disjunction body."""
        same_sign = (x * y > 0).float()
        max_mag = torch.max(torch.abs(x), torch.abs(y))
        core = same_sign * torch.sign(x) * max_mag
        x_zero = (x == 0).float()
        y_zero = (y == 0).float()
        return core + x_zero * y + y_zero * x

    @staticmethod
    def corner_overlap(a, b):
        """Pairwise hyperrectangle-overlap measure on bivector leaves.

        Distinct from ``Ops.overlap`` (mereological parthood-based
        overlap, defined further down). This kernel implements the
        "overlapping corner in any dimension" criterion used by the
        contiguity measure on hoc_shape leaves.

        Args:
            a, b: ``[..., 2]`` tensors -- each is a ``[pos, neg]``
                bivector defining the hyperrectangle interval
                ``[-neg, +pos]`` on its dimension-of-origin.

        Returns:
            tensor of shape ``a.shape[:-1]`` (the leading batch dims
            with the bivector dim collapsed). Each entry in
            ``[-1, +1]``:
              ``+1`` if the intervals overlap (the bivectors share
                    at least one common corner -- equivalent to
                    ``max(-a_neg, -b_neg) <= min(a_pos, b_pos)``).
              ``-1`` if the intervals are disjoint.
              ``0``  if either input is degenerate (no extent --
                    both pos and neg ~ 0), so overlap cannot be
                    decided.

        Vectorized over arbitrary leading dims; no Python iteration.
        Inputs are interpreted as a *single bivector pair* (last
        dim == 2). For higher-D regions the caller stacks bivector
        leaves along an "axis" dim before calling this kernel
        (``a.shape == [..., n_axes, 2]``); in that case the
        any-axis aggregation is the caller's choice (typically
        ``.amax(dim=-1)`` to mark "overlap in any axis = +1").
        """
        if a.shape[-1] != 2 or b.shape[-1] != 2:
            raise ValueError(
                f"Ops.corner_overlap: expected last dim == 2 "
                f"bivector pair, got a={tuple(a.shape)} "
                f"b={tuple(b.shape)}")
        eps = 1e-9
        a_lo = -a[..., 1]
        a_hi =  a[..., 0]
        b_lo = -b[..., 1]
        b_hi =  b[..., 0]
        a_extent = (a[..., 0].abs() + a[..., 1].abs()) > eps
        b_extent = (b[..., 0].abs() + b[..., 1].abs()) > eps
        both_extent = a_extent & b_extent

        lo = torch.maximum(a_lo, b_lo)
        hi = torch.minimum(a_hi, b_hi)
        # Touching corners count as overlap (lo <= hi, inclusive).
        dim_overlap = lo <= hi

        # +1 where both have extent and intervals overlap.
        # -1 where both have extent and intervals are disjoint.
        #  0 where either is degenerate.
        zeros = torch.zeros_like(a_hi)
        ones  = torch.ones_like(a_hi)
        result = torch.where(both_extent & dim_overlap,  ones,  zeros)
        result = torch.where(both_extent & ~dim_overlap, -ones, result)
        return result

    @staticmethod
    def epsilon_delta(in_boxes, out_boxes, norm='inf', eps_floor=1e-6):
        """ε-δ continuity ratios over pairs of bivector hyperrectangles.

        Classical pointwise continuity: ``f`` is continuous at ``x``
        iff ``∀ε > 0, ∃δ > 0`` such that ``‖x' − x‖ < δ ⇒ ‖f(x') −
        f(x)‖ < ε``. Adapted to a finite set of leaf hyperrectangles:
        for each pair ``(R_i, R_j)`` the input-side diameter ``δ_ij``
        and the output-side diameter ``ε_ij`` of the union of the two
        boxes give the empirical ratio ``ε_ij / δ_ij``. The map is
        continuous on the leaf set when this ratio stays bounded.

        Hyperrectangle from a bivector ``[pos, neg]``: per-axis
        interval ``[-|neg|, +pos]``; box diameter on axis ``d`` is
        ``pos_d + |neg_d|``. Union of two boxes has per-axis spread
        ``max(pos_i, pos_j) + max(|neg_i|, |neg_j|)``.

        Args:
            in_boxes:  ``[K, ..., 2]`` -- K leaf input-side bivectors
                (the active higher-order symbol's bivector at each
                leaf's source position).
            out_boxes: ``[K, ..., 2]`` -- matching output-side
                bivectors (the back-projected leaf bivector).
            norm: ``'inf'`` (default; per-axis max -- "did any
                axis blow up") or ``'l2'`` (Euclidean norm of the
                per-axis spread vector).
            eps_floor: clamp on δ to keep the ratio finite when
                input boxes coincide.

        Returns:
            ratios: ``[K, K]`` -- per-pair ε / δ. Symmetric. The
                diagonal carries the per-leaf self-pair ratio
                (``= 1`` when in and out diameters match, e.g. for
                an identity map; consumers typically mask the
                diagonal off when aggregating). Pairs whose δ falls
                below ``eps_floor`` get ``ε / eps_floor``
                (clamped); callers should mask them out with the
                input-extent test instead of trusting the ratio.

        Vectorized over arbitrary leading dims; no Python iteration.
        """
        if in_boxes.shape[-1] != 2 or out_boxes.shape[-1] != 2:
            raise ValueError(
                f"Ops.epsilon_delta: expected last dim == 2 bivector "
                f"pair, got in={tuple(in_boxes.shape)} "
                f"out={tuple(out_boxes.shape)}")
        if in_boxes.shape[0] != out_boxes.shape[0]:
            raise ValueError(
                f"Ops.epsilon_delta: leaf-count mismatch -- "
                f"in_boxes.shape[0]={in_boxes.shape[0]} vs "
                f"out_boxes.shape[0]={out_boxes.shape[0]}")

        def pair_diam(boxes):
            """boxes: [K, ..., 2]. Returns [K, K, ...] pairwise per-
            axis diameters of the union of each box pair."""
            pos = boxes[..., 0]
            neg = boxes[..., 1].abs()
            pos_max = torch.maximum(pos.unsqueeze(0), pos.unsqueeze(1))
            neg_max = torch.maximum(neg.unsqueeze(0), neg.unsqueeze(1))
            return pos_max + neg_max

        in_d  = pair_diam(in_boxes)
        out_d = pair_diam(out_boxes)
        if norm == 'inf':
            delta = in_d.amax(dim=-1)
            eps   = out_d.amax(dim=-1)
        elif norm == 'l2':
            delta = in_d.norm(dim=-1)
            eps   = out_d.norm(dim=-1)
        else:
            raise ValueError(
                f"Ops.epsilon_delta: unknown norm {norm!r}; "
                f"expected 'inf' or 'l2'")

        return eps / delta.clamp_min(eps_floor)

    @staticmethod
    def hyperrectangle_volume(boxes, eps=1e-6):
        """Per-leaf hyperrectangle volume from `[pos, neg]` corner bivectors.

        Companion to :meth:`Ops.corner_overlap` and
        :meth:`Ops.epsilon_delta`; same input contract.

        Args:
            boxes: ``[..., n_axes, 2]`` -- last dim is the bivector
                pair, the preceding dim is the axis count.  Per axis
                ``d`` the side length is ``pos_d + |neg_d|`` (matching
                the box-diameter convention used by
                ``Ops.epsilon_delta.pair_diam``).  Leading dims are
                preserved.
            eps: axes whose side ``< eps`` drop out of the active
                subspace (the rectangle is effectively
                lower-dimensional in that channel).

        Returns:
            ``[...]`` scalar volume per leading-batch element.
            Volume = product of active-axis sides.  An empty active
            subspace returns ``0``.
        """
        if boxes.shape[-1] != 2:
            raise ValueError(
                f"Ops.hyperrectangle_volume: expected last dim == 2 "
                f"bivector pair, got {tuple(boxes.shape)}")
        sides = boxes[..., 0] + boxes[..., 1].abs()           # [..., n_axes]
        active = sides > eps
        # Replace inactive axes' side with 1 so they are neutral in
        # the product; volume returns 0 only when *no* axis is active.
        sides_for_product = torch.where(active, sides,
                                         torch.ones_like(sides))
        any_active = active.any(dim=-1)
        vol = sides_for_product.prod(dim=-1)
        return torch.where(any_active, vol, torch.zeros_like(vol))

    @staticmethod
    def hyperrectangle_overlap_volume(a, b, eps=1e-6):
        """Shared volume of two hyperrectangles in ``[pos, neg]`` form.

        Companion to :meth:`Ops.corner_overlap` (binary corner test) —
        this op returns the actual shared volume, not just an overlap
        indicator.

        Per axis ``d`` the boxes' intervals are ``[-|a_neg|, +a_pos]``
        and ``[-|b_neg|, +b_pos]``.  Shared interval =
        ``max(-|a_neg|, -|b_neg|)`` to ``min(a_pos, b_pos)``;
        width = ``max(0, hi - lo)``.  Axes where either box's side
        ``< eps`` drop out (the rectangle has no extent there).
        Volume = product over the remaining axes' shared widths.

        Args:
            a, b: ``[..., n_axes, 2]`` bivector tensors with matching
                leading dims.
            eps: axis-active threshold.

        Returns:
            ``[...]`` shared-volume scalar per leading element.
        """
        if a.shape[-1] != 2 or b.shape[-1] != 2:
            raise ValueError(
                f"Ops.hyperrectangle_overlap_volume: expected last "
                f"dim == 2 bivector pair, got a={tuple(a.shape)} "
                f"b={tuple(b.shape)}")
        a_lo = -a[..., 1].abs()
        a_hi =  a[..., 0]
        b_lo = -b[..., 1].abs()
        b_hi =  b[..., 0]
        a_side = a[..., 0] + a[..., 1].abs()
        b_side = b[..., 0] + b[..., 1].abs()
        active = (a_side > eps) & (b_side > eps)              # [..., n_axes]

        lo = torch.maximum(a_lo, b_lo)
        hi = torch.minimum(a_hi, b_hi)
        width = (hi - lo).clamp(min=0.0)                      # [..., n_axes]
        # Inactive axes are neutral (side = 1) in the product but
        # only when at least one axis is active overall.
        widths_for_product = torch.where(active, width,
                                          torch.ones_like(width))
        any_active = active.any(dim=-1)
        vol = widths_for_product.prod(dim=-1)
        return torch.where(any_active, vol, torch.zeros_like(vol))

    @staticmethod
    def _conjunction_kernel(x, y, monotonic=False):
        """Conjunction (intersection). Domain/range [-1, 1].

        Thin forwarder to Ops._lower_kernel(mode='AND'):
            monotonic=True  → kind='strict' (lattice min)
            monotonic=False → kind='radial' (RadMin same-sign min magnitude)
        Bit-exact match to the pre-Step-2 body.
        """
        kind = 'strict' if monotonic else 'radial'
        return Ops._lower_kernel(x, y, mode='AND', kind=kind)

    @staticmethod
    def _disjunction_kernel(x, y, monotonic=False):
        """Disjunction (union). Domain/range [-1, 1].

        Thin forwarder to Ops._lift_kernel(mode='OR'):
            monotonic=True  → kind='strict' (lattice max)
            monotonic=False → kind='radial' (RadMax same-sign max magnitude
                              with zero passthrough)
        Bit-exact match to the pre-Step-2 body.
        """
        kind = 'strict' if monotonic else 'radial'
        return Ops._lift_kernel(x, y, mode='OR', kind=kind)

    @staticmethod
    def intersection(x, y, monotonic=False):
        """Set intersection on bivector activations -- the public
        kernel ``IntersectionLayer`` calls.

        Per-axis, per-pole "min toward zero" on ``x`` and ``y``:
            monotonic=False (default) -> RadMin: same-sign min
                magnitude, zero passthrough. The pole closer to
                zero wins per channel.
            monotonic=True            -> strict lattice min on each
                channel.

        Forwards to ``_conjunction_kernel`` so the math is bit-exact
        with the pre-2026-05-04 IntersectionLayer body. Equivalent
        to the alias ``Ops._conjunction_kernel(x, y, monotonic=...)``.
        """
        return Ops._conjunction_kernel(x, y, monotonic=monotonic)

    @staticmethod
    def union(x, y, monotonic=False):
        """Set union on bivector activations -- the public kernel
        ``UnionLayer`` calls.

        Per-axis, per-pole "max toward zero" (in the sense of
        max-magnitude, away from zero) on ``x`` and ``y``:
            monotonic=False (default) -> RadMax: same-sign max
                magnitude with zero passthrough.
            monotonic=True            -> strict lattice max on each
                channel.

        Forwards to ``_disjunction_kernel``; bit-exact with the
        pre-2026-05-04 UnionLayer body.
        """
        return Ops._disjunction_kernel(x, y, monotonic=monotonic)

    @staticmethod
    def _negation_kernel(x, monotonic=False):
        """Negation.
        Bitonic: sign flip (-x).
        Monotonic, last-dim == 2: tetralemma swap on a demuxed [aP, aN] bivector.
        Monotonic, last-dim == 2K: paired-index pair flip."""
        if not monotonic:
            return -x
        n = x.shape[-1]
        if n == 2:
            return x.flip(dims=(-1,))
        if n % 2 == 0:
            pair = x.reshape(*x.shape[:-1], n // 2, 2)
            flipped = pair.flip(dims=(-1,))
            return flipped.reshape(*x.shape)
        raise ValueError(
            f"Ops._negation_kernel(monotonic=True) requires even last dim; "
            f"got shape {tuple(x.shape)}"
        )

    @staticmethod
    def _non_kernel(x, monotonic=False, threshold=None):
        """Non-affirming negation -- the 'indeterminate' commitment.

        Bitonic (default): triangular residual 1 - |clamp(x, -1, 1)|.
            Completes the S-tier trinity partition of unity:
            true(x) + false(x) + non(x) = 1.
        Monotonic: ReLU(x - threshold) if threshold given, else 0."""
        if monotonic:
            if threshold is not None:
                return torch.relu(x - threshold)
            return torch.zeros_like(x)
        return 1.0 - torch.clamp(x, -1.0, 1.0).abs()

    # ---- Inverse logic operations ----------------------------------------
    # ``conjunctionReverse`` / ``disjunctionReverse`` need a codebook ``W``
    # to invert the lossy binary op via search. ``Basis`` supplies this via
    # ``self.getW()`` when delegating; standalone callers may pass ``W`` of
    # shape (K, D) directly.

    @staticmethod
    def negationReverse(x, monotonic=False):
        """Inverse of negation. Self-inverse in both modes.
        Bitonic: sign flip. Monotonic: paired-index flip."""
        return Ops._negation_kernel(x, monotonic=monotonic)

    @staticmethod
    def conjunctionReverse(result, y, W, monotonic=False, unit_ball=False):
        """Inverse of conjunction via codebook search.

        Find the codebook vector x such that conjunction(x, cb_j) ~= result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if W is None or empty.
        ``unit_ball=True`` switches the distance from raw Euclidean to
        wrapped Euclidean (torus geometry).
        """
        return Ops._binary_op_inverse_impl(
            result, W, Ops.conjunction, monotonic, unit_ball=unit_ball)

    @staticmethod
    def disjunctionReverse(result, y, W, monotonic=False, unit_ball=False):
        """Inverse of disjunction via codebook search.

        Find the codebook vector x such that disjunction(x, cb_j) ~= result
        for some cb_j, returning the best-matching left operand.
        Falls back to returning result unchanged if W is None or empty.
        ``unit_ball=True`` switches the distance from raw Euclidean to
        wrapped Euclidean (torus geometry).
        """
        return Ops._binary_op_inverse_impl(
            result, W, Ops.disjunction, monotonic, unit_ball=unit_ball)

    @staticmethod
    def _binary_op_inverse_impl(result, W, op, monotonic, unit_ball=False):
        """Search codebook for pair (cb[i], cb[j]) whose op(cb[i], cb[j]) ~= result.

        Returns cb[i] (the left operand) for each position in result.
        result shape: (..., D).  W (codebook) shape: (K, D).

        Under ``unit_ball=True``, the per-pair distance is computed on
        the wrapped delta (``(d + 1) mod 2 - 1``) so equally-near
        codebook entries on either side of the periodic boundary are
        treated symmetrically.
        """
        if W is None or W.shape[0] == 0:
            return result

        K, D = W.shape
        flat = result.reshape(-1, D)
        N = flat.shape[0]

        cb_i = W.unsqueeze(1).expand(K, K, D)
        cb_j = W.unsqueeze(0).expand(K, K, D)
        composed = op(cb_i, cb_j, monotonic=monotonic)
        composed_flat = composed.reshape(K * K, D)

        chunk_size = max(1, min(N, 2048 // K))
        best_i = torch.empty(N, dtype=torch.long, device=result.device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            diffs = (flat[start:end].unsqueeze(1) - composed_flat.unsqueeze(0))
            if unit_ball:
                diffs = torch.remainder(diffs + 1.0, 2.0) - 1.0
            dists = diffs.pow(2).sum(dim=-1)
            pair_idx = dists.argmin(dim=-1)
            best_i[start:end] = pair_idx // K

        return W[best_i].reshape(result.shape)

    # ---- In-space algebra (lift / lower) ---------------------------------
    # Unified synthesis / analysis dispatchers (Logic.md §8).
    #     lift  = synthesis = many → one (∨), default mode='OR'
    #     lower = analysis  = one → many (∧), default mode='AND'
    # Mode dispatch covers AND / OR / NOT; mode='NOT' is self-inverse.
    # Region operands are 2-tuples (lower, upper); points auto-promote to
    # degenerate regions containing the origin when the body is the region
    # form. The codebook-search inverse routes through Basis (which has
    # access to the codebook W); a standalone Ops caller can use
    # Ops.liftReverse / Ops.lowerReverse for the analytic inverse on the
    # smoothed point body.

    @staticmethod
    def _as_region(x):
        """Promote a point to a degenerate region containing the origin."""
        if isinstance(x, tuple) and len(x) == 2:
            return x
        zero = torch.zeros_like(x)
        return torch.minimum(x, zero), torch.maximum(x, zero)

    @staticmethod
    def top2_select_ste(x, dim=-1):
        """Hard top-2-by-magnitude selection with straight-through gradient.

        Forward: keep only the two largest |x| entries along ``dim``;
        send the rest to 0 (the neutral element for both PiLayer's
        multiplicative AND fold and SigmaLayer's additive OR fold).
        Backward: identity through the mask -- gradient flows to every
        input as if no selection had occurred, so all candidates can
        learn even when they aren't currently in the top-2.

        Used by PiLayer.forward / SigmaLayer.forward when
        ``binary=True`` to realize the 2-operand specialization of the
        N-ary lattice op (binary grammar rules combine exactly two
        constituents; the long tail of latent candidates drops to the
        op's identity element).
        """
        n = x.shape[dim]
        if n <= 2:
            return x
        _, idx = torch.topk(x.abs(), k=2, dim=dim)
        mask = torch.zeros_like(x)
        mask.scatter_(dim, idx, 1.0)
        x_hard = x * mask
        return x + (x_hard - x).detach()

    @staticmethod
    def _lift_kernel(X1, X2=None, mode=_NO_MODE, kind='strict', inverse=False,
             monotonic=False):
        """Synthesis dispatcher: many → one (∨).  Default mode='OR'.

        Forward bodies (point):
            mode='OR'  kind='strict' (default): torch.max(X1, X2)
                       kind='smooth': (X1 + X2) / 2  (arithmetic mean)
                       kind='radial': RadMax — same-sign max magnitude with
                                      zero passthrough (Ops._radmax body)
                       region: (min ℓ, max u) with point auto-promotion
                               (region body unchanged by kind)
            mode='NOT' Ops._negation_kernel(X1, monotonic=monotonic) (self-inverse)
            mode='AND' routes to Ops._lower_kernel(X1, X2, mode='AND', kind=kind)
                       for symmetry

        Inverse (inverse=True):
            mode='OR'  codebook-search via Basis (raises here without W)
            mode='NOT' self-inverse — same as forward
            mode='AND' routes to Ops._lower_kernel(..., inverse=True)

        Legacy positional form Ops._lift_kernel(left, right) (no mode kwarg) was
        the elementwise product.  It emits a DeprecationWarning and
        forwards to Ops._lower_kernel(left, right, mode='AND', kind='smooth'),
        which produces the same elementwise product bit-for-bit.  The
        warning is permanent — the legacy body stays available for
        callers that have not migrated.
        """
        if mode is _NO_MODE:
            if X2 is not None and not inverse:
                warnings.warn(
                    "Ops._lift_kernel(left, right) is the *analysis* product; "
                    "use Ops._lower_kernel(x, y, mode='AND') for the synthesis / "
                    "analysis polarity.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return Ops._lower_kernel(X1, X2, mode='AND', kind='smooth',
                                 monotonic=monotonic)
            mode = 'OR'

        if mode == 'NOT':
            return Ops._negation_kernel(X1, monotonic=monotonic)
        if mode == 'AND':
            return Ops._lower_kernel(
                X1, X2, mode='AND', kind=kind, inverse=inverse,
                monotonic=monotonic,
            )
        if mode == 'OR':
            if inverse:
                raise NotImplementedError(
                    "Ops._lift_kernel(..., mode='OR', inverse=True) requires a "
                    "codebook W; use Basis.lift(..., mode='OR', inverse=True) "
                    "or Ops.disjunctionReverse(result, y, W, ...) directly."
                )
            if isinstance(X1, tuple) or isinstance(X2, tuple):
                l1, u1 = Ops._as_region(X1)
                l2, u2 = Ops._as_region(X2)
                return torch.minimum(l1, l2), torch.maximum(u1, u2)
            if kind == 'strict':
                return torch.max(X1, X2)
            if kind == 'smooth':
                return (X1 + X2) / 2
            if kind == 'radial':
                return Ops._radmax(X1, X2)
            raise ValueError(f"Ops.lift: unknown kind {kind!r}")
        raise ValueError(f"Ops.lift: unknown mode {mode!r}")

    @staticmethod
    def liftReverse(result, right):
        """Analytic inverse of the legacy lift body (elementwise product).

        Pairs with the legacy Ops._lift_kernel(left, right) = left * right body
        that is now delivered through the new Ops._lower_kernel(mode='AND').
        Recovers X1 from result = X1 * right as result / (right +
        epsilon).  Permanent — see Ops.liftReverseAll for the multi-
        return Step 6 form.
        """
        return result / (right + epsilon)

    @staticmethod
    def liftReverseAll(Y, W=None, monotonic=False):
        """Multi-return reverse for the lift dispatcher (Step 6).

        Pairs with the parent plan's Layer-2.5 grammar convention
        ``X1, X2 = liftReverse(Y)``.  For mode='OR' (the analysis-of-
        synthesis direction) the inverse is the codebook-search
        ``Ops.disjunctionReverse``; with W supplied, returns the pair
        ``(recovered_left, recovered_right)`` where recovered_right is
        the search-conditioning operand and recovered_left is the best-
        matching codebook witness.  Without W, falls back to the legacy
        analytic single-operand form by returning ``(Y, Y)`` so callers
        get a tuple of the expected shape; the second slot is the
        identity placeholder.

        This is the new multi-return convention; the existing 2-arg
        ``Ops.liftReverse(result, right)`` remains for analytic-inverse
        callers (it returns a single tensor, not a tuple).  See parent
        plan §Step 6 lines 577–620 and the Step 6 handoff §Risks
        ("Multi-return reverse changes call shapes") for the migration
        path.
        """
        if W is None or (hasattr(W, 'shape') and W.shape[0] == 0):
            return (Y, Y)
        recovered = Ops.disjunctionReverse(Y, Y, W, monotonic=monotonic)
        return (recovered, Y)

    @staticmethod
    def _lower_kernel(X1, X2=None, mode=_NO_MODE, kind='strict', inverse=False,
              monotonic=False):
        """Analysis dispatcher: one → many (∧).  Default mode='AND'.

        Forward bodies (point):
            mode='AND' kind='strict' (default): torch.min(X1, X2)
                       kind='smooth': X1 * X2  (elementwise product)
                       kind='radial': RadMin — same-sign min magnitude with
                                      zero collapse (Ops._radmin body)
                       region: (max ℓ, min u) with point auto-promotion
                               (region body unchanged by kind)
            mode='NOT' Ops._negation_kernel(X1, monotonic=monotonic) (self-inverse)
            mode='OR'  routes to Ops._lift_kernel(X1, X2, mode='OR', kind=kind) for symmetry

        Inverse (inverse=True):
            mode='AND' codebook-search via Basis (raises here without W)
            mode='NOT' self-inverse — same as forward
            mode='OR'  routes to Ops._lift_kernel(..., inverse=True)

        Legacy positional form Ops._lower_kernel(left, right) (no mode kwarg)
        was the arithmetic mean.  It emits a DeprecationWarning and
        forwards to Ops._lift_kernel(left, right, mode='OR', kind='smooth'),
        which produces the same arithmetic mean bit-for-bit.  The
        warning is permanent — the legacy body stays available for
        callers that have not migrated.
        """
        if mode is _NO_MODE:
            if X2 is not None and not inverse:
                warnings.warn(
                    "Ops._lower_kernel(left, right) is the *synthesis* mean; "
                    "use Ops._lift_kernel(x, y, mode='OR') for the synthesis / "
                    "analysis polarity.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return Ops._lift_kernel(X1, X2, mode='OR', kind='smooth',
                                monotonic=monotonic)
            mode = 'AND'

        if mode == 'NOT':
            return Ops._negation_kernel(X1, monotonic=monotonic)
        if mode == 'OR':
            return Ops._lift_kernel(
                X1, X2, mode='OR', kind=kind, inverse=inverse,
                monotonic=monotonic,
            )
        if mode == 'AND':
            if inverse:
                raise NotImplementedError(
                    "Ops._lower_kernel(..., mode='AND', inverse=True) requires a "
                    "codebook W; use Basis.lower(..., mode='AND', inverse=True) "
                    "or Ops.conjunctionReverse(result, y, W, ...) directly."
                )
            if isinstance(X1, tuple) or isinstance(X2, tuple):
                l1, u1 = Ops._as_region(X1)
                l2, u2 = Ops._as_region(X2)
                return torch.maximum(l1, l2), torch.minimum(u1, u2)
            if kind == 'strict':
                return torch.min(X1, X2)
            if kind == 'smooth':
                return X1 * X2
            if kind == 'radial':
                return Ops._radmin(X1, X2)
            raise ValueError(f"Ops.lower: unknown kind {kind!r}")
        raise ValueError(f"Ops.lower: unknown mode {mode!r}")

    @staticmethod
    def lowerReverse(result, right):
        """Analytic inverse of the legacy lower body (arithmetic mean).

        Pairs with the legacy Ops._lower_kernel(left, right) = (left + right) / 2
        body that is now delivered through the new Ops._lift_kernel(mode='OR').
        Recovers X1 from result = (X1 + right) / 2 as 2 * result -
        right.  Permanent — see Ops.lowerReverseAll for the multi-
        return Step 6 form.
        """
        return 2 * result - right

    @staticmethod
    def lowerReverseAll(Y, W=None, monotonic=False):
        """Multi-return reverse for the lower dispatcher (Step 6).

        Pairs with the parent plan's Layer-2.5 grammar convention
        ``X1, X2 = lowerReverse(Y)``.  For mode='AND' (the synthesis-of-
        analysis direction) the inverse is the codebook-search
        ``Ops.conjunctionReverse``; with W supplied, returns the pair
        ``(recovered_left, recovered_right)`` where recovered_right is
        the search-conditioning operand and recovered_left is the best-
        matching codebook witness.  Without W, falls back to the legacy
        analytic single-operand form by returning ``(Y, Y)`` so callers
        get a tuple of the expected shape.

        See ``Ops.liftReverseAll`` for the dual and the parent plan
        §Step 6 lines 577–620 for the convention.
        """
        if W is None or (hasattr(W, 'shape') and W.shape[0] == 0):
            return (Y, Y)
        recovered = Ops.conjunctionReverse(Y, Y, W, monotonic=monotonic)
        return (recovered, Y)

    # ---- Axis selectors (what / where / when) ----------------------------
    # The C -> S boundary demux puts content in the canonical
    # [what | where | when] layout; these selectors zero the non-selected
    # blocks. When ``x.ndim < 3`` the block structure isn't accessible
    # (compose() in scalar mode), so selectors degenerate to identity.

    # ---- Metric ----------------------------------------------------------

    @staticmethod
    def distance(x, y, monotonic=False, dim=-1):
        """Distance in [0, 1]. Bitonic: angular. Monotonic: volume-weighted L2.

        Monotonic distance weights each element by max(|x|, |y|) so that
        matching zeros contribute nothing -- zero-volume elements have no
        bearing on parthood."""
        if monotonic:
            w = torch.max(x.abs(), y.abs())
            total_weight = w.sum(dim=dim).clamp(min=epsilon)
            return (w * (x - y) ** 2).sum(dim=dim) / total_weight
        return (1 - F.cosine_similarity(x, y, dim=dim)) / 2

    # ---- Mereology -------------------------------------------------------
    # Parthood is the fundamental mereological operation. Each member of
    # the suite has a vector form (default) and a scalar form (scalar=True).
    #
    # Vector forms (scalar=False):
    #     part(x, y)     = x * (y / ||y||)                     elementwise
    #     whole(x, y)    = (1 - x) * (y / ||y||)               elementwise
    #     equal(x, y)    = part(x, y) * part(y, x)             elementwise
    #     overlap(x, y)  = min(part(x, y), part(y, x))         elementwise
    #     underlap(x, y) = min(whole(x, y), whole(y, x))       elementwise
    #     boundary(x, y) = |part(x, y) - part(y, x)|           elementwise
    #     copart(x, y)   = y - x                               elementwise

    @staticmethod
    def _part_kernel(x, y, monotonic=False, scalar=False):
        """Part of x under y.

        Vector form (default): x * (y / ||y||) -- elementwise projection
        of x into y's unit direction. Returns a tensor shaped like x.

        Scalar form (scalar=True): clipped cosine projection in [0, 1].
            part(x, y) = max(0, x.y) / (||x|| * ||y||)
        Satisfies Boole's contrapositive: part(x, y) = part(-y, -x).
        Empty-set conventions:
            part(empty, y) = 1, part(x, empty) = 0, part(empty, empty) = 1.
        """
        ny_raw = Ops.norm(y)
        ny = ny_raw.clamp(min=epsilon)
        if not scalar:
            return x * (y / ny.unsqueeze(-1))
        nx = Ops.norm(x)
        dot = (x * y).sum(dim=-1)
        clipped = torch.clamp(dot, min=0.0)
        denom = (nx * ny).clamp(min=epsilon)
        score = (clipped / denom).clamp(0.0, 1.0)
        empty_x = nx < epsilon
        empty_y = ny_raw < epsilon
        ones = torch.ones_like(score)
        zeros = torch.zeros_like(score)
        return torch.where(empty_x, ones, torch.where(empty_y, zeros, score))

    @staticmethod
    def whole(x, y, monotonic=False, scalar=False):
        """Whole of x under y: complement of x in y's unit direction.

        Vector form: (1 - x) * (y / ||y||).
        Scalar form: degree to which x contains y, i.e. part(y, x, scalar=True)."""
        if scalar:
            return Ops._part_kernel(y, x, monotonic=monotonic, scalar=True)
        ny = Ops.norm(y).clamp(min=epsilon).unsqueeze(-1)
        return (1.0 - x) * (y / ny)

    @staticmethod
    def _equal_kernel(x, y, monotonic=False, scalar=False):
        """Mutual parthood.

        Vector form: part(x, y) * part(y, x) (elementwise).
        Scalar form partitions [0, 1]:
            equal == 0     -> underlap (disjoint)
            0 < equal < 1  -> overlap  (strictly partial)
            equal == 1     -> identity (perfect mutual parthood)."""
        p_xy = Ops._part_kernel(x, y, monotonic=monotonic, scalar=scalar)
        p_yx = Ops._part_kernel(y, x, monotonic=monotonic, scalar=scalar)
        return p_xy * p_yx

    @staticmethod
    def overlap(x, y, monotonic=False, scalar=False):
        """Overlap.

        Vector form: elementwise min of part(x, y) and part(y, x).
        Scalar form: boolean region indicator 0 < equal(..., scalar=True) < 1."""
        if scalar:
            e = Ops._equal_kernel(x, y, monotonic=monotonic, scalar=True)
            return (e > 0) & (e < 1)
        return torch.minimum(
            Ops._part_kernel(x, y, monotonic=monotonic),
            Ops._part_kernel(y, x, monotonic=monotonic),
        )

    @staticmethod
    def underlap(x, y, monotonic=False, scalar=False):
        """Underlap.

        Vector form: elementwise min of whole(x, y) and whole(y, x).
        Scalar form: boolean region indicator equal(..., scalar=True) == 0."""
        if scalar:
            e = Ops._equal_kernel(x, y, monotonic=monotonic, scalar=True)
            return e == 0
        return torch.minimum(
            Ops.whole(x, y, monotonic=monotonic),
            Ops.whole(y, x, monotonic=monotonic),
        )

    @staticmethod
    def boundary(x, y, monotonic=False, scalar=False):
        """Directional asymmetry of parthood.

        Vector form: |part(x, y) - part(y, x)| (elementwise).
        Scalar form: zero under clipped-cosine (cosine is symmetric)."""
        m = monotonic
        return torch.abs(
            Ops._part_kernel(x, y, monotonic=m, scalar=scalar)
            - Ops._part_kernel(y, x, monotonic=m, scalar=scalar)
        )

    @staticmethod
    def copart(x, y, monotonic=False, scalar=False):
        """Copart of x under y: the part of y not accounted for by x.

        Vector form: y - x.
        Scalar form: 1 - part(x, y, scalar=True), clamped to [0, 1]."""
        if scalar:
            return (1.0 - Ops._part_kernel(x, y, monotonic=monotonic, scalar=True)).clamp(0.0, 1.0)
        return y - x
#endregion

#region Error Functions
class Loss(nn.Module):
    """Base class for loss computation.

    Subclasses override compute() to define how prediction and target
    tensors are compared. The default compute() is MSE.
    """
    def compute(self, pred, target):
        """Compute loss between pred and target. Override in subclasses."""
        return nn.functional.mse_loss(pred, target)
class ModelLoss(Loss):
    """Weighted reconstruction loss with separate scales for what/where/when.

    Combines a forward task-output loss with a forward reconstruction
    loss; the reconstruction term is decomposed into per-modality
    slices so each (what / where / when) modality carries its own
    weight. Used as the canonical training criterion across the
    model factory.

    The blend between output and reconstruction is set by
    ``reconstruction_scale`` (formerly ``reverse_scale``, retired
    alongside the reverse pipeline 2026-05-14).
    """

    def __init__(self, reconstruction_scale=0.5,
                 what_scale=0.7, where_scale=0.2, when_scale=0.1,
                 embedding_scale=0.1,
                 certainty=False, nOutput=2,
                 conceptualOrder=0,
                 nWhere=None, nWhen=None):
        """Initialize ModelLoss; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__()
        self.reconstruction_scale = float(reconstruction_scale or 0.5)
        self.what_scale = float(what_scale or 0.7)
        self.where_scale = float(where_scale or 0.2)
        self.when_scale = float(when_scale or 0.1)
        self.embedding_scale = float(embedding_scale or 0.1)
        # Resolve nWhere: prefer explicit arg, then architecture-level,
        # then fall back to InputSpace-level (per-space nWhere overrides arch).
        if nWhere is not None and nWhere > 0:
            self.nWhere = nWhere
        else:
            arch_nw = TheXMLConfig.get("architecture.nWhere")
            if arch_nw and arch_nw > 0:
                self.nWhere = arch_nw
            else:
                try:
                    self.nWhere = TheXMLConfig.space("InputSpace", "nWhere")
                except KeyError:
                    self.nWhere = 0
        if nWhen is not None and nWhen > 0:
            self.nWhen = nWhen
        else:
            arch_nn = TheXMLConfig.get("architecture.nWhen")
            if arch_nn and arch_nn > 0:
                self.nWhen = arch_nn
            else:
                try:
                    self.nWhen = TheXMLConfig.space("InputSpace", "nWhen")
                except KeyError:
                    self.nWhen = 0

        if certainty:
            self.output_criterion = CertaintyWeightedCrossEntropy()
        elif nOutput <= 2:
            self.output_criterion = nn.MSELoss()
        elif conceptualOrder > 0:
            self.output_criterion = nn.MSELoss()
        else:
            self.output_criterion = nn.CrossEntropyLoss()

    def output(self, pred, target):
        """Output.
        
        See class docstring for the operation contract.
        """
        return self.output_criterion(pred, target)

    def compute(self, pred, target):
        """Per-slot MSE with what/where/when weighting (legacy).

        Used when nWhere == 0 (no positional encoding).  When nWhere > 0,
        the training loop calls ``compute_piecewise`` instead.
        """
        embSize = pred.shape[-1]
        nWhere = self.nWhere
        nWhen = self.nWhen
        nWhat = embSize - nWhere - nWhen

        loss = pred.new_tensor(0.0)
        if nWhat > 0:
            loss = loss + self.what_scale * F.mse_loss(
                pred[..., :nWhat], target[..., :nWhat])
        if nWhere > 0:
            loss = loss + self.where_scale * F.mse_loss(
                pred[..., nWhat:nWhat + nWhere], target[..., nWhat:nWhat + nWhere])
        if nWhen > 0:
            loss = loss + self.when_scale * F.mse_loss(
                pred[..., nWhat + nWhere:], target[..., nWhat + nWhere:])
        return loss

    def compute_masked(self, pred, target, mask):
        """Sync-free equivalent of ``compute(pred[mask], target[mask])``.

        ``pred`` / ``target``: ``[B, K, *]``; ``mask``: ``[B, K]`` bool.
        Boolean-mask gather (``pred[mask]``) is data-dependent -- its
        row count must be copied to the host, an implicit
        cudaMemcpyDtoH that breaks CUDA-graph capture. This computes the
        identical per-segment weighted-MSE *mean over masked positions*
        with on-device reductions only (masked-sum / (mask_count *
        seg_width) == the gather-then-``F.mse_loss`` mean). ``mask_count``
        stays a device scalar (never ``.item()``-ed). Empty mask -> 0.0
        with no 0/0 (cleaner than, and replacing, the old
        ``nan_to_num`` gate). Matches ``compute``'s ``embSize =
        pred.shape[-1]`` by first clipping to the shared last dim, as
        the call sites did via ``[:, :D]``. fp reduction order differs
        from the gather form (ULP-level), not the value.
        """
        D = min(pred.shape[-1], target.shape[-1])
        p = pred[..., :D]
        t = target[..., :D]
        nWhere = self.nWhere
        nWhen = self.nWhen
        nWhat = D - nWhere - nWhen

        se = (p - t) ** 2                          # [B, K, D]
        mf = mask.to(se.dtype).unsqueeze(-1)       # [B, K, 1]
        mcount = mf.sum()                          # device scalar

        def _seg(a, b):
            denom = (mcount * float(b - a)).clamp(min=1.0)
            return (se[..., a:b] * mf).sum() / denom

        loss = p.new_zeros(())
        if nWhat > 0:
            loss = loss + self.what_scale * _seg(0, nWhat)
        if nWhere > 0:
            loss = loss + self.where_scale * _seg(nWhat, nWhat + nWhere)
        if nWhen > 0:
            loss = loss + self.when_scale * _seg(nWhat + nWhere, D)
        return loss

    def compute_piecewise(self, pred, target):
        """Piecewise reconstruction loss via Chamfer distance.

        Instead of per-slot MSE(pred[v], target[v]), uses bidirectional
        nearest-neighbour matching:

        - **Accuracy**: each predicted token matches its nearest original.
        - **Coverage**: each original token is covered by some prediction.

        This handles token reordering (the grammar pair-merge may swap
        vector positions) and eliminates error shadowing when tokens
        overlap in position space.  The what/where/when component weights
        are applied before computing L2 distances so the matching respects
        the relative importance of content vs position vs time.
        """
        embSize = pred.shape[-1]
        nWhere = self.nWhere
        nWhen = self.nWhen
        nWhat = embSize - nWhere - nWhen

        # Build per-dim weight vector: sqrt so that L2 distance^2 = weighted MSE
        w = pred.new_ones(embSize)
        if nWhat > 0:
            w[:nWhat] = self.what_scale
        if nWhere > 0:
            w[nWhat:nWhat + nWhere] = self.where_scale
        if nWhen > 0:
            w[nWhat + nWhere:] = self.when_scale
        w = w.sqrt()

        p = pred * w
        t = target * w

        # Ensure 3-D [B, N, D]
        if p.dim() == 2:
            p = p.unsqueeze(0)
            t = t.unsqueeze(0)

        # Pairwise squared L2 distances: [B, N_pred, N_target]
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a*b)   (MPS-safe, no cdist)
        p_sq = (p * p).sum(dim=-1, keepdim=True)           # [B, N, 1]
        t_sq = (t * t).sum(dim=-1, keepdim=True)           # [B, N, 1]
        dot = torch.bmm(p, t.transpose(1, 2))              # [B, N, N]
        dists_sq = (p_sq - 2 * dot + t_sq.transpose(1, 2)).clamp(min=0)

        # Accuracy: for each predicted token, squared distance to nearest original
        accuracy = dists_sq.min(dim=2).values.mean()
        # Coverage: for each original token, squared distance to nearest prediction
        coverage = dists_sq.min(dim=1).values.mean()

        return accuracy + coverage

    def forward(self, lossOut, lossIn=None, sbow=None):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        total = lossOut
        if lossIn is not None:
            # A non-finite reconstruction loss (Inf/NaN) is *divergence*,
            # not a recoverable edge case -- it must fail LOUD, never be
            # silently masked (nan_to_num / on-device gate) and left to
            # corrupt the run. (See memory: fail loud on numerical
            # divergence.) The prior ``bool(isfinite.all())`` gate did a
            # per-step cudaMemcpyDtoH. ``torch._assert_async`` keeps the
            # guarantee without that host sync: on CUDA the assertion is
            # enqueued on the stream (no DtoH) and a non-finite loss
            # aborts the process at the next kernel launch -- unmissable,
            # never a silent success; on CPU it asserts immediately
            # (tests still catch divergence). Device handling is
            # internal to the op, so no device-type branching here.
            torch._assert_async(torch.isfinite(lossIn).all())
            rr = self.reconstruction_scale
            total = (1 - rr) * lossOut + rr * lossIn
        if sbow is not None:
            total = total + self.embedding_scale * sbow
        return total

    def total(self, lossOut, lossIn=None, sbow=None):
        """Total.
        
        See class docstring for the operation contract.
        """
        return self(lossOut, lossIn, sbow)

class Error:
    """Central registry for per-batch error/loss terms.

    ``Error`` is a bookkeeping client of ``Loss``: individual sites still
    compute their pred-vs-target comparisons via a ``Loss`` instance (or
    any other path that produces a scalar tensor), and then register the
    result here with a name, weight, originating space, and category.

    Why a registry? There are currently 12+ loss terms accumulated across
    four different call sites (``ModelLoss``, ``BasicModel.runBatch``,
    ``SymbolicSpace.accumulate_symbol_objective``, ``WordSpace.truth_modulated_loss``).
    Debugging convergence problems used to require grepping each site to
    answer "what fraction of today's gradient came from which term?".
    The registry makes that a one-call breakdown, and supports:

      * ``.total()``      -- weighted sum for backprop
      * ``.breakdown()``  -- per-term scalars for logging
      * ``.snapshot()`` + ``.covariance()`` -- running covariance across
                            batches, so you can detect terms that fight
                            each other (anti-correlation) or that carry
                            no signal (zero variance).
      * ``.disable(cat)`` / ``.enable(cat)`` -- one-line ablation by
                            category (``"reconstruction"``, ``"symbol"``,
                            ``"truth"``, ``"discourse"``, ``"embedding"``,
                            ``"prediction"``).

    ``Error`` never enforces specific math -- the caller decides how each
    term is computed and chooses its weight (usually from a config knob).
    The class just collects, sums, and reports.

    Usage pattern inside ``runBatch``:

        TheError.reset()
        TheError.compute("reconstruction", pred, target,
                          method="compute", weight=self.loss.reconstruction_scale,
                          space="InputSpace", category="reconstruction")
        TheError.add("symbol_residual", sym_term, weight=1.0,
                      space="SymbolicSpace", category="symbol")
        total = TheError.total()          # for backprop
        TheError.snapshot()                # record for covariance
    """

    _CATEGORIES = (
        "reconstruction", "prediction", "symbol",
        "truth", "discourse", "embedding", "other",
    )

    def __init__(self, loss: Loss = None, history_max: int = 1024):
        """Initialize Error; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        self._loss: Loss = loss
        self._terms: dict = {}   # name -> {weight, value, space, category, count}
        self._history: list = []  # each entry is {name: weighted_scalar}
        self._history_max = int(history_max)
        self._disabled: set = set()

    # ---- setup ---------------------------------------------------------

    def attach(self, loss: Loss):
        """Associate a ``Loss`` instance for ``compute()`` delegation.

        Optional: callers may use ``.add()`` directly and skip ``attach``.
        """
        self._loss = loss

    def reset(self):
        """Clear the per-batch term store.  Preserves history."""
        self._terms.clear()

    def clear(self):
        """Alias for ``reset``; used by per-subspace transit accumulators.

        When an ``Error`` instance is attached to a ``SubSpace`` as a
        pipeline-carried auxiliary-loss sink, ``runBatch`` calls
        ``clear()`` after harvesting its terms into ``TheError`` so the
        next forward pass starts fresh.
        """
        self._terms.clear()

    def terms(self):
        """Return the current terms as a list of 5-tuples.

        Each tuple is ``(name, tensor, weight, space, category)``.  Same-name
        contributions have been summed at ``add()`` time, so each unique
        name appears once.  Used by ``runBatch`` to fold per-subspace
        accumulators into the module-level ``TheError`` registry while
        preserving tensor references for autograd.
        """
        return [
            (name, rec["value"], rec["weight"], rec["space"], rec["category"])
            for name, rec in self._terms.items()
        ]

    def disable(self, category: str):
        """Exclude all terms of this category from ``.total()``."""
        if category not in self._CATEGORIES:
            warnings.warn(f"Error.disable: unknown category {category!r}; "
                          f"will still work but consider adding it to "
                          f"Error._CATEGORIES for consistency.")
        self._disabled.add(category)

    def enable(self, category: str):
        """Enable.
        
        See class docstring for the operation contract.
        """
        self._disabled.discard(category)

    @property
    def disabled_categories(self):
        """Disabled categories.
        
        See class docstring for the operation contract.
        """
        return frozenset(self._disabled)

    # ---- accumulation --------------------------------------------------

    def add(self, name: str, value, *, weight: float = 1.0,
            space: str = None, category: str = "other"):
        """Register a pre-computed scalar term.

        Repeated ``name`` values are summed (useful when a term is
        contributed from multiple layers).  ``value`` may be ``None``
        (the call becomes a no-op) or a scalar tensor.
        """
        if value is None:
            return
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        rec = self._terms.get(name)
        if rec is None:
            self._terms[name] = {
                "weight": float(weight),
                "value": value,
                "space": space,
                "category": category,
                "count": 1,
            }
        else:
            rec["value"] = rec["value"] + value
            rec["count"] += 1

    def compute(self, name: str, pred, target, *,
                weight: float = 1.0, method: str = "compute",
                space: str = None, category: str = "reconstruction"):
        """Compute a term via the attached ``Loss`` instance, then register it.

        ``method`` names a ``Loss`` method (``"compute"``,
        ``"compute_piecewise"``, ``"output"``, or a subclass extension).
        Returns the raw (unweighted) loss tensor for convenience -- the
        registry stores the unweighted value and applies ``weight`` at
        ``.total()`` time.
        """
        if self._loss is None:
            raise RuntimeError(
                "Error.compute() requires an attached Loss instance; "
                "call TheError.attach(model_loss) first."
            )
        fn = getattr(self._loss, method, None)
        if fn is None:
            raise AttributeError(
                f"Error.compute: Loss has no method {method!r}")
        value = fn(pred, target)
        self.add(name, value, weight=weight, space=space, category=category)
        return value

    # ---- aggregation / inspection --------------------------------------

    def total(self):
        """Return the weighted sum of all enabled terms (or ``None``)."""
        if not self._terms:
            return None
        total = None
        for rec in self._terms.values():
            if rec["category"] in self._disabled:
                continue
            contrib = rec["weight"] * rec["value"]
            total = contrib if total is None else total + contrib
        return total

    def breakdown(self):
        """Per-term snapshot keyed by name.

        Each entry is ``{weight, value, weighted, space, category, count}``
        where ``value`` and ``weighted`` are Python floats when the term
        is a scalar, or ``None`` for multi-element tensors.
        """
        out = {}
        for name, rec in self._terms.items():
            v_tensor = rec["value"]
            if isinstance(v_tensor, torch.Tensor) and v_tensor.numel() == 1:
                value_f = float(v_tensor.detach().item())
            else:
                value_f = None
            out[name] = {
                "weight": rec["weight"],
                "value": value_f,
                "weighted": rec["weight"] * value_f if value_f is not None else None,
                "space": rec["space"],
                "category": rec["category"],
                "count": rec["count"],
            }
        return out

    def snapshot(self):
        """Record the current weighted breakdown for covariance analysis.

        Only scalar terms are recorded; multi-element tensors are
        skipped (they contribute to ``.total()`` via backprop but not
        to the history).
        """
        snap = {}
        for name, rec in self._terms.items():
            v = rec["value"]
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                snap[name] = rec["weight"] * float(v.detach().item())
        if snap:
            self._history.append(snap)
            if len(self._history) > self._history_max:
                self._history.pop(0)

    def covariance(self, n_steps: int = None):
        """Running covariance of weighted term values across recent snapshots.

        Returns ``{"names": [...], "cov": tensor[T,T]}`` where T is the
        number of terms that appeared in the selected window.  Any term
        absent from a given snapshot is treated as zero for that step.
        """
        if not self._history:
            return {"names": [], "cov": torch.zeros(0, 0)}
        hist = self._history if n_steps is None else self._history[-n_steps:]
        names = sorted({k for s in hist for k in s.keys()})
        if not names:
            return {"names": [], "cov": torch.zeros(0, 0)}
        mat = torch.zeros(len(hist), len(names))
        for i, s in enumerate(hist):
            for j, n in enumerate(names):
                mat[i, j] = s.get(n, 0.0)
        centered = mat - mat.mean(dim=0, keepdim=True)
        denom = max(len(hist) - 1, 1)
        cov = (centered.transpose(0, 1) @ centered) / denom
        return {"names": names, "cov": cov}

    def format_breakdown(self) -> str:
        """Single-line summary of all terms, ordered by weighted magnitude."""
        bd = self.breakdown()
        rows = []
        for name, entry in bd.items():
            w = entry["weighted"]
            if w is None:
                continue
            rows.append((abs(w), name, entry["weight"], entry["value"], w,
                         entry["category"]))
        rows.sort(reverse=True)
        parts = [
            f"{name}[{cat}]={val:.4g}*{wt:.4g}={wval:.4g}"
            for _, name, wt, val, wval, cat in rows
        ]
        return " | ".join(parts) if parts else "<empty>"

# Module-level singleton.  Callers do ``from Layers import TheError`` and
# then ``TheError.reset() / .add(...) / .total()``.  A single ``Error``
# instance is kept at module scope so every space and every loss term
# registers into the same bookkeeping store per process.
TheError = Error()

class CertaintyWeightedMAELoss(Loss):
    """MAE loss weighted by prediction magnitude (certainty).

    High-magnitude (confident) predictions are penalized more when wrong,
    blended with unweighted MAE via ``alpha``.
    """
    def __init__(self, alpha=0.5):
        """Initialize CertaintyWeightedMAELoss; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        abs_error = torch.abs(targets - predictions)
        certainty = torch.abs(predictions)
        loss = abs_error * (self.alpha * certainty + (1 - self.alpha))
        return torch.mean(loss)
class CertaintyWeightedMSELoss(Loss):
    """MSE loss weighted by prediction magnitude (certainty).

    Hybrid of certainty-weighted MSE and plain MSE, blended by ``alpha``.
    ``alpha=0`` collapses to plain MSE; ``alpha=1`` is purely weighted.
    """
    def __init__(self, alpha=0.5):
        """Initialize CertaintyWeightedMSELoss; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        certainty = outputs.abs().sum(dim=1)                # per-sample confidence
        mse_loss = ((outputs - targets) ** 2).sum(dim=1)
        cw_mse_loss = mse_loss * certainty
        hybrid_loss = self.alpha * cw_mse_loss.mean() + (1 - self.alpha) * mse_loss.mean()
        return hybrid_loss
class CertaintyWeightedCrossEntropy(Loss):
    """Cross-entropy weighted by predicted probability of the true class.

    Hybrid of certainty-weighted CE and plain CE, blended by ``alpha``.
    ``alpha=0`` collapses to plain CE; ``alpha=1`` is purely weighted.
    ``epsilon`` floors log inputs to keep gradients finite.
    """
    def __init__(self, alpha=0.5, epsilon=1e-8):
        """Initialize CertaintyWeightedCrossEntropy; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # If targets are one-hot, convert to indices
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        if targets.dim() == 2 and targets.size(1) == logits.size(1):
            targets = targets.argmax(dim=1)
        # Ensure targets are int64 and on the same device as logits
        targets = targets.to(dtype=torch.long, device=logits.device)
        batch_size = logits.size(0)
        batch_indices = torch.arange(batch_size, device=logits.device)

        log_probs = F.log_softmax(logits, dim=1)  # [batch, num_classes]
        ce_loss = -log_probs[batch_indices, targets]
        #norm_logits = torch.norm(logits, dim=1)
        #cwce_loss = -norm_logits * log_probs[batch_indices, targets]
        p_true = torch.exp(log_probs[batch_indices, targets])
        cwce_loss = -p_true * log_probs[batch_indices, targets]  # possibly erroneous
        hybrid_loss = self.alpha * cwce_loss + (1 - self.alpha) * ce_loss
        #norm = torch.norm(hybrid_loss, p=2)
        return hybrid_loss.mean()
#endregion

#region Various kinds of memory
class Mem:
    """Base class for temporal memory filters (exponential, gamma, mean, etc.).

    Subclasses implement ``delta()`` to update internal state from a new
    observation.  ``get()`` returns the current filtered output.  The
    ``removeRC``/``insertRC``/``setRC`` helpers support dynamic resizing
    of the output matrix (1-indexed for legacy compatibility).
    """
    def __init__(self, sz=None):
        """Initialize Mem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        self.lr = 0.01
        self.nTrials = 0
        self.output = None
        self.reset(sz)

    def delta(self, *args):
        """Increment nTrials by one. Extra arguments are ignored."""
        self.nTrials += 1
    def get(self):
        """Return the current output array."""
        return self.output
    def set(self, in_val):
        """Set the output array to a new value."""
        self.output = in_val
    def reset(self, sz=None):
        """
        Reset the output to a zeros array and nTrials to 0.
        If sz is not provided, use the current output's shape (or (0, 0) if undefined).
        """
        if sz is None:
            if self.output is None:
                sz = (0, 0)
            else:
                sz = self.output.shape
        self.output = torch.zeros(sz, device=TheDevice.get())
        self.nTrials = 0
    def removeRC(self, r=None, c=None):
        """
        Remove a row and/or column from the output.
        The row (r) and column (c) indices are assumed to be given in MATLAB 1-indexed form.
        """
        if r is not None:
            self.output = np.delete(self.output, r - 1, axis=0)
        if c is not None:
            self.output = np.delete(self.output, c - 1, axis=1)
    def insertRC(self, r=None, c=None):
        """
        Insert a row and/or column of zeros into the output.
        The indices r and c are assumed to be 1-indexed.
        """
        if r is not None:
            new_row = np.zeros((1, self.output.shape[1]))
            self.output = np.insert(self.output, r - 1, new_row, axis=0)
        if c is not None:
            new_col = np.zeros((self.output.shape[0], 1))
            self.output = np.insert(self.output, c - 1, new_col, axis=1)
    def setRC(self, r, c, val):
        """
        Set the (r, c) element to val and its symmetric element (c, r) to val.
        Indices are assumed to be 1-indexed.
        """
        self.output[r - 1, c - 1] = val
        self.output[c - 1, r - 1] = val

    @staticmethod
    def test():
        """
        Test method that creates instances of several Mem-derived classes
        and calls their testImpulse method.
        """
        names = ['ExponentialMem', 'GammaMem', 'MeanMem']
        for name in names:
            # Create an instance using globals() (similar to MATLAB's feval)
            m = globals()[name]()
            Mem.testImpulse(m, name)

    @staticmethod
    def testOne(m):
        """
        Test one impulse by incrementing and plotting the output.
        """
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        x = np.arange(1, 1001)
        y = np.zeros(1000)
        m.delta(1, 1)  # extra parameters are ignored
        for i in range(len(x)):
            y[i] = m.get()
            m.delta(1, 1)
        plt.plot(x, y)
        plt.title("Test One")
        plt.show(block=False)

    @staticmethod
    def testImpulse(m, name):
        """
        Test impulse response by incrementing and plotting the output.
        """
        import matplotlib.pyplot as plt
        plt.figure(1)
        x = np.arange(1, 1001)
        y = np.zeros(1000)
        m.delta(1.0)
        for i in range(len(x)):
            y[i] = m.get()
            m.delta(0)
        plt.plot(x, y)
        plt.title(f"Impulse Response: {name}")
        plt.show(block=False)
# ZOHMem subclass: Zero-Order Hold memory.
class ZOHMem(Mem):
    """Zero-Order Hold memory: ``output`` is whatever was last written.

    Simplest memory primitive in the family -- every ``delta(in1)``
    call overwrites ``self.output`` with ``in1``, ignoring trial count.
    Useful as a baseline / control in memory comparisons.
    """
    def __init__(self, sz=1):
        """Initialize ZOHMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)

    def delta(self, in1, in2=None):
        # Call the base class delta and then set output to in1.
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        self.output = in1
# StateMem subclass: adds a 'state' property.
class StateMem(Mem):
    """Memory primitive with a parallel ``state`` tensor alongside ``output``.

    Subclasses that need a recurrent / hidden state (RLSMem, GammaMem)
    inherit from this. ``reset`` / ``removeRC`` / ``insertRC`` /
    ``setRC`` all mirror their changes into the state tensor so it
    stays shape-aligned with output.
    """
    def __init__(self, sz=1):
        """Initialize StateMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)
        self.state = None
        self.reset(sz)

    def reset(self, sz=None):
        """
        Reset output and state to zeros and nTrials to 0.
        """
        if sz is None:
            if self.output is None:
                sz = (0, 0)
            else:
                sz = self.output.shape
        super().reset(sz)
        self.state = torch.zeros(sz, device=TheDevice.get())

    def delta(self, *args):
        # Just call the base class delta.
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta(*args)

    def removeRC(self, r=None, c=None):
        """
        Remove a row/column from both output and state.
        """
        super().removeRC(r, c)
        if r is not None:
            self.state = np.delete(self.state, r - 1, axis=0)
        if c is not None:
            self.state = np.delete(self.state, c - 1, axis=1)

    def insertRC(self, r=None, c=None):
        """
        Insert a row/column of zeros into both output and state.
        """
        super().insertRC(r, c)
        if r is not None:
            new_row = np.zeros((1, self.state.shape[1]))
            self.state = np.insert(self.state, r - 1, new_row, axis=0)
        if c is not None:
            new_col = np.zeros((self.state.shape[0], 1))
            self.state = np.insert(self.state, c - 1, new_col, axis=1)

    def setRC(self, r, c, val):
        """
        Set the (r,c) element in both output and state (and symmetrically).
        """
        super().setRC(r, c, val)
        self.state[r - 1, c - 1] = val
        self.state[c - 1, r - 1] = val
# RLSMem subclass: Recursive Least Squares memory.
class RLSMem(StateMem):
    """Recursive least-squares memory: ``output += L2(output - in1) * in1``.

    Adaptive update keyed off the residual norm; large errors move the
    output toward ``in1`` more aggressively. ``momLR`` is the optional
    momentum learning rate for the (currently unused) state update path.
    """
    def __init__(self, sz=1):
        """Initialize RLSMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)
        self.momLR = 0.2

    def delta(self, in1, in2=None):
        # Call the base class (Mem) delta.
        """Delta.
        
        See class docstring for the operation contract.
        """
        Mem.delta(self)
        # Compute error using the L2 norm.
        err = np.linalg.norm(self.output - in1, 2)
        self.output = self.output + err * in1
        # Optionally update state (commented out in original code):
        # self.state = self.state + self.momLR * err * in1
        # self.output = self.output + self.state
# ProbMem subclass: probabilistic memory update.
class ProbMem(Mem):
    """Conditional-probability memory: each cell tracks P(in1_r | in2_c).

    On each ``delta(in1, in2)`` step, the per-cell value is updated as a
    running-mean toward +1 / -1 / 0 depending on the sign agreement of
    the corresponding inputs. Encodes positive vs negative co-occurrence.
    """
    def __init__(self, sz=1):
        """Initialize ProbMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)

    def delta(self, in1, in2):
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        # Iterate over the indices of in1 and in2.
        for r in range(len(in1)):
            for c in range(len(in2)):
                # Increase or decrease conditional probability based on activation sign.
                if in2[c] > 0 and in1[r] > 0:
                    self.output[r, c] = ((self.nTrials - 1) / self.nTrials) * self.output[r, c] + (1 / self.nTrials) * 1
                elif in2[c] > 0 and in1[r] < 0:
                    self.output[r, c] = ((self.nTrials - 1) / self.nTrials) * self.output[r, c] + (1 / self.nTrials) * -1
# MeanMem subclass: computes a running mean.
class MeanMem(Mem):
    """Running arithmetic mean of all inputs seen so far.

    ``delta(in1)`` updates ``output`` toward ``in1`` with weight
    ``1 / nTrials``, so the result is the exact unweighted mean of the
    sequence. Bias-free; new samples have diminishing influence.
    """
    def __init__(self, sz=1):
        """Initialize MeanMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)

    def delta(self, in1, in2=None):
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        self.output = ((self.nTrials - 1) / self.nTrials) * self.output + (1 / self.nTrials) * in1
# GammaMem subclass: blends state with output using a second learning rate.
class GammaMem(StateMem):
    """Two-stage exponential filter: state then output.

    ``state`` integrates ``in1`` at rate ``self.lr``; ``output``
    integrates ``state`` at rate ``self.lr2``. Effectively a cascaded
    low-pass with separately tunable bandwidths.
    """
    def __init__(self, sz=1, lr2=0.05):
        """Initialize GammaMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)
        self.lr2 = lr2

    def delta(self, in1, in2=None):
        # Call the StateMem delta method.
        """Delta.
        
        See class docstring for the operation contract.
        """
        StateMem.delta(self)
        self.state = (1 - self.lr) * self.state + self.lr * in1
        self.output = (1 - self.lr2) * self.output + self.lr2 * self.state

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        Mem.testOne(GammaMem())
# ExponentialMem subclass: exponential memory update.
class ExponentialMem(Mem):
    """Exponential-moving-average memory: ``output = (1-lr)*output + lr*in1``.

    Single-rate IIR low-pass; equivalent to MeanMem at lr=1/n but with
    a fixed lr that gives exponentially decaying weight on old samples.
    Default lr inherits from the base ``Mem`` class.
    """
    def __init__(self, sz=1, lr=None):
        """Initialize ExponentialMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)
        if lr is not None:
            self.lr = lr

    def delta(self, in1, in2=None):
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        self.output = (1 - self.lr) * self.output + self.lr * in1
    @staticmethod
    def test():
        # Create an instance of ExponentialMem and run a test.
        """Self-test; verifies the round-trip / invariant."""
        m_exp = ExponentialMem(sz=(5, 5), lr=0.1)
        # Simulate a delta update with an input (for example, a 5x5 array).
        x = np.ones((5, 5))
        m_exp.delta(x)
        x = x = np.zeros((5, 5))
        m_exp.delta(x)
        print("ExponentialMem output after two delta calls:")
        print(m_exp.get())
# CorrMem subclass: correlational memory update.
class CorrMem(Mem):
    """Correlation matrix memory; tracks normalized product of two streams.

    On each ``delta(in1, in2)`` step, per-cell ``output[r, c]`` is
    updated toward ``saturate(in1[r]*in2[c] / (|in1[r]|*|in2[c]|))``
    with weight ``max(|in1[r]|, |in2[c]|) / nTrials`` -- favours
    high-amplitude updates.
    """
    def __init__(self, sz=1):
        """Initialize CorrMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)

    def delta(self, in1, in2):
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        for r in range(len(in1)):
            for c in range(len(in2)):
                val = in1[r] * in2[c]
                # Avoid division by zero.
                denom = np.sqrt(in1[r]**2 * in2[c]**2)
                if denom != 0:
                    val = Ops.saturate(val / denom)
                else:
                    val = 0
                amt = max(abs(in1[r]), abs(in2[c]))
                self.output[r, c] = ((self.nTrials - amt) / self.nTrials) * self.output[r, c] + (amt / self.nTrials) * val
#endregion



# Two distance modes:
#   ``use_cosine_sim=False`` (Euclidean, the default):
#     ``argmin_i ||x - c_i||^2 = argmax_i (x . c_i - 0.5 ||c_i||^2)``
#     implemented as one matmul + cached row-norm subtract. ``||c_i||^2``
#     is held in the ``_b_norms_sq`` buffer and refreshed in the
#     codebook setter and at the end of each EMA update. Same FLOPs as
#     ``torch.cdist``'s mm-trick path, but skips the sqrt and the
#     per-row ``||x||^2`` add (both invariant under argmin over codes).
#     The codebook lives in the [-1, +1] hypercube; magnitude carries
#     information (feature intensity) and there is no codebook-level
#     negation operator, so two patterns differ when their coordinates
#     differ. Right metric for PerceptualSpace and SymbolicSpace.
#
#   ``use_cosine_sim=True`` (dot product, codebook unit-norm):
#     ``indices = (flat @ codebook.T).argmax(dim=-1)``
#     The EMA path renormalizes the codebook to unit L2 each step
#     (see below), so ``a_i`` is unit-norm. For a fixed query ``b``,
#     ``cos(a_i, b) = (a_i . b) / ||b||`` is monotone in ``a_i . b``
#     because ``||b|| >= 0`` is constant across i, so ranking by the
#     raw dot product matches ranking by cosine. The input is NOT
#     normalized -- in ConceptualSpace the input magnitude in
#     [-1, +1] encodes belief certainty (1 = known true, 0 = unknown,
#     -1 = known false) and must be preserved end-to-end.
#
# See doc/Spaces.md "Codebook similarity metric" for the full theory.
class VectorQuantize(nn.Module):
    """Vector-quantization codebook with EMA / cosine / rotation-trick updates.

    Standard VQ-VAE building block: matches each input vector to its
    nearest codebook entry under Euclidean or cosine distance, returns
    a straight-through quantized output, and tracks a commitment loss.
    EMA decay smooths the codebook; dead-code revival rotates unused
    rows toward fresh inputs. See module comment for the two distance
    modes and ``doc/Spaces.md`` "Codebook similarity metric".
    """
    def __init__(
        self,
        dim,
        codebook_size,
        commitment_weight=1.0,
        use_cosine_sim=False,
        decay=0.8,
        threshold_ema_dead_code=0,
        rotation_trick=False,
        eps=1e-5,
        codebook_retire=False,
        **kwargs,
    ):
        """Initialize VectorQuantize; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.use_cosine_sim = use_cosine_sim
        self.decay = float(decay)
        self.threshold_ema_dead_code = int(threshold_ema_dead_code)
        self.rotation_trick = bool(rotation_trick)
        self.eps = float(eps)
        # Gate for the dead-code replacement path. Off by default because
        # reseeding expired rows with fresh samples can blow up the effective
        # number of distinct codes on non-stationary data.
        self.codebook_retire = bool(codebook_retire)
        self.codebook = torch.randn(codebook_size, dim)
        # EMA accumulators used by ``update_ema``. ``cluster_size`` counts
        # how many inputs snap to each code (bootstrapped at 1.0 to match
        # the library default so the first-step smoothed divisor is never
        # zero). ``embed_avg`` is the running sum of the assigned inputs;
        # the codebook is rewritten as ``embed_avg / cluster_size`` (with
        # laplace smoothing) each training step.
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_avg", self.codebook.data.clone())
        # Cached squared L2 norms of each codebook row, refreshed in the
        # codebook setter and at the end of each EMA update. Used by the
        # Euclidean retrieval path to do
        # ``argmin_i ||x - c_i||^2 = argmax_i (x . c_i - 0.5 ||c_i||^2)``
        # via a single matmul instead of ``torch.cdist``: same FLOPs as
        # cdist's mm-trick, but skips the sqrt and the per-row ||x||^2 add
        # (both invariant under argmin over codes).
        self.register_buffer(
            "_b_norms_sq",
            (self.codebook.data ** 2).sum(dim=-1),
        )
        # Owner-space tag for error messages (set post-construction by the
        # owning Space via ``_tag_vq_name``).
        self.name = ""
        # Per-instance row-chunk override for the matmul/argmax intermediate.
        # ``None`` => derive from ``_VQ_CHUNK_TARGET_BYTES`` at forward time.
        self._vq_chunk_rows = None

    @property
    def codebook(self):
        """Codebook.
        
        See class docstring for the operation contract.
        """
        return self._parameters["_codebook"]

    @codebook.setter
    def codebook(self, value):
        """Codebook.
        
        See class docstring for the operation contract.
        """
        param = value if isinstance(value, nn.Parameter) else nn.Parameter(value.detach().clone())
        if "_codebook" in self._parameters:
            self._parameters["_codebook"] = param
        else:
            self.register_parameter("_codebook", param)
        self.codebook_size = param.shape[0]
        # Keep EMA buffers consistent when the codebook is reassigned
        # externally (e.g. the post-init rescale in Codebook.addVectors).
        # Reset embed_avg to the new values; reshape cluster_size if the
        # codebook size changed.
        if "embed_avg" in self._buffers:
            self._buffers["embed_avg"] = param.data.clone()
        if "cluster_size" in self._buffers:
            cs = self._buffers["cluster_size"]
            if cs.shape[0] != param.shape[0]:
                self._buffers["cluster_size"] = torch.ones(
                    param.shape[0], device=cs.device, dtype=cs.dtype
                )
        # Refresh the cached ||c_i||^2 buffer so the Euclidean retrieval
        # path stays consistent with the codebook. Allocate fresh when
        # the row count changed.
        if "_b_norms_sq" in self._buffers:
            self._buffers["_b_norms_sq"] = (param.data ** 2).sum(dim=-1)

    def _sync_ema_buffers(self):
        """Repair EMA buffers when they drift from the live codebook shape.

        Training can reassign the codebook Parameter after construction
        (for example when resizing or reloading weights). The EMA buffers
        must stay row-aligned with that Parameter or the dead-code refresh
        path can end up indexing the wrong rows.
        """
        codebook = self.codebook
        V, D = int(codebook.shape[0]), int(codebook.shape[1])
        cluster_size = self.cluster_size
        if (
            cluster_size.ndim != 1
            or int(cluster_size.shape[0]) != V
            or cluster_size.device != codebook.device
        ):
            dtype = cluster_size.dtype if cluster_size.is_floating_point() else torch.float32
            self._buffers["cluster_size"] = torch.ones(
                V, device=codebook.device, dtype=dtype
            )
        embed_avg = self.embed_avg
        if (
            embed_avg.ndim != 2
            or tuple(embed_avg.shape) != (V, D)
            or embed_avg.device != codebook.device
        ):
            dtype = embed_avg.dtype if embed_avg.is_floating_point() else codebook.dtype
            self._buffers["embed_avg"] = codebook.detach().to(dtype=dtype).clone()
        b_norms_sq = self._b_norms_sq
        if (
            b_norms_sq.ndim != 1
            or int(b_norms_sq.shape[0]) != V
            or b_norms_sq.device != codebook.device
        ):
            self._buffers["_b_norms_sq"] = (
                codebook.detach() ** 2
            ).sum(dim=-1)

    @staticmethod
    def _rotate_to(src, tgt):
        """Rotation-trick STE from arXiv:2410.06424 -- forwards ``tgt`` but
        rotates the upstream gradient from ``tgt``'s direction back to
        ``src``'s direction and rescales by ``||tgt|| / ||src||`` so
        magnitude is preserved. Mirrors ``vector_quantize_pytorch``'s
        ``rotate_to`` / ``efficient_rotation_trick_transform``.
        """
        eps = 1e-8
        orig_shape = src.shape
        e = src.reshape(-1, src.shape[-1])
        q = tgt.reshape(-1, tgt.shape[-1])
        norm_e = e.norm(dim=-1, keepdim=True).clamp(min=eps)
        norm_q = q.norm(dim=-1, keepdim=True).clamp(min=eps)
        e_hat = e / norm_e
        q_hat = q / norm_q
        w = e_hat + q_hat
        w = w / w.norm(dim=-1, keepdim=True).clamp(min=eps)
        w = w.detach()
        e_dot_w = (e * w).sum(dim=-1, keepdim=True)
        e_dot_u = (e * e_hat.detach()).sum(dim=-1, keepdim=True)
        out = e - 2.0 * e_dot_w * w + 2.0 * e_dot_u * q_hat.detach()
        scale = (norm_q / norm_e).detach()
        return (out * scale).reshape(orig_shape)

    # Default byte budget for one VQ distance/similarity tile.  4 GiB
    # leaves ample headroom on 64 GB unified-memory accelerators (the
    # AMD Strix Halo target) while keeping the per-tile allocation well
    # below per-buffer caps on current GPU stacks.  Small VQs (V < 64K)
    # never trigger chunking on first call.  Override per-instance via
    # ``vq._vq_chunk_rows``.
    _VQ_CHUNK_TARGET_BYTES = 4 * (1 << 30)  # 4 GiB

    def forward(self, x, return_all_codes=False, freeze_codebook=False, **kwargs):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        codebook = self.codebook
        N = flat.shape[0]
        V = codebook.shape[0]
        D = codebook.shape[1]
        gib = (N * V * 4) / (1024 ** 3)  # float32 bytes -> GiB

        # Chunk over rows of `flat`.  The [chunk, V] intermediate
        # (cdist or matmul) is the dominant allocation; the EMA path
        # below avoids the [N, V] one_hot entirely via bincount +
        # index_add_, so chunk size is bounded only by this matrix.
        chunk = self._vq_chunk_rows
        if chunk is None:
            max_pairs = self._VQ_CHUNK_TARGET_BYTES // 4  # float32 bytes
            chunk = max(1, max_pairs // max(V, 1))
        chunk = min(chunk, N) if N > 0 else 1

        # ``indices`` is the argmin/argmax of distances; the gather that
        # follows (codebook[indices]) carries no gradient back through the
        # selection, so the entire indices computation can run under
        # ``no_grad`` to skip saving cdist/matmul intermediates for the
        # backward pass.  EMA update has its own ``no_grad`` block below.
        try:
            with torch.no_grad():
                if self.use_cosine_sim:
                    # Dot-product mode: codebook is unit-norm (maintained by
                    # the EMA path below), so ``argmax_i (flat . codebook_i)``
                    # equals ``argmax_i cos(flat, codebook_i)`` -- the per-row
                    # ``||flat||`` is a positive constant across i and
                    # cancels out of the ranking. Skipping the input
                    # normalization preserves the input magnitude (the
                    # belief-certainty signal in ConceptualSpace) end-to-end
                    # and saves an O(N*D) normalize op per chunk.
                    if N <= chunk:
                        indices = (flat @ codebook.T).argmax(dim=-1)
                    else:
                        parts = []
                        for s in range(0, N, chunk):
                            parts.append((flat[s:s+chunk] @ codebook.T).argmax(dim=-1))
                        indices = torch.cat(parts, dim=0)
                else:
                    # Euclidean mode via the matmul / cached-norm trick:
                    #   ||x - c_i||^2 = ||x||^2 + ||c_i||^2 - 2 (x . c_i)
                    # ||x||^2 is a constant across i, so it drops from
                    # argmin. After negation:
                    #   argmin_i ||x - c_i||^2
                    #     = argmax_i (x . c_i - 0.5 ||c_i||^2)
                    # ``self._b_norms_sq`` (refreshed in the codebook
                    # setter and after every EMA update) is ``[V]``;
                    # ``flat @ codebook.T`` is ``[chunk, V]``. Same
                    # FLOPs as cdist's mm-trick path, skips sqrt and
                    # the per-row ||x||^2 add, and gives Inductor a
                    # cleaner graph (no autograd plumbing for cdist).
                    half_b_norms_sq = (0.5 * self._b_norms_sq).to(flat.dtype)
                    if N <= chunk:
                        scores = flat @ codebook.T
                        scores = scores - half_b_norms_sq
                        indices = scores.argmax(dim=-1)
                    else:
                        parts = []
                        for s in range(0, N, chunk):
                            scores = flat[s:s+chunk] @ codebook.T
                            scores = scores - half_b_norms_sq
                            parts.append(scores.argmax(dim=-1))
                        indices = torch.cat(parts, dim=0)
        except RuntimeError as e:
            owner = self.name or "<unnamed VQ>"
            raise RuntimeError(
                f"VectorQuantize[{owner}]: distance matrix allocation "
                f"failed even after chunking. flat={tuple(flat.shape)}, "
                f"codebook={tuple(codebook.shape)}, chunk={chunk}, "
                f"pairwise matrix = [{N}, {V}] float32 = {gib:.2f} GiB. "
                f"Reduce {owner}.nVectors, reduce batchSize, or set "
                f"vq._vq_chunk_rows to a smaller value. Original error: {e}"
            ) from e
        quantized_raw = codebook[indices].reshape(original_shape)
        commit_loss = self.commitment_weight * F.mse_loss(
            x, quantized_raw.detach()
        )

        # EMA codebook update + dead-code replacement. Without these the
        # codebook gets no gradient (commit_loss detaches the quantized
        # side and STE is a no-op for the codes) so codes never track the
        # data distribution. Mirrors ``EuclideanCodebook.update_ema`` and
        # ``expire_codes_`` in vector_quantize_pytorch.
        if self.training and not freeze_codebook:
            with torch.no_grad():
                self._sync_ema_buffers()
                flat_f = flat.float()
                # Equivalent to ``one_hot(indices, V).t() @ flat_f`` but
                # without ever allocating the [N, V] one-hot tensor: at
                # body-scale microbatch (N in the millions) that matrix
                # alone passes the GPU per-buffer cap.
                #
                # ``torch.bincount`` has a data-dependent output shape
                # (``max(minlength, indices.max()+1)``) so dynamo graph-
                # breaks on it ("Dynamic shape operator aten.bincount.
                # default"). ``indices`` is an argmax/argmin over the V
                # codebook rows, so every value is in ``[0, V)`` and
                # ``bincount(minlength=V)`` would always yield exactly
                # ``[V]``. A fixed ``[V]`` zeros + ``index_add_`` of a
                # ones source is the identical per-entry count with a
                # static output shape -- the same idiom ``embed_sum``
                # uses immediately below.
                embed_sum = torch.zeros(
                    V, D, device=flat_f.device, dtype=flat_f.dtype,
                )
                embed_sum.index_add_(0, indices, flat_f)
                cluster_size_batch = torch.zeros(
                    V, device=flat_f.device, dtype=flat_f.dtype,
                )
                cluster_size_batch.index_add_(
                    0, indices,
                    torch.ones_like(indices, dtype=flat_f.dtype),
                )
                self.cluster_size.mul_(self.decay).add_(
                    cluster_size_batch.to(self.cluster_size.dtype),
                    alpha=1.0 - self.decay,
                )
                self.embed_avg.mul_(self.decay).add_(
                    embed_sum.to(self.embed_avg.dtype),
                    alpha=1.0 - self.decay,
                )
                n = self.cluster_size.sum()
                cs_smooth = (
                    (self.cluster_size + self.eps)
                    / (n + V * self.eps)
                    * n
                )
                new_embed = self.embed_avg / cs_smooth.unsqueeze(-1)
                if self.use_cosine_sim:
                    new_embed = F.normalize(new_embed, dim=-1)
                self.codebook.copy_(new_embed.to(self.codebook.dtype))
                # Refresh the cached ||c_i||^2 so the Euclidean retrieval
                # path on the next forward sees the updated codebook. In
                # cosine mode every row is unit-norm by construction, so
                # we still write 1.0 here -- harmless for the dot-product
                # path (which doesn't read this buffer) and keeps the
                # invariant true if useDotProduct is later toggled off.
                self._b_norms_sq.copy_(
                    (self.codebook.detach() ** 2).sum(dim=-1)
                )
                if self.codebook_retire and self.threshold_ema_dead_code > 0:
                    expired = (
                        self.cluster_size < self.threshold_ema_dead_code
                    ).reshape(-1)
                    if int(expired.numel()) != V:
                        raise RuntimeError(
                            "VectorQuantize EMA state corrupt: "
                            f"expired mask has {int(expired.numel())} rows "
                            f"for codebook size {V}"
                        )
                    expired_idx = torch.nonzero(
                        expired, as_tuple=False
                    ).flatten()
                    if int(expired_idx.numel()) > 0:
                        n_expired = int(expired_idx.numel())
                        sample_idx = torch.randint(
                            0, flat.shape[0], (n_expired,), device=flat.device,
                        )
                        sampled = flat[sample_idx].to(self.codebook.dtype)
                        if self.use_cosine_sim:
                            sampled = F.normalize(sampled, dim=-1)
                        self.codebook.index_copy_(0, expired_idx, sampled)
                        thr = float(self.threshold_ema_dead_code)
                        self.cluster_size.index_fill_(0, expired_idx, thr)
                        self.embed_avg.index_copy_(
                            0,
                            expired_idx,
                            sampled.to(self.embed_avg.dtype) * thr,
                        )

        # Gradient estimator for the quantized output: rotation trick when
        # requested and the input carries gradient, otherwise vanilla STE.
        if self.rotation_trick and self.training and x.requires_grad:
            quantized = self._rotate_to(x, quantized_raw)
        else:
            quantized = x + (quantized_raw - x).detach()
        indices = indices.reshape(original_shape[:-1])
        if return_all_codes:
            return quantized, indices, commit_loss, quantized.unsqueeze(0)
        return quantized, indices, commit_loss

    @torch.no_grad()
    def grow_on_novelty(self, x, eps, free_threshold=None):
        """ε-growing codebook: insert novel inputs into dead slots.

        Two-phase logic:

          1. **Dead-slot detection.** A slot is considered "dead"
             (available for re-use) when its EMA ``cluster_size``
             falls below ``free_threshold``.  Default threshold is
             ``self.threshold_ema_dead_code`` if that's > 0, otherwise
             a small fraction of the codebook's initial unit
             ``cluster_size`` (``0.5``) so EMA-decayed never-assigned
             slots count as dead after a few batches.

          2. **Novelty test.** For each row of the flattened encoder
             output ``x``, compute the min-distance to the **live**
             entries (cluster_size > threshold).  Rows whose min-
             distance exceeds ``eps`` are "novel" and get inserted
             into the available dead slots, seeding
             ``cluster_size``/``embed_avg`` so subsequent EMA blends
             pull toward the inserted point.

        Returns the count of inserted slots.  Idempotent when every
        novel row already maps within ``eps`` of a live entry.
        No-op when ``eps <= 0`` or the layer is in eval mode.
        """
        if eps <= 0.0 or not self.training:
            return 0
        if free_threshold is None:
            free_threshold = (float(self.threshold_ema_dead_code)
                              if self.threshold_ema_dead_code > 0
                              else 0.5)
        flat = x.reshape(-1, x.shape[-1])
        if self.use_cosine_sim:
            flat = F.normalize(flat, dim=-1)
        live_mask = (self.cluster_size > free_threshold).reshape(-1)
        free_idx = torch.nonzero(
            ~live_mask, as_tuple=False).flatten()
        if int(free_idx.numel()) == 0:
            return 0
        live_idx = torch.nonzero(
            live_mask, as_tuple=False).flatten()
        seed_val = max(1.0, float(self.threshold_ema_dead_code))
        if int(live_idx.numel()) == 0:
            # No live entries yet -- seed the first n free slots with
            # the leading rows of x.  Mirrors the dead-code retire's
            # cold-start path.
            n_seed = min(int(free_idx.numel()), int(flat.shape[0]))
            target = free_idx[:n_seed]
            sampled = flat[:n_seed].to(self.codebook.dtype)
            self.codebook.index_copy_(0, target, sampled)
            self.cluster_size.index_fill_(0, target, seed_val)
            self.embed_avg.index_copy_(
                0, target,
                sampled.to(self.embed_avg.dtype) * seed_val)
            return n_seed
        live_codes = self.codebook.index_select(0, live_idx)
        if self.use_cosine_sim:
            live_codes = F.normalize(live_codes, dim=-1)
        # Pairwise squared L2; min over codes per row.
        x_sq = (flat * flat).sum(dim=-1, keepdim=True)
        c_sq = (live_codes * live_codes).sum(dim=-1, keepdim=True).T
        dots = flat @ live_codes.T
        dists_sq = (x_sq - 2 * dots + c_sq).clamp(min=0.0)
        min_dist = dists_sq.min(dim=-1).values.sqrt()
        novel_mask = min_dist > float(eps)
        novel_rows = torch.nonzero(
            novel_mask, as_tuple=False).flatten()
        if int(novel_rows.numel()) == 0:
            return 0
        n_insert = min(int(novel_rows.numel()),
                       int(free_idx.numel()))
        sampled = flat[novel_rows[:n_insert]].to(
            self.codebook.dtype)
        target = free_idx[:n_insert]
        self.codebook.index_copy_(0, target, sampled)
        self.cluster_size.index_fill_(0, target, seed_val)
        self.embed_avg.index_copy_(
            0, target, sampled.to(self.embed_avg.dtype) * seed_val)
        return n_insert


def test():
    """Self-test; verifies the round-trip / invariant."""
    torch.autograd.set_detect_anomaly(True)

    TruthLayer.test()
    LinearLayer.test()


    InvertibleLinearLayer.test()

    SigmaLayer.test()
    PiLayer.test()

    AttentionLayer.test()
    Mem.test()
    DecisionBoundaryLayer.test()

def main():
    """Main.
    
    See class docstring for the operation contract.
    """
    try:
        test()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc


# =====================================================================
# Ops-as-Layer-container (2026-05-02 consolidation pass).
#
# Per the user's design intent: Ops should not own grammar operations
# itself; it should delegate to GrammarLayer instances. Each
# `Ops.<grammar_op>` is now an `_OpHandle` whose:
#   * direct call ``Ops.<op>(args, **kwargs)`` routes to the math
#     kernel (renamed to `Ops._<op>_kernel`) -- preserves backward
#     compat for callers passing `monotonic=` / `mode=` / `kind=`
#     flags.
#   * `.forward(...)` / `.reverse(...)` / `.compose(...)` /
#     `.generate(...)` route through the corresponding GrammarLayer
#     subclass (NotLayer, ConjunctionLayer, etc.). These satisfy the
#     "every Layer has forward and reverse" contract and are the
#     surfaces SyntacticLayer / Chart use.
# =====================================================================
class _OpHandle:
    """Container that exposes both the legacy kernel signature
    (callable; takes flags) AND the GrammarLayer interface
    (forward/reverse/compose/generate)."""

    def __init__(self, kernel, layer):
        """Initialize _OpHandle; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        self._kernel = kernel
        self._layer = layer

    def __call__(self, *args, **kwargs):
        """Invoke this callable."""
        return self._kernel(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return self._layer.forward(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return self._layer.reverse(*args, **kwargs)

    def compose(self, *args, **kwargs):
        """Compose the input via this layer's parse contract."""
        return self._layer.compose(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Drive the reverse / generation pass."""
        return self._layer.generate(*args, **kwargs)

    @property
    def rule_name(self):
        return self._layer.rule_name

    @property
    def arity(self):
        return self._layer.arity

    @property
    def invertible(self):
        return self._layer.invertible

    @property
    def lossy(self):
        return self._layer.lossy

    @property
    def layer(self):
        """The wrapped GrammarLayer instance (singleton)."""
        return self._layer


def _bind_ops_singletons():
    """Bind `Ops.<grammar_op>` to `_OpHandle` instances so the same
    name resolves either to the legacy kernel (via __call__) or the
    GrammarLayer surface (via .forward / .reverse / .compose /
    .generate).

    Called once at module load. Each `_OpHandle`'s kernel is captured
    BEFORE rebinding, so internal `Ops._<op>_kernel(...)` self-references
    inside Ops static methods continue to route directly to the kernel.
    """
    # Map: legacy op name -> (kernel staticmethod, GrammarLayer subclass).
    # ``absorb`` was retired 2026-05-04 -- the marker lives on
    # ``GrammarLayer.absorb`` (base class) so no kernel/class binding.
    bindings = (
        ('negation',    Ops._negation_kernel,    NotLayer),
        ('non',         Ops._non_kernel,         NonLayer),
        ('conjunction', Ops._conjunction_kernel, ConjunctionLayer),
        ('disjunction', Ops._disjunction_kernel, DisjunctionLayer),
        ('lift',        Ops._lift_kernel,        LiftLayer),
        ('lower',       Ops._lower_kernel,       LowerLayer),
        ('equal',       Ops._equal_kernel,       EqualLayer),
        ('part',        Ops._part_kernel,        PartLayer),
    )
    for name, kernel, cls in bindings:
        try:
            inst = cls()
        except TypeError:
            continue
        setattr(Ops, name, _OpHandle(kernel, inst))


_bind_ops_singletons()


# Self-test: run the LogicNet smoke test when executed as a script.
if __name__ == "__main__":
    main()
