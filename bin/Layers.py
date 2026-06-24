"""Core layer primitives used by BasicModel.

This module mixes conventional neural-network utilities with a set of
custom reversible, ergodic, and memory-style layers.  Most higher-level
model construction happens in ``BasicModel.py``; this file provides the
building blocks and the update rules they share.
"""
from __future__ import annotations  # allow X | Y union syntax on Python 3.9

import os
import warnings
import collections
import numpy as np
import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import torch.optim as optim
import time
from typing import Dict, List, Optional, Tuple
from collections import namedtuple

epsilon = 1e-7  # to avoid log(0)

# Device used by all layers.
import util
from util import TheXMLConfig, TheDevice
from util import TheMessage

# ---------------------------------------------------------------------------
# Meronomy constants (MeronomySpec §3-§4; MeronomyPlan Stage 0).
#
# EPS_LOG is the log-floor near m = 0 for the membership folds: PiLayer2 /
# SigmaLayer2 run on memberships m in [0, 1] through log/exp, and m = 0
# (the bottom element, "nothing") is floored to EPS_LOG before the log so
# the fold stays finite while remaining within tolerance of the absorber
# laws. Distinct from the global ``epsilon`` (1e-7), which guards the
# legacy odds/atanh charts.
EPS_LOG = 1e-6

# Default upper clamp for the contractive diagonal (d >= 1) when
# ``stable=True`` on ContractiveInvertibleLinearLayer: in log-membership
# space "stable" means BOUNDED AMPLIFICATION of negative log-mass --
# the opposite regime of the legacy (eps, 1.0] clamp, which wanted
# non-expansion of |x| in the odds chart. Config-overridable; see
# ``meronomy_d_max_stable()``.
D_MAX_STABLE = 4.0


def meronomy_enabled():
    """Resolve the ``<architecture><meronomy>`` mode knob (off | on).

    The single meronomy option (MeronomyPlan §0): ``on`` binds the
    meronymic slots to the membership kernels and enables the interface
    factoring and the reference table. Default ``off`` — stages land
    dark; existing model XMLs are untouched until cutover. Accepts the
    bare element text (``<meronomy>on</meronomy>``) or the attributed
    form (text under ``"_"`` when attributes like ``dMaxStable`` are
    present).
    """
    try:
        raw = TheXMLConfig.get("architecture.meronomy", default=None)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    if isinstance(raw, dict):
        raw = raw.get("_", None)
    if raw is None:
        return False
    return str(raw).strip().lower() in ("on", "true", "1", "yes")


def meronomy_d_max_stable():
    """Resolve the stable-clamp upper bound for contractive diagonals.

    Reads ``<architecture><meronomy dMaxStable="...">`` from the XML
    config; falls back to the module default ``D_MAX_STABLE`` when the
    key is absent, unparsable, or no config is loaded.
    """
    try:
        raw = TheXMLConfig.get("architecture.meronomy.dMaxStable",
                               default=None)
    except (KeyError, TypeError, ValueError, AttributeError):
        return D_MAX_STABLE
    if raw is None:
        return D_MAX_STABLE
    try:
        return float(raw)
    except (TypeError, ValueError):
        return D_MAX_STABLE


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


class ConceptualCombine(Layer):
    r"""Square, exactly-invertible 2-STREAM conceptual bind (C-10).

    Asymmetric-vq plan sec.5 (rev. 2026-06-09; slot geometry corrected
    2026-06-10): a concept is the REVERSIBLE BIND of its perceptual form
    (PS) and symbolic form (WS). The two streams are stacked along the
    VECTOR axis -- ``N`` slots each, ``2N`` total -- and ONE cascade runs
    over the flattened ``2N * D`` slab (``16 * nDim`` for the production
    parallel config: ``2^14`` exactly, zero pad), so the bind mixes ACROSS
    slots, not merely within each position. Wraps an
    :class:`InvertibleLinearLayer` (the dense LDU, ``full`` / ``last``) or
    the base :class:`GrammarLayer` butterfly cascade (``butterfly``)::

        CS_t = ILL_t( stack[ PS_t ; WS_t ] )        # forward bind
        [ PS_t ; WS_t ] = ILL_t^{-1}( CS_t )        # exact reverse

    with ``PS_t, WS_t`` of shape ``[..., N, D]`` (the FULL muxed event
    width: the where/when band PARTICIPATES in the bind, option B -- the
    spec's ``16 * nDim`` arithmetic is the event width, ``nDim`` being the
    config event width). ``CS_t`` is the
    WHOLE mixed bind, flattened (``[..., M]``, ``M = next_pow2(2N*D)``);
    the PS / WS VIEWS are its slot-halves -- ``views(CS)[0] ==``
    vectors ``0..N-1``, ``views(CS)[1] ==`` vectors ``N..2N-1`` -- exactly
    the row-windows a Phase-5/6 ``.active`` index view can express. There
    is NO augment and NO carrier/augment split. This retires the 3-stream
    design's pathologies at the source (doc/plans/2026-06-09-asymmetric-
    vq-symbolic-ws.md sec.5):

      * **contraction gone** -- nothing is shed to an augment; the carrier
        holds the full bind, so there is no rank bottleneck and no
        per-stage scale collapse (the old design read the carrier from the
        ps-slot only -- 1/3 rank -- bleeding 2/3 to the augment per stage);
      * **exact inversion, no augment threading** -- the layer is a true
        bijection on its operating width, so ``CS -> (PS, WS)`` is exact by
        construction; the whole augment-pairing reverse is deleted
        (``reverse_dropped`` retired with it);
      * **right semantics** -- the mix of the three alphas is ABSORBED into
        the learned weights; the carrier is the bind, not a slice.

    The bind is kept PURELY LINEAR: a saturating squash inside the combine
    would break exact inversion; PS/WS arrive already bounded (codebook /
    fold), norm-carried through the layer (the +-1 OPERATING RANGE is an
    input contract, not a tanh).

    The span is selected by the global ``<sigmaPi>`` knob via
    :meth:`Spaces.Space.sigma_pi_mode` (``last`` | ``butterfly`` | ``full``),
    matching the PS.Pi / WS.Sigma construction in Spaces.py so behaviour is
    consistent across the codebase:

      * ``full``      -- ONE dense LDU over the whole ``[..., 2D]`` slab
                         (the wide<->deep invertible bridge).
      * ``last``      -- per-slot square fold over the last dim; for this
                         per-position bind the slot IS the ``2D`` content
                         vector, so it is the same single dense LDU as
                         ``full`` (no cross-position flatten).
      * ``butterfly`` -- the O(M log M) cross-element 2x2-LDU cascade over a
                         POWER-OF-TWO width ``M = next_pow2(2D)``. The base
                         GrammarLayer cascade is PURELY LINEAR: each node is a
                         bare 2x2 LDU pair op (:meth:`GrammarLayer.
                         _butterfly_pair_forward` /
                         :meth:`GrammarLayer._butterfly_pair_reverse`), whose
                         closed-form per-node inverse (sign-flip on the
                         off-diagonals + reciprocals on the diagonals) makes
                         the cascade an exact bijection on ``R^M`` -- there is
                         no pi (atanh/tanh) fold. (PiLayer / SigmaLayer inject
                         that nonlinearity via their ``forward`` / ``reverse``
                         overrides, which the BARE GrammarLayer here never
                         calls; ``_butterfly_pair_op`` in SigmaLayer is dead
                         code on this path.) Sizing the layer at ``M`` (so the
                         layer's ``N == M_total == M``) is what makes the
                         round-trip exact: nothing is zero-padded or stripped,
                         so the per-node bijection covers every coordinate the
                         cascade writes into.

    Unlike PiLayer/SigmaLayer this combine is a PLAIN invertible linear map
    (no multiplicative log-domain fold): the conceptual carrier is a linear
    mixture, so the dense path uses ``InvertibleLinearLayer`` directly and the
    butterfly path uses the bare linear cascade.

    Shape conventions (both paths): arbitrary leading batch dims ``[..., D]``
    per stream. ``ConceptualCombine`` itself flattens all leading dims to a
    single batch dim before calling the wrapped layer and restores them on the
    outputs, so the butterfly path no longer relies on GrammarLayer's own
    multi-dim flatten (which would mix the wrong axes for ``[B, T, D]``).

    This is a self-contained layer; wiring into the Models forward path is a
    later task.
    """

    def __init__(self, content_dim, n_vectors=1, naive=False,
                 sigma_pi_mode="full", hasBias=True, ergodic=False,
                 n_streams=2):
        """Build the per-stage square 2-stream slot bind.

        Args:
            content_dim: ``D`` -- the per-vector width: the FULL muxed
                event width (``cs.muxedSize`` == the config ``nDim``), so
                the where/when band participates in the bind (option B).
            n_vectors: ``N`` -- the per-stream slot count. The streams
                stack to ``2N`` vectors and the wrapped layer is square
                over ``M = next_pow2(2 * N * D)`` (``== 2ND`` when that is
                a power of two -- the production 16*1024 = 2^14 case).
            naive: forwarded to ``InvertibleLinearLayer`` (dense path only);
                ``naive=False`` routes reverse through the exact structured
                ``_solve_ldu`` (the design's required path -- do NOT use
                ``naive=True`` / pinv).
            sigma_pi_mode: the global ``<sigmaPi>`` span; one of
                ``last`` | ``butterfly`` | ``full`` (legacy ``<butterfly>``
                boolean accepted). Normalised via ``Space.sigma_pi_mode``.
            hasBias: forwarded to the dense ``InvertibleLinearLayer``. The
                butterfly cascade is bias-free by construction.
            ergodic: forwarded to the dense ``InvertibleLinearLayer`` (noise
                injection preserving the exact LDU inverse). Ignored by the
                butterfly path (the base cascade has no ergodic mode).
        """
        D = int(content_dim)
        Nv = int(n_vectors)
        if D < 1 or Nv < 1:
            raise ValueError(
                f"ConceptualCombine: content_dim and n_vectors must be "
                f">= 1; got D={D}, N={Nv}")
        S = int(n_streams)
        if S < 2:
            raise ValueError(
                f"ConceptualCombine: n_streams must be >= 2; got {S}")
        N = S * Nv * D
        super().__init__(N, N)
        self.content_dim = D
        self.n_vectors = Nv
        self.n_streams = S
        self.combine_dim = N  # flattened S*N*D stack [stream0-slots ; stream1 ; ...]

        # Normalise the <sigmaPi> span the same way the spaces do; a lazy
        # import keeps Layers.py free of a Spaces import cycle (Spaces
        # imports Layers). Fall back to the documented enum if Spaces is
        # unavailable at construction time.
        try:
            from Spaces import Space
            mode = Space.sigma_pi_mode(sigma_pi_mode)
        except Exception:
            if isinstance(sigma_pi_mode, bool):
                mode = "butterfly" if sigma_pi_mode else "last"
            elif sigma_pi_mode is None:
                mode = "last"
            else:
                v = str(sigma_pi_mode).strip().lower()
                if v in ("last", "butterfly", "full"):
                    mode = v
                elif v in ("true", "1"):
                    mode = "butterfly"
                elif v in ("false", "0", ""):
                    mode = "last"
                else:
                    raise ValueError(
                        "ConceptualCombine: sigma_pi_mode must be "
                        f"last|butterfly|full; got {sigma_pi_mode!r}")
        self.sigma_pi_mode = mode

        if mode == "butterfly":
            # Cross-element linear 2x2-LDU cascade over the WHOLE flattened
            # 2N*D stack -- the bind mixes across slots, the same
            # cross-position reach PS.sigma / WS.pi get from their
            # flattened cascades. The base GrammarLayer cascade is
            # plain-linear and exactly invertible (closed-form per-node
            # inverse); no pi (atanh/tanh) fold.
            #
            # CRITICAL: size the cascade at M = next power of two >= 2N*D,
            # NOT at 2N*D. The GrammarLayer butterfly cascade structurally
            # runs over next_pow2(N) and, if N is not itself a power of
            # two, _butterfly_flatten zero-pads up to that width while
            # _butterfly_unflatten strips back down to N -- discarding the
            # coords the cascade wrote into the padded tail, which the
            # zero-re-padding reverse cannot recover (round-trip error at
            # non-identity weights). By constructing the layer with
            # N == M (a power of two) the layer's N == M_total == M, so
            # NOTHING is padded or stripped and the cascade is an exact
            # bijection on the full R^M. ConceptualCombine owns the
            # 2ND->M zero-pad (forward) and the M->2ND read-back (reverse)
            # itself. For the production parallel config (N=8, D=1024)
            # M == 16384 == 2^14 exactly: zero pad, zero waste.
            M = 1 << ((N - 1).bit_length())  # next pow2 >= 2ND
            self.combine_padded = M          # cascade operating width
            self.layer = GrammarLayer(M, M, butterfly=True, N=M)
        else:
            # "full" and "last": a single dense LDU over the whole
            # flattened 2N*D stack (per-position and flattened coincide
            # once the stack is flattened), so both map to one
            # InvertibleLinearLayer. Square, no padding. Practical only
            # at small test widths -- production uses butterfly.
            self.combine_padded = N
            self.layer = InvertibleLinearLayer(
                N, N, naive=naive, ergodic=ergodic,
                hasBias=hasBias, stable=True)
        # The carrier IS the full bind: the layer's operating width.
        self.carrier_dim = self.combine_padded
        self.layers.append(self.layer)
        # The CORPUS CALLOSUM (Alec, 2026-06-10): when CS glues the bind's
        # two views for operation (until OutputSpace accepts both views as
        # two subspaces), the halves are literally STACKED along the vector
        # axis (2N vectors) and glued by a learned [2N, N] matrix over the
        # SLOT axis (channel-preserving). Initialised to AVERAGING
        # (0.5 * [I_N ; I_N]) so the glue starts as the balanced mix of the
        # two hemispheres and learns its own routing from there. Tiny
        # parameter cost (2N*N scalars; 128 for the production 16x8).
        _cal = torch.zeros(S * Nv, Nv)
        _eye = torch.eye(Nv)
        for _s in range(S):                       # averaging init: (1/S) I per stream
            _cal[_s * Nv:(_s + 1) * Nv, :] = (1.0 / S) * _eye
        self.callosum = nn.Parameter(_cal)

    def _combine(self, *streams):
        """Stack the ``n_streams`` ``[..., N, D]`` streams along the VECTOR axis
        into ``[..., S*N, D]`` (order: stream 0, 1, ...; the 2-stream case is
        ``[ps-slots ; ws-slots]``)."""
        return torch.cat(streams, dim=-2)

    def _split_streams(self, x):
        """Split ``[..., S*N, D]`` into ``n_streams`` slices each ``[..., N, D]``
        (2-stream case returns ``(ps, ws)``)."""
        Nv = self.n_vectors
        return tuple(x[..., s * Nv:(s + 1) * Nv, :]
                     for s in range(self.n_streams))

    def views(self, carrier):
        """The PS / WS views of a bind: its slot-halves.

        ``carrier`` is the ``[..., M]`` flat bind ``forward`` returned.
        Returns ``(ps_view, ws_view)``, each ``[..., N, D]`` -- vectors
        ``0..N-1`` and ``N..2N-1`` of the STORED mix (the row-windows a
        Phase-5/6 ``.active`` index view expresses; NOT the inverse's
        halves, so a consumer of the views keeps the bind in its gradient
        path).
        """
        leading = tuple(carrier.shape[:-1])
        body = carrier[..., :self.combine_dim].reshape(
            *leading, self.n_streams * self.n_vectors, self.content_dim)
        return tuple(v.contiguous() for v in self._split_streams(body))

    def glue(self, carrier):
        """The corpus-callosum glue: the STACKED views reduced to N vectors.

        Reads the bind's ``[..., 2N, D]`` stack (both views, literally
        stacked) and applies the learned ``[2N, N]`` callosum over the
        slot axis: ``out[..., n, :] = sum_k callosum[k, n] *
        stack[..., k, :]``. Channel-preserving; init = averaging the two
        hemispheres. This is the head-facing N-vector operation event
        until OutputSpace accepts both views directly.
        """
        leading = tuple(carrier.shape[:-1])
        stack = carrier[..., :self.combine_dim].reshape(
            *leading, self.n_streams * self.n_vectors, self.content_dim)
        # [..., S*N, D] x [S*N, N] -> [..., N, D] over the slot axis.
        glued = torch.einsum("...kd,kn->...nd", stack, self.callosum)
        return glued.contiguous()

    @staticmethod
    def _flatten_leading(x):
        """Flatten all leading dims to a single batch dim.

        ``[..., W] -> ([B', W], leading_shape)`` where ``B'`` is the
        product of the leading dims. The combine owns this flatten (rather
        than delegating to GrammarLayer's multi-dim flatten, which would
        mix the wrong axes for ``[B, T, D]`` inputs) so the wrapped layer
        always sees a clean ``[B', W]`` matrix.
        """
        leading = tuple(x.shape[:-1])
        return x.reshape(-1, x.shape[-1]), leading

    @staticmethod
    def _restore_leading(x, leading):
        """Inverse of :meth:`_flatten_leading`: ``[B', W] -> [*leading, W]``."""
        return x.reshape(*leading, x.shape[-1])

    def forward(self, *streams):
        """The bind: ``CS = layer(flatten(stack[stream0 ; stream1 ; ...]))``.

        Stacks the ``n_streams`` ``[..., N, D]`` streams along the vector axis
        into ``[..., S*N, D]``, flattens the trailing ``(S*N, D)`` into ``S*N*D``
        (and all leading dims to a single batch dim), zero-pads
        ``S*N*D -> M`` (the operating width; empty for the production
        ``16*1024 = 2^14`` case), applies the wrapped invertible layer, and
        restores the leading dims.

        Returns the FULL ``[..., M]`` flat carrier ``CS`` -- the whole
        mixed bind. There is no augment: the carrier and the bind are the
        same tensor, which is what makes :meth:`reverse` an exact inversion
        with nothing threaded alongside. Use :meth:`views` for the
        ``[..., N, D]`` PS / WS slot-half windows.
        """
        x = self._combine(*streams)                # [..., S*N, D]
        leading = tuple(x.shape[:-2])
        flat = x.reshape(-1, self.combine_dim)     # [B', S*N*D]
        if self.combine_padded > self.combine_dim:
            pad = flat.new_zeros(
                flat.shape[0], self.combine_padded - self.combine_dim)
            flat = torch.cat([flat, pad], dim=-1)  # [B', M]
        out = self.layer.forward(flat)             # [B', M]
        # A5 fullgraph fix (kept from the 3-stream design): force a
        # materialised, non-aliasing output so no view of a padded
        # intermediate escapes the forward (AOT autograd's output-alias
        # epilogue chokes on views whose recorded base width differs from
        # the runtime base).
        return out.contiguous().reshape(*leading, self.carrier_dim)

    def reverse(self, carrier):
        """Exact reverse: ``[ps ; ws] = layer^{-1}(CS)``.

        Flattens leading dims, applies the layer's exact structured inverse
        (dense ``naive=False`` -> ``_solve_ldu``; butterfly -> the per-node
        closed-form pair reverse), reads back the first ``2N*D`` coords
        (the ``M - 2N*D`` tail is the recovered zero-pad, ~0), un-flattens
        to ``[..., 2N, D]``, and splits into ``(ps, ws)`` each
        ``[..., N, D]``. Exact to the solve / cascade tolerance -- by
        construction, with NOTHING threaded: the carrier IS the whole bind.
        """
        flat, leading = self._flatten_leading(carrier)  # [B', M]
        x = self.layer.reverse(flat)                    # [B', M]
        body = x[..., :self.combine_dim].reshape(
            *leading, self.n_streams * self.n_vectors, self.content_dim)
        return tuple(v.contiguous() for v in self._split_streams(body))


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


class ContractiveInvertibleLinearLayer(NonNegativeInvertibleLinearLayer):
    r"""Non-negative LDU layer under the meronymic weight law
    (MeronomySpec §4; MeronomyPlan Stage 1).

    A NEW class beside :class:`NonNegativeInvertibleLinearLayer`, not a
    modification of it -- legacy layers and checkpoints are untouched.
    All constraints hold BY CONSTRUCTION (softplus reparametrization),
    at init and after arbitrary optimizer steps (spec §10.10):

        diag(W)    >= 1   d = 1 + softplus(raw); init raw = -5 => d ~ 1
        offdiag(W) >= 0   softplus-LDU as in the parent
        b          <= 0   b = -softplus(raw); init raw = -5 => b ~ 0

    Applied to log-memberships (log m <= 0, PiLayer2), contraction is a
    theorem of the parametrization: the diagonal amplifies negative
    log-mass, off-diagonal mixing only deepens the part, and the bias
    can only deepen it further -- pi(m) <= m for all m. This is the
    load-bearing member of the meronomy design: a minted whole
    geometrically dominates its parts at mint, not after training.

    ``stable=True`` clamps d to ``[1.0, d_max]`` -- the OPPOSITE regime
    of the parent's ``(eps, 1.0]`` clamp: in log-membership space
    "stable" means bounded amplification of negative log-mass, not
    non-expansion of |x| (the odds chart's concern). ``d_max`` defaults
    to ``meronomy_d_max_stable()`` (config-overridable D_MAX_STABLE).

    ``blocks=k`` declares the k-operand fold form (binary folds use
    ``blocks=2`` with ``nInput == 2 * nOutput``): the effective weight
    is the stack of operand blocks ``W = [W_1; ...; W_k]`` and the law
    holds PER BLOCK -- every block diagonal ``diag(W_b) >= 1`` per row
    (spec §4: this yields the acceptance bound pi <= min(operands)).
    Realized inside the LDU by lifting the block-diagonal entries of L
    (rows ``b * nOutput + i``, col ``i``, ``b >= 1``; strict lower
    triangle, so unit-triangularity and the exact triangular-solve
    inverse are untouched) to the same ``1 + softplus`` law as d:
    ``W[b*D + i, i] >= L[b*D + i, i] * d_i >= 1``.
    """

    def __init__(self, nInput, nOutput, naive=False, ergodic=False,
                 hasBias=True, stable=False, blocks=1, d_max=None):
        """Initialize ContractiveInvertibleLinearLayer; see class docstring.

        ``blocks > 1`` requires ``nInput == blocks * nOutput`` (operand
        blocks stack along the input axis). ``d_max=None`` resolves via
        ``meronomy_d_max_stable()`` at construction time.
        """
        if int(blocks) < 1:
            raise ValueError(
                f"ContractiveInvertibleLinearLayer: blocks must be >= 1; "
                f"got {blocks}")
        if int(blocks) > 1 and nInput != int(blocks) * nOutput:
            raise ValueError(
                f"ContractiveInvertibleLinearLayer: blocks={blocks} "
                f"requires nInput == blocks * nOutput; got "
                f"nInput={nInput}, nOutput={nOutput}")
        super().__init__(nInput, nOutput, naive=naive, ergodic=ergodic,
                         hasBias=hasBias, stable=stable)
        self.blocks = int(blocks)
        self.d_max = float(d_max) if d_max is not None \
            else float(meronomy_d_max_stable())
        with torch.no_grad():
            # Parent fills d at softplus_inverse(1.0) for softplus(d) ~ 1;
            # the contractive law is d = 1 + softplus(raw), so re-init the
            # raw at -5 => d ~ 1.0067 ~ 1 (near-identity, just above it).
            self.d.fill_(-5.0)
            if hasBias:
                # b = -softplus(raw); raw = -5 => b ~ -0.0067 ~ 0. The
                # parent leaves biasWeight at zeros, which here would be
                # b = -log(2) -- too deep for a near-identity init.
                self.biasWeight.fill_(-5.0)
        # +1.0 lift at the block-diagonal entries of L (see class
        # docstring). None when blocks == 1 (unary form: no lift).
        if self.blocks > 1:
            lift = torch.zeros(nInput, nInput)
            for b in range(1, self.blocks):
                for i in range(nOutput):
                    lift[b * nOutput + i, i] = 1.0
            self.register_buffer('_block_diag_lift', lift)
        else:
            self.register_buffer('_block_diag_lift', None)
        # Cached identity matrices for the unit-triangular factors.
        # The parent materializes ``torch.eye`` on every call; under
        # torch.export → executorch edge dialect, eye decomposes to an
        # ``aten.arange`` whose dtype (uint16 for small ranges) the
        # edge verifier rejects. Constant buffers trace as lifted
        # constants instead, and skip the per-call build (non-persistent:
        # derived data, not state).
        self.register_buffer('_eye_in', torch.eye(nInput), persistent=False)
        self.register_buffer('_eye_out', torch.eye(nOutput),
                             persistent=False)

    # --- Contractive factor helpers ---
    def _L(self):
        """Unit-lower-triangular, non-negative off-diagonals; block
        diagonals (blocks > 1) carry the 1 + softplus(raw) >= 1 law."""
        off = torch.tril(nn.functional.softplus(self.raw_L), diagonal=-1)
        if self._block_diag_lift is not None:
            off = off + self._block_diag_lift
        return off + self._eye_in

    def _U(self):
        """Unit-upper-triangular with non-negative off-diagonals
        (cached-identity variant of the parent's; export-safe)."""
        off = torch.triu(nn.functional.softplus(self.raw_U), diagonal=1)
        return off + self._eye_out

    def _d_effective(self):
        """Diagonal d = 1 + softplus(raw) >= 1; stable=True clamps to
        [1.0, d_max] (bounded amplification of negative log-mass)."""
        d = 1.0 + nn.functional.softplus(self.d)
        gate = getattr(self, '_current_gate', None)
        if gate is not None:
            d = d * gate
        if self.stable:
            d = d.clamp(1.0, self.d_max)
        return d

    # --- Ergodic overrides (noise enters raw domain; law preserved) ---
    def _L_eff(self):
        """Ergodic _L: perturb raw, then reapply the per-entry law."""
        raw = self.raw_L + self.var * self.noise_raw_L
        off = torch.tril(nn.functional.softplus(raw), diagonal=-1)
        if self._block_diag_lift is not None:
            off = off + self._block_diag_lift
        return off + self._eye_in

    def _U_eff(self):
        """Ergodic _U (cached-identity variant; export-safe)."""
        raw = self.raw_U + self.var * self.noise_raw_U
        off = torch.triu(nn.functional.softplus(raw), diagonal=1)
        return off + self._eye_out

    def _d_eff(self):
        """Ergodic d: noise (and the anneal bias) enter the RAW domain
        and the ``1 + softplus`` law is applied ONCE — the ``_L_eff``
        convention. ``bias = 1, var = 0`` therefore reproduces the
        clean ``_d_effective`` exactly (init ~1.0067, near-identity).

        Review fix 2026-06-11: the previous form softplused ``self.d``
        before the ergodic mix and then re-softplused, inflating the
        init diagonal to ~1.697 — ergodic meronymy folds started far
        more contractive than the documented near-identity point.
        """
        raw = self.bias * self.d + self.var * self.noise_d
        d = 1.0 + nn.functional.softplus(raw)
        gate = getattr(self, '_current_gate', None)
        if gate is not None:
            d = d * gate
        if self.stable:
            d = d.clamp(1.0, self.d_max)
        return d

    # --- Non-positive bias (b <= 0 by construction) ---
    # The legacy positive path (NonNegativeLinearLayer._effective_bias,
    # b = +softplus) and the unconstrained invertible path (raw
    # biasWeight) are both untouched; this class owns the third regime.
    def _effective_bias(self):
        """Bias b = -softplus(raw) <= 0; ergodic noise enters the raw."""
        if not self.hasBias:
            return 0
        if self.ergodic:
            raw = self.bias * self.biasWeight + self.var * self.biasNoise
            return -nn.functional.softplus(raw)
        return -nn.functional.softplus(self.biasWeight)

    def forwardBias(self, x):
        """Add the constrained (non-positive) bias."""
        if self.hasBias:
            x = x + self._effective_bias()
        return x

    def reverseBias(self, y):
        """Subtract the constrained (non-positive) bias."""
        if self.hasBias:
            y = y - self._effective_bias()
        return y


# =====================================================================
# SurfaceSchema -- universal surface-realization templates (UG = shared
# templates). doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter
# .md ("Absorb / Emit / Swap codification"). Each operator declares one
# of five templates describing how its surface marker and operand order
# behave in analysis (absorb / PS forward) and synthesis (emit / PS
# reverse). Only the marker SLOTS are declared here; the marker CONTENT
# is learned at runtime and bound per-operator (GrammarLayer.bind_marker
# / emit). Two operators "share one template" when they reference the
# same singleton -- conjunction / disjunction / isEqual are surface-
# indiscriminable and all use T2, discriminated by the slot-0 operator
# vector rather than by distinct schemas.
# =====================================================================
class SurfaceSchema:
    """One universal surface-realization template (T1-T5).

    Fields:
      template_id      'T1'..'T5'.
      name             human-readable template name.
      arity            operand count the template realizes (1 or 2).
      has_marker       True iff the template carries a learned marker slot.
      marker_position  where the marker sits: 'PRE' / 'INFIX' / 'SUF' /
                       'CIRCUM' / 'LEARNED' (position itself learned) /
                       'NONE'.
      order_mode       how operand order is realized: 'TRIVIAL' (unary),
                       'FREE' ({id,swap} either, unrecorded), 'MARKED'
                       ({id,swap} recorded), 'ELISION' (one survives, the
                       other absorbed).
      selects_op       True iff the marker may select which operator fires
                       (e.g. and / or / copula share one INFIX schema).

    Equality is by ``template_id`` so equal templates compare equal even
    if not the same object; the canonical templates below are singletons,
    so ``is`` also holds for operators that share a template.
    """

    __slots__ = ('template_id', 'name', 'arity', 'has_marker',
                 'marker_position', 'order_mode', 'selects_op')

    def __init__(self, template_id, name, arity, has_marker,
                 marker_position, order_mode, selects_op=False):
        """Store the template fields; see class docstring."""
        self.template_id = template_id
        self.name = name
        self.arity = int(arity)
        self.has_marker = bool(has_marker)
        self.marker_position = marker_position
        self.order_mode = order_mode
        self.selects_op = bool(selects_op)

    def __eq__(self, other):
        return (isinstance(other, SurfaceSchema)
                and other.template_id == self.template_id)

    def __hash__(self):
        return hash(self.template_id)

    def __repr__(self):
        return (f"SurfaceSchema({self.template_id} {self.name} "
                f"arity={self.arity} marker={self.has_marker} "
                f"order={self.order_mode})")


# The five universal templates. T4 (bare juxtapose) is the default base
# schema (GrammarLayer.surface_schema) so any operator round-trips by bare
# concatenation even before a marker is learned.
T1_UNARY_AFFIX = SurfaceSchema(
    'T1', 'UNARY_AFFIX', 1, True, 'LEARNED', 'TRIVIAL')
T2_BINARY_INFIX = SurfaceSchema(
    'T2', 'BINARY_INFIX', 2, True, 'INFIX', 'FREE', selects_op=True)
T3_BINARY_DIRECTIONAL = SurfaceSchema(
    'T3', 'BINARY_DIRECTIONAL', 2, True, 'LEARNED', 'MARKED')
T4_BINARY_JUXTAPOSE = SurfaceSchema(
    'T4', 'BINARY_JUXTAPOSE', 2, False, 'NONE', 'MARKED')
T5_BINARY_ELISION = SurfaceSchema(
    'T5', 'BINARY_ELISION', 2, False, 'NONE', 'ELISION')


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
        space_role:        which space the operator runs in:
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
    space_role             = 'SS'
    reads_activation = False

    # Surface-realization template (SurfaceSchema, above). T4 BINARY_
    # JUXTAPOSE is the default base schema -- bare concatenation, no
    # marker -- so any operator round-trips even before it learns a
    # marker. Concrete operators override this with their template
    # (see GRAMMAR_LAYER_CLASSES schema assignment in Language.py).
    surface_schema   = T4_BINARY_JUXTAPOSE

    # Chart authority registration -- Surface-3 wiring (see
    # doc/research/three-surfaces.md). When a SyntacticLayer registers
    # itself as the class-level authority, every GrammarLayer instance
    # constructed afterwards adds itself to the authority's roster, and
    # its `gated_run(x, fn, *args)` helper consults the authority
    # before running `fn`. The chart can then gate Pi / Sigma /
    # Not / etc. via `rule_probability(rule_name)`. Default: no
    # authority -> layers run unconditionally (backward-compat).
    _chart_authority = None

    def __init__(self, nInput=0, nOutput=0, butterfly=False, N=None):
        """Initialize GrammarLayer; allocate state for the class contract.

        See class docstring for invariants.

        Stage 5 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md;
        2026-05-27 revision): when ``butterfly=True``, allocate an
        FFT-style element-pair butterfly cascade over a flattened
        ``[B, M]`` view where ``M`` is the total element count
        (rounded up to the next power of two). Each cascade node is a
        single ``2 x 2`` LDU-factored block (``L``: unit lower-tri /
        one off-diag scalar; ``d``: 2 diagonal scalars; ``U``: unit
        upper-tri / one off-diag scalar). The 2-element per-pair op
        mixes pairs of elements; ``log2(M)`` levels with bit-reversal
        permutations between them connect every pair of elements
        across the whole flattened vector.

        ``N`` is the **flattened element count** (e.g., ``nInput *
        nInputDim`` for a per-position-D layer). Padded to the next
        power of two internally; the pad slots are zero-init and
        contribute identity at forward / reverse. Callers may pass
        ``N=None`` to defer; the wrapping space (e.g. ``PartSpace``)
        auto-computes it.

        Per-node LDU storage (vs the prior single packed ``W`` Parameter
        and ``torch.linalg.solve`` inverse): for ``2 x 2`` nodes the
        inverse is closed-form (sign-flip on off-diagonals + reciprocals
        on diagonals), no matrix solve required. This matches what
        PiLayer / SigmaLayer's invertibility already does at the layer
        level — the cascade uses the same parameterization on each
        node.
        """
        super().__init__(nInput, nOutput)
        # -- Butterfly cascade state (Stage 5, 2026-05-27 revision) --
        # FFT-style element-pair cascade over a flattened element axis.
        # Each node is a 2-element op with a 2x2 LDU block (4 scalars).
        # Identity init makes ``forward(x) == x`` at init.
        self.butterfly = bool(butterfly)
        if self.butterfly:
            if N is None or int(N) <= 0:
                raise ValueError(
                    "GrammarLayer: butterfly=True requires N >= 2 "
                    "(total flattened element count); got N={N!r}")
            N_in = int(N)
            if N_in < 2:
                raise ValueError(
                    f"GrammarLayer.butterfly: N must be >= 2; got {N_in}")
            # Pad to next power of two (FFT cascade structurally
            # requires 2^k); the extra slots are zero-padded at
            # forward time and stripped at the end.
            M = 1
            while M < N_in:
                M *= 2
            n_levels = int(math.log2(M))
            self.N = N_in            # nominal (unpadded) element count
            self.M_total = M         # padded power-of-two for the cascade
            self.n_levels = n_levels
            # The cascade composes square (nInput == nOutput required
            # so reverse round-trips at the layer level).
            if int(nInput) != int(nOutput):
                raise ValueError(
                    "GrammarLayer.butterfly: nInput must equal "
                    f"nOutput (got {nInput} vs {nOutput})")
            # Packed per-pair LDU storage. Each node owns 4 scalars:
            #   L: single sub-diagonal entry (the [1, 0] of the 2x2 L)
            #   d0, d1: the two diagonal entries
            #   U: single super-diagonal entry (the [0, 1] of the 2x2 U)
            # Identity init: L=0, d0=d1=1, U=0 → W=I per node.
            pair_count = M // 2
            self.butterfly_L = nn.Parameter(
                torch.zeros(n_levels, pair_count))
            self.butterfly_d = nn.Parameter(
                torch.ones(n_levels, pair_count, 2))
            self.butterfly_U = nn.Parameter(
                torch.zeros(n_levels, pair_count))
            # Per-level bit-reversal permutations placing XOR-
            # neighbour elements adjacent before pair-pack.
            perms = self._build_butterfly_perms(M, n_levels)
            self.register_buffer('butterfly_perms', perms)
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

    # ---------------------------------------------------------------
    # Butterfly cascade machinery (Stage 5).
    # ---------------------------------------------------------------
    @staticmethod
    def _build_butterfly_perms(N, n_levels):
        """Per-level permutation that makes XOR-neighbours adjacent.

        At level ``k`` (0-indexed) the cascade pairs slot ``i`` with
        slot ``i XOR (1 << k)``.  The permutation reorders the slot
        axis so that XOR-neighbours land in adjacent positions; the
        pair-pack reshape then groups them, the per-pair op fires,
        and the inverse permutation re-scatters the result.

        Returns a buffer ``[n_levels, N]`` of forward permutation
        indices.  ``_butterfly_inverse_perm`` is computed on demand
        from this -- not stored separately because each forward perm
        already costs ``N`` ints per level and the inverse is just
        a scatter.
        """
        perms = torch.empty((n_levels, N), dtype=torch.long)
        for k in range(n_levels):
            stride = 1 << k
            block = stride << 1
            order = []
            for start in range(0, N, block):
                for offset in range(stride):
                    order.append(start + offset)
                    order.append(start + offset + stride)
            perms[k] = torch.tensor(order, dtype=torch.long)
        return perms

    @staticmethod
    def _butterfly_inverse_perm(perm):
        """Return the inverse permutation: ``inv[perm[i]] = i``."""
        # ``empty`` (contiguous), not ``empty_like``: ``perm`` is a non-contiguous
        # view, so ``empty_like`` lowers to ``aten.empty_permuted`` (no portable
        # ExecuTorch kernel). ``inv`` is fully overwritten below -> identical.
        inv = torch.empty(perm.shape, dtype=perm.dtype, device=perm.device)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return inv

    # -- 2x2 LDU per-pair scalar ops (the cascade primitive) -----------
    #
    # Each node owns 4 scalars: L (sub-diag), d0, d1 (diag), U (super-
    # diag). The full 2x2 matrix is W = L_node @ D_node @ U_node where
    #   L_node = [[1, 0], [L, 1]]
    #   D_node = diag(d0, d1)
    #   U_node = [[1, U], [0, 1]]
    # Forward pair op: y = L_node @ D_node @ U_node @ x
    #   u_x_0 = x0 + U*x1
    #   u_x_1 = x1
    #   du_x_0 = d0 * u_x_0
    #   du_x_1 = d1 * u_x_1
    #   y0 = du_x_0
    #   y1 = L * du_x_0 + du_x_1
    # Reverse pair op: x = U_node^-1 @ D_node^-1 @ L_node^-1 @ y
    #   L_inv @ y: [y0, y1 - L*y0]
    #   D_inv @ ...: [.../d0, .../d1]
    #   U_inv @ ...: [a - U*b, b]
    # No torch.linalg.solve; no matrix inverse. Pure scalar ops.

    # Lower bound for the 2x2-LDU diagonal magnitude. Larger than the
    # full-LDU ``epsilon`` (1e-7) because the butterfly is a CASCADE: the
    # reverse divides by ``|d|`` at EVERY level, so ``(1/eps)^n_levels`` must
    # stay finite. 1e-3 bounds the worst-case reverse amplification while
    # leaving ``d`` essentially free near its unit (identity) init.
    _BUTTERFLY_D_EPS = 1e-3

    @staticmethod
    def _stable_pair_d(d_node):
        """Clamp the 2x2-LDU diagonal away from zero: ``|d|`` in
        ``[_BUTTERFLY_D_EPS, 1]``, sign preserved, and ``d == 0 -> +eps``
        (a plain ``sign(0)=0`` would otherwise leave it at zero).

        The reverse pair op divides by ``d``; an unclamped learnable diagonal
        that drifts to ~0 makes ``1/d`` blow up -- the radix reverse
        divergence (a ``d`` reaching 0 / a flushed denormal on MPS). Applied
        IDENTICALLY in the forward and reverse pair ops so ``x*d`` then
        ``/d`` cancel exactly: the cascade stays structurally invertible.
        """
        sign = torch.sign(d_node) + (d_node == 0).to(d_node.dtype)
        return sign * d_node.abs().clamp(GrammarLayer._BUTTERFLY_D_EPS, 1.0)

    def _butterfly_pair_forward(self, x_pair, L_node, d_node, U_node):
        """Apply 2x2 LDU pair op forward: y = L @ D @ U @ x_pair.

        Args:
            x_pair: ``[B, M, 2]`` paired input (M = total // 2).
            L_node: ``[M]`` sub-diagonal scalars.
            d_node: ``[M, 2]`` diagonal scalars.
            U_node: ``[M]`` super-diagonal scalars.

        Returns:
            ``[B, M, 2]`` paired output.
        """
        d_node = self._stable_pair_d(d_node)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]
        # U: [[1, U], [0, 1]]
        u_x0 = x0 + U_node * x1
        u_x1 = x1
        # D: diag(d0, d1)
        du_x0 = d_node[..., 0] * u_x0
        du_x1 = d_node[..., 1] * u_x1
        # L: [[1, 0], [L, 1]]
        y0 = du_x0
        y1 = L_node * du_x0 + du_x1
        return torch.stack([y0, y1], dim=-1)

    def _butterfly_pair_reverse(self, y_pair, L_node, d_node, U_node):
        """Apply 2x2 LDU pair op reverse: x = U^-1 @ D^-1 @ L^-1 @ y.

        Closed-form inverse from the LDU scalars (sign-flips on the
        off-diagonals + reciprocals on the diagonals). No matrix
        solve required.
        """
        d_node = self._stable_pair_d(d_node)
        y0 = y_pair[..., 0]
        y1 = y_pair[..., 1]
        # L^-1 = [[1, 0], [-L, 1]]
        a0 = y0
        a1 = y1 - L_node * y0
        # D^-1 = diag(1/d0, 1/d1)
        b0 = a0 / d_node[..., 0]
        b1 = a1 / d_node[..., 1]
        # U^-1 = [[1, -U], [0, 1]]
        x0 = b0 - U_node * b1
        x1 = b1
        return torch.stack([x0, x1], dim=-1)

    def _butterfly_flatten(self, x):
        """Flatten ``x`` to ``[B, M_total]`` (padded with zeros).

        Accepts any shape with leading batch dim B; flattens the
        remaining dims, then right-pads with zeros to reach
        ``self.M_total`` (the next power of two ≥ ``self.N``).

        Returns ``(flat, original_shape)`` where ``original_shape`` is
        the tuple of dims after B (for unflatten).
        """
        original_shape = tuple(x.shape[1:])
        B = x.shape[0]
        flat = x.reshape(B, -1)
        n = flat.shape[1]
        if n < self.M_total:
            pad = torch.zeros(
                B, self.M_total - n,
                device=flat.device, dtype=flat.dtype)
            flat = torch.cat([flat, pad], dim=1)
        elif n > self.M_total:
            # Should not happen if constructor sized M_total correctly,
            # but guard against caller passing larger input than spec'd.
            raise RuntimeError(
                f"butterfly_flatten: input has {n} elements per batch "
                f"row but cascade is sized for M_total={self.M_total}.")
        return flat, original_shape

    def _butterfly_unflatten(self, flat, original_shape):
        """Inverse of ``_butterfly_flatten``: strip pad, reshape."""
        B = flat.shape[0]
        n = 1
        for d in original_shape:
            n *= d
        return flat[:, :n].reshape((B,) + original_shape)

    def _butterfly_forward(self, x):
        """FFT-style element-pair butterfly cascade forward.

        Flattens ``x`` to a 1-D-per-batch vector of length ``M_total``
        (padded power-of-two). At each level:
          1. Permute along the element axis (XOR-neighbour pairs
             adjacent).
          2. Pack adjacent pairs into ``[B, M_total // 2, 2]``.
          3. Apply 2x2 LDU pair op with the level's L, d, U scalars.
          4. Unpack to ``[B, M_total]``.
          5. Inverse-permute to restore the natural element order.

        Identity init (L=0, d0=d1=1, U=0) makes ``forward(x) == x``
        at init (modulo the zero-pad strip).
        """
        flat, original_shape = self._butterfly_flatten(x)
        for level in range(self.n_levels):
            perm = self.butterfly_perms[level]
            inv_perm = self._butterfly_inverse_perm(perm)
            flat = flat[:, perm]
            pair = flat.reshape(flat.shape[0], -1, 2)
            pair = self._butterfly_pair_forward(
                pair,
                self.butterfly_L[level],
                self.butterfly_d[level],
                self.butterfly_U[level])
            flat = pair.reshape(flat.shape[0], -1)
            flat = flat[:, inv_perm]
        return self._butterfly_unflatten(flat, original_shape)

    def _butterfly_reverse(self, y):
        """Inverse of ``_butterfly_forward`` (LDU sign-flip + reciprocal).

        Runs the per-level inverse pair op in reverse level order.
        Composed with ``_butterfly_forward`` this restores the input
        up to numerical precision (zero-pad slots stay at zero
        through the inverse cascade since identity init).
        """
        flat, original_shape = self._butterfly_flatten(y)
        for level in reversed(range(self.n_levels)):
            perm = self.butterfly_perms[level]
            inv_perm = self._butterfly_inverse_perm(perm)
            flat = flat[:, perm]
            pair = flat.reshape(flat.shape[0], -1, 2)
            pair = self._butterfly_pair_reverse(
                pair,
                self.butterfly_L[level],
                self.butterfly_d[level],
                self.butterfly_U[level])
            flat = pair.reshape(flat.shape[0], -1)
            flat = flat[:, inv_perm]
        return self._butterfly_unflatten(flat, original_shape)

    def forward(self, x):
        """Default forward dispatch.

        When ``self.butterfly`` is True, runs the butterfly cascade.
        Otherwise raises: subclasses must override ``forward`` for
        their non-butterfly behaviour.
        """
        if self.butterfly:
            return self._butterfly_forward(x)
        raise NotImplementedError(
            f"{type(self).__name__}.forward: not implemented "
            "(set butterfly=True to use the cascade or override "
            "forward in the subclass)")

    def reverse(self, y):
        """Default reverse dispatch.

        When ``self.butterfly`` is True, runs the butterfly cascade
        inverse. Otherwise raises.
        """
        if self.butterfly:
            return self._butterfly_reverse(y)
        raise NotImplementedError(
            f"{type(self).__name__}.reverse: not implemented")

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
    def absorb(self, left, right, *, marker_id=None, weight=1.0):
        """Absorb a sub-span as the operator's surface marker (analysis /
        PS forward). Returns ``left`` -- the content operand survives; the
        marker ``right`` is consumed (not a content child).

        doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md
        ("Absorb / Emit / Swap codification"): the surface marker is
        learned and owned by the operator. When the analyzer knows the
        absorbed sub-span's PS codebook identity it passes ``marker_id``;
        absorb then binds it to this operator (many-to-one: many surface
        markers -> one operator) with co-occurrence ``weight`` so the
        heaviest becomes the canonical default for ``emit``. Called with
        only ``(left, right)`` (the legacy sugar path) it is a pure
        left-pass and records nothing.

        Subclasses may override to record richer side-effects; the base
        is the left-pass + optional marker binding.
        """
        if marker_id is not None:
            self.bind_marker(marker_id, weight=weight)
        return left

    # -- Learned surface markers (owned by the operator) ----------------
    #
    # The marker SLOT is declared by ``surface_schema`` (SurfaceSchema);
    # the marker CONTENT is learned here from co-occurrence. Binding is
    # many-to-one (markers -> operator) with a canonical operator ->
    # default-marker for emit. Stored as ``{marker_id: weight}`` on the
    # instance; ``marker_id`` is a hashable PS codebook identity.
    def _markers(self):
        """Return the per-instance marker-binding dict, creating it lazily
        (robust to subclasses that don't call ``super().__init__``)."""
        markers = getattr(self, '_marker_bindings', None)
        if markers is None:
            markers = {}
            self._marker_bindings = markers
        return markers

    def bind_marker(self, marker_id, *, weight=1.0):
        """Bind a surface marker (PS codebook id) to this operator from
        co-occurrence (many-to-one). Repeated / multiple markers
        accumulate weight; the heaviest is the canonical default."""
        markers = self._markers()
        markers[marker_id] = markers.get(marker_id, 0.0) + float(weight)

    def bound_markers(self):
        """Return ``{marker_id: weight}`` of all markers bound to this
        operator (a copy; many-to-one marker -> operator map)."""
        return dict(self._markers())

    def canonical_marker(self):
        """Operator -> default marker (PS codebook id) for emit: the
        most co-occurring bound marker, or ``None`` if none bound."""
        markers = self._markers()
        if not markers:
            return None
        return max(markers.items(), key=lambda kv: kv[1])[0]

    def emit(self, *, marker_id=None):
        """Realize the operator's surface marker (synthesis / PS reverse)
        -- the inverse of :meth:`absorb`.

        Uses recorded route metadata, NEVER the lossy
        ``generate()``=(parent, parent) inverse (vector ``generate`` is a
        lossy ``(parent, parent)`` split, so a faithful round-trip MUST
        replay the recorded marker). Returns the marker PS codebook id:
        the route-recorded ``marker_id`` when supplied (exact replay),
        else the operator's canonical (most-bound) default. Returns
        ``None`` for a marker-free schema (T4 / T5) -- those round-trip by
        bare juxtaposition / elision with no marker to place.
        """
        if not self.surface_schema.has_marker:
            return None
        if marker_id is not None:
            return marker_id
        return self.canonical_marker()

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


class SigmaLayer(GrammarLayer):
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
                 monotonic=False, butterfly=False, N=None):
        """Initialize SigmaLayer; allocate state for the class contract.

        See class docstring for invariants.

        ``butterfly`` / ``N`` (Stage 5, doc/plans/2026-05-26-two-loop-
        pi-sigma-substrate.md): when ``butterfly=True``, the layer
        delegates ``forward`` / ``reverse`` to the inherited
        GrammarLayer butterfly cascade. ``N`` is the per-position
        slot count and must be a power of two; ``nInput == nOutput``
        defines D, the per-slot feature width. The per-pair op is
        sigma-style (atanh -> matmul -> tanh) applied at each
        cascade node.
        """
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        self.invertible = invertible
        self.ergodic    = ergodic
        self.nonlinear  = nonlinear
        self.stable     = stable
        self.monotonic  = monotonic
        self.activation = torch.zeros(1, nOutput, 1)
        if self.butterfly:
            # In butterfly mode the cascade owns the parameterised
            # weight; no separate inner LDU layer is needed for the
            # unary feature fold (per-pair op runs on packed pairs).
            # ``self.layer`` is left unset; the legacy unary forward
            # path is replaced by ``_butterfly_forward``.
            self.layer = None
        else:
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

    # -- Butterfly per-pair op -----------------------------------
    def _butterfly_pair_op(self, x_pair, W_node):
        """Sigma-style per-pair op for the butterfly cascade.

        atanh -> einsum -> tanh applied per pair (the same math the
        unary ``forward`` path uses, but per-pair-batched over the
        packed pair axis).
        """
        if self.nonlinear:
            x_pair = torch.atanh(x_pair.clamp(-1 + epsilon, 1 - epsilon))
        out = torch.einsum('bmi,mij->bmj', x_pair, W_node)
        if self.nonlinear:
            out = torch.tanh(out)
        return out

    def _butterfly_pair_op_reverse(self, y_pair, W_inv_node):
        """Reverse of ``_butterfly_pair_op``; inverts the per-pair op."""
        if self.nonlinear:
            y_pair = torch.atanh(y_pair.clamp(-1 + epsilon, 1 - epsilon))
        out = torch.einsum('bmi,mij->bmj', y_pair, W_inv_node)
        if self.nonlinear:
            out = torch.tanh(out)
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

        Butterfly mode (``self.butterfly``): wrap the FFT-style
        element-pair cascade (per the 2026-05-27 revision) with the
        same atanh / tanh transforms as the per-position path. The
        cascade replaces the ``W @ x`` linear core; the surrounding
        nonlinearity preserves the sigma semantics. ``binary`` and
        ``gate`` are not honoured in butterfly mode.
        """
        if self.butterfly:
            if self.nonlinear:
                x = torch.atanh(x.clamp(-1 + epsilon, 1 - epsilon))
            y = self._butterfly_forward(x)
            if self.nonlinear:
                y = torch.tanh(y)
            self.activation = y.detach()
            return y
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

        Butterfly mode: wrap ``_butterfly_reverse`` with atanh / tanh
        symmetric to forward.
        """
        if self.butterfly:
            if self.nonlinear:
                y = torch.atanh(y.clamp(-1 + epsilon, 1 - epsilon))
            x = self._butterfly_reverse(y)
            if self.nonlinear:
                x = torch.tanh(x)
            self.activation = x.detach()
            return x
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

    def synthesize_over_set(self, part_codes):
        """σ-fold a SET of M codes ``[..., M, D]`` into ONE code ``[..., D]``
        that geometrically contains them (doc/specs/mereological-order-
        raising.md: a higher-order PART synthesized from its constituents).

        The M-way generalization of :meth:`compose`'s binary atanh-sum: every
        constituent's logit contributes additively before one shared
        ``self.layer.forward`` + ``tanh``, so the parent contains all M.
        Order-invariant, ``M -> 1``, reuses the SAME ``self.layer`` (NO new
        parameters), and for ``M == 2`` reduces exactly to ``compose``. The
        balanced split in :meth:`generate` inverts it (each constituent
        ``= tanh(s / M)``) when the inner layer is invertible -- the explicit
        ``part_chain`` provenance is the fallback when it is not."""
        if self.nonlinear:
            a = torch.atanh(part_codes.clamp(-1 + epsilon, 1 - epsilon))
        else:
            a = part_codes
        a_sum = a.sum(dim=-2)
        out = self.layer.forward(a_sum)
        if self.nonlinear:
            out = torch.tanh(out)
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
        # Local import: Layers.py is imported by Optimizer's eventual
        # callers; keep the dep local to this smoke-test path to avoid
        # any import-order surprise during module load.
        from Optimizer import Adam
        optimizer = Adam(layer.getParameters(), lr=0.01)
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


class RunStructureLayer(Layer):
    """Fixed-shape mereological run / gap / containment structure over a SET of
    ``.where`` brackets (``doc/specs/mereological-order-raising.md``).

    Given the sibling part spans under a candidate whole -- ``spans``
    ``[..., K, 2]`` of ``(start, end)`` with a ``valid`` mask -- it reports how
    many CONTIGUOUS RUNS they form (``n_runs`` == the part count == the
    part/whole ratio numerator) plus the pairwise containment mask (the
    ``A isa B`` test). Subsymbolic + compiled-forward-safe: NO host-side control
    flow, no sort, no variable shapes. A span ``i`` starts a new run iff NO
    valid earlier span (by start, index tie-broken) reaches ``start_i - tol`` --
    one ``[..., K, K]`` broadcast of comparisons + an ``any``-reduction, O(K^2)
    over the small slot count K. Parameter-free (a measure), but a real
    ``Layer`` so the Start/End/Reset/paramUpdate cascade reaches it via
    ``self.layers``.
    """

    def __init__(self, nWhere=2, contiguity_tol=0.5):
        """``nWhere`` is the bracket width (2); ``contiguity_tol`` is the error
        band around "the spans touch" (gap ~= 0): touching / overlapping spans
        (gap 0) merge into one run, while a skipped position (gap >= 1, e.g. a
        space that is not itself a part -- a word boundary) opens a new run."""
        super().__init__(nWhere, nWhere)
        self.contiguity_tol = float(contiguity_tol)

    def forward(self, spans, valid=None, tol=None):
        """``spans`` ``[..., K, 2]`` (start, end). ``valid`` ``[..., K]`` bool
        (default ``end > start`` -- analysis-span pads are ``(0, 0)``).

        Returns a dict of FIXED-SHAPE tensors:
            ``n_runs`` / ``n_gaps`` ``[...]`` (long);
            ``is_run_start`` / ``valid`` ``[..., K]`` (bool);
            ``contained_mask`` ``[..., K, K]`` bool (``[..., i, j]`` = span i
                within span j: ``start_i >= start_j - tol`` and
                ``end_i <= end_j + tol``, both valid; the diagonal is True);
            ``ratio`` ``[...]`` (== ``n_runs``, the routing input: one run =>
                singleton / refine, many => integrate / disintegrate);
            ``span_extent`` ``[..., K]`` per-part width (0 for pads);
            ``max_extent`` / ``total_extent`` ``[...]`` -- the extent dimension
                the run-count misses: a singleton with large ``max_extent`` is a
                long part that property-tiling should analyse;
            ``tightest_container`` ``[..., K]`` (long) -- force #1: for each span
                the index of its SMALLEST strict container (the smallest whole
                over it), or ``-1`` when none. The "A isa B" edge with no
                intervening smaller whole;
            ``route_hint`` ``[...]`` (long) -- the three-aspect routing class:
                ``0`` NULL (no valid runs), ``1`` REFINE (one contiguous run /
                singleton), ``2`` RAISE (>1 run / discontiguous).
        """
        tol = self.contiguity_tol if tol is None else float(tol)
        starts = spans[..., 0]
        ends = spans[..., 1]
        if valid is None:
            valid = ends > starts
        K = int(starts.shape[-1])
        idx = torch.arange(K, device=spans.device)
        si = starts.unsqueeze(-1)                  # [..., K, 1] (i)
        sj = starts.unsqueeze(-2)                  # [..., 1, K] (j)
        ei = ends.unsqueeze(-1)
        ej = ends.unsqueeze(-2)                    # [..., 1, K]
        vj = valid.unsqueeze(-2)                   # [..., 1, K]
        # j precedes i (start order, index tie-break) AND j's end reaches i's
        # start within tol -> i is connected to an earlier span.
        precedes = (sj < si) | ((sj == si)
                                & (idx.unsqueeze(-1) > idx.unsqueeze(-2)))
        reaches = ej >= (si - tol)
        has_earlier = (precedes & reaches & vj).any(dim=-1)        # [..., K]
        is_run_start = valid & (~has_earlier)
        n_runs = is_run_start.sum(dim=-1)                           # [...]
        n_gaps = (n_runs - 1).clamp(min=0)
        contained_mask = ((si >= sj - tol) & (ei <= ej + tol)
                          & valid.unsqueeze(-1) & vj)               # [..., K, K]
        # Extent signals (the "how WIDE is each part" dimension, fixed-shape):
        # the run-count alone cannot tell a basic-level word from a long
        # single part (e.g. "antidisestablishmentarianism" -> 1 run, extent 28).
        # A singleton (n_runs == 1) with a LARGE max_extent is the
        # property-tiling-analysis trigger that n_runs misses; routing combines
        # the two. Pads (invalid) contribute 0.
        span_extent = (ends - starts).clamp(min=0) * valid.to(ends.dtype)  # [..., K]
        max_extent = span_extent.amax(dim=-1)                              # [...]
        total_extent = span_extent.sum(dim=-1)                            # [...]
        # Force #1 -- the TIGHTEST link (doc/specs/mereological-order-raising.md:
        # "link the largest part with the smallest whole; skip when a bigger
        # part or smaller whole already subsumes the relation"). For each span i
        # (a candidate part), pick its SMALLEST STRICT container j (i != j, i in
        # j, minimal extent) -- that is the smallest whole over it; -1 when none.
        # Fixed-shape: mask the diagonal out of contained_mask, fill non-container
        # extents with +big, argmin over j. (Ties -> lowest index, the repo's
        # index tie-break.) This is the "A isa B" edge with no intervening whole.
        eye = torch.eye(K, dtype=torch.bool, device=spans.device)
        eye = eye.view((1,) * (contained_mask.dim() - 2) + (K, K))
        strict = contained_mask & (~eye)                            # [..., K, K]
        has_container = strict.any(dim=-1)                          # [..., K]
        _BIG = float(2 ** 30)
        cand_ext = torch.where(
            strict, span_extent.unsqueeze(-2),
            torch.full_like(span_extent.unsqueeze(-2), _BIG))       # [..., K, K]
        argmin_j = cand_ext.argmin(dim=-1)                          # [..., K]
        tightest_container = torch.where(
            has_container, argmin_j, torch.full_like(argmin_j, -1))  # [..., K]
        # Three-aspect routing hint (doc/specs/mereological-order-raising.md
        # "contiguity routes refine vs raise"): per whole, classify its .where
        # run-structure -- 0 = NULL (no valid runs, nothing to act on);
        # 1 = REFINE (one contiguous run = a localized singleton -> chunk finer
        # / property-tile); 2 = RAISE (>1 run = discontiguous -> raise order).
        # A computed signal from n_runs; the routing ACTION (the feedback loop)
        # consumes it later.
        route_hint = torch.where(
            n_runs <= 0, torch.zeros_like(n_runs),
            torch.where(n_runs == 1, torch.ones_like(n_runs),
                        torch.full_like(n_runs, 2)))                  # [...]
        return {"n_runs": n_runs, "n_gaps": n_gaps,
                "is_run_start": is_run_start, "valid": valid,
                "contained_mask": contained_mask, "ratio": n_runs,
                "span_extent": span_extent, "max_extent": max_extent,
                "total_extent": total_extent,
                "tightest_container": tightest_container,
                "route_hint": route_hint}


# Char-class property tiling (doc/specs/mereological-order-raising.md "Analysis
# = property-tiling"). A char-class property is a BINARY TILING of a byte input
# into {has-class} / {not} -- the analysis-side (WholeSpace π) dual of a
# synthesis chunk. ASCII byte ranges (inclusive) per class; selecting a SET of
# classes ORs them (the analysis-side OR over .index-selected properties, dual
# to PartSpace's AND over particles).
LETTER, DIGIT, WHITESPACE, PUNCT = 0, 1, 2, 3
# Key-only sentinel for the generic "word" whole-TYPE (a whole, not a tiling
# predicate): used as a property_class_whole key, never passed to
# char_class_region (whose spans come from the whitespace analysis cut).
WORD = 100
# Key-only sentinels for the two POLES of the relation-only CS symbol lattice
# (doc/specs/mereological-order-raising.md, the word/object/meta creation). A
# freshly-minted OBJECT-symbol B starts maximally unspecified: its only WHOLE is
# the UNIVERSE (the top -- the entire input extent, more general than a word)
# and its only PART is the ATOM level (the bottom -- "made of some atoms",
# unspecified). Successive refinement (σ synthesizes the atoms into higher-order
# parts; π splits the universe into finer wholes -- the lifecycle loop) replaces
# these poles with concrete part / whole codes. Like WORD, these are key-only
# whole/part-code sentinels in the relation-only table, never tiling predicates.
UNIVERSE = 101
ATOM = 102
_CHAR_CLASS_RANGES = {
    LETTER: [(65, 90), (97, 122)],          # A-Z, a-z
    DIGIT: [(48, 57)],                       # 0-9
    WHITESPACE: [(9, 9), (10, 10), (13, 13), (32, 32)],  # tab, LF, CR, space
    PUNCT: [(33, 47), (58, 64), (91, 96), (123, 126)],   # ASCII punctuation
}


def char_class_region(byte_input, class_ids):
    """Binary-tiling region of ``byte_input`` for a SET of char classes.

    ``byte_input`` ``[..., N]`` (byte values 0..255); ``class_ids`` an iterable
    of ``LETTER`` / ``DIGIT`` / ``WHITESPACE`` / ``PUNCT``. Returns a signed
    region ``[..., N]`` in ``{+1, -1}``: ``+1`` where the byte is in ANY
    selected class (the OR-union -- the has-property whole), ``-1`` otherwise
    (the complement whole). Same ``>0 has / <=0 not`` convention as
    :meth:`Codebook.materialize_property`. A fixed 256-entry membership LUT
    (built from the selected classes, then gathered) keeps this a fixed-shape
    tensor op -- no host-side per-position control flow (the class loop is over
    the fixed selected set, not the data)."""
    if not torch.is_tensor(byte_input):
        byte_input = torch.as_tensor(byte_input)
    b = byte_input.long()
    lut = torch.zeros(256, dtype=torch.bool)
    for cid in class_ids:
        for (lo, hi) in _CHAR_CLASS_RANGES[int(cid)]:
            lut[lo:hi + 1] = True
    member = lut.to(b.device)[b.clamp(min=0, max=255)]
    return member.to(torch.float32) * 2.0 - 1.0


class PropertyTilingLayer(Layer):
    """WholeSpace char-class property tiling (doc/specs/mereological-order-
    raising.md): an ANALYSIS op that binary-tiles a byte input into
    ``{has-class}`` / ``{not}`` for a selected SET of char classes
    (``LETTER`` / ``DIGIT`` / ``WHITESPACE`` / ``PUNCT``), OR-unioned -- the
    analysis-side dual of a synthesis chunk. Parameter-free, fixed-shape (like
    :class:`RunStructureLayer`). ``forward(byte_input, class_ids)`` returns the
    signed region ``[..., N]`` in ``{+1, -1}`` (``+1`` = the has-property whole,
    ``-1`` = the complement whole), each carrying a ``.what`` (the class) and,
    via the run/containment measure over the ``+1`` runs, a ``.where``."""

    LETTER, DIGIT, WHITESPACE, PUNCT = LETTER, DIGIT, WHITESPACE, PUNCT

    def __init__(self):
        super().__init__(1, 1)              # parameter-free measure; nominal widths

    def forward(self, byte_input, class_ids):
        """Signed ``{+1,-1}`` tiling region for the OR-union of ``class_ids``."""
        return char_class_region(byte_input, class_ids)


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

# NotLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).

# NonLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


# IntersectionLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


# UnionLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).
# LiftLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


# LowerLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


# SymbolizeLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


# ConjunctionLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


# DisjunctionLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


# IsEqualLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


class EqualLayer(GrammarLayer):
    """``C -> equal(C, C)`` -- geometric identity on concept bivectors.

    C-space_role identity: ``equal(A, B)`` returns the mutual-parthood
    score ``part(A, B) * part(B, A)`` on bivector activations,
    yielding 1 where A and B are co-located on the bivector cone
    and 0 where they are disjoint. Acts directly on the concept-space_role
    bivector activation; no codebook lookup is required.

    Compare with the S-space_role ``isEqual`` which produces a single
    parent symbol asserting identity — a higher-epistemic-level
    wholeness operation that cannot be expressed at the subsymbolic
    level.

    Lossy with ``(parent, parent)`` pseudo-inverse on reverse.
    """
    rule_name        = "equal"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'CS'
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


# PartLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


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
    space_role             = 'CS'
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
    space_role             = 'CS'
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
    space_role             = 'CS'
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
    space_role             = 'CS'
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


# QueryLayer -- moved to Language.py (2026-05-29 grammar-file-refactor §5).


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
    """Pairwise luminosity ``area - overlapArea * |t_A - t_B|`` in [-1, 1].

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
    """One-step kernel overlap ``K(child, parent)`` in (0, 1] per the
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
    space_role             = 'CS'
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
    space_role             = 'CS'
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
    space_role             = 'SS'
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


# GRAMMAR_LAYER_CLASSES -- moved to Language.py (2026-05-29 grammar-file-refactor §5). Backward-compat lookup is provided by the module-level __getattr__ at the end of this file.


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


class PiLayer(GrammarLayer):
    r"""Multiplicative boundary fold feature of the subsymbolic loop:
    ``[-1, 1] -> [-1, 1]``.

    Substrate, not a grammar operation. Instantiated directly by spaces
    (``PartSpace.pi_input`` / ``pi_concept``) and used by the
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
                 monotonic=False, nonlinear=True,
                 butterfly=False, N=None):
        """Initialize PiLayer; allocate state for the class contract.

        See class docstring for invariants.

        ``butterfly`` / ``N`` (Stage 5, doc/plans/2026-05-26-two-loop-
        pi-sigma-substrate.md): when ``butterfly=True``, the layer
        delegates ``forward`` / ``reverse`` to the inherited
        GrammarLayer butterfly cascade. ``N`` is the per-position
        slot count and must be a power of two; ``nInput == nOutput``
        defines D. The per-pair op is pi-style (atanh -> matmul ->
        tanh) applied at each cascade node.
        """
        super().__init__(nInput, nOutput, butterfly=butterfly, N=N)
        self.invertible = invertible
        self.stable     = stable
        self.hasBias    = hasBias
        self.monotonic  = monotonic
        self.nonlinear  = nonlinear
        if self.butterfly:
            # Butterfly mode owns the per-position weight cascade;
            # the legacy unary inner layer is not constructed.
            self.layer = None
        else:
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
    def bias(self):
        if self.layer is None:
            return None
        return self.layer.bias
    @property
    def var(self):
        if self.layer is None:
            return None
        return self.layer.var

    def resample_noise(self):
        """Resample noise.

        See class docstring for the operation contract.
        """
        if self.layer is not None:
            self.layer.resample_noise()

    # -- Symmetric domain transforms ----------------------------------

    def _to_mult(self, x):
        """Map [-1, 1] -> (0, inf), identity at 0 -> 1.

        The clamp into the open ``(-1, 1)`` interval is UNCONDITIONAL:
        ``(1 + x)/(1 - x)`` is positive only there, and its result
        always feeds a ``log`` (the forward fold for both ``nonlinear``
        settings, and the ``nonlinear=True`` reverse). An input at or
        outside ``+-1`` -- e.g. an un-normalized percept on an untrained
        model, since percept normalization runs AFTER ``pi.forward`` --
        would otherwise make the ratio ``<= 0`` and inject
        ``log(<=0) = NaN``. Enforcing the documented ``[-1, 1]`` domain
        keeps the fold finite; strictly in-domain values
        (``|x| < 1 - eps``) are unchanged. The clamp does NOT swallow
        non-finite input (``clamp(nan/inf)`` stays nan/inf), so a genuine
        upstream divergence still propagates and fails loud.
        """
        x = x.clamp(-1 + self._eps, 1 - self._eps)
        return (1 + x) / (1 - x)

    def _from_mult(self, y):
        """Map (0, inf) -> (-1, 1), identity at 1 -> 0."""
        return (y - 1) / (y + 1)

    # -- Butterfly per-pair op (Stage 5) ------------------------------
    def _butterfly_pair_op(self, x_pair, W_node):
        """Pi-style per-pair op for the butterfly cascade.

        atanh -> einsum -> tanh applied per pair. The atanh / tanh
        framing is the unary pi math (the multiplicative AND fold in
        log-mult domain reduces to the same atanh-> linear -> tanh
        kernel once the bias term is dropped, which the cascade does
        intentionally -- the per-node weights carry all parametric
        flexibility).
        """
        if self.nonlinear:
            x_pair = torch.atanh(x_pair.clamp(-1 + epsilon, 1 - epsilon))
        out = torch.einsum('bmi,mij->bmj', x_pair, W_node)
        if self.nonlinear:
            out = torch.tanh(out)
        return out

    def _butterfly_pair_op_reverse(self, y_pair, W_inv_node):
        """Reverse of ``_butterfly_pair_op``; inverts the per-pair op."""
        if self.nonlinear:
            y_pair = torch.atanh(y_pair.clamp(-1 + epsilon, 1 - epsilon))
        out = torch.einsum('bmi,mij->bmj', y_pair, W_inv_node)
        if self.nonlinear:
            out = torch.tanh(out)
        return out

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

        Butterfly mode: wrap the FFT-style element-pair cascade with
        the pi log-domain transforms (``_to_mult`` / log / cascade /
        exp / ``_from_mult``). The cascade replaces the ``W @ l``
        linear core in log-mult space; the surrounding nonlinearity
        preserves the pi semantics.
        """
        if self.butterfly:
            if binary:
                x = Ops.top2_select_ste(x)
            m = self._to_mult(x)
            l = torch.log(m)
            wl = self._butterfly_forward(l)
            if self.nonlinear:
                out = torch.tanh(wl / 2)
            else:
                out = torch.exp(wl)
            return out
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

        Butterfly mode: wrap ``_butterfly_reverse`` with pi log-domain
        transforms symmetric to forward.
        """
        if self.butterfly:
            if self.nonlinear:
                m = self._to_mult(y)
                l = torch.log(m)
            else:
                # ``nonlinear=False`` forward emits a positive mult-domain
                # value (``exp(wl) in (0, inf)``); the reverse ``log`` needs
                # that domain. A reconstruction seed driven from an arbitrary
                # (signed) idea can land <= 0 here -- clamp to the log domain
                # so the leg stays finite instead of injecting
                # ``log(<=0) = NaN``. NaN/Inf inputs (a genuine upstream
                # divergence) still propagate through the clamp and fail loud
                # downstream -- the clamp rescues only finite out-of-domain
                # values, it does not swallow non-finite ones.
                l = torch.log(y.clamp(min=self._eps))
            lx = self._butterfly_reverse(l)
            # ``tanh(lx/2)`` equals ``_from_mult(exp(lx))`` for finite ``lx``
            # but is saturation-safe (no ``inf/inf = NaN`` on overflow), so it
            # serves both the nonlinear and the former ``_from_mult(exp)``
            # branch.
            out = torch.tanh(lx / 2)
            return out
        # Inline inner-reverse: log-domain multiplicative AND fold inverse.
        self.layer._current_gate = gate
        try:
            W_inv = self.layer.compute_Winverse_current()
            y = y.to(W_inv.device)
            if self.nonlinear:
                m = self._to_mult(y)
                l = torch.log(m)
            else:
                # See the butterfly branch above: ``nonlinear=False`` reverse
                # expects ``y`` in the positive mult-domain the forward emits.
                # Clamp to the log domain so a signed reconstruction seed
                # stays finite, while NaN/Inf still propagate (fail-loud
                # preserved -- the clamp does not swallow non-finite input).
                l = torch.log(y.clamp(min=self._eps))
            b = self.layer._effective_bias()
            lx = (l - b) @ W_inv
            # ``tanh(lx/2)`` == ``_from_mult(exp(lx))`` but saturation-safe.
            out = torch.tanh(lx / 2)
            if self.layer.ergodic:
                self.resample_noise()
        finally:
            self.layer._current_gate = None
        return out

    def factorize_over_set(self, y, M, gate=None):
        r"""π-FACTORIZE a fold into M EQUAL factors -- the log-domain DUAL of
        :meth:`SigmaLayer.generate`'s balanced split (doc/specs/mereological-
        order-raising.md "Ramsification with the towers"; the π-analyse split's
        property-source). The π forward folds factors MULTIPLICATIVELY (their
        log-memberships SUM through ``W``); this recovers the summed
        log-membership ``lx = (log(_to_mult(y)) - b) @ W^-1`` exactly as
        :meth:`reverse`, then SPLITS IT EQUALLY by ``M`` and exits, returning ONE
        representative factor ``[..., nInput]`` in ``[-1, 1]`` (the M factors are
        equal, exactly as ``σ.generate`` returns equal halves). ``M == 1``
        reduces to :meth:`reverse`. Where σ splits a SUM by M in the atanh
        domain, π splits a PRODUCT by M in the log-mult domain. Requires
        ``invertible=True``; round-trips (re-folding M equal copies recovers
        ``y``)."""
        M = max(1, int(M))
        if self.butterfly:
            if self.nonlinear:
                l = torch.log(self._to_mult(y))
            else:
                l = torch.log(y.clamp(min=self._eps))
            lx = self._butterfly_reverse(l)
            return torch.tanh((lx / float(M)) / 2.0)
        self.layer._current_gate = gate
        try:
            W_inv = self.layer.compute_Winverse_current()
            y = y.to(W_inv.device)
            if self.nonlinear:
                l = torch.log(self._to_mult(y))
            else:
                l = torch.log(y.clamp(min=self._eps))
            b = self.layer._effective_bias()
            lx = (l - b) @ W_inv
            out = torch.tanh((lx / float(M)) / 2.0)
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


class PiLayer2(Layer):
    r"""Membership-domain analysis fold (MeronomySpec §4; MeronomyPlan
    Stage 2): the part-maker / intersection.

    A NEW class beside the legacy :class:`PiLayer`, which stays untouched
    for non-meronymic consumers. Runs on MEMBERSHIPS ``m in [0, 1]^D``
    (the meronomy chart) -- ground content natively; epistemic scalars
    enter only through the evaluation chart ``Ops.eval_chart`` (§3).
    NO odds transforms anywhere in this class: the legacy odds embedding
    ``(1+x)/(1-x)`` -- the ``p+n=1`` zero-ignorance slice, Bayesian
    sharpening rather than contraction -- is retired from the meronymic
    path.

        forward:  z = exp(W · log(clamp(m, EPS_LOG, 1)) + b)
        reverse:  m = exp(W^-1 · (log z − b))           (exact LDU solve)

    The inner :class:`ContractiveInvertibleLinearLayer` enforces
    ``diag(W) >= 1`` (per operand block for binary folds),
    ``offdiag(W) >= 0``, ``b <= 0`` by construction, so CONTRACTION IS A
    THEOREM: ``pi(m) <= m`` for all m, and for the binary ``2D -> D``
    form (``blocks=2``) the acceptance bound ``pi <= min(operands)``
    holds elementwise -- operand mixing is safe because off-diagonal
    terms add negative log-mass (they can only deepen the part).

    Identities / absorbers (§4): ``1`` is the pi-identity (intersecting
    with everything is a no-op -- exact at near-identity init, to eps in
    general) and ``0`` is the pi-absorber (a single zero witness
    annihilates; exclusion propagates -- a THEOREM for all admissible
    weights, since ``diag >= 1`` forces ``z <= EPS_LOG`` there).

    The forward floors m at ``EPS_LOG`` (the §4 eps-floor near m = 0)
    and ceils at 1 (log m <= 0 is what makes the contraction theorem
    bite); NaN/Inf inputs propagate through the clamp and fail loud.
    The binary reverse stays recommender-shaped as today
    (``Ops.conjunctionReverse`` -- codebook search; the layer's own
    ``reverse`` is the exact unary inverse / canonical binary preimage).
    """

    def __init__(self, nInput, nOutput, naive=False, ergodic=False,
                 hasBias=True, stable=True, blocks=1, d_max=None):
        """Initialize PiLayer2; see class docstring for the kernel law.

        ``stable=True`` (default) bounds the diagonal in
        ``[1, d_max]`` -- bounded amplification of negative log-mass.
        ``blocks=2`` declares the binary ``2D -> D`` fold form.
        """
        super().__init__(nInput, nOutput)
        self.blocks = int(blocks)
        self.layer = ContractiveInvertibleLinearLayer(
            nInput, nOutput, naive=naive, ergodic=ergodic,
            hasBias=hasBias, stable=stable, blocks=blocks, d_max=d_max)
        self.layers.append(self.layer)

    @staticmethod
    def _clamp_membership(m):
        """Membership-chart domain guard: floor EPS_LOG, ceil 1.

        The floor keeps ``log`` finite at the ``m -> 0`` corner (and is
        what the corner round-trips recover); the ceil keeps
        ``log m <= 0`` so the §4 contraction theorem holds. NaN/Inf
        propagate (the clamp rescues finite out-of-domain values only).
        """
        return m.clamp(EPS_LOG, 1.0)

    def forward(self, m, gate=None):
        """pi(m) = exp(W · log(clamp(m, EPS_LOG, 1)) + b).

        Memberships in, memberships out (``z in (0, 1]`` is automatic:
        every log-domain term is <= 0 under the weight law).
        ``gate`` flows to the inner LDU diagonal as in the legacy folds.
        """
        m = m.to(self.layer.d.device)
        l = torch.log(self._clamp_membership(m))
        wl = (self.layer.forward(l, gate=gate) if gate is not None
              else self.layer.forward(l))
        return torch.exp(wl)

    def reverse(self, z, gate=None):
        """m = exp(W^-1 · (log z − b)) -- the exact §4 inverse.

        ``z`` is floored at the dtype's tiny (NOT at EPS_LOG: a fold of
        EPS_LOG-floored operands lands far below EPS_LOG, and the corner
        round-trip must recover the floor exactly) and ceiled at 1.
        Output is clamped back to the membership chart ``[0, 1]``.
        """
        z = z.to(self.layer.d.device)
        z = z.clamp(torch.finfo(z.dtype).tiny, 1.0)
        lz = torch.log(z)
        lx = (self.layer.reverse(lz, gate=gate) if gate is not None
              else self.layer.reverse(lz))
        return torch.exp(lx).clamp(0.0, 1.0)

    def compose(self, left, right, gate=None):
        """Binary AND fold on memberships: requires ``blocks == 2``.

        Concatenates the operands along the feature axis and runs the
        2D -> D forward; ``pi(left, right) <= min(left, right)``
        elementwise is the §10.1 acceptance bound.
        """
        if self.blocks != 2:
            raise RuntimeError(
                "PiLayer2.compose requires the binary form "
                f"(blocks=2); this layer has blocks={self.blocks}")
        return self.forward(torch.cat([left, right], dim=-1), gate=gate)


class SigmaLayer2(Layer):
    r"""Membership-domain synthesis fold (MeronomySpec §4; MeronomyPlan
    Stage 2): the whole-maker / union, as the De Morgan wrap of a pi
    kernel:

        sigma(m) = 1 − pi_kernel(1 − m)        (probabilistic-sum family)

    ONE FOLD KERNEL PLUS ONE INVOLUTION implements both operators: this
    class owns no weights of its own -- ``self.kernel`` is a
    :class:`PiLayer2` (constructed here, or passed in to share an
    existing kernel object), and extension ``sigma(m) >= m`` (binary:
    ``sigma >= max(operands)``, §10.1) is the mirrored theorem under the
    same weight law. No separate sigma constraint surface exists.

    Identities / absorbers mirror pi's: ``0`` is the sigma-identity and
    ``1`` the sigma-absorber (everything absorbs -- a theorem for all
    admissible weights). Single-lump doctrine (§4): no adjacency exists
    over witness dims, so every sigma extent is a single lump; no
    abstraction classifier is implemented here or anywhere.

    Legacy :class:`SigmaLayer` (tanh/atanh additive fold) stays untouched
    for non-meronymic consumers.
    """

    def __init__(self, nInput=None, nOutput=None, kernel=None, **kw):
        """Initialize SigmaLayer2 around a pi kernel.

        Either pass ``nInput`` / ``nOutput`` (+ PiLayer2 kwargs) to
        construct a fresh kernel, or pass ``kernel=`` an existing
        :class:`PiLayer2` to share its object (the §10.4 De Morgan
        exactness setup: one shared kernel, two operators).
        """
        if kernel is None:
            if nInput is None or nOutput is None:
                raise ValueError(
                    "SigmaLayer2: pass nInput/nOutput or kernel=")
            kernel = PiLayer2(nInput, nOutput, **kw)
        elif kw:
            raise ValueError(
                "SigmaLayer2: kernel= and PiLayer2 kwargs are exclusive")
        super().__init__(kernel.nInput, kernel.nOutput)
        self.kernel = kernel
        self.layers.append(kernel)
        self.blocks = kernel.blocks

    def forward(self, m, gate=None):
        """sigma(m) = 1 − pi(1 − m); memberships in, memberships out."""
        return 1.0 - self.kernel.forward(1.0 - m, gate=gate)

    def reverse(self, z, gate=None):
        """Exact inverse through the involution: 1 − pi.reverse(1 − z)."""
        return 1.0 - self.kernel.reverse(1.0 - z, gate=gate)

    def compose(self, left, right, gate=None):
        """Binary OR fold on memberships: requires ``blocks == 2``.

        ``sigma(left, right) >= max(left, right)`` elementwise is the
        mirrored §10.1 bound. The binary reverse stays recommender-shaped
        (``Ops.disjunctionReverse``) as today.
        """
        return 1.0 - self.kernel.compose(1.0 - left, 1.0 - right, gate=gate)


class MeronymicFoldAdapter(Layer):
    """K3-wire adapter binding a membership kernel at a meronymic slot
    (MeronomyPlan Stage 9 cutover; MeronomySpec §2, §4).

    The wire keeps carrying the K3 scalar ``a in [-1, +1]`` (the
    BivectorRetirementPlan invariant); the fold computes in the
    membership chart: ``χ`` in, kernel, ``χ⁻¹`` out — epistemic scalars
    enter membership folds only through the evaluation chart (§3),
    applied exactly once per crossing, here at the slot boundary. The
    same chart pair on ``reverse`` makes the adapter exactly invertible
    wherever the kernel is.

    At near-identity init the membership kernel is approximately the
    identity on memberships (d ≈ 1, offdiag ≈ 0, b ≈ 0), so the adapter
    starts close to a pass-through: cutover begins from a benign
    operating point and training shapes the fold from there.

    ``kind``: ``'sigma'`` (PS slot — synthesis/whole-maker) or ``'pi'``
    (WS slot — analysis/part-maker). ``binary=`` is accepted for
    call-surface parity with the legacy folds and ignored (operand
    selection belongs to the binary ``blocks=2`` form, not the unary
    slot). Non-meronymic consumers keep the odds-kernel layers (plan §4
    item 4 — those operators are NOT legacy; only this slot rebinds).

    **Butterfly mode (author, 2026-06-11 — the cutover correction).**
    ``butterfly=True`` rebuilds the slot's cascade in the membership
    chart instead of dropping it: the order-preservation law of the
    meronymic slots constrains the KERNEL CLASS (contractive,
    non-negative log-mass weights), never the fold's TOPOLOGY — and
    order-preserving maps compose, so the FFT-style cross-slot cascade
    with the contractive law at every 2×2 node preserves the partial
    order end-to-end while keeping the cascade's cross-position reach
    (which per-slot folds lack, and which functions like XOR over the
    slab require: distinct word codes sit INCOMPARABLE in the partial
    order, so monotonicity never excluded them — only the lost reach
    did). Each node is the 2×2 contractive LDU law in log-mass via
    the SQUARE reparam: off-diagonal taps ``raw² >= 0``, diagonal
    ``(1 + raw_d²)`` clamped to ``[1, d_max]``, bias-free; the
    REVERSE divides by ``d >= 1`` (contractive — strictly tamer than
    the signed cascade's ``1/d`` blow-up guard). χ applies once at
    the slot boundary; ``kind='sigma'`` conjugates the π-law cascade
    by the complement involution (``σ = 1 − π(1 − ·)``), which
    composes through the levels exactly. Near-identity init
    (raw = 0.05: taps ~0.0025 with gradient scale ~0.10 — squares
    decouple tap size from gradient scale, where softplus locks them
    together and strangles learning) keeps the cutover's benign-start
    contract. ``gate`` is not honoured in butterfly mode (same
    contract as the legacy cascade).

    **Recorded risk + contingency (author, 2026-06-11).** The lawful
    node matrices are SPECTRALLY EXPANSIVE in log-mass by
    construction: ``det(LDU) = d₀·d₁ >= 1`` per 2×2 node, so the
    spectral radius of every level — hence of any composition of
    membership folds — is ``>= 1`` there. One application per pump
    is benign (near-identity init; ``d_max`` bounds per-application
    drift), but ITERATED application necessarily migrates mass
    toward the chart corners — π toward 𝟘 (the 0000 code), σ toward
    𝟙 (the 1111 code) — and a saturated code carries no usable
    specification. The eigenvalue floor is the law itself (order
    preservation + memberships-stay-memberships forces non-negative
    log-mass weights with ``d >= 1``), so it cannot be trained away —
    only bounded (``d_max``, shallow application counts, the
    σπσ = σ registration pressure on stored symbols).
    **Contingency:** if the geometric constraint proves too
    restrictive in practice — folds saturating, or solutions
    excluded — the meronomy moves to an explicit PART-WHOLE TREE ON
    THE CODES (parthood as stored structure over code atoms, the
    order law enforced on the tree) rather than a geometric
    constraint baked into the fold operators.
    ``test_meronymic_butterfly.py`` pins both sides of this boundary:
    discontiguous specifications survive — and are PRODUCED by —
    repeated lawful application (scattered wholes are first-class;
    the FFT pairing carries no contiguity bias), and the saturation
    regime under an aggressive law is characterized explicitly.
    """
    invertible = True
    # Legacy slot-surface attributes: consumers of the slot layers read
    # these (e.g. the WS parallel-fold dispatch reads ``fold.N`` to
    # match the construction-time flat total; ``butterfly`` is probed
    # with getattr). In unary mode the adapter is a per-slot membership
    # fold and carries the legacy total for dispatch only; in butterfly
    # mode the cascade is real and ``butterfly`` flips True per
    # instance.
    butterfly = False
    nonlinear = False
    monotonic = True

    def __init__(self, kind, nInput, nOutput, stable=True, ergodic=False,
                 naive=False, legacy_N=None, butterfly=False, d_max=None):
        super().__init__(nInput, nOutput)
        if kind not in ('sigma', 'pi'):
            raise ValueError(
                f"MeronymicFoldAdapter: kind must be 'sigma' or 'pi'; "
                f"got {kind!r}")
        self.kind = kind
        self.N = int(legacy_N) if legacy_N is not None else int(nInput)
        self.butterfly = bool(butterfly)
        if self.butterfly:
            # Membership-chart cascade over the flattened slab (same
            # pairing topology as GrammarLayer's signed cascade).
            M = 1
            while M < max(2, self.N):
                M *= 2
            self.M_total = M
            self.n_levels = int(math.log2(M))
            self.d_max = float(d_max) if d_max is not None \
                else float(meronomy_d_max_stable())
            pair_count = M // 2
            # SQUARE reparam (tap = raw², d = 1 + raw_d²), NOT the
            # unary kernel's softplus: softplus locks the gradient
            # scale to the tap size (both ~ e^raw near identity), so a
            # depth-compensated benign start would strangle learning —
            # the signed cascade trains precisely because its raws are
            # linear (taps init 0, gradient 1). Squares decouple them:
            # at raw = 0.05 the taps are ~0.0025 (near-identity across
            # a 7-level stack: the benign-start contract) while the
            # gradient scale is 2·raw ~ 0.10. Non-negativity is
            # guaranteed, exact identity stays reachable (raw → 0), and
            # the sign symmetry leaves no dead zone.
            self.raw_bfly_L = nn.Parameter(
                torch.full((self.n_levels, pair_count), 0.05))
            self.raw_bfly_d = nn.Parameter(
                torch.full((self.n_levels, pair_count, 2), 0.05))
            self.raw_bfly_U = nn.Parameter(
                torch.full((self.n_levels, pair_count), 0.05))
            self.register_buffer(
                'butterfly_perms',
                GrammarLayer._build_butterfly_perms(M, self.n_levels))
            # Pad hygiene (exact-invertibility requirement the signed
            # cascade lacks): when N is not a power of two, pairs that
            # touch a pad lane are pinned to the IDENTITY law. Pads
            # then stay exactly 𝟘 (log-mass 0) through every level —
            # they never absorb real mass that the unflatten would
            # discard — so the cascade restricted to the N real lanes
            # is a bijection and the reverse is exact. Real↔real pairs
            # are unaffected (full law, full reach).
            mask = torch.ones(self.n_levels, pair_count)
            for k in range(self.n_levels):
                perm = self.butterfly_perms[k]
                for p in range(pair_count):
                    if (int(perm[2 * p]) >= self.N
                            or int(perm[2 * p + 1]) >= self.N):
                        mask[k, p] = 0.0
            self.register_buffer('_bfly_pair_mask', mask)
            self.fold = None
        elif kind == 'sigma':
            self.fold = SigmaLayer2(nInput, nOutput, stable=stable,
                                    ergodic=ergodic, naive=naive)
        else:
            self.fold = PiLayer2(nInput, nOutput, stable=stable,
                                 ergodic=ergodic, naive=naive)
        if self.fold is not None:
            self.layers.append(self.fold)
        self.activation = torch.zeros(1, nOutput, 1)

    # -- membership cascade (butterfly mode) ---------------------------
    # Borrow the signed cascade's flatten/permutation plumbing — the
    # adapter carries the same ``M_total`` / ``butterfly_perms``
    # surface, only the per-node law differs.
    _bfly_flatten = GrammarLayer._butterfly_flatten
    _bfly_unflatten = GrammarLayer._butterfly_unflatten

    def _mem_law(self):
        """The per-node contractive law via the square reparam:
        L, U = raw² >= 0; d = (1 + raw_d²) in [1, d_max]. Pad-touching
        pairs are pinned to identity (L = U = 0, d = 1) so pads stay
        exactly 𝟘 and the real-lane cascade inverts."""
        mask = self._bfly_pair_mask
        L = self.raw_bfly_L.square() * mask
        d = (1.0 + self.raw_bfly_d.square()).clamp(1.0, self.d_max)
        d = d * mask.unsqueeze(-1) + (1.0 - mask.unsqueeze(-1))
        U = self.raw_bfly_U.square() * mask
        return L, d, U

    def _mem_cascade(self, l, inverse=False):
        """Run the contractive cascade over flattened log-mass.

        GATHER-FREE pairing (MPS-compile requirement): bounds-checked
        index ops each demand an error-report buffer argument in the
        generated Metal kernels, and an Inductor fusion holding the
        cascade's ``2·n_levels`` permutation gathers exhausts Metal's
        31-buffer limit ("no 'buffer' resource location available for
        'error_buf'"). At stride ``s = 2^k`` the XOR-partners
        ``(i, i+s)`` sit at positions ``(off, s+off)`` of each
        contiguous ``2s`` block, so a ``[B, M/2s, 2, s]`` VIEW
        separates the pair members on pure reshapes — no advanced
        indexing anywhere on the compute path. The per-pair math and
        the parameter layout are identical to the permuted form
        (pair ``p = blk·s + off`` — the legacy perm order;
        ``butterfly_perms`` stays registered for the pad-mask
        construction and lane-pair lookups in tests).

        Per node forward (all entries >= 0: order-preserving;
        log-mass stays <= 0 so memberships stay in (0, 1] — the
        §10.10 weight law): ``u0 = l0 + U·l1``; ``y0 = d0·u0``;
        ``y1 = L·y0 + d1·l1``. Reverse is the closed-form 2×2 LDU
        inverse, dividing by ``d >= 1`` (contractive — no blow-up).

        ``l``: ``[B, ...]`` log-mass (<= 0); flattened and zero-padded
        to ``M_total`` — a log-mass pad of 0 is membership 𝟙, the
        π-identity, so pad lanes contribute exactly nothing to real
        lanes (the §10.2 identity theorem at the pad)."""
        flat, original_shape = self._bfly_flatten(l)
        B = flat.shape[0]
        L, d, U = self._mem_law()
        levels = range(self.n_levels)
        if inverse:
            levels = reversed(levels)
        for level in levels:
            s = 1 << level
            blocks = flat.reshape(B, -1, 2, s)
            l0 = blocks[..., 0, :]
            l1 = blocks[..., 1, :]
            # Per-pair scalars at [nblocks, s]; flat pair index
            # blk·s + off matches the legacy perm-ordered layout.
            Lk = L[level].reshape(-1, s)
            d0 = d[level, :, 0].reshape(-1, s)
            d1 = d[level, :, 1].reshape(-1, s)
            Uk = U[level].reshape(-1, s)
            if inverse:
                a1 = l1 - Lk * l0
                b0 = l0 / d0
                b1 = a1 / d1
                x0 = b0 - Uk * b1
                flat = torch.stack([x0, b1], dim=-2).reshape(B, -1)
            else:
                u0 = l0 + Uk * l1
                y0 = d0 * u0
                y1 = Lk * y0 + d1 * l1
                flat = torch.stack([y0, y1], dim=-2).reshape(B, -1)
        return self._bfly_unflatten(flat, original_shape)

    def forward(self, x, binary=False, gate=None):
        """χ → membership fold → χ⁻¹ on the K3 wire.

        Butterfly mode: χ once at the boundary, complement involution
        for the σ kind, log-mass cascade (width-agnostic over the
        flattened slab, like the signed cascade it re-charters).

        Unary mode: width-mismatched calls (the legacy cascade was
        width-agnostic; the per-slot membership fold is not) fall back
        to identity for the batch — the same convention the WS
        parallel-fold dispatch uses for mismatched totals.
        """
        if self.butterfly:
            m = Ops.eval_chart(x.clamp(-1.0, 1.0))
            if self.kind == 'sigma':
                m = 1.0 - m
            l = torch.log(m.clamp(EPS_LOG, 1.0))
            z = torch.exp(self._mem_cascade(l))
            if self.kind == 'sigma':
                z = 1.0 - z
            out = Ops.eval_chart_inv(z)
            self.activation = out.detach()
            return out
        if x.shape[-1] != self.nInput:
            self.activation = x.detach()
            return x
        m = Ops.eval_chart(x.clamp(-1.0, 1.0))
        z = self.fold.forward(m, gate=gate)
        out = Ops.eval_chart_inv(z)
        self.activation = out.detach()
        return out

    def reverse(self, y, gate=None):
        """Exact inverse through the same chart pair (identity on
        width-mismatched unary calls, mirroring forward; in butterfly
        mode the cascade inverts level-by-level, dividing by
        d >= 1)."""
        if self.butterfly:
            z = Ops.eval_chart(y.clamp(-1.0, 1.0))
            if self.kind == 'sigma':
                z = 1.0 - z
            lz = torch.log(z.clamp(torch.finfo(z.dtype).tiny, 1.0))
            m = torch.exp(self._mem_cascade(lz, inverse=True)).clamp(
                0.0, 1.0)
            if self.kind == 'sigma':
                m = 1.0 - m
            x = Ops.eval_chart_inv(m)
            self.activation = x.detach()
            return x
        if y.shape[-1] != self.nOutput:
            self.activation = y.detach()
            return y
        z = Ops.eval_chart(y.clamp(-1.0, 1.0))
        m = self.fold.reverse(z, gate=gate)
        x = Ops.eval_chart_inv(m)
        self.activation = x.detach()
        return x

    def getParameters(self):
        """Optimizable parameters of the wrapped kernel / cascade."""
        return [p for _, p in self.named_parameters()]

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
            symbolic_space: WholeSpace for concept<->symbol projection.
        Returns:
            [B, N, D] lifted subject concepts.
        """
        ws = symbolic_space

        # 1. Project concept vectors to symbol space via WholeSpace.forward()
        ws.subspace.set_event(verb)
        verb_syms = ws.forward(ws.subspace).materialize()     # [B, N, symbol_dim]
        ws.subspace.set_event(obj)
        obj_syms = ws.forward(ws.subspace).materialize()      # [B, N, symbol_dim]

        # 2. Intersect: restrict verb by object (monotonic -> min)
        restricted_syms = torch.min(verb_syms, obj_syms)      # [B, N, symbol_dim]

        # 3. Symbols are already in concept-space coordinates
        # (symbol_dim == concept_dim is enforced in WholeSpace).
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
            """Test double for WholeSpace bound to a PiLayer.

            Implements just enough of WholeSpace's API for
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

class SparsityRegLayer(Layer):
    """Soft-threshold L1 proximal operator.

    Shared by PartSpace, ConceptualSpace, and WholeSpace so a
    single sparsity implementation is reused across space_roles. Extracted from
    ``WholeSpace.l1_proximal``.

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
    *distinct* concepts only. See basicmodel/doc/Philosophy.md
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

    See basicmodel/doc/Philosophy.md for the tetralemma (catuskoti)
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
# attached.
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
    """Truth store on WholeSpace: encoded truth statements scaled by DoT.

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

    Propositional structure is defined by the S-space_role grammar:
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
        # (SymbolSpace.truth_modulated_loss) needs *emptiness*, not the
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
        basicmodel/doc/Philosophy.md for the tetralemma mapping.

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
                ``valid_mask``).
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
            This layout is **not** the current WholeSpace codebook layout,
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
        new leading-bivector WholeSpace codebook.
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
            The current WholeSpace codebook layout is
            ``[pos_pole, neg_pole, where..., when...]`` — a **single**
            leading bivector plus trailing positional template data, not
            repeated pairs. If passing a symbol activation from the new
            codebook, slice the leading bivector first
            (``sym_act[..., :2]``) to avoid applying tetralemma corner
            policy to positional-template dims. See
            basicmodel/doc/Spaces.md "Codebook shape" and
            basicmodel/doc/Philosophy.md.

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

        Under the current WholeSpace codebook layout
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
            symbolic_space: WholeSpace for projection.
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
        ws = symbolic_space
        luminosity_before = float(model.Luminosity(truth_layer=self))

        # K(X, Y): original action SVO
        result_svo = lifting_layer.forward_transitive_svo(
            subject, verb, obj, ws)

        # K(Y, X): dual action OVS
        result_ovs = lifting_layer.forward_transitive_svo(
            obj, verb, subject, ws)

        # Project results to symbol space via WholeSpace.forward()
        ws.subspace.set_event(result_svo)
        svo_syms = ws.forward(ws.subspace).materialize()  # [B, N, symbol_dim]
        ws.subspace.set_event(result_ovs)
        ovs_syms = ws.forward(ws.subspace).materialize()  # [B, N, symbol_dim]

        # Temporarily extend truth store (average over batch and vectors)
        saved_count = self.count.item()
        basis = getattr(getattr(ws, 'subspace', None), 'basis', None)
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
        ``WholeSpace.forwardEnd``, and ``symbol_states`` should be
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
        """Catuṣkoṭi luminosity of the truth store, computed on the codes.

        MeronomySpec §3 (rev 2026-06-10b). Per conceptual dimension
        ``k``, the stored signed references split into the true/false
        pole bivector — coverage, not mass: trust is already baked into
        stored magnitudes, and duplicate truths must not inflate the
        area —

            T_k = max_i relu(+truths[i, k])
            F_k = max_i relu(-truths[i, k])

            luminosity = mean_k [ (T_k - F_k) - min(T_k, F_k) ]

        clamped to ``[-1, 1]``: total area weighted by sign, minus the
        regions where the sign differs (contradictory evidence — the
        catuṣkoṭi B corner). Per-dimension corners: T-only ⇒ ``+T``,
        F-only ⇒ ``-F``, both ⇒ ``-min(T, F)``, neither ⇒ ``0``.

        Computed directly over the stored codes — no pullback to the
        mereological ground: the §4 weight law puts the parthood
        geometry on the codes themselves, so code-space_role areas are a
        registration-maintained approximation of ground areas (§5;
        exact at mint, degrading conservatively under mixing). The
        previous implementation decoded every row through
        ``sym.decode_to_concept`` and folded sequentially (order-
        dependent, and effectively stubbed: it returned 0.0 whenever
        ``sym`` was absent); ``sym`` is retained for call-site
        compatibility and ignored.

        Paired-index bivector rows (``record`` with a monotonic basis)
        are outside this measure's domain: their poles are orthogonal
        by construction (no sign opposition arises), so such stores
        read as pure positive area; B-corner policy for that layout
        lives in ``tetralemma_balance_penalty``.
        """
        n = int(self.count.item())
        if n == 0:
            return 0.0
        stored = self.truths[:n]
        T = stored.clamp(min=0).max(dim=0).values
        F = (-stored).clamp(min=0).max(dim=0).values
        lum = ((T - F) - torch.minimum(T, F)).mean().item()
        return max(-1.0, min(1.0, lum))

    # -- The absolute corpus's duties (GrammarOpsPass §6; author
    # sign-off 2026-06-11) ----------------------------------------------
    #
    # The absolute truth set is a CONSISTENT CORPUS governing admission:
    # it (1) governs the admissibility of new truths/beliefs, (2) serves
    # as the basis for causal reasoning (relations between ideas live in
    # the sibling ``RelativeTruthStore`` and NEVER enter this store or
    # its luminosity), and (3) provides user feedback on the truth of a
    # statement. The conflict region ``min(T_k, F_k)`` is MEASURED,
    # never stored; the preemption/admissibility statistic is its
    # per-dimension MAX — one sharply contested witness interrupts,
    # where a mean would dilute it away across nDim.

    def conflict_profile(self, extra=None):
        """Per-dimension contested mass ``min(T_k, F_k)`` over the
        absolute store, optionally with candidate row(s) ``extra``
        added (measured, never stored — the candidate is NOT recorded).
        Returns a ``[nDim]`` tensor; zeros when the store is empty and
        no candidate is given."""
        n = int(self.count.item())
        rows = [self.truths[:n]] if n else []
        if extra is not None:
            rows.append(
                extra.reshape(-1, self.nDim).to(self.truths))
        if not rows:
            return torch.zeros(self.nDim)
        stored = torch.cat(rows, dim=0)
        T = stored.clamp(min=0).max(dim=0).values
        F = (-stored).clamp(min=0).max(dim=0).values
        return torch.minimum(T, F)

    def conflict_mass(self, extra=None) -> float:
        """The preemption / admissibility statistic (§6): the
        per-dimension MAX of :meth:`conflict_profile`."""
        prof = self.conflict_profile(extra=extra)
        return float(prof.max().item()) if prof.numel() else 0.0

    def admissible(self, candidate, threshold=0.5) -> bool:
        """Admission governance (duty 1): a candidate truth/belief is
        admissible iff admitting it would keep the corpus's conflict
        mass at or below ``threshold``. A commit-point gate — the
        measure itself is soft and stored nowhere; callers that want
        legacy unconditional recording simply don't consult it."""
        return self.conflict_mass(extra=candidate) <= float(threshold)

    def preemption_signal(self, threshold=0.5, hysteresis=0.1,
                          extra=None):
        """Preattention trigger (§6): threshold + hysteresis on the
        absolute set's conflict mass. ``extra`` carries the percept
        stream's contested candidate(s) — conflict between incoming
        evidence and the parse's commitments captures the serial
        thread. Fires when the mass exceeds ``threshold``; once fired,
        stays fired until the mass falls below ``threshold -
        hysteresis`` (chatter guard). Returns ``(mass, fired)``.

        The latch (``_preempt_active``) is a plain transient attribute:
        measured-not-stored binds epistemic content; this is controller
        state, absent from ``state_dict``.
        """
        mass = self.conflict_mass(extra=extra)
        active = bool(getattr(self, '_preempt_active', False))
        th = float(threshold)
        if active:
            active = mass > th - float(hysteresis)
        else:
            active = mass > th
        self._preempt_active = active
        return mass, active

    def truth_of(self, statement, threshold=0.6):
        """User truth feedback (duty 3): the graded truth of one
        statement code against the corpus.

        Returns ``dict(truth, contested, grounded)``: ``truth`` in
        ``[-1, 1]`` is the signed cosine agreement of the
        best-matching stored truth (0.0 when nothing matches above
        ``threshold`` — unknown, not false); ``contested`` is the
        conflict mass the statement would join (duty 1's measure,
        reused); ``grounded`` says whether any stored truth spoke.
        """
        st = statement.reshape(-1)[: self.nDim].to(self.truths)
        n = int(self.count.item())
        contested = self.conflict_mass(extra=st)
        if n == 0 or float(st.norm()) == 0.0:
            return {'truth': 0.0, 'contested': contested,
                    'grounded': False}
        stored = self.truths[:n]
        sims = F.normalize(stored, dim=-1) @ (st / st.norm())
        best = sims[sims.abs().argmax()]
        grounded = bool(best.abs().item() > float(threshold))
        return {'truth': float(best.item()) if grounded else 0.0,
                'contested': contested,
                'grounded': grounded}

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
            from Language import GRAMMAR_LAYER_CLASSES
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
        model handle for the WholeSpace/ConceptualSpace context, the
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

        ws = getattr(model, 'wholeSpace', None) if model is not None else None
        from Language import GRAMMAR_LAYER_CLASSES
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

                    if ws is not None and model is not None:
                        lum_before = model.Luminosity(truth_layer=self)
                    else:
                        lum_before = 0.0
                    saved_count = self.count.item()

                    dot_i = stored[i].norm().item()
                    dot_j = stored[j].norm().item()
                    degree = attenuation * min(dot_i, dot_j)

                    direction = F.normalize(candidate.unsqueeze(0), dim=-1).squeeze(0)
                    self.record(direction, degree, basis=basis)
                    if ws is not None and model is not None:
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
        consumer (``SymbolSpace.truth_modulated_loss``) does not force a
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
        # surface was removed when WholeSpace.luminosity migrated.
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
        # surface after the C->S swap to SigmaLayer on WholeSpace).
        _sigma = SigmaLayer(N, nSym, monotonic=True, invertible=True,
                            nonlinear=True)
        class _MockSS:
            """Test double for WholeSpace exposing only a ``sigma`` attribute.

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


class RelativeTruthStore(Layer):
    """The second of the two truth sets (GrammarOpsPass §6; author
    sign-off 2026-06-11): **relative truths are relations between
    ideas** — causal implication is the worked example.

    A relative truth contains two ideas and a relation. Model it as
    ``NP = VP NP``, but ``VP(NP)`` cannot be collapsed without making
    it specific, so **all three components are stored uncollapsed**
    (``np1`` — the state of affairs at t₁; ``vp`` — the change; ``np2``
    — the consequent at t₂; the codes' ``.when`` band carries the
    temporal shape) and enforced as a **structural constraint** over
    references: this structures the references and creates universal
    truths.

    NOT expressible as material truth: recording ``if a then b`` as
    ``¬a ∨ b`` both loses the temporal content and corrupts the
    absolute set with a region assertion the causal rule never
    licensed. Accordingly this store is a SIBLING of ``TruthLayer``
    (the absolute store) — relative entries never enter the luminosity
    measure, and no coverage computation ever has to mask them out.

    Consumed only by the reasoning loop: evaluation is by SIMULATION or
    RELATIONAL EVALUATION, never by coverage — evaluate NP₁, run
    one-or-more VP reasoning steps (``consequents`` is the per-step
    expansion), check NP₂ (``evaluate`` is the relational form).
    """

    def __init__(self, nDim: int, max_triples: int = 1024):
        super().__init__(nDim, nDim)
        self.nDim = int(nDim)
        self.max_triples = int(max_triples)
        # The three components, stored UNCOLLAPSED and aligned by row.
        self.register_buffer('np1', torch.zeros(max_triples, nDim))
        self.register_buffer('vp', torch.zeros(max_triples, nDim))
        self.register_buffer('np2', torch.zeros(max_triples, nDim))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        # Per-triple relation trust (DoT), aligned by row with np1/vp/np2.
        # A REGISTERED BUFFER (not a plain list) so it survives a
        # state_dict save/load round-trip alongside the magnitude buffers
        # -- otherwise a reloaded relation silently reverts to the 1.0
        # fallback in ConceptualSpace.reason / verify_relation. Zero-init
        # means an OLD checkpoint lacking this key loads (non-strict) with
        # all-zero trust rather than erroring.
        self.register_buffer('trust', torch.zeros(max_triples))

    @property
    def _trusts(self):
        """Read-only back-compat view of the per-triple trust as a Python
        list over the live rows. The canonical store is the ``trust``
        buffer; this exists for callers/docstrings that referred to the
        old list. Writes must target the ``trust`` buffer directly (a
        list snapshot would not write back)."""
        return self.trust[: int(self.count.item())].tolist()

    def __len__(self):
        return int(self.count.item())

    @torch.no_grad()
    def record_triple(self, np1, vp, np2, degree: float = 1.0) -> int:
        """Store one relation between ideas — all three components,
        uncollapsed; ``degree`` (the relation's DoT/trust) is baked
        into the stored magnitudes like ``TruthLayer.record``. Returns
        the row index, or ``-1`` when the store is full."""
        n = int(self.count.item())
        if n >= self.max_triples:
            return -1
        d = float(degree)
        self.np1[n] = np1.reshape(-1)[: self.nDim].to(self.np1) * d
        self.vp[n] = vp.reshape(-1)[: self.nDim].to(self.vp) * d
        self.np2[n] = np2.reshape(-1)[: self.nDim].to(self.np2) * d
        self.trust[n] = d
        self.count.fill_(n + 1)
        return n

    def triple(self, idx: int):
        """The stored ``(np1, vp, np2)`` at ``idx`` (views)."""
        n = int(self.count.item())
        if not (0 <= int(idx) < n):
            raise IndexError(f"triple {idx} of {n}")
        return self.np1[idx], self.vp[idx], self.np2[idx]

    @staticmethod
    def _sims(query, rows):
        q = query.reshape(-1)[: rows.shape[-1]].to(rows)
        qn = q / q.norm().clamp_min(1e-12)
        rn = F.normalize(rows, dim=-1)
        return rn @ qn

    @torch.no_grad()
    def consequents(self, state, vp=None, threshold: float = 0.7):
        """One causal reasoning step (the loop's expansion): stored
        relations whose antecedent matches ``state`` (graded cosine
        over ``np1``; further filtered by ``vp`` similarity when a
        change is specified) yield their consequents.

        Returns a list of ``(idx, match, vp_row, np2_row)`` sorted by
        descending match — the serial loop steps from a state of
        affairs to its licensed next states.
        """
        n = int(self.count.item())
        if n == 0:
            return []
        match = self._sims(state, self.np1[:n])
        if vp is not None:
            match = match * self._sims(vp, self.vp[:n]).clamp_min(0.0)
        keep = (match > float(threshold)).nonzero(as_tuple=True)[0]
        out = [(int(i), float(match[i]), self.vp[int(i)],
                self.np2[int(i)]) for i in keep]
        out.sort(key=lambda t: -t[1])
        return out

    @torch.no_grad()
    def evaluate(self, np1, vp, np2) -> float:
        """Relational evaluation of a queried relation against the
        corpus of stored relations (the §6 evaluation that coverage
        can never provide): the best joint match
        ``max_i sim(np1, np1_i) · sim(vp, vp_i) · sim(np2, np2_i)``
        over non-negative component similarities, in ``[0, 1]``.

        0.0 = no stored relation licenses it; 1.0 = exactly the stored
        universal. Verification by SIMULATION (running the VP steps)
        composes from :meth:`consequents`; this is the one-shot
        relational form.
        """
        n = int(self.count.item())
        if n == 0:
            return 0.0
        j = (self._sims(np1, self.np1[:n]).clamp_min(0.0)
             * self._sims(vp, self.vp[:n]).clamp_min(0.0)
             * self._sims(np2, self.np2[:n]).clamp_min(0.0))
        return float(j.max().item())

    @torch.no_grad()
    def constraint_residuals(self):
        """The structural-constraint face (sign-off: 'store all three
        and then enforce them as a structural constraint'): for every
        pair of stored relations whose ``(np1, vp)`` agree, their
        consequents must agree — a universal truth is one rule, not a
        coincidence of instances. Returns ``[n]`` per-row residuals:
        ``max_j agree_ij · (1 − sim(np2_i, np2_j))`` — 0 when the
        corpus is functionally consistent. Measured, never stored."""
        n = int(self.count.item())
        if n == 0:
            return torch.zeros(0)
        a1 = F.normalize(self.np1[:n], dim=-1)
        av = F.normalize(self.vp[:n], dim=-1)
        a2 = F.normalize(self.np2[:n], dim=-1)
        agree = ((a1 @ a1.T).clamp_min(0.0)
                 * (av @ av.T).clamp_min(0.0))
        agree.fill_diagonal_(0.0)
        disagree = 1.0 - (a2 @ a2.T)
        return (agree * disagree).max(dim=-1).values.clamp_min(0.0)

    @torch.no_grad()
    def reset(self):
        """Clear the store (idempotent)."""
        self.np1.zero_()
        self.vp.zero_()
        self.np2.zero_()
        self.count.zero_()
        self.trust.zero_()


class TernaryTruthStore(Layer):
    """The unified LTM + relative-truth store (Truth / Ideas consolidation,
    Alec 2026-06-18): ONE tensor of ternary rows that combines the discourse
    LTM end-state chain and the ``RelativeTruthStore`` relation corpus.

    Each row is ``(NP1, VP, NP2)`` -- three FULL idea vectors (event width
    ``nDim``; ``Null`` = the zero vector) -- plus a per-row ``timestamp`` and
    a per-row scalar ``trust`` in ``[-1, 1]``:

      * ``NP  .   .``  -> an IDEA (absolute truth)
      * ``NP  VP  .``  -> a unary predication
      * ``NP  VP  NP`` -> an IDEA-RELATION-IDEA (relative truth)

    The relation kind is tagged in ``rel_type`` (``REL_NONE`` for an absolute
    idea; ``REL_PARTOF`` / ``REL_IMPLIES`` are the two canonical relations;
    ``REL_OTHER`` for any learned predicate carried in the VP slot).
    ``partOf`` rows feed the meronomy / parthood; ``implies`` rows feed
    modus-ponens reasoning.

    UNLIKE ``RelativeTruthStore`` the idea vectors are stored UNSCALED (trust
    is a SEPARATE column, not baked into the magnitude), so a reader needs no
    un-baking. All state is registered buffers, so the corpus rides the
    state_dict -- it is PERSISTENT (provisioned from an XML TruthSet at load
    time, grown by conversation at runtime). The monotonic ``timestamp`` gives
    the single ordering both readers need: the AR predictor takes the recent
    slice (:meth:`recent`); reasoning scans the whole corpus
    (:meth:`relations`).

    Storage-only foundation (stage 1 of the consolidation): nothing
    constructs it yet, so it is inert / byte-identical until wired.
    """

    REL_NONE = 0
    REL_PARTOF = 1
    REL_IMPLIES = 2
    REL_OTHER = 3

    def __init__(self, nDim: int, capacity: int = 1024, content_width=None):
        super().__init__(nDim, nDim)
        self.nDim = int(nDim)
        self.capacity = int(capacity)
        # Content ('what') width a reasoning reader slices to (the leading
        # band of the event vector); defaults to the full width. Stored for
        # forward-compat -- this class keeps the FULL idea vectors.
        self.content_width = int(content_width) if content_width else self.nDim
        # ONE tensor: [capacity, 3, nDim] -- slot 0 = NP1, 1 = VP, 2 = NP2.
        self.register_buffer('slots', torch.zeros(capacity, 3, nDim))
        self.register_buffer('rel_type',
                             torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('timestamp', torch.zeros(capacity))
        self.register_buffer('trust', torch.zeros(capacity))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        # Monotonic logical clock: each append without an explicit timestamp
        # takes the next tick. XML provisioning appends first (earliest
        # ticks); conversation appends continue from there.
        self.register_buffer('_next_ts', torch.tensor(0, dtype=torch.long))

    def __len__(self):
        return int(self.count.item())

    def _fit(self, vec):
        """Conform an idea operand to the store width (leading ``nDim``,
        zero-padded). ``None`` -> the zero (Null) vector."""
        if vec is None:
            return self.slots.new_zeros(self.nDim)
        v = vec.reshape(-1).to(self.slots)
        if v.numel() == self.nDim:
            return v
        out = self.slots.new_zeros(self.nDim)
        k = min(self.nDim, v.numel())
        out[:k] = v[:k]
        return out

    @torch.no_grad()
    def append(self, np1, vp=None, np2=None, *, rel_type=REL_NONE,
               trust=0.0, timestamp=None) -> int:
        """Append one ternary row. ``Null`` slots (``None``) store the zero
        vector. ``trust`` is clamped to ``[-1, 1]``. ``timestamp`` defaults to
        the next monotonic tick. Returns the row index, or ``-1`` when the
        store is full."""
        n = int(self.count.item())
        if n >= self.capacity:
            return -1
        self.slots[n, 0] = self._fit(np1)
        self.slots[n, 1] = self._fit(vp)
        self.slots[n, 2] = self._fit(np2)
        self.rel_type[n] = int(rel_type)
        self.trust[n] = max(-1.0, min(1.0, float(trust)))
        if timestamp is None:
            self.timestamp[n] = float(self._next_ts.item())
            self._next_ts.fill_(int(self._next_ts.item()) + 1)
        else:
            ts = float(timestamp)
            self.timestamp[n] = ts
            # keep the clock monotonic past any explicit (e.g. provisioned) ts
            if ts >= float(self._next_ts.item()):
                self._next_ts.fill_(int(ts) + 1)
        self.count.fill_(n + 1)
        return n

    def append_idea(self, np1, *, trust=0.0, timestamp=None) -> int:
        """Append an ABSOLUTE truth (``NP . .``)."""
        return self.append(np1, None, None, rel_type=self.REL_NONE,
                           trust=trust, timestamp=timestamp)

    def append_relation(self, np1, vp, np2, *, rel_type=REL_OTHER,
                        trust=0.0, timestamp=None) -> int:
        """Append an IDEA-RELATION-IDEA (``NP VP NP``); ``np2=None`` ->
        ``NP VP .``."""
        return self.append(np1, vp, np2, rel_type=rel_type,
                           trust=trust, timestamp=timestamp)

    def row(self, idx: int) -> dict:
        """The stored row as ``{np1, vp, np2, rel_type, timestamp, trust}``
        (vectors are views; ``rel_type`` int, the rest floats)."""
        n = int(self.count.item())
        if not (0 <= int(idx) < n):
            raise IndexError(f"row {idx} of {n}")
        i = int(idx)
        return {
            'np1': self.slots[i, 0], 'vp': self.slots[i, 1],
            'np2': self.slots[i, 2], 'rel_type': int(self.rel_type[i].item()),
            'timestamp': float(self.timestamp[i].item()),
            'trust': float(self.trust[i].item()),
        }

    def recent(self, n: int):
        """Row indices of the ``n`` most-recent rows (descending timestamp) --
        the AR predictor's recency window."""
        c = int(self.count.item())
        if c == 0 or n <= 0:
            return self.count.new_zeros(0)
        order = torch.argsort(self.timestamp[:c], descending=True)
        return order[: int(n)]

    def relations(self, rel_type=None):
        """Row indices of RELATION rows (``rel_type != REL_NONE``), optionally
        filtered to a single ``rel_type`` -- the reasoning scan."""
        c = int(self.count.item())
        if c == 0:
            return self.count.new_zeros(0)
        rt = self.rel_type[:c]
        mask = rt != self.REL_NONE if rel_type is None else rt == int(rel_type)
        return mask.nonzero(as_tuple=True)[0]

    def ideas(self):
        """Row indices of ABSOLUTE-idea rows (``rel_type == REL_NONE``)."""
        c = int(self.count.item())
        if c == 0:
            return self.count.new_zeros(0)
        return (self.rel_type[:c] == self.REL_NONE).nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def set_trust(self, idx: int, trust: float):
        """Overwrite a row's scalar trust (verification / re-assertion)."""
        i = int(idx)
        if not (0 <= i < int(self.count.item())):
            raise IndexError(f"row {i} of {int(self.count.item())}")
        self.trust[i] = max(-1.0, min(1.0, float(trust)))

    @torch.no_grad()
    def reset(self):
        """Clear the store (idempotent)."""
        self.slots.zero_()
        self.rel_type.zero_()
        self.timestamp.zero_()
        self.trust.zero_()
        self.count.zero_()
        self._next_ts.zero_()


class InterSentenceLayer(Layer):
    """Inter-sentence ARMA(p, q) next-sentence predictor.

    **What it is.** Sentence-level autoregression lives here, not in
    the within-sentence body (which is now IR-only / masked-LM).  Each
    sentence is summarised into a flat ``[B, sentence_dim]`` rep
    ``s_t`` (the body's S-space_role root slot pooled per row); the
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
    ``SymbolSpace.layers`` so the Layer ergodic walk reaches the MLP
    predictor's parameters via ``SymbolSpace.params``.

    Replaces the pre-2026-05-14 contrastive cosine machinery
    (``context_window`` recent buffer, ``centroid_history`` repulsive
    ring, ``lam`` cosine push) and the AttentionLayer-based
    predictive head.  See ``doc/Architecture.md`` §"Sentence-level
    AR (InterSentenceLayer)" for the design rationale.
    """

    name = "Discourse"

    def __init__(self, n_symbols, max_depth, n_dim,
                 p=5, q=2, hidden_dim=None,
                 concept_dim=None, batch=1, ltm_capacity=1024,
                 # Legacy-named kwargs accepted for back-compat; ignored.
                 context_window=None, centroid_history=None, lam=None):
        """Initialize the ARMA(p, q) predictor.

        ``n_symbols``, ``max_depth``, ``n_dim`` describe the legacy
        ``[S | W]`` snapshot layout the caller used to hand in; under
        ARMA we only need the flat ``sentence_dim`` (defaults to
        ``n_symbols * n_dim``) but keep the constructor signature for
        smooth migration of the SymbolSpace instantiation site.

        ``p`` = AR lag count (number of past sentence reps consumed
        by the predictor); ``q`` = MA lag count (past residuals).
        ``hidden_dim`` defaults to ``2 * sentence_dim``.
        ``concept_dim`` is the target dim for the optional priming
        cast that lifts ``s_hat_{t+1}`` into C-space_role for the chat-
        loop's ``_c_prior`` injection.

        ``ltm_capacity`` bounds the long-term-memory (LTM) chain of
        per-sentence STM end-states (the AR sequence used for
        inter-sentence prediction; see ``observe_stm_end_state`` /
        ``get_stm_chain``).  Default 1024 — separate from TruthLayer's
        ``truthMaxEntries`` and from the ARMA orders ``p`` / ``q``.

        ``context_window`` / ``centroid_history`` / ``lam`` are
        accepted for signature back-compat with the retired
        contrastive constructor and ignored.
        """
        del context_window, centroid_history, lam  # retired knobs
        n_symbols = int(n_symbols)
        n_dim = int(n_dim)
        # Sentence-rep pooling: the per-sentence rep ``s_t`` is the
        # **root S-space_role slot** (row 0 of the body's final-stage S-space_role
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

        # -- LTM: chain of per-sentence STM end-states (Task 7, plan §8).
        # The AR sequence used for INTER-sentence prediction. Unlike the
        # ARMA ``_s_history`` ring (fixed-shape ``[B, p, D]`` tensor), an
        # end-state is VARIABLE depth (1 for an absolute sentence, 3 for
        # a relative ``[predicate, idea1, idea2]`` end-state), so a fixed
        # ``register_buffer`` tensor does not fit. Use one bounded
        # ``collections.deque(maxlen=ltm_capacity)`` PER ROW (parallel
        # document streams — mirror how ``_s_history`` is per-row), each
        # holding time-ordered ``(depth:int, payload:[depth,D] tensor,
        # tetralemma:tuple|None)`` tuples. This is TRANSIENT host-side
        # state (NOT checkpoint material — mirror the ARMA rings'
        # ``persistent=False``); it is NOT a registered buffer / not in
        # ``state_dict`` and the boundary-only ``observe_stm_end_state``
        # writer is ``@torch.compiler.disable``'d like ``observe``.
        self.ltm_capacity = int(ltm_capacity)
        self._stm_end_states = [
            collections.deque(maxlen=self.ltm_capacity)
            for _ in range(self._batch)]

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
        # the cast lifts s_hat_{t+1} into C-space_role so it can be added
        # as a bias before the body's sigma-pi loop.
        self.cast = None
        if self.concept_dim is not None and self.sentence_dim > 0:
            self.cast = LinearLayer(self.sentence_dim, self.concept_dim)
            self.layers.append(self.cast)

        # -- Inter-level next-end-state predictor (Task 8, plan §9). -----
        # The SAME predictor class as the intra-sentence in-STM predictor
        # (``IntraSentenceLayer``), instantiated HERE at the inter-sentence
        # level. It reads the last ``K`` STM end-states (the LTM chain) and
        # predicts the ROOT of the next end-state. ``predict_next_end_state``
        # wraps it into a ``(depth_hat, payload_hat[depth_hat, D])`` shape.
        #
        # ``stm_capacity`` here is the CHAIN WINDOW ``K`` (the number of
        # most-recent end-states fed to the predictor), NOT the intra-level
        # STM slot count. The plan suggests ``truthMaxEntries``; that knob
        # (default 1024) is the TruthLayer WS-codebook cap and is NOT
        # plumbed to this constructor, and a 1024-wide predictor input would
        # be wasteful for a short AR context. So we use a small bounded
        # window ``K = min(ltm_capacity, 8)`` (documented deviation): the AR
        # signal that predicts the next end-state lives in the last handful
        # of sentences, and the IntraSentenceLayer's ``stm_capacity`` only
        # feeds its serial-``reverse`` fan-out default (unused on our
        # forward-only path), so ``K`` is the meaningful quantity.
        self._inter_chain_window = max(1, min(int(self.ltm_capacity), 8))
        self._inter_predictor = None
        if self.concept_dim is not None and self.concept_dim > 0:
            self._inter_predictor = IntraSentenceLayer(
                concept_dim=self.concept_dim,
                stm_capacity=self._inter_chain_window,
                routing_dim=self.concept_dim)
            self.layers.append(self._inter_predictor)

        # L_inter accumulation (live grad tensor) + the per-row last
        # predicted root, mirroring ConceptualSpace's intra-loss accumulator
        # (Spaces.py ``_accumulate_intra_loss`` / ``consume_intra_loss``).
        # ``_inter_loss_accum`` is a live (grad-carrying) running SUM of
        # per-sentence ``MSE(payload_hat_root, actual_root)``; ``None`` until
        # the first scored sentence. ``_inter_loss_count`` is the mean
        # denominator. ``_inter_last_pred_root`` holds the most-recent
        # predicted root PER ROW (``[D]`` tensor or ``None``) so the next
        # ``observe_stm_end_state`` can score the prediction it made against
        # the end-state that actually arrived. ``_inter_loss_weight`` gates
        # accumulation off when the knob is non-positive (set by the host at
        # construction; defaults to 0.1 so the layer is self-contained).
        self._inter_loss_accum = None
        self._inter_loss_count = 0
        self._inter_last_pred_root = [None] * self._batch
        self._inter_loss_weight = 0.1
        # Second accumulator: the InfoNCE next-idea CONTRASTIVE term -- ranks the
        # actual next root above the chain's past roots under cosine(pred, .).
        # Off (weight 0) -> the layer carries only the legacy MSE L_inter
        # (byte-identical). Set by the host from <interContrastiveWeight>.
        self._inter_contrastive_accum = None
        self._inter_contrastive_count = 0
        self._inter_contrastive_weight = 0.0
        self._inter_contrastive_temp = 0.1

        # LTM consolidation FU3 (Change 2, 2026-06-18): when wired to the
        # unified ``TernaryTruthStore`` (``_ltm_store`` set by the host at
        # construction, ``_ltm_consolidation`` True), the AR predictor reads
        # the GLOBAL store's recency window (:meth:`get_stm_chain` ->
        # ``ltm_store.recent``) instead of this layer's per-row deque
        # ``_stm_end_states``, and ``observe_stm_end_state`` STOPS appending
        # to that deque (the store-append at the Models observe site is the
        # single source). ``None`` / ``False`` (the default) -> the legacy
        # per-row deque path, byte-identical. NOTE: the unified store is
        # GLOBAL (one recency window, not per row), so the store-backed read
        # is the correct semantics for B=1 / a single conversation; B>1
        # batched training shares one global recency window across rows.
        self._ltm_store = None
        self._ltm_consolidation = False

    # -- per-batch resize ---------------------------------------------
    def ensure_batch(self, batch):
        """Resize per-row substrate to a new batch size.  Cascaded
        from ``SymbolSpace.ensure_batch`` so the body's per-B state is
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
        # Resize the per-row LTM chains to match. Reallocate fresh
        # deques: the ARMA rings above zero on resize, so the LTM chain
        # follows the same "new batch -> clean per-row state" semantic
        # (cascaded from ``SymbolSpace.ensure_batch`` at the start of a
        # microbatch; the old document's chain does not survive a batch
        # reshape any more than ``_s_history`` does).
        self._stm_end_states = [
            collections.deque(maxlen=self.ltm_capacity)
            for _ in range(batch)]
        # Per-row last-predicted-root parks reset on a batch reshape too
        # (the prior document's pending prediction does not survive, same
        # as the LTM chain / ARMA rings above).
        self._inter_last_pred_root = [None] * batch

    # -- sentence-rep pooling -----------------------------------------
    def _pool_sentence_rep(self, s_tensor):
        """Pool an S-space_role event into a per-row ``[B, sentence_dim]``
        sentence rep.

        Accepts:
          * ``[B, N, D]`` (microbatch S-space_role event) -- take row 0
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
        # Take the root S-space_role slot (row 0) per batch row.
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

    # -- LTM: long-term memory chain of STM end-states (Task 7) --------
    @torch.compiler.disable
    def observe_stm_end_state(self, depths, payloads, tetralemmas=None):
        """Append one STM end-state PER ROW to the LTM chain.

        Called from the sentence-boundary hook AFTER the reduce /
        relative-preserve step has decided each row's end-state. EVERY
        sentence's end-state lands in LTM regardless of
        ``truthCriterion`` — LTM is the AR sequence the inter-sentence
        predictor consumes; ``truthCriterion`` only gates the separate
        WS-codebook insertion of *learned relations* (Task 6c), not
        this chain.

        Args:
          depths: ``[B]`` ints/long tensor (or python sequence) — the
            end-state depth for each row (1 for an absolute sentence, 3
            for a relative ``[predicate, idea1, idea2]`` end-state).
          payloads: ragged list of length ``B``; ``payloads[b]`` is the
            ``[depth_b, D]`` tensor of the end-state slots for row ``b``
            (since depths differ per row a ragged list is natural, not a
            single padded tensor).
          tetralemmas: OPTIONAL list (length ``B``) / ``None``. The per-row
            scalar trust to store with the end-state; ``None`` (the default)
            parks ``None`` in every row's trust slot.

        Stored per row as a time-ordered tuple
        ``(depth:int, payload:[depth,D] tensor, trust:float|None)``.
        Bounded at ``ltm_capacity`` (deque ``maxlen``): the oldest
        end-state is evicted once the chain is full.

        Fail-loud: a non-finite (NaN/Inf) payload RAISES — a corrupt
        end-state must never be silently stored (user memory:
        "fail loud on numerical divergence").
        """
        if depths is None or payloads is None:
            return
        # Normalise ``depths`` to a python list of ints without forcing
        # a per-row host sync inside any captured region (this method is
        # ``@torch.compiler.disable``'d and boundary-only, so a single
        # ``.tolist()`` host hop here is fine).
        if isinstance(depths, torch.Tensor):
            depth_list = [int(d) for d in depths.detach().reshape(-1).tolist()]
        else:
            depth_list = [int(d) for d in depths]
        B = len(payloads)
        if len(self._stm_end_states) != B:
            # The chain list lags an un-cascaded batch reshape; grow /
            # shrink to match this boundary's row count. (ensure_batch
            # normally keeps these in lockstep; this is defensive.)
            self.ensure_batch(B)
        for b in range(B):
            payload = payloads[b]
            depth = depth_list[b] if b < len(depth_list) else (
                int(payload.shape[0]) if payload is not None else 0)
            if payload is not None:
                if not torch.isfinite(payload).all():
                    raise FloatingPointError(
                        "InterSentenceLayer.observe_stm_end_state: row "
                        f"{b} STM end-state payload contains NaN/Inf "
                        "(depth={}, shape={}). Refusing to store a "
                        "corrupt end-state in the LTM chain."
                        .format(depth, tuple(payload.shape)))
                # L_inter (Task 8, plan §9): if a prediction was made for
                # THIS row's end-state (``predict_next_end_state`` stashed a
                # live predicted root on ``_inter_last_pred_root[b]``), score
                # it against the end-state that actually arrived. Compare the
                # predicted ROOT against the actual end-state's ROOT (the
                # OLDEST slot under newest-at-slot-0, via
                # ``_reduce_end_state_to_root``; reduced to the predictor
                # width) — this is robust to a depth mismatch between
                # prediction and reality (the plan's "when depths differ,
                # compare roots"). The actual root is DETACHED
                # so the loss trains ``_inter_predictor``, not the perception
                # path. Score BEFORE the detach/clone below so the live
                # payload's slot-0 is available; the target is detached
                # regardless.
                if (self._inter_predictor is not None
                        and b < len(self._inter_last_pred_root)
                        and self._inter_last_pred_root[b] is not None):
                    pd = payload.detach()
                    if (self._ltm_store is not None and pd.dim() == 2
                            and pd.shape[0] >= 2):
                        # Consolidated: the incoming payload is STM-order
                        # [idea2, idea1, predicate] (newest-at-slot-0); the AR
                        # root is idea1 (slot depth-2). Wrap as [1, D] so
                        # _reduce (consolidated -> slot 0) returns it, matching
                        # the infix slot-0 root get_stm_chain feeds the
                        # predictor (so context and actual roots agree).
                        actual_root = self._reduce_end_state_to_root(
                            pd[pd.shape[0] - 2].unsqueeze(0))
                    else:
                        actual_root = self._reduce_end_state_to_root(pd)
                    pred_root = self._inter_last_pred_root[b]
                    if actual_root is not None:
                        actual_root = actual_root.to(
                            device=pred_root.device, dtype=pred_root.dtype)
                        self._accumulate_inter_loss(pred_root, actual_root)
                        # InfoNCE: rank the actual next root above the chain's
                        # PAST roots (negatives) under cosine(pred, .). Best-
                        # effort -- a short/odd chain just yields fewer
                        # negatives (the accumulator no-ops with none; the MSE
                        # term above still runs). Gated by the contrastive weight.
                        if float(getattr(self, "_inter_contrastive_weight",
                                         0.0)) > 0.0:
                            negs = []
                            try:
                                for (_d, _pl, _t) in self.get_stm_chain(
                                        n=8, b=b):
                                    _rt = self._reduce_end_state_to_root(_pl)
                                    if _rt is not None and torch.is_tensor(_rt):
                                        negs.append(_rt.detach().to(
                                            device=pred_root.device,
                                            dtype=pred_root.dtype))
                            except Exception:
                                negs = []
                            self._accumulate_inter_contrastive(
                                pred_root, actual_root.detach(), negs)
                # Detach + clone so the stored end-state is a stable
                # snapshot decoupled from the live STM buffer (which the
                # next sentence overwrites in place) and carries no
                # autograd history into the host-side ring.
                payload = payload.detach().clone()
            # The pending prediction for this row has now been scored (or
            # there was none); clear it so a row that observes WITHOUT a
            # fresh predict_next_end_state isn't double-counted next time.
            if b < len(self._inter_last_pred_root):
                self._inter_last_pred_root[b] = None
            tet = None
            if tetralemmas is not None and b < len(tetralemmas):
                tet = tetralemmas[b]
            # LTM consolidation FU3 (Change 2): when wired to the unified
            # store the per-row deque is NOT the source of truth -- the
            # Models observe site appends each end-state to the global
            # ``ltm_store`` (sink (a)). Skip the deque append here so the AR
            # chain is read SOLELY from the store (via ``get_stm_chain``);
            # the L_inter predict/score cycle ABOVE still runs (it reads
            # context through ``get_stm_chain`` -> store, and scores the
            # ``payload`` arg against the staged prediction). Legacy /
            # non-consolidated path is unchanged (deque append as today).
            if self._ltm_store is None:
                self._stm_end_states[b].append((int(depth), payload, tet))

    @torch.compiler.disable
    def predict_and_observe_stm_end_state(self, depths, payloads,
                                          tetralemmas=None):
        """Stage the next-end-state prediction THEN observe the arriving
        end-state, in one call — the foolproof AR ordering for the TRAINING
        sentence-boundary hook.

        The two-call staging protocol (``predict_next_end_state`` then
        ``observe_stm_end_state``) is correct but easy to get wrong: the
        training boundary hook historically called only ``observe`` and so
        never staged a prediction, leaving ``L_inter`` permanently empty (the
        inter-predictor never trained). This method makes the order
        impossible to get wrong by composing the two existing pieces.

        AR order (the WHOLE point): for EVERY batch row, run
        ``predict_next_end_state(b)`` FIRST — predicting from the chain state
        BEFORE this boundary's end-state is appended, so the prediction is
        genuinely "the next end-state from history". THEN call
        ``observe_stm_end_state`` once for the whole batch — which scores each
        row's just-staged prediction against the end-state that actually
        arrived (accumulating ``L_inter``) and only then appends it to the
        chain. No double-append, no double-score: prediction reads the old
        chain, observation supervises against the new end-state and grows the
        chain by exactly one per row.

        Args / semantics are identical to ``observe_stm_end_state`` (see its
        docstring) — this is purely the correct ordering wrapper. Cold-start
        safety is inherited unchanged: the first sentence predicts from an
        empty chain (degenerate zeros root, ``_inter_last_pred_root[b]``
        cleared to ``None``), so ``observe`` finds nothing pending to score
        and just appends — exactly as the staged path degenerates today. When
        there is no inter-predictor (``sentencePrediction`` off / absolute-
        only configs) ``predict_next_end_state`` returns ``None`` and stages
        nothing, so this degenerates to a bare ``observe`` — byte-identical
        to the pre-change boundary behaviour.
        """
        if depths is None or payloads is None:
            return
        # Stage the next-end-state prediction for EVERY row from the chain
        # state BEFORE the new end-states are appended (``observe`` below does
        # the appends). ``predict_next_end_state`` records the live predicted
        # root on ``_inter_last_pred_root[b]`` (or clears it on cold start /
        # no predictor), which ``observe_stm_end_state`` then scores.
        if self._inter_predictor is not None:
            B = len(payloads)
            for b in range(B):
                self.predict_next_end_state(b)
        # Score each staged prediction against the arriving end-state
        # (accumulating ``L_inter``) and append to the chain. Reusing the
        # existing method keeps the fail-loud / detach / ragged-payload /
        # tetralemma handling identical and avoids any duplication.
        self.observe_stm_end_state(depths, payloads, tetralemmas=tetralemmas)

    def get_stm_chain(self, n=None, b=0):
        """Return the last ``n`` LTM end-states for row ``b``.

        Args:
          n: number of most-recent end-states to return; ``None`` (the
            default) returns the entire chain for the row.
          b: batch row index (default 0 — the test uses B=1 / row 0).

        Returns a python ``list`` of time-ordered tuples (oldest first,
        most-recent last), each ``(depth:int, payload:[depth,D] tensor,
        trust:float|None)``. Returns ``[]`` for an out-of-range row
        or an empty chain.

        LTM consolidation FU3 (Change 2): when wired to the unified store
        (``self._ltm_store is not None``) the chain is read from the GLOBAL
        ``TernaryTruthStore`` recency window, NOT this layer's per-row deque.
        The store is global (one recency window), so ``b`` is IGNORED -- this
        is the correct semantics for B=1 / a single conversation; B>1 batched
        training shares the one global recency window across rows.
        ``store.recent(n)`` returns DESCENDING-timestamp indices, so we
        reverse to OLDEST-FIRST time order and reconstruct each row as a
        ``(depth, payload, tet)`` tuple matching the deque convention:
          * an ABSOLUTE row (``rel_type == REL_NONE``) ->
            ``(1, np1[None, :], trust)``;
          * a RELATION row -> ``(3, stack([np1, vp, np2]), trust)`` -- INFIX
            ``[idea1, predicate, idea2]`` (the store's native ``[NP1, VP, NP2]``
            order; idea1 may be present without a predicate, so it anchors the
            triple). The consolidated ``_reduce_end_state_to_root`` reads slot
            0 = idea1 (the subject / always-present anchor) as the AR root.
        """
        store = self._ltm_store
        if store is not None:
            total = len(store)
            if total == 0:
                return []
            if n is not None and int(n) <= 0:
                return []                            # match the deque path
            k = total if n is None else min(int(n), total)
            # recent() -> descending timestamp; reverse for oldest-first.
            idxs = store.recent(k)
            idx_list = list(reversed([int(i) for i in idxs.tolist()]))
            out = []
            for i in idx_list:
                r = store.row(i)
                tet = r["trust"]
                if r["rel_type"] == store.REL_NONE:
                    out.append((1, r["np1"].reshape(1, -1).clone(), tet))
                else:
                    payload = torch.stack(
                        [r["np1"], r["vp"], r["np2"]], dim=0).clone()
                    out.append((3, payload, tet))
            return out
        bi = int(b)
        if bi < 0 or bi >= len(self._stm_end_states):
            return []
        chain = self._stm_end_states[bi]
        if n is None:
            return list(chain)
        n = int(n)
        if n <= 0:
            return []
        if n >= len(chain):
            return list(chain)
        # Last ``n``, preserving time order.
        return list(chain)[-n:]

    # -- inter-sentence next-end-state prediction (Task 8, plan §9) -----
    def _reduce_end_state_to_root(self, payload):
        """Reduce one (ragged-depth) end-state payload ``[depth, D]`` to a
        single representative ``[D]`` vector: its ROOT.

        The chain's payloads are ragged (depth 1 for an absolute end-state,
        depth 3 for a relative ``[predicate, idea1, idea2]`` end-state). To
        feed a FIXED-width ``[1, K, D]`` context into the predictor we
        collapse each end-state to ONE vector.

        Newest-at-slot-0 convention: the STM end-state is stored
        newest-first, so the root/predicate lives at the OLDEST slot
        (index ``depth-1``), NOT slot 0. We take that last slot — for an
        absolute end-state (depth 1) it is slot 0 (the collapsed idea), and
        for a relative end-state (depth 3) it is slot 2, the predicate (the
        head the relative structure hangs off). This recovers the SAME root
        vector the old oldest-first ``x[0]`` read returned. (Documented
        reduction; mean-over-depth is the obvious alternative but the root
        carries the sentence-level signal.)

        Right-pads / left-truncates to ``concept_dim`` so a chain payload
        whose D differs from the predictor's width stays well-defined.
        """
        if payload is None:
            return None
        x = payload
        if x.dim() == 1:
            root = x
        elif x.dim() == 2 and x.shape[0] >= 1:
            if self._ltm_store is not None:
                # Consolidated / INFIX payloads [idea1, predicate, idea2]: the
                # AR root is idea1 (slot 0) -- the subject/topic, present even
                # when there is no predicate (Alec 2026-06-18, infix order).
                root = x[0]
            else:
                root = x[x.shape[0] - 1]             # legacy newest-at-slot-0: oldest slot
        else:
            return None
        D = int(self._inter_predictor.concept_dim)
        cur = int(root.shape[-1])
        if cur == D:
            return root
        if cur > D:
            return root[:D]
        return F.pad(root, (0, D - cur))

    @torch.compiler.disable
    def predict_next_end_state(self, b=0):
        """Predict the SHAPE of the next STM end-state from the LTM chain.

        Returns ``(depth_hat:int, payload_hat:[depth_hat, D] tensor)`` where
        ``depth_hat`` is the predicted depth of the next end-state and
        ``payload_hat`` is the predicted end-state slots. Runs the
        inter-level ``IntraSentenceLayer`` (``self._inter_predictor``) over
        the chain's last-``K`` end-state ROOTS (``K`` = ``_inter_chain_window``)
        to produce the predicted ROOT, then broadcasts that root across
        ``depth_hat`` slots.

        Mechanism:
          * **Chain reduction** (ragged -> fixed): reduce each of the last
            ``K`` end-states to its root (``_reduce_end_state_to_root`` —
            the OLDEST slot under newest-at-slot-0)
            to form a ``[1, K, D]`` context, LEFT-padding with zeros when the
            chain is shorter than ``K`` (so the most-recent end-state is
            always at the tail, mirroring the ARMA ring's newest-at-``-1``).
          * **Root prediction**: ``self._inter_predictor.forward(context,
            routing=None, parallel=False)`` -> ``[1, D]``, the predicted root
            of the next end-state.
          * **depth_hat**: a simple AR prior — the depth of the MOST RECENT
            end-state in the chain (a relative sentence tends to be followed
            by structure of the same shape; an absolute by an absolute). The
            test only requires ``depth in {1, 3}`` + finite payload; this AR
            prior delivers exactly that without a separate learned head.
            (Documented: a tiny ``concept_dim -> 2`` argmax head is the
            richer alternative; the AR prior is the agreed scaffold.)
          * **payload_hat**: the predicted root broadcast across
            ``depth_hat`` slots -> ``[depth_hat, D]``.

        Cold start (empty chain): returns ``(1, zeros[1, D])`` — a
        well-defined degenerate shape the caller can stage or ignore.

        Records the predicted root on ``_inter_last_pred_root[b]`` so the
        next ``observe_stm_end_state`` can score it (``L_inter``).

        Fail-loud: a non-finite predicted root RAISES (user memory: fail
        loud on numerical divergence).
        """
        if self._inter_predictor is None:
            return None
        bi = int(b)
        D = int(self._inter_predictor.concept_dim)
        device = next(self._inter_predictor.parameters()).device
        dtype = next(self._inter_predictor.parameters()).dtype
        K = int(self._inter_chain_window)
        chain = self.get_stm_chain(n=K, b=bi)
        if not chain:
            # Cold start: no AR signal yet. Degenerate (1, zeros[1, D]).
            self._inter_last_pred_root[bi] = None
            zero_root = torch.zeros(D, device=device, dtype=dtype)
            return 1, zero_root.unsqueeze(0)
        # Reduce each end-state to its root and LEFT-pad to K so the
        # most-recent sits at the tail (newest-at--1, like the ARMA ring).
        roots = []
        for (_depth, payload, _tet) in chain:
            r = self._reduce_end_state_to_root(payload)
            if r is None:
                r = torch.zeros(D, device=device, dtype=dtype)
            roots.append(r.to(device=device, dtype=dtype))
        if len(roots) < K:
            pad = [torch.zeros(D, device=device, dtype=dtype)
                   for _ in range(K - len(roots))]
            roots = pad + roots
        context = torch.stack(roots, dim=0).unsqueeze(0)   # [1, K, D]
        payload_root_hat = self._inter_predictor.forward(
            context, routing=None, parallel=False)         # [1, D]
        if not torch.isfinite(payload_root_hat).all():
            raise FloatingPointError(
                "InterSentenceLayer.predict_next_end_state: predicted "
                "end-state root contains NaN/Inf. Refusing to emit a "
                "corrupt prediction.")
        root_vec = payload_root_hat.reshape(-1)[:D]         # [D]
        # depth_hat: AR prior = depth of the most-recent end-state.
        depth_hat = int(chain[-1][0])
        if depth_hat <= 0:
            depth_hat = 1
        # Record the predicted root so the NEXT ``observe_stm_end_state``
        # can score it (``L_inter``). When grad is enabled we keep the LIVE
        # (grad-carrying) root tensor so the deferred MSE backprops into
        # ``_inter_predictor``'s params; under no-grad (eval / generation)
        # ``forward`` already produced a detached value, so this is just the
        # prediction's value with no graph attached.
        self._inter_last_pred_root[bi] = root_vec
        payload_hat = root_vec.unsqueeze(0).expand(depth_hat, -1)
        return depth_hat, payload_hat

    def _accumulate_inter_loss(self, pred_root, actual_root):
        """Accumulate one sentence of ``L_inter = MSE(pred_root,
        actual_root)`` into the live running sum, gated on grad-enabled +
        positive weight (mirrors ``ConceptualSpace._accumulate_intra_loss``).

        ``pred_root`` / ``actual_root``: ``[D]`` (or broadcastable) live
        tensors. No-op when grad is disabled (eval) or the inter-loss weight
        is non-positive — avoids eval-time graph growth and a wasted MSE
        when the term is off.

        Fail-loud: a non-finite step loss RAISES.
        """
        if not torch.is_grad_enabled():
            return
        if float(self._inter_loss_weight) <= 0.0:
            return
        if pred_root is None or actual_root is None:
            return
        step_loss = F.mse_loss(pred_root, actual_root)
        if not torch.isfinite(step_loss).all():
            raise FloatingPointError(
                "InterSentenceLayer._accumulate_inter_loss: L_inter step "
                "is NaN/Inf. Refusing to accumulate a corrupt loss term.")
        if self._inter_loss_accum is None:
            self._inter_loss_accum = step_loss
        else:
            self._inter_loss_accum = self._inter_loss_accum + step_loss
        self._inter_loss_count += 1

    def consume_inter_loss(self):
        """Return the per-sentence MEAN of the accumulated inter-sentence
        prediction loss and RESET the accumulator.

        Returns a live scalar tensor (mean over the scored sentences)
        carrying grad to ``_inter_predictor``'s params, or ``None`` when
        nothing was accumulated this batch (eval-time, weight off, or no
        sentence was both predicted-for and observed). Mirrors
        ``ConceptualSpace.consume_intra_loss`` — ``runBatch`` consumes it
        once, post-body / pre-backward, next to the ARMA + intra terms.
        """
        if self._inter_loss_accum is None:
            self._inter_loss_count = 0
            return None
        mean_loss = self._inter_loss_accum / max(1, self._inter_loss_count)
        self._inter_loss_accum = None
        self._inter_loss_count = 0
        return mean_loss

    def set_inter_loss_weight(self, weight):
        """Set the inter-loss accumulation gate (read from the
        ``interLossWeight`` knob by the host at construction)."""
        self._inter_loss_weight = float(weight)

    def set_inter_contrastive(self, weight, temp=0.1):
        """Set the InfoNCE next-idea contrastive gate + softmax temperature
        (read from ``interContrastiveWeight`` / ``interContrastiveTemp``)."""
        self._inter_contrastive_weight = float(weight)
        self._inter_contrastive_temp = max(1e-4, float(temp))

    def _accumulate_inter_contrastive(self, pred_root, pos_root, neg_roots):
        """Accumulate one InfoNCE step: rank ``pos_root`` (the actual next root)
        above ``neg_roots`` (the chain's past roots) under
        ``cosine(pred_root, .)/temp``. ``pred_root`` ``[D]`` is grad-bearing (the
        staged prediction into ``_inter_predictor``); ``pos_root`` + ``neg_roots``
        are DETACHED, so the gradient trains the predictor, never the targets.
        No-op under no-grad / weight<=0 / no negatives (the caller's MSE term
        still runs). Fail-loud on NaN."""
        if not torch.is_grad_enabled():
            return
        if float(self._inter_contrastive_weight) <= 0.0:
            return
        if pred_root is None or pos_root is None or not neg_roots:
            return
        q = pred_root.reshape(-1)
        q = q / q.norm().clamp_min(1e-12)
        cand = torch.stack([pos_root.reshape(-1)]
                           + [n.reshape(-1) for n in neg_roots], dim=0)  # [1+N,D]
        cn = cand / cand.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits = (cn @ q) / float(self._inter_contrastive_temp)          # [1+N]
        step = F.cross_entropy(
            logits.unsqueeze(0),
            torch.zeros(1, dtype=torch.long, device=logits.device))
        if not torch.isfinite(step).all():
            raise FloatingPointError(
                "InterSentenceLayer._accumulate_inter_contrastive: InfoNCE "
                "step is NaN/Inf. Refusing to accumulate a corrupt loss term.")
        if self._inter_contrastive_accum is None:
            self._inter_contrastive_accum = step
        else:
            self._inter_contrastive_accum = self._inter_contrastive_accum + step
        self._inter_contrastive_count += 1

    def consume_inter_contrastive_loss(self):
        """Per-sentence MEAN of the accumulated InfoNCE next-idea loss; resets.
        ``None`` when nothing accumulated (eval / weight off / no warm chain).
        Mirrors :meth:`consume_inter_loss`; ``runBatch`` consumes it once,
        post-body / pre-backward."""
        if self._inter_contrastive_accum is None:
            self._inter_contrastive_count = 0
            return None
        mean = self._inter_contrastive_accum / max(1, self._inter_contrastive_count)
        self._inter_contrastive_accum = None
        self._inter_contrastive_count = 0
        return mean

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
            # Clear the LTM chain on a hard discourse boundary. The plan
            # (§8) defines LTM as the AR sequence for INTER-sentence
            # prediction, so a document-boundary reset clears it the same
            # way ``_s_history`` zeros — the next document starts cold.
            for dq in self._stm_end_states:
                dq.clear()
            # Drop any pending inter-sentence prediction + the live loss
            # accumulator (Task 8): the next document predicts cold, and a
            # boundary must not leak a half-formed grad term across the
            # reset (mirrors the chain/ring clear above).
            self._inter_last_pred_root = [None] * self._batch
            self._inter_loss_accum = None
            self._inter_loss_count = 0
            self._inter_contrastive_accum = None
            self._inter_contrastive_count = 0
        else:
            bi = int(batch)
            self._s_history[bi].zero_()
            self._s_count[bi] = 0
            self._e_history[bi].zero_()
            self._e_count[bi] = 0
            if 0 <= bi < len(self._stm_end_states):
                self._stm_end_states[bi].clear()
            if 0 <= bi < len(self._inter_last_pred_root):
                self._inter_last_pred_root[bi] = None
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
        downstream callers (``SymbolSpace.stm_residual_microbatch``)
        gate the priming bias on ``conf is None`` so we emit a
        placeholder rather than break that wiring.  Real cold-start
        gating happens via ``_s_count``: rows whose ring is empty
        return ``(None, None)``.

        Compiled path: ``predict()`` is called from two traced sites
        (``BasicModel._forward_per_stage`` and
        ``SymbolSpace.stm_residual_microbatch``), always with ``b is
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
        consumes only the S-space_role sentence rep so ``w_tensor`` is
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
        """Lift a predicted sentence rep into C-space_role for the chat-
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


class IntraSentenceLayer(Layer):
    """In-STM autoregressive predictor: STM[1:end] -> predicted STM[0]
    (serial) / STM_prev[0:end] -> predicted STM_new[0:end] (parallel).
    Combined PI-then-Sigma, no intermediate tanh; the routing
    distribution from SymbolSubSpace.current_rules conditions the
    Sigma collapse (rule-aware predictor).

    Architecture (PI first, per the design note):

        pi:    PiLayer(concept_dim -> working_dim), invertible, nonlinear
               (log-domain multiplicative boundary fold). The PI body
               lifts low-rank STM slots into the working width.
        sigma: SigmaLayer(working_dim -> concept_dim), invertible,
               **nonlinear=False** (raw linear W @ x + b). Sigma
               collapses the lifted slots into the predicted idea.

    The defining requirement of this layer is that there is NO
    intermediate activation between the PI body and the Sigma body:
    ``sigma`` is built ``nonlinear=False`` so the two linear cores fuse
    cleanly (PI's log-domain exp/tanh boundary still bounds PI's own
    output, but no extra tanh is interposed before Sigma's matmul).

    ``working_dim`` defaults to ``concept_dim`` so both sublayers are
    square isomorphisms and the per-slot (parallel) round trip
    ``reverse(forward(x, parallel=True)) ~= x`` is exact up to the LDU
    inverse tolerance. The serial path's cross-slot Sigma collapse is
    many-to-one and therefore only **approximately** invertible (see
    ``reverse``).

    Routing conditioning: when ``routing`` is not None it is projected
    ``[B, routing_dim] -> [B, concept_dim]`` by ``self.routing_proj`` and
    added as a bias to the Sigma output. This makes the predictor
    rule-aware without a separate attention block. ``routing=None``
    skips the bias entirely so the layer is usable before the per-word
    router is wired (Task 4 of the STM serial/parallel plan).

    Subclasses ``Layer`` (not ``Space``): no SubSpace, no tensor-map
    contract. ``self.pi`` / ``self.sigma`` are appended to
    ``self.layers`` (an ``nn.ModuleList``) so the Layer ergodic /
    paramUpdate cascade and the Space's Start/Reset walk reach their
    parameters. ``self.routing_proj`` is a plain ``nn.Linear`` (no
    ergodic interface) and registers via attribute assignment.
    """

    name = "IntraSentence"

    def __init__(self, concept_dim, stm_capacity, routing_dim,
                 working_dim=None, naive=True, ergodic=False,
                 stable=True, monotonic=False, batch=1):
        """Build the combined PI-then-Sigma in-STM predictor.

        ``concept_dim``: per-slot C-space_role feature width D (the input and
        output width of the predictor).
        ``stm_capacity``: STM slot count; the default fan-out ``k`` for
        the serial ``reverse`` (``k = stm_capacity - 1``: the predictor
        consumes STM[1:end], i.e. all but the freshly-predicted slot 0).
        ``routing_dim``: width of the soft routing distribution injected
        as the Sigma bias. Not yet populated (Task 4); the
        ``routing_proj`` is built so the conditioning path exists, but
        ``forward``/``reverse`` accept ``routing=None`` and skip it.
        ``working_dim``: PI lift width (Sigma input width). Defaults to
        ``concept_dim`` to keep both sublayers square (exact per-slot
        inverse).

        ``naive`` / ``ergodic`` / ``stable`` / ``monotonic`` follow the
        C-space_role substrate pi/sigma construction convention (see
        ``ConceptualSpace.sigma_in`` / ``sigma_cs``).
        """
        concept_dim = int(concept_dim)
        working_dim = int(working_dim) if working_dim is not None else concept_dim
        super().__init__(concept_dim, concept_dim)

        self.concept_dim = concept_dim
        self.working_dim = working_dim
        self.stm_capacity = int(stm_capacity)
        self.routing_dim = int(routing_dim)
        self._batch = int(batch)

        # PI first: lift each STM slot from concept_dim into the working
        # width. Log-domain multiplicative boundary fold (nonlinear=True,
        # the PiLayer default) bounds the lifted slots to [-1, 1] so the
        # downstream raw-linear Sigma sees a well-scaled operand.
        self.pi = PiLayer(
            concept_dim, working_dim,
            naive=naive, ergodic=ergodic, invertible=True,
            hasBias=True, stable=stable, monotonic=monotonic,
            nonlinear=True)
        # Sigma SECOND with nonlinear=False: raw linear W @ x + b, so NO
        # extra tanh is interposed between the PI body and the Sigma body.
        # NOTE: this is NOT a linear fusion -- the PI body above is
        # nonlinear=True (its symmetric log-domain (1+x)/(1-x) embedding is
        # an intrinsic nonlinearity), so the composite is pi(nonlinear)
        # followed by a raw-linear Sigma, not two fusable linear cores. The
        # only guarantee is "no interposed activation between the two".
        self.sigma = SigmaLayer(
            working_dim, concept_dim,
            naive=naive, ergodic=ergodic, invertible=True,
            nonlinear=False, stable=stable, monotonic=monotonic)

        # Expose the sublayers' params to the Layer ergodic / paramUpdate
        # cascade AND the Space Start/Reset walk via the ModuleList.
        self.layers = nn.ModuleList([self.pi, self.sigma])

        # Routing conditioning: project the soft rule distribution into
        # concept_dim and add to the Sigma output. Built unconditionally
        # so the conditioning path exists; only applied when a non-None
        # routing tensor is supplied (Task 4 wires the producer).
        self.routing_proj = nn.Linear(self.routing_dim, concept_dim)

    # -- core PI->Sigma transform ------------------------------------
    def _pi_sigma(self, x):
        """Apply ``sigma(pi(x))`` on a ``[..., concept_dim]`` operand
        (no routing bias, no cross-slot collapse). Returns
        ``[..., concept_dim]``. Used per-slot by both the serial and
        parallel forward paths.
        """
        return self.sigma.forward(self.pi.forward(x))

    def _apply_routing_bias(self, y, routing, sign=1):
        """Add (``sign=+1``, forward) or subtract (``sign=-1``, reverse)
        the projected routing bias on a Sigma output ``y``.

        ``y``: ``[B, D]`` (serial) or ``[B, N, D]`` (parallel).
        ``routing``: ``[B, routing_dim]`` or None. When None this is a
        no-op (keeps the layer usable before the router is wired). When
        present the bias is projected to ``[B, D]`` and broadcast over
        the slot axis in the parallel case. The reverse path passes
        ``sign=-1`` to exactly undo the additive forward bias before
        inverting the Sigma body.
        """
        if routing is None:
            return y
        bias = self.routing_proj(routing)            # [B, D]
        if y.dim() == 3:
            bias = bias.unsqueeze(1)                 # [B, 1, D] -> broadcast over N
        return y + sign * bias

    def forward(self, prior_slots, routing=None, parallel=False):
        """Predict the next idea(s) from prior STM slots.

        ``prior_slots``: ``[B, K, D]`` (K prior slots).
        ``routing``: ``[B, routing_dim]`` soft rule distribution, or
            None to skip the rule-aware bias.

        Serial (``parallel=False``, the primary regime): PI-lift each of
        the K slots, then Sigma-**collapse across the K slots** (sum-fold
        over the slot axis) into ONE predicted idea ``[B, D]``. This is
        the "Sigma collapses the lifted slots into the predicted idea".

        Parallel (``parallel=True``): apply PI->Sigma **per slot** (no
        cross-slot collapse), returning ``[B, N, D]`` with the same N as
        the input.

        The routing bias (if any) is added to the Sigma output in both
        regimes (broadcast over the slot axis when parallel).
        """
        if prior_slots.dim() != 3:
            raise ValueError(
                f"IntraSentenceLayer.forward expects [B, K, D]; got "
                f"shape {tuple(prior_slots.shape)}")

        if parallel:
            # Per-slot PI->Sigma; no cross-slot mixing. [B, N, D].
            y = self._pi_sigma(prior_slots)
            return self._apply_routing_bias(y, routing)

        # Serial: PI-lift every slot, then fold (sum) the lifted slots
        # over the slot axis BEFORE the Sigma collapse -> one idea.
        lifted = self.pi.forward(prior_slots)        # [B, K, W]
        folded = lifted.sum(dim=1)                    # [B, W]  (raw slot-axis sum)
        y = self.sigma.forward(folded)                # [B, D]
        return self._apply_routing_bias(y, routing)

    def reverse(self, predicted, routing=None, parallel=False, k=None):
        """Approximate inverse of ``forward`` (recon roundtrip helper).

        Both paths first subtract the routing bias (exactly, since the
        bias was an additive term), then invert the Sigma and PI bodies.

        Parallel / per-slot path (``parallel=True``): the sublayers are
        invertible and width-preserving, so
        ``reverse(forward(x, parallel=True), parallel=True) ~= x`` up to
        the LDU inverse tolerance (tight).

        Serial / collapse path (``parallel=False``): the Sigma collapse
        sums over the K lifted slots, which is **many-to-one**, so the
        inverse is necessarily approximate. We reconstruct by
        ``sigma.reverse`` to the working width, dividing the recovered
        fold equally across the ``k`` slots, then ``pi.reverse`` each
        slot back to concept width. ``k`` defaults to
        ``stm_capacity - 1`` (the STM[1:end] fan-out the predictor
        consumes). Returns ``[B, k, D]``.
        """
        if parallel:
            if predicted.dim() != 3:
                raise ValueError(
                    f"IntraSentenceLayer.reverse(parallel=True) expects "
                    f"[B, N, D]; got {tuple(predicted.shape)}")
            y = self._apply_routing_bias(predicted, routing, sign=-1)
            return self.pi.reverse(self.sigma.reverse(y))

        # Serial collapse inverse (approximate).
        if predicted.dim() != 2:
            raise ValueError(
                f"IntraSentenceLayer.reverse(parallel=False) expects "
                f"[B, D]; got {tuple(predicted.shape)}")
        if k is None:
            k = max(1, self.stm_capacity - 1)
        k = int(k)
        y = self._apply_routing_bias(predicted, routing, sign=-1)
        folded = self.sigma.reverse(y)               # [B, W]  (recovered slot-sum)
        per_slot = (folded / float(k)).unsqueeze(1)  # [B, 1, W]
        per_slot = per_slot.expand(-1, k, -1)        # [B, k, W]
        return self.pi.reverse(per_slot)             # [B, k, D]

    # -- loss helper (wired into training by a later task) -----------
    def intra_loss(self, pred, target):
        """L_intra = MSE(pred, target).

        ``pred`` / ``target``: matching shapes (``[B, D]`` serial or
        ``[B, N, D]`` parallel). Returns the scalar mean-squared error.
        The actual accumulation into the training backward path is wired
        by a later task (Task 3); this helper exists so that wiring is a
        one-liner and the loss math lives next to the layer that defines
        it.
        """
        return F.mse_loss(pred, target)

    def Reset(self, batch=None, hard=True):
        """Per-sentence / per-document state reset.

        ``IntraSentenceLayer`` holds no per-call recurrent buffers (the
        STM substrate it reads lives on ``ConceptualSpace``), so Reset
        is a structural no-op beyond honoring an optional batch resize
        hint. The PI / Sigma sublayers carry no per-call state to clear.
        Present so the Space's Start/Reset cascade reaches the layer
        without error (mirrors ``InterSentenceLayer.Reset``).
        """
        if batch is not None:
            self._batch = int(batch)


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
        # Task 4 (shared byte codebook): the PerceptStore that owns byte
        # identity across synthesis front ends. Wired by PartSpace
        # (``mirror_to_store``) in bpe/mphf modes; when set, every vocab
        # entry is mirrored into the store (percept_id == chunk_id) and
        # ``bytes_for`` resolves through it -- ``id_to_bytes`` becomes
        # the segmentation-side mirror, not the authority. Stashed
        # without module registration: the store is owned (and
        # registered) by the PartSpace, not this layer.
        object.__setattr__(self, "percept_store", None)
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
        # Task 4: an artifact load replaces the vocabulary wholesale --
        # re-mirror it into the shared byte store (no-op when unwired).
        self.mirror_to_store()
        return self

    # -- Shared byte store (Task 4) -------------------------------------

    def mirror_to_store(self, store=None):
        """Mirror the chunk vocabulary into the shared ``PerceptStore``.

        Task 4 (2026-06-09 build-batch plan): byte identity lives in ONE
        store across synthesis front ends -- segmentation (BPE merges /
        MPHF keys) is a strategy OVER the shared byte/percept codebook,
        not a private table. Entries are inserted in chunk-id order and
        ``RadixLayer.insert`` allocates sequential ids, so
        ``percept_id == chunk_id`` for every mirrored entry. That
        alignment is ASSERTED (fail loud): the reverse path
        (``bytes_for`` -> ``store.bytes_for``) relies on it, and a store
        that was grown by another writer cannot host this mirror.
        Idempotent -- re-mirroring an already-mirrored vocabulary is a
        no-op (``insert`` returns the existing id).

        Args:
            store: optional store to adopt as ``self.percept_store``
                before mirroring. ``None`` keeps the current wiring.

        Returns:
            The number of vocabulary entries mirrored (0 when unwired).
        """
        if store is not None:
            object.__setattr__(self, "percept_store", store)
        store = self.percept_store
        if store is None:
            return 0
        mirrored = 0
        for cid in sorted(self.id_to_bytes):
            bt = self.id_to_bytes[cid]
            pid = store.insert(bytes(bt))
            assert int(pid) == int(cid), (
                f"ChunkLayer.mirror_to_store: percept_id {pid} != "
                f"chunk_id {cid} for {bt!r} -- the shared store must be "
                f"dedicated to this vocabulary (insertion in chunk-id "
                f"order keeps the id spaces aligned).")
            mirrored += 1
        return mirrored

    def bytes_for(self, chunk_id):
        """Canonical bytes for ``chunk_id``, via the shared byte store.

        Task 4: when the ``PerceptStore`` is wired, it is the byte-
        identity authority and this resolves through
        ``store.bytes_for`` (the same reverse surface radix uses);
        ``id_to_bytes`` is the segmentation-side mirror and serves only
        as the unwired fallback. Returns ``None`` for unknown ids.
        """
        cid = int(chunk_id)
        store = self.percept_store
        if store is not None and 0 <= cid < len(store):
            return store.bytes_for(cid)
        bt = self.id_to_bytes.get(cid)
        return bytes(bt) if bt is not None else None

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
            # Task 4: promotions mirror into the shared byte store as
            # they happen (id alignment asserted; see mirror_to_store).
            if self.percept_store is not None:
                _pid = self.percept_store.insert(bytes(new_key))
                assert int(_pid) == int(new_id), (
                    f"ChunkLayer.train_step: percept_id {_pid} != "
                    f"chunk_id {new_id} for promoted merge {new_key!r}; "
                    f"the shared store and the vocab id spaces must stay "
                    f"aligned.")
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

#region RadixLayer (canonical PerceptStore)


# ---------------------------------------------------------------------------
# RadixTrie
# ---------------------------------------------------------------------------


class _RadixNode:
    """Single node in the :class:`RadixTrie`.

    Each node holds a (possibly empty) ``prefix`` byte-string and a dict
    of ``children`` keyed by the first byte of the child's prefix. A
    non-``None`` ``percept_id`` marks the node as terminal -- the
    walk from the root concatenating prefixes reaches the canonical
    byte sequence assigned that ID.

    Internal nodes (``percept_id is None``) carry shared prefixes; the
    longest-match walk records the deepest terminal it has seen so
    far and returns it when no further child matches.
    """

    __slots__ = ("prefix", "percept_id", "children")

    def __init__(self, prefix: bytes = b"",
                 percept_id: Optional[int] = None) -> None:
        self.prefix: bytes = prefix
        self.percept_id: Optional[int] = percept_id
        self.children: Dict[int, "_RadixNode"] = {}


class RadixTrie:
    """Radix trie over byte-strings with percept-ID payloads.

    Supports:

    * :meth:`insert` -- online insertion of ``(bytes -> percept_id)``;
      forks existing edges as needed.
    * :meth:`longest_match` -- O(L) longest-prefix match; returns the
      ``(percept_id, length)`` of the deepest terminal prefix that
      matches the input.
    * :meth:`get` -- exact lookup for an existing key; ``None`` if
      absent or only a partial match exists.
    * :meth:`serialize` / :meth:`load_state` -- flat ``(prefix,
      percept_id, [child_byte])`` triples for persistence.
    """

    def __init__(self) -> None:
        self.root: _RadixNode = _RadixNode(prefix=b"")
        self._size: int = 0

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: bytes) -> bool:
        return self.get(key) is not None

    def get(self, key: bytes) -> Optional[int]:
        """Exact lookup: return the percept ID for ``key`` or ``None``."""
        node = self.root
        remaining = key
        while remaining:
            first = remaining[0]
            child = node.children.get(first)
            if child is None:
                return None
            cp = child.prefix
            if not remaining.startswith(cp):
                return None
            remaining = remaining[len(cp):]
            node = child
        return node.percept_id

    def longest_match(self, key: bytes) -> Tuple[Optional[int], int]:
        """Return ``(percept_id, match_length)`` for the longest known
        prefix of ``key``.

        If no inserted key is a prefix of ``key``, returns
        ``(None, 0)`` so the caller can route the entire input to the
        byte-fallback path.
        """
        node = self.root
        consumed = 0
        best_id: Optional[int] = None
        best_len: int = 0
        if node.percept_id is not None:
            best_id = node.percept_id
            best_len = 0
        remaining = key
        while remaining:
            first = remaining[0]
            child = node.children.get(first)
            if child is None:
                break
            cp = child.prefix
            if not remaining.startswith(cp):
                break
            consumed += len(cp)
            remaining = remaining[len(cp):]
            node = child
            if node.percept_id is not None:
                best_id = node.percept_id
                best_len = consumed
        return best_id, best_len

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(self, key: bytes, percept_id: int) -> bool:
        """Insert ``key`` with the supplied ``percept_id``.

        Returns ``True`` if a new terminal was created and ``False`` if
        the key was already terminal (existing ID is preserved -- this
        method does not overwrite; callers should check :meth:`get`
        first if they want re-key semantics).
        """
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError(
                f"RadixTrie.insert: key must be bytes, got {type(key)!r}")
        key = bytes(key)
        if len(key) == 0:
            if self.root.percept_id is None:
                self.root.percept_id = int(percept_id)
                self._size += 1
                return True
            return False
        node = self.root
        remaining = key
        while True:
            first = remaining[0]
            child = node.children.get(first)
            if child is None:
                # Pure tail-append: new leaf carrying the remainder.
                new_leaf = _RadixNode(prefix=remaining,
                                      percept_id=int(percept_id))
                node.children[first] = new_leaf
                self._size += 1
                return True
            cp = child.prefix
            # Find shared prefix length with the child's edge.
            common = 0
            limit = min(len(cp), len(remaining))
            while common < limit and cp[common] == remaining[common]:
                common += 1
            if common == len(cp):
                # The whole child edge matches; descend.
                remaining = remaining[common:]
                node = child
                if not remaining:
                    if node.percept_id is None:
                        node.percept_id = int(percept_id)
                        self._size += 1
                        return True
                    return False
                continue
            # Partial overlap -> fork the child edge at `common`.
            old_child = child
            split = _RadixNode(prefix=cp[:common])
            # Re-key the old child by its first remaining byte.
            old_child.prefix = cp[common:]
            split.children[old_child.prefix[0]] = old_child
            node.children[first] = split
            if common == len(remaining):
                # New key terminates at the split point.
                split.percept_id = int(percept_id)
                self._size += 1
                return True
            # Branch sibling for the new tail.
            tail = remaining[common:]
            new_leaf = _RadixNode(prefix=tail, percept_id=int(percept_id))
            split.children[tail[0]] = new_leaf
            self._size += 1
            return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def serialize(self) -> List[Tuple[bytes, Optional[int], List[int]]]:
        """Pre-order flat dump: ``(prefix, percept_id, child_keys)`` per
        node. ``child_keys`` is the ordered list of first-byte keys for
        the node's children (so :meth:`load_state` can reconstruct the
        edges by re-walking the same list).
        """
        out: List[Tuple[bytes, Optional[int], List[int]]] = []
        stack: List[_RadixNode] = [self.root]
        while stack:
            node = stack.pop()
            keys = sorted(node.children.keys())
            out.append((node.prefix, node.percept_id, list(keys)))
            # Push in reverse so traversal pops in sorted order.
            for k in reversed(keys):
                stack.append(node.children[k])
        return out

    def load_state(
        self, dump: List[Tuple[bytes, Optional[int], List[int]]]
    ) -> None:
        """Rebuild the trie from a :meth:`serialize` dump."""
        if not dump:
            self.root = _RadixNode(prefix=b"")
            self._size = 0
            return
        it = iter(dump)
        prefix, percept_id, keys = next(it)
        root = _RadixNode(prefix=prefix, percept_id=percept_id)
        size = 1 if percept_id is not None else 0
        stack: List[Tuple[_RadixNode, List[int]]] = [(root, list(keys))]
        # Build children depth-first so the iterator matches the
        # pre-order serialisation.
        for node_prefix, node_id, node_keys in it:
            parent, pending = stack[-1]
            while not pending:
                stack.pop()
                if not stack:
                    raise ValueError(
                        "RadixTrie.load_state: dangling node after the "
                        "tree was fully consumed")
                parent, pending = stack[-1]
            first_key = pending.pop(0)
            child = _RadixNode(prefix=node_prefix, percept_id=node_id)
            parent.children[first_key] = child
            if node_id is not None:
                size += 1
            stack.append((child, list(node_keys)))
        self.root = root
        self._size = size


# ---------------------------------------------------------------------------
# BytesFallbackEncoder
# ---------------------------------------------------------------------------


class BytesFallbackEncoder(nn.Module):
    """Per-byte vector codebook + hit-count bookkeeping for unknown chunks.

    The encoder owns a ``[256, D]`` ``nn.Parameter`` and computes the
    fallback vector for an unknown chunk as

        v(chunk) = sum_{b in chunk} byte_codebook[b] / sqrt(len(chunk))

    The division by ``sqrt(L)`` keeps the encoded norm O(1) instead of
    growing linearly with the chunk length, matching the
    expectation of downstream layers operating on unit-scale vectors.

    Hit counts are tracked in a plain Python dict so promotion logic can
    poll them cheaply; they are persisted via the parent
    RadixLayer's ``vocab_extras`` blob.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(
                f"BytesFallbackEncoder: dim must be positive, got {dim}")
        self.dim: int = int(dim)
        # Small-stddev init so the fallback contribution starts near
        # zero and the codebook can learn whatever scale it wants.
        self.byte_codebook = nn.Parameter(
            torch.randn(256, self.dim) * 0.02)
        # Pure-Python hit counters; persisted via vocab_extras.
        self.hit_counts: Dict[bytes, int] = {}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, chunk: bytes) -> torch.Tensor:
        """Return the fallback vector for ``chunk`` and bump its hit
        counter.

        ``chunk`` must be a non-empty ``bytes`` object.
        """
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError(
                f"BytesFallbackEncoder.encode: chunk must be bytes, got "
                f"{type(chunk)!r}")
        if len(chunk) == 0:
            raise ValueError(
                "BytesFallbackEncoder.encode: chunk must be non-empty.")
        chunk_b = bytes(chunk)
        self.hit_counts[chunk_b] = self.hit_counts.get(chunk_b, 0) + 1
        idx = torch.tensor(list(chunk_b),
                           dtype=torch.long,
                           device=self.byte_codebook.device)
        rows = self.byte_codebook.index_select(0, idx)
        return rows.sum(dim=0) / math.sqrt(len(chunk_b))

    def hits(self, chunk: bytes) -> int:
        """Read the hit counter for ``chunk`` (0 if never seen)."""
        return int(self.hit_counts.get(bytes(chunk), 0))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def extras_dump(self) -> Dict[bytes, int]:
        """Serialise the hit-count dict for storage in ``vocab_extras``."""
        return dict(self.hit_counts)

    def extras_load(self, hit_counts: Dict[bytes, int]) -> None:
        """Restore hit counters from a :meth:`extras_dump` payload."""
        self.hit_counts = {bytes(k): int(v) for k, v in hit_counts.items()}


# ---------------------------------------------------------------------------
# RadixLayer (canonical class; formerly PerceptStore)
# ---------------------------------------------------------------------------


class RadixLayer(Layer):
    """Authoritative store for perceptual identity + invertibility.

    Formerly ``PerceptStore`` in ``bin/PerceptStore.py``; promoted to a
    proper ``Layer`` subclass so the layer-API ``forward()`` /
    ``reverse()`` symmetry holds and the standard Layer cascades
    (Start/End, paramUpdate, sigma) apply uniformly.

    See module docstring for the broader architectural context; see the
    class spec in
    ``doc/plans/2026-05-27-perceptstore-meta-taxonomy-reentrancy.md``
    Sec. Design - PerceptStore for the verbatim contract.

    Components:

    * ``radix_trie`` -- canonical byte sequences with longest-match
      lookup. Authoritative.
    * ``hash_map: dict[bytes -> int]`` -- fast cache mirroring the
      trie's terminal nodes; rebuilt on load.
    * ``inverse_table: list[bytes]`` -- index by percept ID; maps ID
      back to canonical bytes. Structural invertibility.
    * ``codebook`` -- ``nn.Parameter[V, D]`` learned vector payload per
      percept. ``V`` grows by doubling as inserts overflow.
    * ``byte_fallback`` -- :class:`BytesFallbackEncoder` for unknown
      chunks; tracks promotion-candidate hit counts.
    * ``promotion_threshold`` / ``promotion_min_length`` -- when a chunk's
      hit count reaches the threshold AND its length is at least the
      minimum, it is promoted into the trie + hash_map + inverse_table
      + codebook.

    Forward path (chunk bytes -> vector):

      1. ``hash_map.get(chunk)`` -> existing percept ID (fast cache hit).
      2. Else ``radix_trie.longest_match(chunk)`` -> permanent percept
         ID + residual bytes. The residual falls through to
         :meth:`byte_fallback.encode`.
      3. ``byte_fallback.encode(residual)`` -> temporary fallback
         vector; increments hit count.
      4. If hit count >= ``promotion_threshold`` AND
         ``len(residual) >= promotion_min_length`` and the residual is
         the whole chunk (so we're learning the full chunk, not just
         the suffix), promote: insert into trie + hash_map +
         inverse_table + codebook (init from the byte-fallback
         encoding).

    Reverse path (percept ID -> canonical bytes):

      1. ``inverse_table[percept_id]`` -> bytes. Exact, no learning.
    """

    # Cap at construction time; grow by doubling when an insert would
    # otherwise overflow.
    DEFAULT_INITIAL_CAP: int = 64

    def __init__(
        self,
        dim: int,
        *,
        initial_cap: Optional[int] = None,
        promotion_threshold: int = 4,
        promotion_min_length: int = 2,
        basis=None,
    ) -> None:
        super().__init__(nInput=int(dim), nOutput=int(dim))
        if dim <= 0:
            raise ValueError(
                f"RadixLayer: dim must be positive, got {dim}")
        self.dim: int = int(dim)
        self.promotion_threshold: int = int(promotion_threshold)
        self.promotion_min_length: int = int(promotion_min_length)
        cap = int(initial_cap) if initial_cap is not None \
            else self.DEFAULT_INITIAL_CAP
        if cap <= 0:
            raise ValueError(
                f"RadixLayer: initial_cap must be positive, got {cap}")
        self._capacity: int = cap
        self._size: int = 0
        # 2026-06-04: vector storage is a Codebook *Basis*, not a raw
        # nn.Parameter. When ``basis`` is supplied (PartSpace passes
        # ``subspace.what`` in radix mode) the percept vectors live on that
        # shared Basis -- ONE percept codebook, on ``.what``, so the SubSpace
        # can store ``.active`` selections and materialize lazily (no vector
        # copy). The shared basis is stashed WITHOUT module registration
        # (object.__setattr__) so it is not double-counted in this layer's
        # parameters / state_dict (``.what`` owns it). When ``basis`` is None
        # (standalone / unit tests) we own a private Codebook. The
        # ``codebook`` property exposes the ``[V, D]`` prototype tensor so the
        # existing read / seed call sites are unchanged.
        if basis is not None:
            object.__setattr__(self, "_basis", basis)
            _w = basis.getW()
            if _w is not None:
                self._capacity = int(_w.shape[0])
        else:
            from Spaces import Codebook
            _own = Codebook()
            _own.create(1, self._capacity, self.dim, customVQ=False)
            self._basis = _own
        # Authoritative trie + cache + inverse table.
        self.radix_trie: RadixTrie = RadixTrie()
        self.hash_map: Dict[bytes, int] = {}
        self.inverse_table: List[bytes] = []
        # Byte-fallback encoder. Owns its own [256, D] Parameter.
        self.byte_fallback: BytesFallbackEncoder = BytesFallbackEncoder(
            self.dim)
        # Frequency-driven concatenation (2026-06-04): per-chunk sighting
        # counts. A spelled-out chunk is promoted to a single percept once
        # it recurs ``promotion_threshold`` times (see ``observe_chunk``).
        self._chunk_hits: Dict[bytes, int] = {}

    @property
    def codebook(self):
        """The ``[V, D]`` percept prototype tensor (an ``nn.Parameter`` when
        owned), backed by the Codebook Basis. Read / seed call sites use
        ``self.codebook`` / ``self.codebook.data`` transparently; growth
        goes through ``_grow_to`` -> ``Basis.grow_to``."""
        return self._basis.getW()

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def __contains__(self, key: bytes) -> bool:
        return bytes(key) in self.hash_map

    def get_id(self, chunk: bytes) -> Optional[int]:
        """Hash-map lookup. Returns ``None`` if ``chunk`` is unknown."""
        return self.hash_map.get(bytes(chunk))

    def bytes_for(self, percept_id: int) -> bytes:
        """Reverse path: percept ID -> canonical bytes. Raises
        :class:`IndexError` for invalid IDs."""
        pid = int(percept_id)
        if pid < 0 or pid >= self._size:
            raise IndexError(
                f"RadixLayer.bytes_for: percept_id {pid} out of range "
                f"[0, {self._size})")
        return self.inverse_table[pid]

    def vector_for(self, percept_id: int) -> torch.Tensor:
        """Codebook row for ``percept_id`` (size [D])."""
        pid = int(percept_id)
        if pid < 0 or pid >= self._size:
            raise IndexError(
                f"RadixLayer.vector_for: percept_id {pid} out of range "
                f"[0, {self._size})")
        return self.codebook[pid]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(
        self,
        chunk: bytes,
        *,
        init_vector: Optional[torch.Tensor] = None,
    ) -> int:
        """Insert ``chunk`` (canonical bytes) and return its percept ID.

        If ``chunk`` is already known, returns the existing ID without
        modifying state. Otherwise grows the codebook if needed and
        seeds the new row with ``init_vector`` if supplied (else with
        a small-stddev random vector).
        """
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError(
                f"RadixLayer.insert: chunk must be bytes, got "
                f"{type(chunk)!r}")
        if len(chunk) == 0:
            raise ValueError(
                "RadixLayer.insert: chunk must be non-empty.")
        chunk_b = bytes(chunk)
        existing = self.hash_map.get(chunk_b)
        if existing is not None:
            return int(existing)
        # Allocate the next percept ID.
        new_id = self._size
        # Grow the codebook if the new ID would overflow.
        if new_id >= self._capacity:
            self._grow_to(max(new_id + 1, self._capacity * 2))
        # Seed the new row.
        with torch.no_grad():
            if init_vector is None:
                self.codebook.data[new_id, :].normal_(
                    mean=0.0, std=0.02)
            else:
                if not torch.is_tensor(init_vector):
                    raise TypeError(
                        "RadixLayer.insert: init_vector must be a "
                        f"Tensor, got {type(init_vector)!r}")
                if init_vector.shape != (self.dim,):
                    raise ValueError(
                        f"RadixLayer.insert: init_vector shape "
                        f"{tuple(init_vector.shape)} != ({self.dim},)")
                self.codebook.data[new_id, :].copy_(
                    init_vector.detach().to(
                        self.codebook.device, self.codebook.dtype))
        # Update the auxiliary structures.
        self.radix_trie.insert(chunk_b, new_id)
        self.hash_map[chunk_b] = new_id
        self.inverse_table.append(chunk_b)
        self._size += 1
        return new_id

    # ------------------------------------------------------------------
    # Lookup path
    # ------------------------------------------------------------------

    def lookup(self, chunk: bytes) -> torch.Tensor:
        """Forward path: canonical bytes -> a vector of size [D].

        Walks the spec'd path:

          1. hash-map hit -> permanent codebook row.
          2. radix longest-match -> known prefix + residual fallback.
          3. byte fallback over residual; bumps hit counter; may promote.

        See :meth:`lookup_with_id` for the same path plus the resolved
        percept ID (``None`` for unpromoted slots).
        """
        vec, _ = self.lookup_with_id(chunk)
        return vec

    def lookup_with_id(
        self, chunk: bytes
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Forward path with percept-ID resolution.

        Returns ``(vector, percept_id_or_None)``:
          * Hash-map hit -> ``(codebook[pid], pid)``.
          * Whole input matched a known prefix exactly -> ``(vec, pid)``.
          * Promotion fired during the call -> ``(vec, new_pid)``.
          * Unpromoted byte-fallback / partial-match -> ``(vec, None)``.

        Sibling of :meth:`lookup`; both share this single implementation
        so the spec'd Forward path is one routine.
        """
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError(
                f"RadixLayer.lookup_with_id: chunk must be bytes, got "
                f"{type(chunk)!r}")
        if len(chunk) == 0:
            raise ValueError(
                "RadixLayer.lookup_with_id: chunk must be non-empty.")
        chunk_b = bytes(chunk)
        # Step 1: hash-map fast path.
        cached = self.hash_map.get(chunk_b)
        if cached is not None:
            return self.codebook[cached], int(cached)
        # Step 2: radix longest-match on the input.
        match_id, match_len = self.radix_trie.longest_match(chunk_b)
        prefix_vec: Optional[torch.Tensor] = None
        if match_id is not None and match_len > 0:
            prefix_vec = self.codebook[match_id]
        residual = chunk_b[match_len:]
        if not residual:
            # Whole chunk matched (rare here since the hash-map miss
            # came first; defensive guard).
            vec = prefix_vec if prefix_vec is not None \
                else self.codebook[match_id]
            return vec, (int(match_id) if match_id is not None else None)
        # Step 3: byte-fallback encode the residual.
        fallback_vec = self.byte_fallback.encode(residual)
        if prefix_vec is None:
            combined = fallback_vec
        else:
            combined = prefix_vec + fallback_vec
        # Step 4: promotion check + hit-counter bookkeeping.
        # Three regimes:
        #   (a) full miss (match_len == 0): encode() already bumped
        #       hit_counts[chunk_b] since residual == chunk_b.
        #   (b) partial miss, first sight: hit_counts[chunk_b] is 0
        #       because encode() bumped only the residual; seed it to 1.
        #   (c) partial miss, repeat sight: hit_counts[chunk_b] was
        #       seeded on a prior call but encode() bumped only the
        #       residual this time, so bump the full-chunk counter too.
        full_hits = self.byte_fallback.hits(chunk_b)
        if (full_hits == 0 and len(residual) < len(chunk_b)):
            self.byte_fallback.hit_counts[chunk_b] = 1
            full_hits = 1
        elif len(residual) < len(chunk_b):
            self.byte_fallback.hit_counts[chunk_b] = full_hits + 1
            full_hits = full_hits + 1
        if (full_hits >= self.promotion_threshold
                and len(chunk_b) >= self.promotion_min_length):
            new_pid = self.insert(chunk_b, init_vector=combined.detach())
            return combined, int(new_pid)
        return combined, None

    # ------------------------------------------------------------------
    # Spell-out emission (2026-06-04)
    # ------------------------------------------------------------------

    def spell_out(self, chunk: bytes) -> List[int]:
        """Emit ``chunk`` as a sequence of percept IDs -- the model's
        "send the next-largest percept, up to the size of the word"
        contract.

        At each position take the LONGEST known prefix-percept; when no
        multi-byte prefix is known, fall to a SINGLE-BYTE percept (seeded
        on demand so every byte is always a valid one-row percept). So an
        unfamiliar word is "spelled out" as a run of byte/prefix percepts,
        and a word the trie has already concatenated (via promotion) comes
        back as one percept. EVERY returned id indexes a single
        ``codebook`` (``.what``) row -- which is what lets the SubSpace
        store a ``.active`` selection and materialize without copying.

        Returns the list of percept-store row ids (length >= 1 for a
        non-empty chunk).
        """
        chunk_b = bytes(chunk)
        if len(chunk_b) == 0:
            return []
        pids: List[int] = []
        i = 0
        n = len(chunk_b)
        while i < n:
            remaining = chunk_b[i:]
            match_id, match_len = self.radix_trie.longest_match(remaining)
            if match_id is not None and match_len > 0:
                pids.append(int(match_id))
                i += int(match_len)
                continue
            # No known prefix at this position: emit a single byte-percept,
            # seeding it on first sight so the spell-out always bottoms out.
            one = chunk_b[i:i + 1]
            pid = self.hash_map.get(one)
            if pid is None:
                pid = self.insert(one)
            pids.append(int(pid))
            i += 1
        return pids

    def observe_chunk(self, chunk: bytes) -> Optional[int]:
        """Frequency-driven concatenation: count a full chunk's sightings
        and promote it to a SINGLE percept once it recurs
        ``promotion_threshold`` times (and is at least
        ``promotion_min_length`` bytes). The promoted row is seeded from
        the mean of the chunk's current spell-out percepts so the new
        concatenated percept starts meaningful (a continuation of what was
        being spelled out). Returns the new percept id on promotion, else
        ``None`` (already a percept, too short, or below threshold).
        """
        chunk_b = bytes(chunk)
        if len(chunk_b) < self.promotion_min_length:
            return None
        if chunk_b in self.hash_map:
            return None
        self._chunk_hits[chunk_b] = self._chunk_hits.get(chunk_b, 0) + 1
        if self._chunk_hits[chunk_b] < self.promotion_threshold:
            return None
        pids = self.spell_out(chunk_b)
        with torch.no_grad():
            init = (self.codebook[pids].mean(dim=0).detach()
                    if pids else None)
        new_id = self.insert(chunk_b, init_vector=init)
        self._chunk_hits.pop(chunk_b, None)
        return int(new_id)

    # ------------------------------------------------------------------
    # Layer-API forward / reverse
    # ------------------------------------------------------------------

    def forward(self, chunks):
        """Layer-API forward: alias of :meth:`lookup_with_id`.

        Accepts a single ``bytes`` / ``bytearray`` chunk and returns
        the ``(vector, percept_id_or_None)`` tuple. Batched / tensor
        inputs are NOT supported here -- the PartSpace's
        ``_embed_radix`` loop calls :meth:`lookup_with_id` per slot for
        clarity; this overload just exposes the same routine under the
        canonical Layer name so the symmetric ``forward()`` /
        ``reverse()`` contract holds.
        """
        return self.lookup_with_id(chunks)

    def reverse(self, vec, *, symbolic_space=None):
        """Layer-API reverse: vec -> canonical bytes (or list-of-bytes).

        Stage 8 structural decode formerly resident on
        :meth:`BasicModel._reverse_decode_one`; relocated here so
        ``RadixLayer.forward`` / ``RadixLayer.reverse`` are symmetric
        Layer-API operations and the cross-space reach lives on the
        layer that owns the inverse table.

        Accepts:
          * ``vec`` of shape ``[D]`` -> returns ``bytes``.
          * ``vec`` of shape ``[N, D]`` -> returns ``List[bytes]``.
          * ``vec`` of shape ``[B, N, D]`` -> returns ``List[List[bytes]]``.

        When ``symbolic_space`` is supplied, the walk goes
        ``vec -> WS.codebook nearest -> META taxonomy children
        -> positive PS percept id -> bytes_for(pid)``. When it is
        ``None`` the fallback is nearest-PS-codebook directly (no META
        walk); useful for standalone tests without a full WholeSpace
        wired.

        Numerical-divergence policy:
          * NaN/Inf entries raise ``RuntimeError`` (fail loud); silently
            masking them via ``nan_to_num`` would hide upstream bugs.
          * Width mismatch / empty codebook / near-zero norm slots
            return ``b""``; those are legitimate "no symbol here"
            conditions, not divergences.
        """
        if not torch.is_tensor(vec):
            return b""
        if vec.dim() == 3:
            return [
                [self.reverse(vec[b, n], symbolic_space=symbolic_space)
                 for n in range(vec.shape[1])]
                for b in range(vec.shape[0])
            ]
        if vec.dim() == 2:
            return [
                self.reverse(vec[n], symbolic_space=symbolic_space)
                for n in range(vec.shape[0])
            ]
        # 1-D path: actual structural decode.
        # Fail loud on NaN/Inf: with NaN, all (diffs*diffs) comparisons
        # are False and argmin silently returns row 0, masking a real
        # numerical divergence upstream.
        finite = torch.isfinite(vec.detach().cpu())
        if not bool(finite.all()):
            raise RuntimeError(
                "RadixLayer.reverse: input vector contains NaN/Inf. "
                "Numerical divergence must surface, not be silently "
                "masked. "
                f"vec[finite]={int(finite.sum().item())}/"
                f"{int(vec.numel())}.")
        # Pick the comparison codebook. WS-driven walk if a WS peer is
        # supplied (preferred); else nearest-PS-codebook fallback.
        if symbolic_space is not None:
            cb = getattr(symbolic_space.subspace, "what", None)
            W = cb.getW() if cb is not None else None
            if W is None:
                # WS codebook absent (e.g. <codebook>none</codebook>): there
                # is no WS taxonomy to walk, so fall back to the standalone
                # PS-table decode (nearest ACTIVE percept -> inverse_table)
                # rather than emitting an empty slot for every word.
                symbolic_space = None
                W = self.codebook
        else:
            W = self.codebook
        if W is None:
            return b""
        # Empty codebook is a clean no-symbol situation, not a
        # divergence -- bail before argmin sees a [0, D] tensor.
        if W.shape[0] == 0:
            return b""
        # Bound the nearest-row search to ACTIVE percept rows [0, _size).
        # The codebook is preallocated to capacity (nVectors); the unused
        # reserve rows are untrained noise that must not win the argmin (and
        # for the PS-table decode an out-of-active match has no inverse_table
        # entry -> IndexError -> empty slot). Restricting to active rows keeps
        # the decode on real percepts.
        if symbolic_space is None and 0 < self._size < W.shape[0]:
            W = W[:self._size]
        # Move both to the same device + dtype; collapse to [D].
        target = vec.detach().to(W.device, W.dtype).reshape(-1)
        if target.shape[0] != W.shape[1]:
            # CS->PS demux: the recon vector is the muxed [what|where|when]
            # event, wider than the content-width codebook row -- take the
            # leading .what slice (mirrors insert_symbol's demux). A narrower
            # vec is a genuine mismatch; bail.
            if target.shape[0] > W.shape[1]:
                target = target[:W.shape[1]]
            else:
                return b""
        # Null-slot guard (2026-05-28). Padding / inter-word-space slots
        # produce near-zero recon vectors. Matching them against any
        # codebook via argmin would land on whichever row is closest to
        # the origin and surface a spurious word. Return b"" instead so
        # the slot renders as empty in the join. Threshold is
        # conservative: well below the typical recon norm (~1-2 in
        # MM_xor) but above numerical noise.
        if float(target.norm()) < 1e-3:
            return b""
        # Nearest-row search.
        diffs = W - target.unsqueeze(0)
        sq = (diffs * diffs).sum(dim=1)
        nearest_row = int(torch.argmin(sq).item())
        # Standalone fallback: no WS, decode directly via PS table.
        if symbolic_space is None:
            try:
                return self.bytes_for(nearest_row)
            except IndexError:
                return b""
        # WS-walk: nearest may be the META itself, or its WS child.
        # Resolve the row's position via WholeSpace's lookup tables;
        # an unbound row (not in _ws_row_to_pos) has no taxonomy entry
        # and no surface-bytes path -- return b"" rather than
        # mis-indexing the PS table with an WS row id.
        ws = symbolic_space
        nearest_pos = ws._ws_row_to_pos.get(nearest_row)
        if nearest_pos is None:
            return b""
        # Walk to a META node: either the nearest row IS a META, or
        # it's the WS-child of one (auto-bound under word learning
        # initializes both the META row and its WS child to the same
        # seed vector; the META row drifts during training while the
        # child stays close to seed, so the nearest match often lands
        # on the child, not the META). Climb one level via
        # ``taxonomy_parent`` to recover the META in that case.
        meta_pos = None
        children = ws.taxonomy_children(nearest_pos)
        if children:
            meta_pos = nearest_pos
        else:
            parent = ws.taxonomy_parent(nearest_pos)
            if parent is not None and ws.is_meta(int(parent)):
                meta_pos = int(parent)
                children = ws.taxonomy_children(meta_pos)
        if meta_pos is None or not children:
            return b""
        for child in children:
            ci = int(child)
            if ws._pos_kind.get(ci) == "ps":
                ps_row = ws._ps_pos_to_row.get(ci)
                if ps_row is None:
                    continue
                try:
                    return self.bytes_for(int(ps_row))
                except IndexError:
                    return b""
        return b""

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------

    def _grow_to(self, new_cap: int) -> None:
        """Grow the codebook to ``new_cap`` rows, preserving existing rows.

        Mirrors the ``Codebook.grow_to`` pattern: rebuild the
        ``nn.Parameter`` with the new capacity, copy the old rows,
        small-stddev init the new ones. Existing optimizer state is
        not preserved -- callers that hold an optimizer over
        :attr:`codebook` must rebuild it (PartSpace does this via
        the standard ``rebuild_optimizer`` plumbing on insert).
        """
        new_cap = int(new_cap)
        if new_cap <= self._capacity:
            return
        # Delegate growth to the Codebook Basis: it preserves the existing
        # rows and zero-inits the new ones (``insert`` seeds the new row
        # immediately after). Replaces the old raw-Parameter rebuild.
        self._basis.grow_to(new_cap)
        self._capacity = new_cap

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def vocab_extras(self) -> Dict[str, object]:
        """Pure-Python dump of the non-tensor state for ``vocab_extras``.

        The trie, hash map, inverse table, byte-fallback hit counts,
        and the promotion knobs all live in pure-Python structures; the
        codebook + byte-fallback's ``[256, D]`` Parameters travel via
        the standard ``state_dict`` mechanism.
        """
        return {
            "dim": self.dim,
            "capacity": self._capacity,
            "size": self._size,
            "promotion_threshold": self.promotion_threshold,
            "promotion_min_length": self.promotion_min_length,
            "radix_trie": self.radix_trie.serialize(),
            "hash_map": dict(self.hash_map),
            "inverse_table": list(self.inverse_table),
            "byte_fallback_hits": self.byte_fallback.extras_dump(),
        }

    def load_vocab_extras(self, extras: Dict[str, object]) -> None:
        """Restore the pure-Python state from a :meth:`vocab_extras` blob.

        Note: ``codebook`` Parameter shape must already match the
        ``capacity`` recorded in ``extras`` (the caller should have
        loaded the state_dict first, or be calling this on a freshly
        constructed RadixLayer with the matching ``initial_cap``).
        """
        cap = int(extras["capacity"])
        size = int(extras["size"])
        if cap != self._capacity:
            # Re-allocate the codebook to the saved capacity, copying
            # what we already have (typically the freshly-init'd state).
            self._grow_to(cap)
        self._size = size
        self.promotion_threshold = int(extras["promotion_threshold"])
        self.promotion_min_length = int(extras["promotion_min_length"])
        self.radix_trie = RadixTrie()
        # The serialised entries are lists of ints (JSON-friendly);
        # rebuild them as proper tuples of (bytes, Optional[int], list[int]).
        dump_in = extras["radix_trie"]
        dump_norm: List[Tuple[bytes, Optional[int], List[int]]] = []
        for prefix, percept_id, keys in dump_in:
            dump_norm.append((bytes(prefix),
                              None if percept_id is None
                              else int(percept_id),
                              [int(k) for k in keys]))
        self.radix_trie.load_state(dump_norm)
        self.hash_map = {bytes(k): int(v)
                         for k, v in extras["hash_map"].items()}
        self.inverse_table = [bytes(b) for b in extras["inverse_table"]]
        self.byte_fallback.extras_load(extras["byte_fallback_hits"])


#endregion

#region Tokenizer GPU Layers

class BPEGpuLayer(Layer):
    """GPU BPE tokenizer for the FROZEN-vocab training path.

    The legacy ``ChunkLayer`` greedy longest-match is a Python trie walk
    that needs ``byte_indices.tolist()`` -- a per-step cudaMemcpyDtoH. When
    the BPE vocab is frozen (``word_learning <= 0`` -- the CPU-pretrain ->
    freeze -> GPU-train workflow), the whole tokenizer becomes static
    tensor ops with zero host sync. ``PartSpace`` owns an instance
    of this layer and caches the per-(frozen) vocab static tables on
    itself (``self._bpe_static_tables``); this layer holds the algorithm
    only.

    Bit-identical to the trie walk is asserted by ``test/bpe_gpu_equiv.py``
    before the switch -- a one-token silent divergence corrupts all
    training (fail-loud memory).
    """

    class _BPEGpuUnavailable(Exception):
        """Raised when the static GPU tables cannot be built (e.g. a frozen
        codebook/vocab key mismatch). The caller falls back to the verified
        trie reference -- never a silent wrong result."""

    # Polynomial rolling hash over bytes. 1099511628211 is the FNV-style
    # 64-bit prime; arithmetic wraps mod 2**64 via int64 overflow, which is
    # fine -- the hash only needs to be a consistent bucket; correctness
    # comes from the explicit byte-verify, not from the hash being perfect.
    _HASH_MUL = 1099511628211

    def __init__(self):
        super().__init__(nInput=0, nOutput=0)

    @staticmethod
    def _poly_hash(windows):
        """``windows`` [..., L] int64 byte values -> [...] int64 hash."""
        h = torch.zeros(windows.shape[:-1], dtype=torch.int64,
                        device=windows.device)
        L = windows.shape[-1]
        for k in range(L):
            h = h * BPEGpuLayer._HASH_MUL + (windows[..., k] + 1)
        return h

    @staticmethod
    def build_static_tables(chunk_layer, codebook, device):
        """Frozen vocab -> static device tensors. Call ONCE per (frozen)
        chunk_layer/codebook on the target device; cache the result.

        Returns a dict the GPU tokenizer + the rewired _embed_bpe consume.
        All Python-dict / latin1 work happens HERE (one-time, host), never
        per batch.
        """
        vocab = chunk_layer.vocab               # {byte-tuple: id}
        id_to_bytes = chunk_layer.id_to_bytes   # {id: byte-tuple}
        maxL = int(chunk_layer._max_merge_len)
        V = int(chunk_layer._next_id)           # ids are 0..V-1
        boundary = set(int(b) for b in chunk_layer.BOUNDARY_BYTES)

        tok_bytes = torch.full((V, maxL), -1, dtype=torch.int64)
        tok_len = torch.zeros(V, dtype=torch.int64)
        is_boundary = torch.zeros(V, dtype=torch.bool)
        for i in range(V):
            key = id_to_bytes.get(i)
            if key is None:
                # 0..255 are always seeded; a gap above that would be a
                # vocab bug -- single-byte fallback id == byte value.
                key = (i,) if i < 256 else ()
            kl = len(key)
            tok_len[i] = kl
            if kl:
                tok_bytes[i, :kl] = torch.tensor(
                    [int(b) for b in key], dtype=torch.int64)
                is_boundary[i] = all(int(b) in boundary for b in key)

        # chunk_id -> codebook row (frozen resolve chain). -1 == "skip"
        # (boundary chunk, or byte_mode key not in the codebook -- the
        # original returns None and the sub-token is dropped).
        byte_mode = bool(getattr(codebook, 'byte_mode', False))
        key_to_index = codebook.pretrain.key_to_index
        chunk_to_cb = torch.full((V,), -1, dtype=torch.int64)
        for i in range(V):
            if bool(is_boundary[i]):
                continue
            key = id_to_bytes.get(i, (i,) if i < 256 else ())
            if not key:
                continue
            latin1 = "".join(chr(int(b) & 0xFF) for b in key)
            idx = key_to_index.get(latin1)
            if idx is None:
                if byte_mode:
                    continue          # original: _resolve -> None -> skip
                raise AssertionError(
                    f"build_static_tables: key {latin1!r} missing from "
                    f"frozen codebook.pretrain (word_learning<=0) -- .kv "
                    f"load mismatch.")
            chunk_to_cb[i] = int(idx)

        # Per-length sorted (hash -> id) for longest-match searchsorted.
        by_len = {}
        for L in range(1, maxL + 1):
            ids = [i for i in range(V) if int(tok_len[i]) == L]
            if not ids:
                by_len[L] = None
                continue
            ids_t = torch.tensor(ids, dtype=torch.int64)
            windows = tok_bytes[ids_t, :L]                 # [K, L]
            hashes = BPEGpuLayer._poly_hash(windows)       # [K]
            order = torch.argsort(hashes)
            by_len[L] = (hashes[order].to(device),
                         ids_t[order].to(device))

        return {
            "maxL": maxL, "V": V,
            "tok_bytes": tok_bytes.to(device),
            "tok_len": tok_len.to(device),
            "is_boundary": is_boundary.to(device),
            "chunk_to_cb": chunk_to_cb.to(device),
            "by_len": by_len,
        }

    @staticmethod
    def gpu_longest_match(byte_buf, tables):
        """``byte_buf`` [B, N] long (0..255; 0 terminates a row).

        Returns ``best_id`` [B, N] long and ``best_len`` [B, N] long: the
        id / length of the longest vocab entry starting at each position
        (single-byte fallback guarantees ``best_len >= 1``). Pure tensor
        ops, no host sync.
        """
        B, N = byte_buf.shape
        dev = byte_buf.device
        maxL = tables["maxL"]
        tok_bytes = tables["tok_bytes"]
        # Single-byte baseline: id == byte value (ids 0..255 seeded), len 1.
        best_id = byte_buf.clone()
        best_len = torch.ones(B, N, dtype=torch.int64, device=dev)

        for L in range(2, maxL + 1):
            entry = tables["by_len"].get(L)
            if entry is None or N < L:
                continue
            keys_sorted, ids_sorted = entry
            # [B, N-L+1, L] windows; pad-free positions only.
            win = byte_buf.unfold(1, L, 1).to(torch.int64)     # [B, M, L]
            h = BPEGpuLayer._poly_hash(win)                    # [B, M]
            pos = torch.searchsorted(keys_sorted, h)
            pos = pos.clamp(max=keys_sorted.numel() - 1)
            hit = keys_sorted[pos] == h
            cand_id = ids_sorted[pos]                          # [B, M]
            # Byte-verify (collision-proof): window == tok_bytes[cand_id].
            cb = tok_bytes[cand_id][..., :L]                   # [B, M, L]
            verified = hit & (win == cb).all(dim=-1)           # [B, M]
            # A match of length L at position i beats any shorter one.
            M = win.shape[1]
            sl = slice(0, M)
            take = verified
            best_id[:, sl] = torch.where(take, cand_id, best_id[:, sl])
            best_len[:, sl] = torch.where(
                take, torch.full_like(best_len[:, sl], L),
                best_len[:, sl])
        return best_id, best_len

    @staticmethod
    def segment_words(chunk_ids, tok_count, tables, nObj):
        """Tensor word-segmentation, **fully static** (no ``.item()``, no
        boolean-mask compaction -> zero DtoH, fullgraph/CUDA-graph safe).
        Bit-identical semantics to ``_embed_bpe``'s Python sweep:

          * boundary chunk (all bytes in BOUNDARY_BYTES) separates words;
          * non-boundary chunk resolves via ``chunk_to_cb`` (-1 == skip:
            byte_mode key not in codebook);
          * a word = a maximal run of non-boundary chunks, EMITTED only if
            it has >=1 resolved sub-token (empty/all-unresolved runs take
            no slot); per-row emitted words are slotted 0..nObj-1 in
            left-to-right order, later words dropped.

        Everything stays ``[B,T]`` (T = chunk buffer width, static). The
        emitter scatters these into static ``[B*nObj]`` buffers, so the
        target id ``b*nObj+slot`` is the only thing needed -- no dense
        word-id / word-count, hence no host sync.

        Returns (all ``[B,T]``, long/bool): ``sub_cb`` (codebook row, -1
        where not a kept sub-token), ``sub_target`` (``b*nObj+slot``, or
        ``B*nObj`` trash bucket where not kept), ``sub_pos`` (token index,
        for first-sub-token tiebreak), ``keep`` (bool).
        """
        B, T = chunk_ids.shape
        dev = chunk_ids.device
        pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        valid = pos < tok_count.unsqueeze(1)
        ids = chunk_ids.clamp(min=0)
        is_bnd = tables["is_boundary"][ids] | (~valid)          # pad => sep
        cb = tables["chunk_to_cb"][ids]                          # -1 = skip
        resolved = valid & (~is_bnd) & (cb != -1)                # [B,T]

        # run_key = #boundary chunks strictly before this position (a run
        # of non-boundary chunks between two boundaries shares one key).
        bnd_i = is_bnd.to(torch.int64)
        run_key = torch.cumsum(bnd_i, dim=1) - bnd_i             # [B,T]

        # Per (b, run_key): does the run hold >=1 resolved sub-token?
        has_res = torch.zeros(B, T, dtype=torch.int64, device=dev)
        has_res.scatter_reduce_(
            1, run_key, resolved.to(torch.int64),
            reduce="amax", include_self=True)                    # 0/1 grid

        # Slot for an emitted run = #emitted runs with smaller run_key
        # (exclusive cumsum of the has_res grid over the run-key axis).
        emit_excl = torch.cumsum(has_res, dim=1) - has_res       # [B,T]
        slot = emit_excl.gather(1, run_key)                      # [B,T]
        keep = resolved & (slot < nObj)                          # [B,T] bool

        b_idx = (torch.arange(B, device=dev)
                 .unsqueeze(1).expand(B, T))                     # [B,T]
        trash = B * nObj
        sub_target = torch.where(keep, b_idx * nObj + slot,
                                 torch.full_like(slot, trash))
        sub_cb = torch.where(keep, cb, torch.full_like(cb, -1))
        return sub_cb, sub_target, pos, keep

    @staticmethod
    def gpu_chunk_ids(byte_buf, best_id, best_len):
        """Greedy left-to-right consumption -> token-major ``chunk_ids``
        [B, N] long (padded with -1) and ``tok_count`` [B] long.

        Sequential by nature (token k+1 starts at cursor + L[cursor]); a
        bounded on-device scan over a ``[B]`` cursor, fixed N iterations
        with masking -- no ``.any()``/``.item()`` host sync. N is the byte
        buffer width; #tokens per row <= N so N iterations always suffice.
        """
        B, N = byte_buf.shape
        dev = byte_buf.device
        chunk_ids = torch.full((B, N), -1, dtype=torch.int64, device=dev)
        cursor = torch.zeros(B, dtype=torch.int64, device=dev)
        tok_count = torch.zeros(B, dtype=torch.int64, device=dev)
        ar = torch.arange(B, device=dev)
        for t in range(N):
            c = cursor.clamp(max=N - 1)
            cur_byte = byte_buf[ar, c]
            active = (cursor < N) & (cur_byte != 0)
            cur_id = best_id[ar, c]
            cur_len = best_len[ar, c]
            chunk_ids[ar, torch.full_like(cursor, t)] = torch.where(
                active, cur_id, chunk_ids[ar, torch.full_like(cursor, t)])
            tok_count = tok_count + active.to(torch.int64)
            cursor = cursor + torch.where(
                active, cur_len, torch.zeros_like(cur_len))
        return chunk_ids, tok_count



class MPHFGpuLayer(Layer):
    """Static minimal-perfect-hash (MPHF) percept->row index for the
    FROZEN-vocab training path -- the Rework A core mechanism.

    Per the consolidated two-loop spec (§"Percept -> MPHF -> table",
    §IMPLEMENTATION DETAILS D2): each percept's byte slot passes through an
    MPHF producing an ``index in [0, V_percept)``. That index addresses a
    **table** whose every entry holds BOTH (1) the literal surface word
    string and (2) the ConceptualSpace activation vector for that token.
    The table's halves are reused directly from the frozen ``Embedding``
    codebook (``codebook.wv._vectors`` and ``codebook.wv.index_to_key``)
    -- no parallel structures.

    Build/verify pattern mirrors ``BPEGpuLayer``: frozen, built ONCE at
    the CPU->GPU handoff over the frozen lexicon key set, cached on the
    owning ``PartSpace`` as ``self._mphf_static_tables``; runtime is
    pure static tensor ops -- poly-hash + ``searchsorted`` + collision-
    proof byte-verify -- ZERO host sync, O(1) per slot. The MPHF is
    **non-invertible**: the reverse map is the table lookup, never an
    inverse hash. A one-row silent mis-resolution corrupts all training,
    so the byte-verify is exact regardless of hash collisions (fail-loud
    memory).
    """

    class _MPHFUnavailable(Exception):
        """Raised when the static MPHF table cannot be built (e.g. an empty
        or non-Embedding codebook). The caller falls back to the existing
        verified path -- never a silent wrong result."""

    def __init__(self):
        super().__init__(nInput=0, nOutput=0)

    @staticmethod
    def build_mphf_table(codebook, device):
        """Frozen lexicon key set -> static device tensors. Call ONCE per
        (frozen) codebook on the target device; cache the result.

        The frozen key set is ``codebook.wv.index_to_key`` (the ``.kv``
        lexicon + ASCII bootstrap + NULL rows -- frozen at
        ``word_learning<=0``). All Python-dict / utf-8 work happens HERE
        (one-time, host), never per batch.

        Returns a dict the runtime ``mphf_index`` consumes plus the
        ``surface`` list (== ``index_to_key``) for the reverse map.
        """
        wv = getattr(codebook, "wv", None)
        if wv is None or getattr(wv, "index_to_key", None) is None:
            raise MPHFGpuLayer._MPHFUnavailable(
                "build_mphf_table: codebook has no wv.index_to_key "
                "(non-Embedding / numeric codebook).")
        keys = list(wv.index_to_key)
        V = len(keys)
        if V == 0:
            raise MPHFGpuLayer._MPHFUnavailable(
                "build_mphf_table: empty lexicon.")

        # Each key -> its utf-8 byte tuple, EXACTLY as InputSpace lexes a
        # token slot into ``subspace.what.W`` (null-terminated utf-8). The
        # MPHF keys on the same byte representation the percept slot carries
        # so the lookup is byte-for-byte the frozen ``key_to_index`` row.
        key_bytes = []
        maxL = 1
        for k in keys:
            try:
                kb = k.encode("utf-8")
            except Exception:
                # Non-str keys are not lexable percept slots; map to a
                # zero-length entry (never matched -> NULL row fallback).
                kb = b""
            key_bytes.append(kb)
            if len(kb) > maxL:
                maxL = len(kb)

        # Padded row->bytes (-1 pad) + lengths for the collision-proof
        # byte-verify, mirroring ``BPEGpuLayer``'s ``tok_bytes``/``tok_len``.
        tok_bytes = torch.full((V, maxL), -1, dtype=torch.int64)
        tok_len = torch.zeros(V, dtype=torch.int64)
        for i, kb in enumerate(key_bytes):
            kl = len(kb)
            tok_len[i] = kl
            if kl:
                tok_bytes[i, :kl] = torch.tensor(
                    [int(b) for b in kb], dtype=torch.int64)

        # Per-length sorted (poly-hash -> row idx) index for the O(1)
        # exact-key ``searchsorted`` lookup. A length-0 key (empty / NULL
        # sentinel slot) is never matched -- the runtime maps an empty slot
        # to ``null_row`` explicitly, not via the hash.
        by_len = {}
        for L in range(1, maxL + 1):
            ids = [i for i in range(V) if int(tok_len[i]) == L]
            if not ids:
                by_len[L] = None
                continue
            ids_t = torch.tensor(ids, dtype=torch.int64)
            windows = tok_bytes[ids_t, :L]                 # [K, L]
            hashes = BPEGpuLayer._poly_hash(windows)       # [K]
            order = torch.argsort(hashes)
            by_len[L] = (hashes[order].to(device),
                         ids_t[order].to(device))

        # The NULL row: ``\x00`` (byte 0) is the per-row cursor seal / pad
        # sentinel and ``Embedding.create`` seeds it at row 0. An empty /
        # all-zero percept slot resolves here (the NULL char surface), NOT
        # via the hash.
        null_row = int(wv.key_to_index.get("\x00", 0))

        return {
            "maxL": int(maxL),
            "V": int(V),
            "tok_bytes": tok_bytes.to(device),
            "tok_len": tok_len.to(device),
            "by_len": by_len,
            "null_row": null_row,
            # The surface-string half of the D2 table == index_to_key
            # (ASCII-prefilled + NULL; the reverse map's lookup target).
            "surface": keys,
        }

    @staticmethod
    def mphf_index(token_byte_slots, tables, return_verified=False):
        """``token_byte_slots`` [B, K, W] long (0..255; 0 terminates a slot,
        EXACTLY ``InputSpace.subspace.what.W``'s per-token null-terminated
        utf-8 layout).

        Returns ``row`` [B, K] long: the frozen lexicon row index for each
        percept slot. Pure static tensor ops (poly-hash + ``searchsorted``
        + collision-proof byte-verify), ZERO host sync, O(1) per slot.

        Resolution per slot:
          * effective key length L = #leading non-zero bytes (utf-8 token,
            null-terminated -- the byte AFTER the token is the 0 sentinel);
          * L == 0 (empty / pad / NULL-sentinel slot) -> ``null_row``;
          * else poly-hash the L-byte window, ``searchsorted`` the
            length-L sorted hash table, byte-verify against ``tok_bytes``
            (collision-proof: exact regardless of hash); a verified hit ->
            that row, a miss (key not in the frozen lexicon -- OOV) ->
            ``null_row`` (the documented frozen-vocab fallback: the table
            only holds the frozen key set; an OOV percept reconstructs to
            the NULL surface rather than silently aliasing a wrong row).

        When ``return_verified=True``, returns ``(row, verified)`` where
        ``verified`` ``[B, K]`` bool is True only for slots that hit the
        frozen lexicon via the collision-proof byte-verify (False for
        L==0 slots AND OOV slots; the row at those False positions is the
        ``null_row`` fallback). The ``synthesis_mode='mphf'`` selectable
        runtime path uses this signal to gate the OOV->BPE-trie fallback
        per spec.
        """
        if token_byte_slots.dim() == 2:
            token_byte_slots = token_byte_slots.unsqueeze(-1)
        B, K, W = token_byte_slots.shape
        dev = token_byte_slots.device
        bb = token_byte_slots.to(torch.int64).clamp(0, 255)  # [B,K,W]
        null_row = tables["null_row"]
        maxL = tables["maxL"]

        # Effective key length = #leading non-zero bytes (the slot is one
        # token's utf-8 bytes then a 0). ``cumprod`` over (byte != 0) gives
        # 1 for every position up to (not including) the first 0; the sum
        # is the leading-run length -- pure tensor, no host sync.
        nonzero = (bb != 0).to(torch.int64)                  # [B,K,W]
        lead = torch.cumprod(nonzero, dim=-1)                # [B,K,W]
        eff_len = lead.sum(dim=-1)                            # [B,K]

        row = torch.full((B, K), null_row, dtype=torch.int64, device=dev)
        verified_any = torch.zeros((B, K), dtype=torch.bool, device=dev)
        for L in range(1, min(maxL, W) + 1):
            entry = tables["by_len"].get(L)
            if entry is None:
                continue
            keys_sorted, ids_sorted = entry
            win = bb[..., :L]                                 # [B,K,L]
            h = BPEGpuLayer._poly_hash(win)                   # [B,K]
            pos = torch.searchsorted(keys_sorted, h)
            pos = pos.clamp(max=keys_sorted.numel() - 1)
            hit = keys_sorted[pos] == h
            cand = ids_sorted[pos]                            # [B,K]
            cb = tables["tok_bytes"][cand][..., :L]           # [B,K,L]
            verified = hit & (win == cb).all(dim=-1) & (eff_len == L)
            row = torch.where(verified, cand, row)
            verified_any = verified_any | verified
        if return_verified:
            return row, verified_any
        return row

    @staticmethod
    def reverse_map_rows(concept_vectors, codebook):
        """Non-invertible reverse map: a concept-activation vector ->
        nearest ``wv._vectors`` row index.

        The MPHF is NOT inverted; the reverse map is the **table lookup**
        (nearest concept-activation row). ``concept_vectors`` is ``[..., D]``
        (D == ``wv._vectors`` width). Returns the matching ``[...]`` long
        row indices; ``surface[idx]`` (== ``codebook.wv.index_to_key[idx]``)
        is the reconstructed literal surface word. Pure tensor op, no host
        sync (the host ``index_to_key`` indexing is the caller's choice,
        done only off the training-critical path).
        """
        W = codebook.wv._vectors                              # [V, D]
        flat = concept_vectors.reshape(-1, concept_vectors.shape[-1])
        # Cosine-style nearest row (the codebook lives on the periodic unit
        # cell; dot-product nearest is the same neighbour the codebook
        # search uses). Static [N, V] then argmax -- no host sync.
        sims = flat @ W.t()                                   # [N, V]
        idx = sims.argmax(dim=-1)                              # [N]
        return idx.reshape(concept_vectors.shape[:-1])


class SymbolLearningLayer(Layer):
    """Symbol-learning statistics + promotion policy.

    Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
    §Phase 2 / step 6 — scaffold; deferred §Phase 2 → Phase 3 closeout —
    policy + ``extend_artifact`` integration.

    Disabled-by-default and detached from autograd. Inlines the former
    ``SymbolLearningStats`` (accumulator), ``SymbolLearningPolicy``
    (MDL-flavored thresholds), and ``flush_and_promote`` orchestration
    into a single Layer that ``ConceptualSpace`` owns. Promotion still
    happens only at explicit flush boundaries -- never inside autograd
    ``forward``.

    Hook signals:
      * Zero-order / QE via ``observe_qe(activation, snapped)``: squared
        quantization error + online leader-clustering producing per-cluster
        (centroid, qe_ema, stability) summaries.
      * Higher-order / PMI via ``observe_reduce(left_ref, right_ref,
        parent_ref, *, parent_scalar=None, parent_category=None,
        parent_order=None)``: pair counts + the rule's LHS metadata.

    Numerical-divergence policy: NaN / Inf inputs to ``observe_qe`` raise
    ``ValueError`` rather than getting silently ``nan_to_num``-cleaned.
    """

    class _Cluster:
        """One leader-clustering cluster: centroid + EMA of squared QE +
        stability count. Centroid is a detached 1-D tensor."""
        __slots__ = ('centroid', 'qe_ema', 'stability')

        def __init__(self, centroid, qe_ema, stability):
            self.centroid = centroid
            self.qe_ema = float(qe_ema)
            self.stability = int(stability)

    def __init__(self, enabled=False, *,
                 leader_radius=0.5, ema_alpha=0.1,
                 qe_promote_threshold=0.5, stability_n=10,
                 pmi_threshold=1.0, count_threshold=5,
                 default_zero_order_category='NEW',
                 default_zero_order_parent_ref_id=0,
                 default_higher_order_category='NEW',
                 default_higher_order_order=1):
        super().__init__(nInput=0, nOutput=0)
        self.enabled = bool(enabled)
        # Stats hyperparameters.
        self.leader_radius = float(leader_radius)
        self.ema_alpha = float(ema_alpha)
        # Policy thresholds.
        self.qe_promote_threshold = float(qe_promote_threshold)
        self.stability_n = int(stability_n)
        self.pmi_threshold = float(pmi_threshold)
        self.count_threshold = int(count_threshold)
        self.default_zero_order_category = str(default_zero_order_category)
        self.default_zero_order_parent_ref_id = int(
            default_zero_order_parent_ref_id)
        self.default_higher_order_category = str(default_higher_order_category)
        self.default_higher_order_order = int(default_higher_order_order)
        # Accumulator state — plain Python so nothing leaks into autograd.
        self.qe_count = 0
        self.qe_sum_squared = 0.0
        self.pair_counts = {}
        self.pair_info = {}
        self._clusters = []

    @staticmethod
    def enabled_from_config():
        """Read ``<architecture><symbolLearning enabled="..."/>`` from XML.

        Defaults to ``False`` when the key is absent. Accepts ``"true"`` /
        ``"false"`` (case-insensitive) and ``"1"`` / ``"0"``. Any other
        value falls back to ``False``.
        """
        try:
            raw = TheXMLConfig.get("architecture.symbolLearning.enabled",
                                   default=False)
        except (KeyError, TypeError, ValueError, AttributeError):
            return False
        if raw is None:
            return False
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        return s in ("true", "1", "yes", "on")

    def observe_qe(self, activation, snapped):
        """Record one quantization-error measurement.

        Computes squared distance between the pre-snap activation and
        the snapped prototype, updates the aggregate QE counter, and
        runs online leader clustering on the un-snapped activation.

        NaN / Inf inputs raise ``ValueError`` per the project's "fail
        loud on numerical divergence" rule.
        """
        if not self.enabled:
            return
        if torch.isnan(activation).any() or torch.isinf(activation).any():
            raise ValueError(
                "SymbolLearningLayer.observe_qe: NaN/Inf in activation")
        if torch.isnan(snapped).any() or torch.isinf(snapped).any():
            raise ValueError(
                "SymbolLearningLayer.observe_qe: NaN/Inf in snapped")
        with torch.no_grad():
            act = activation.detach()
            snap = snapped.detach()
            diff = act - snap
            sq = float(diff.pow(2).sum().item())
            flat = act.reshape(-1)
        self.qe_count += 1
        self.qe_sum_squared += sq
        self._update_clusters(flat, sq)

    def _update_clusters(self, vec, sq_qe):
        """Online leader clustering."""
        if not self._clusters:
            self._clusters.append(self._Cluster(
                centroid=vec.clone(), qe_ema=sq_qe, stability=1))
            return
        radius_sq = self.leader_radius ** 2
        nearest = None
        nearest_d = math.inf
        for c in self._clusters:
            if c.centroid.shape != vec.shape:
                continue
            d = float((vec - c.centroid).pow(2).sum().item())
            if d < nearest_d:
                nearest_d = d
                nearest = c
        if nearest is None or nearest_d > radius_sq:
            self._clusters.append(self._Cluster(
                centroid=vec.clone(), qe_ema=sq_qe, stability=1))
            return
        a = self.ema_alpha
        nearest.qe_ema = (1.0 - a) * nearest.qe_ema + a * sq_qe
        nearest.centroid = (1.0 - a) * nearest.centroid + a * vec
        nearest.stability += 1

    def observe_reduce(self, left_ref, right_ref, parent_ref, *,
                       parent_scalar=None, parent_category=None,
                       parent_order=None):
        """Record one REDUCE outcome with rule LHS metadata."""
        if not self.enabled:
            return
        key = (int(left_ref), int(right_ref))
        self.pair_counts[key] = self.pair_counts.get(key, 0) + 1
        info = self.pair_info.setdefault(key, {
            'count': 0,
            'parent_scalar_sum': 0.0,
            'parent_ref': int(parent_ref),
            'parent_category': None,
            'parent_order': None,
        })
        info['count'] += 1
        info['parent_ref'] = int(parent_ref)
        if parent_scalar is not None:
            info['parent_scalar_sum'] += float(parent_scalar)
        if parent_category is not None:
            info['parent_category'] = str(parent_category)
        if parent_order is not None:
            info['parent_order'] = int(parent_order)

    def flush(self):
        """Return a snapshot of accumulated stats and reset state."""
        snap = {
            'qe_count': self.qe_count,
            'qe_sum_squared': self.qe_sum_squared,
            'qe_mean': (self.qe_sum_squared / self.qe_count
                        if self.qe_count > 0 else 0.0),
            'pair_counts': dict(self.pair_counts),
            'pair_info': {k: dict(v) for k, v in self.pair_info.items()},
            'clusters': [
                {'centroid': c.centroid.clone(),
                 'qe_ema': c.qe_ema,
                 'stability': c.stability}
                for c in self._clusters
            ],
        }
        self.qe_count = 0
        self.qe_sum_squared = 0.0
        self.pair_counts = {}
        self.pair_info = {}
        self._clusters = []
        return snap

    def propose_candidates(self, snapshot):
        """Apply MDL-flavored thresholds to ``snapshot`` and return a list
        of ``embed.NewRef`` candidates. May be empty.

        Zero-order:
          Promote a cluster with ``stability >= stability_n`` AND
          ``qe_ema >= qe_promote_threshold`` to a new order-0 ref at
          ``default_zero_order_category`` /
          ``default_zero_order_parent_ref_id``. The cluster centroid's
          scalar mean becomes the new ref's learned scalar.

        Higher-order:
          Promote a pair ``(a, b)`` with ``count >= count_threshold`` AND
          ``log(P(a,b) / (P(a) P(b))) >= pmi_threshold`` to a new ref at
          the recorded ``parent_category`` / ``parent_order`` (from the
          rule's LHS metadata). The new ref's scalar is the mean of the
          reduce-time parent activations.
        """
        from embed import NewRef
        candidates = []
        # ---- Zero-order (QE) ----
        for c in snapshot.get('clusters', []):
            if c['stability'] < self.stability_n:
                continue
            if c['qe_ema'] < self.qe_promote_threshold:
                continue
            centroid = c['centroid']
            if isinstance(centroid, torch.Tensor):
                scalar = float(centroid.mean().item())
            else:
                scalar = float(centroid)
            candidates.append(NewRef(
                scalar=scalar,
                order=0,
                parent_ref_id=int(self.default_zero_order_parent_ref_id),
                category=str(self.default_zero_order_category),
            ))
        # ---- Higher-order (PMI × frequency) ----
        pair_counts = snapshot.get('pair_counts', {})
        pair_info = snapshot.get('pair_info', {})
        total = sum(pair_counts.values())
        if total > 0:
            left_marginal = {}
            right_marginal = {}
            for (a, b), n in pair_counts.items():
                left_marginal[a] = left_marginal.get(a, 0) + n
                right_marginal[b] = right_marginal.get(b, 0) + n
            for (a, b), n in pair_counts.items():
                if n < self.count_threshold:
                    continue
                la = left_marginal[a]
                rb = right_marginal[b]
                if la <= 0 or rb <= 0:
                    continue
                # PMI = log[ (n * total) / (la * rb) ]
                pmi = math.log((n * total) / (la * rb))
                if pmi < self.pmi_threshold:
                    continue
                info = pair_info.get((a, b), {})
                count = max(info.get('count', n), 1)
                scalar_sum = info.get('parent_scalar_sum', 0.0)
                scalar = scalar_sum / count if count > 0 else 0.0
                category = info.get('parent_category') \
                    or self.default_higher_order_category
                order = info.get('parent_order')
                if order is None:
                    order = self.default_higher_order_order
                parent_ref = info.get(
                    'parent_ref', self.default_zero_order_parent_ref_id)
                candidates.append(NewRef(
                    scalar=float(scalar),
                    order=int(order),
                    parent_ref_id=int(parent_ref),
                    category=str(category),
                ))
        return candidates

    def flush_and_promote(self, artifact_path):
        """Flush accumulated stats, run the policy, call
        ``embed.extend_artifact`` on any candidates. Returns the
        candidate list (empty when nothing crossed the thresholds).

        Flush boundary is the caller's responsibility (e.g., end of
        training batch). By construction this never mutates the
        artifact inside an autograd forward.
        """
        from embed import extend_artifact
        snap = self.flush()
        candidates = self.propose_candidates(snap)
        if candidates:
            extend_artifact(artifact_path, candidates)
        return candidates


class _RuleScorer(nn.Module):
    """Small MLP scoring rules from top-of-stack payloads.

    Private to ``ShortTermMemory``; replaces the standalone
    ``stm_driver.RuleScorer`` (2026-05-21 SymbolSubSpace/STM Layer
    refactor). Input: top operand payload(s) — ``left`` (required) and
    ``right`` (optional, ``None`` for unary REDUCE). Concatenates and
    projects to a per-rule logit vector. Architecturally minimal -- one
    hidden layer + linear head -- because admissibility masking does
    the hard discrimination; the scorer just orders the survivors.
    """

    def __init__(self, payload_dim, n_rules, hidden_dim=None):
        super().__init__()
        self.payload_dim = int(payload_dim)
        self.n_rules = int(n_rules)
        # Concatenated input width: 2 * payload_dim. For unary REDUCE we
        # zero out the right slot so the same head handles both shapes.
        in_features = 2 * self.payload_dim
        hidden = int(hidden_dim) if hidden_dim else max(in_features, 16)
        self.body = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden, self.n_rules)

    def forward(self, left, right):
        """Return ``[n_rules]`` logits for the given operand payload(s)."""
        if right is None:
            right = torch.zeros_like(left)
        x = torch.cat([left, right], dim=-1)
        return self.head(self.body(x))


class ShortTermMemory(Layer):
    """Short-term memory (STM) on ConceptualSpace.

    Carries two roles after the 2026-05-21 SymbolSubSpace/STM Layer refactor:

    1. **Per-batch idea stack** (legacy chart consumer surface). A
       ``[B, capacity, concept_dim]`` payload buffer that the chart's
       CKY compose path pushes "ideas" (continuous compositions) onto.
       Distinct from ``SymbolSpace._stm_fired`` (once-per-sentence
       discourse-priming flag). Spec §"Removed Public Surfaces" calls
       for ``ShortTermMemory`` to become data-free; the legacy push /
       peek / snapshot surface is retained transitionally for the
       chart's CKY consumers in ``Models.py`` and ``embed.py``.

    2. **STM shift/reduce driver** (new in this refactor). Inlines the
       former ``stm_driver.STMDriver`` + ``stm_driver.RuleScorer`` +
       ``stm_trainer.train_step`` behavior as methods that take a
       ``word_subspace`` argument (the typed-STM data carrier). The
       lazy-initialised ``self.scorer`` is a small MLP that scores
       admissible REDUCE actions; ``ConceptualSpace.attach_knowledge``
       wires the rule signature list in.

    Semantics of the idea stack:
        * Each slot holds a ``[concept_dim]`` vector.
        * Pushes are bottom-up: ``peek(b, 0)`` returns the most
          recent idea; ``peek(b, n)`` returns the n-th most recent.
        * Capacity is configurable via
          ``<ConceptualSpace><stmCapacity>N</stmCapacity></ConceptualSpace>``.

    Lifecycle:
        * Built by ``ConceptualSpace.__init__`` at construction time
          (capacity from XML; default 8).
        * Cleared on hard ``Reset`` (sentence boundary): all rows set
          to zero, depth pointers reset to 0.

    Storage of the idea stack is a plain registered buffer
    (``persistent=False``); STM contents are runtime working state,
    not learned weights.
    """

    DEFAULT_CAPACITY = 8

    def __init__(self, batch=1, capacity=None, concept_dim=0):
        super().__init__(nInput=0, nOutput=0)
        # Phase E completion of doc/specs/2026-05-21-wordsubspace-stm-
        # layer-refactor.md: the Layer is data-free. ``_init_capacity``
        # / ``_init_concept_dim`` are the constructor's record of the
        # requested sizes — used only to seed SymbolSubSpace's
        # ``_idea_buffer`` if a standalone (no-SymbolSubSpace) test
        # constructs a ShortTermMemory directly. Once
        # ``attach_word_subspace`` is called the data lives on
        # SymbolSubSpace's ``_idea_*`` buffers and the proxy properties /
        # methods below route there.
        self._init_capacity = int(capacity or self.DEFAULT_CAPACITY)
        self._init_concept_dim = int(concept_dim)
        self._init_batch = int(batch)
        # Standalone fallback: when no SymbolSubSpace has been attached
        # yet (e.g. test_conceptual_stm.py constructs ShortTermMemory
        # bare), we allocate a local idea-stack so the legacy push /
        # peek / snapshot surface keeps working. ``attach_word_subspace``
        # later switches the proxy onto the SymbolSubSpace's buffers and
        # drops the local allocation.
        self._word_subspace = None
        # A5 fullgraph fix (doc/plans/2026-06-06-parallel-conceptual-
        # recurrence.md sec 2 & 4): the per-forward STM working buffer is a
        # PLAIN attribute (stashed via ``object.__setattr__`` so it never
        # lands in ``nn.Module._buffers`` / ``_parameters`` and is never
        # captured as a torch.compile graph INPUT). ``begin_forward``
        # re-creates it fresh INSIDE the traced forward each pass, so it is a
        # graph INTERMEDIATE -- not a persisted, mutated, output-aliased
        # input. The retired ``_fallback_buffer`` was a persistent tensor
        # whose ``requires_grad`` oscillated across forwards (flipping a
        # Dynamo guard -> recompile) and whose in-place mutation crashed AOT
        # autograd's ``gen_alias_from_base`` once it was resized between
        # calls. The standalone surface keeps working because the constructor
        # seeds the live buffer eagerly (bare push/peek tests never call a
        # compiled forward, so a plain in-Python tensor there is fine).
        object.__setattr__(
            self, '_live_buffer',
            torch.zeros(self._init_batch, self._init_capacity,
                        self._init_concept_dim))
        object.__setattr__(
            self, '_live_depth',
            torch.zeros(self._init_batch, dtype=torch.long))
        self._live_capacity = self._init_capacity
        self._live_max_depth_host = 0
        # STM shift/reduce driver state. Initialised lazily on first
        # ``shift`` / ``train_scorer_step`` call (or eagerly via
        # ``init_scorer``); ``rule_signatures`` is a plain Python list
        # of dicts, stashed via ``object.__setattr__`` to bypass
        # nn.Module's child-type checks.
        self.scorer = None
        object.__setattr__(self, 'rule_signatures', None)

    def attach_word_subspace(self, word_subspace):
        """Wire a ``SymbolSubSpace`` so the Layer's data-accessor methods
        route to its ``_idea_*`` buffers. Stored via
        ``object.__setattr__`` to bypass nn.Module's submodule
        registration (the SymbolSubSpace owns the back-reference; this is
        a non-owning routing pointer).

        Called from ``SymbolSubSpace.__init__`` (Phase E completion of the
        2026-05-21 refactor). Once attached, the local ``_fallback_*``
        buffers are dropped.
        """
        object.__setattr__(self, '_word_subspace', word_subspace)
        # If the SymbolSubSpace's idea-stack capacity is smaller than the
        # one this ShortTermMemory was constructed for, grow it now so
        # we honour the constructor's sizing contract.
        if hasattr(word_subspace, 'idea_ensure_capacity'):
            word_subspace.idea_ensure_capacity(self._init_capacity)

    # -- per-forward live working buffer (A5 fullgraph fix) ---------------
    #
    # The STM data lives on the PLAIN ``_live_*`` attributes (stashed via
    # ``object.__setattr__`` so they stay out of ``nn.Module._buffers`` and
    # are never captured as a torch.compile graph input). ``begin_forward``
    # re-creates ``_live_buffer`` fresh inside the traced forward each pass,
    # making it a graph intermediate -- so the in-place STM writes mutate the
    # intermediate (legal) instead of a persisted, output-aliased input.
    # When a SymbolSubSpace is attached the buffer is still data-free in the
    # nn.Module sense (the live data is not a registered buffer); the
    # ``capacity`` proxy honours ``ss._idea_capacity`` so a caller that grows
    # the attached idea-stack capacity (e.g. test_bounded_stm_fold) still
    # sizes the next forward's fresh buffer correctly.

    @property
    def _buffer(self):
        return self._live_buffer

    @_buffer.setter
    def _buffer(self, value):
        object.__setattr__(self, '_live_buffer', value)

    @property
    def _depth(self):
        return self._live_depth

    @_depth.setter
    def _depth(self, value):
        object.__setattr__(self, '_live_depth', value)

    @property
    def _max_depth_host(self):
        return self._live_max_depth_host

    @_max_depth_host.setter
    def _max_depth_host(self, value):
        self._live_max_depth_host = int(value)

    @property
    def capacity(self):
        ss = self._word_subspace
        if ss is None:
            return self._live_capacity
        return ss._idea_capacity

    @property
    def concept_dim(self):
        ss = self._word_subspace
        if ss is None:
            return self._init_concept_dim
        return ss._stm_payload_dim

    def begin_forward(self, batch, device=None, dtype=None):
        """Seed a FRESH per-forward working buffer (A5 fullgraph fix).

        doc/plans/2026-06-06-parallel-conceptual-recurrence.md sec 2 & 4
        (the load-bearing principle): per-batch DATA threads THROUGH the
        forward as tensors and is NEVER persisted as accumulated state on a
        space/Layer across forwards. The idea buffer used to be a persistent
        registered buffer (``ss._idea_buffer`` when attached, the standalone
        ``_fallback_buffer`` otherwise) that the compiled forward mutated
        IN PLACE. That made it a graph INPUT that was mutated and
        output-aliased, so torch.compile/AOT autograd had to regenerate it
        as an alias of its base every call -- which (a) flipped a Dynamo
        ``requires_grad`` guard each batch (the buffer's grad status
        oscillated False->True), forcing a recompile, and (b) crashed AOT's
        ``gen_alias_from_base`` once the buffer was resized between calls.

        Calling this at the TOP of the forward body REPLACES the buffer
        attribute with a freshly-allocated ``torch.zeros`` created INSIDE the
        traced region. A fresh in-trace allocation is a graph INTERMEDIATE,
        not an input, so the subsequent in-place STM writes mutate the
        intermediate (legal, no input-mutation alias) and the persisted
        attribute is never read-as-input nor output-aliased. The STM starts
        every forward empty -- which is exactly the existing sentence-
        boundary semantics (parallel: a fresh sentence; serial: the prelude
        ``clear()``), so this changes no behaviour, only the storage's graph
        identity.

        Routed through the ``_buffer`` / ``_depth`` setters so it works for
        both the SymbolSubSpace-attached proxy and the standalone fallback.
        """
        batch = int(batch)
        cap = int(self.capacity)
        dim = int(self.concept_dim)
        # IMPORTANT: do NOT read ``self._buffer`` here -- reading the
        # persisted buffer inside the traced forward would re-introduce it as
        # a graph INPUT (the very thing we are trying to avoid). Resolve
        # device/dtype from the explicit args only; default to float32 on the
        # caller-supplied device.
        if dtype is None:
            dtype = torch.float32
        self._buffer = torch.zeros(
            batch, cap, dim, device=device, dtype=dtype)
        self._depth = torch.zeros(batch, dtype=torch.long, device=device)
        self._max_depth_host = 0

    # -- idea-stack API (chart compose consumer surface) ------------------
    #
    # A5: all operations act on the single live store (the ``_buffer`` /
    # ``_depth`` / ``_max_depth_host`` properties -> the ``_live_*`` plain
    # attributes). The former ``ss is None`` branch (SymbolSubSpace-attached
    # vs standalone ``_fallback_*``) is gone -- the live data is no longer a
    # registered buffer on either, so the compiled forward never sees it as a
    # graph input. ``capacity`` still proxies to ``ss._idea_capacity`` when
    # attached so an externally-grown idea-stack capacity sizes the next
    # ``begin_forward`` correctly.

    def ensure_batch(self, batch):
        """Resize the idea-stack to ``batch`` rows. Idempotent."""
        batch = int(batch)
        buf = self._buffer
        if buf is not None and int(buf.shape[0]) == batch:
            return
        device = buf.device if buf is not None else None
        dtype = buf.dtype if buf is not None else torch.float32
        self._buffer = torch.zeros(
            batch, int(self.capacity), int(self.concept_dim),
            device=device, dtype=dtype)
        self._depth = torch.zeros(batch, dtype=torch.long, device=device)
        self._max_depth_host = 0

    def ensure_capacity(self, capacity):
        """Grow the per-slot capacity to at least ``capacity`` (grow-only)."""
        capacity = int(capacity)
        buf = self._buffer
        old_cap = int(buf.shape[1]) if buf is not None else 0
        if capacity <= old_cap:
            # Keep the standalone capacity record in sync (grow-only) so
            # ``self.capacity`` reflects the request when not attached.
            if self._word_subspace is None and capacity > self._live_capacity:
                self._live_capacity = capacity
            return
        device = buf.device
        B = int(buf.shape[0])
        new_buf = torch.zeros(
            B, capacity, int(self.concept_dim),
            device=device, dtype=buf.dtype)
        if old_cap > 0:
            new_buf[:, :old_cap, :] = buf
        self._buffer = new_buf
        if self._word_subspace is None:
            self._live_capacity = capacity

    def push(self, b, idea):
        # Newest-at-slot-0: write the new idea at slot 0 and shift the
        # existing occupants RIGHT. Overflow RAISES (strict-bound
        # primitive; the rolling-window drop is _stm_shift_and_push).
        cap = int(self.capacity)
        depth = int(self._depth[b].item())
        if depth >= cap:
            raise RuntimeError(
                f"ShortTermMemory.push: row {b} is at capacity "
                f"({cap}); reduce before pushing further.")
        buf = self._buffer
        if depth > 0:
            buf[b, 1:depth + 1] = buf[b, 0:depth].clone()
        buf[b, 0] = idea
        self._depth[b] = depth + 1
        if depth + 1 > self._max_depth_host:
            self._max_depth_host = depth + 1

    def push_step(self, ideas):
        # Newest-at-slot-0: shift RIGHT by one slot, write slot 0.
        buf = self._buffer
        cap = int(buf.shape[1])
        if cap > 1:
            buf[:, 1:cap] = buf[:, 0:cap - 1].clone()
        buf[:, 0] = ideas
        self._depth = self._depth + 1
        self._max_depth_host = self._max_depth_host + 1

    def push_step_masked(self, ideas, gate_b_1):
        # Newest-at-slot-0 masked push: gated rows shift RIGHT + write
        # slot 0; un-gated rows are unchanged.
        B, D = ideas.shape
        buf = self._buffer
        cap = int(buf.shape[1])
        gate_b = gate_b_1.view(B)
        gate_bool = gate_b.bool()
        gate_col = gate_bool.view(B, 1, 1)
        shifted = buf.clone()
        if cap > 1:
            shifted[:, 1:cap] = buf[:, 0:cap - 1]
        shifted[:, 0] = ideas
        self._buffer = torch.where(gate_col, shifted, buf)
        self._depth = self._depth + gate_bool.long()

    def push_window_batch(self, ideas):
        # Newest-at-slot-0: shift RIGHT by W; write the window REVERSED
        # into slots [0, W) so the window's newest position lands at 0.
        B, W, D = ideas.shape
        if W == 0:
            return
        buf = self._buffer
        cap = int(buf.shape[1])
        if cap > W:
            buf[:, W:cap] = buf[:, 0:cap - W].clone()
        buf[:, 0:W] = torch.flip(ideas, dims=[1])
        self._depth = self._depth + W
        self._max_depth_host = self._max_depth_host + int(W)

    def pop(self, b):
        # Newest-at-slot-0: pop slot 0, shift the rest LEFT.
        depth = int(self._depth[b].item())
        if depth == 0:
            return None
        buf = self._buffer
        idea = buf[b, 0].clone()
        if depth > 1:
            buf[b, 0:depth - 1] = buf[b, 1:depth].clone()
        buf[b, depth - 1].zero_()
        self._depth[b] = depth - 1
        return idea

    def peek(self, b, n=0):
        # Newest-at-slot-0: the n-th-from-newest item is at slot n.
        depth = int(self._depth[b].item())
        if depth <= n:
            return None
        return self._buffer[b, n]

    def snapshot(self, detach=False):
        buf = self._buffer
        if buf is None:
            return None
        B = int(buf.shape[0])
        if B == 0:
            return None
        max_depth = int(self._max_depth_host)
        if max_depth == 0:
            return None
        cap = int(self.capacity)
        if max_depth > cap:
            max_depth = cap
        snap = buf[:, :max_depth, :]
        if detach:
            snap = snap.detach().clone()
        else:
            snap = snap.clone()
        return snap

    def size(self, b):
        return int(self._depth[b].item())

    def is_full(self, b):
        return self.size(b) >= self.capacity

    def is_empty(self, b):
        return self.size(b) == 0

    def clear(self, b=None):
        buf = self._buffer
        if buf is None:
            return
        if b is None:
            buf.zero_()
            self._depth.zero_()
            self._max_depth_host = 0
            return
        b = int(b)
        if b < 0 or b >= int(buf.shape[0]):
            return
        buf[b].zero_()
        self._depth[b] = 0
        self._max_depth_host = int(self._depth.max().item())

    # -- STM shift/reduce driver (formerly stm_driver.STMDriver) -----------

    def init_scorer(self, rule_signatures, payload_dim, hidden_dim=None):
        """Allocate the rule scorer for the STM driver path. Called by
        ``SymbolSubSpace._init_stm_driver`` once the knowledge artifact's
        rule signature list is known. Idempotent — re-calling with the
        same shape is a no-op; a shape change replaces the scorer.
        """
        n_rules = len(rule_signatures)
        if (self.scorer is not None
                and getattr(self.scorer, 'payload_dim', None) == int(payload_dim)
                and getattr(self.scorer, 'n_rules', None) == n_rules):
            object.__setattr__(self, 'rule_signatures', list(rule_signatures))
            return
        self.scorer = _RuleScorer(
            payload_dim=int(payload_dim), n_rules=n_rules,
            hidden_dim=hidden_dim)
        object.__setattr__(self, 'rule_signatures', list(rule_signatures))

    def shift(self, word_subspace, b, payload, *, category, order, ref_id):
        """SHIFT one token onto ``word_subspace``'s typed STM at row ``b``.

        Mirrors the old ``STMDriver.shift``; the operand state now lives
        on the SymbolSubSpace's typed buffers (Phase D of the refactor).
        """
        word_subspace.push(
            b, payload,
            category_id_str=category,
            order=int(order),
            ref_id=int(ref_id))

    def reduce_step(self, word_subspace, b):
        """Pick the highest-scoring admissible rule for row ``b``.

        Returns ``{'rule_index': int, 'rule_signature': dict}``.
        Raises ``RuntimeError`` when no rule is admissible.
        """
        out = self._score_reduce(word_subspace, b)
        rule_index = int(torch.argmax(out['masked_logits']).item())
        return {
            'rule_index': rule_index,
            'rule_signature': self.rule_signatures[rule_index],
        }

    def reduce_step_soft(self, word_subspace, b):
        """Same as ``reduce_step`` but additionally returns the
        post-softmax probability distribution over rules. Inadmissible
        rules carry probability 0.

        NaN softmax probabilities raise ``ValueError`` per the
        "fail loud on numerical divergence" rule.
        """
        out = self._score_reduce(word_subspace, b)
        masked_logits = out['masked_logits']
        probs = torch.softmax(masked_logits, dim=-1)
        if torch.isnan(probs).any():
            raise ValueError(
                "ShortTermMemory.reduce_step_soft: NaN in softmax "
                "probabilities")
        rule_index = int(torch.argmax(masked_logits).item())
        return {
            'rule_index': rule_index,
            'rule_signature': self.rule_signatures[rule_index],
            'probabilities': probs,
            'masked_logits': masked_logits,
            'admissibility_mask': out['mask'],
        }

    def _score_reduce(self, word_subspace, b):
        """Shared scoring kernel for the hard and soft reduce paths.

        Builds the admissibility mask via ``word_subspace``'s typed
        STM, forwards the scorer over the operand payloads, applies
        the mask to the logits.
        """
        from embed import mask_logits as _mask_logits
        if self.scorer is None or self.rule_signatures is None:
            raise RuntimeError(
                "ShortTermMemory._score_reduce: scorer not initialised. "
                "Call ``init_scorer(rule_signatures, payload_dim)`` "
                "first.")
        mask = word_subspace.reduce_admissibility(b, self.rule_signatures)
        if not mask.any():
            raise RuntimeError(
                "ShortTermMemory: no admissible rule for current "
                "stack top.")
        d = int(word_subspace._depth[b].item())
        right = word_subspace.top(b, k=1)['payload'] if d >= 1 else None
        left = word_subspace.top(b, k=2)['payload'] if d >= 2 else right
        if d == 1:
            logits = self.scorer(left, None)
        else:
            logits = self.scorer(left, right)
        masked = _mask_logits(logits, mask)
        return {
            'mask': mask,
            'logits': logits,
            'masked_logits': masked,
            'left': left,
            'right': right,
        }

    def train_scorer_step(self, word_subspace, input_vectors,
                          target_rule_ids, *, snap_fn, optimizer=None):
        """Supervised training step for the rule scorer (formerly
        ``stm_trainer.train_step``).

        Replays the SHIFT/REDUCE loop on ``input_vectors[0]``
        (single-row, batch 1) and accumulates cross-entropy loss
        between ``softmax(masked_logits)`` and the next
        ``target_rule_ids[i]`` at each REDUCE position. When
        ``optimizer`` is given, steps it after backward. Returns the
        scalar loss tensor.
        """
        if input_vectors.ndim != 3 or input_vectors.shape[0] < 1:
            raise ValueError(
                "train_scorer_step: expected input_vectors of shape "
                "[B, N, D]")
        if optimizer is not None:
            optimizer.zero_grad()
        # Reset row 0 of word_subspace's typed STM.
        while int(word_subspace._depth[0].item()) > 0:
            word_subspace.pop(0)
        N = int(input_vectors.shape[1])
        reduces_done = 0
        losses = []
        max_reduces = max(1, N * 3 + 4)
        for n in range(N):
            payload = input_vectors[0, n]
            ref_id, category, order = snap_fn(payload)
            self.shift(word_subspace, 0, payload,
                       category=category, order=order, ref_id=ref_id)
            while reduces_done < max_reduces:
                try:
                    score_out = self._score_reduce(word_subspace, 0)
                except RuntimeError:
                    break
                if reduces_done >= len(target_rule_ids):
                    break
                target_id = int(target_rule_ids[reduces_done])
                mask = score_out['mask']
                if not bool(mask[target_id].item()):
                    raise ValueError(
                        f"train_scorer_step: target rule_id={target_id} "
                        "is inadmissible at REDUCE position "
                        f"{reduces_done}. Supervision is inconsistent "
                        "with the typed admissibility mask.")
                masked_logits = score_out['masked_logits']
                log_probs = F.log_softmax(masked_logits, dim=-1)
                loss = -log_probs[target_id]
                losses.append(loss)
                # Apply the target rule to advance the stack (not the
                # argmax — supervision drives the trajectory).
                sig = self.rule_signatures[target_id]
                arity = len(sig.get('rhs_categories', ()))
                d_after_shift = int(word_subspace._depth[0].item())
                if arity == 2 and d_after_shift >= 2:
                    right = word_subspace.pop(0)
                    left = word_subspace.pop(0)
                    parent_payload = (
                        left['payload'] + right['payload']) / 2.0
                elif arity == 1 and d_after_shift >= 1:
                    only = word_subspace.pop(0)
                    parent_payload = only['payload']
                else:
                    break
                word_subspace.push(
                    0, parent_payload,
                    category_id_str=str(sig.get('lhs_category', 'UNK')),
                    order=int(sig.get('lhs_order', 0)), ref_id=-1)
                reduces_done += 1
        # Final cleanup: REDUCE until depth==1 or out of targets.
        while reduces_done < min(max_reduces, len(target_rule_ids)):
            if int(word_subspace._depth[0].item()) <= 1:
                break
            try:
                score_out = self._score_reduce(word_subspace, 0)
            except RuntimeError:
                break
            target_id = int(target_rule_ids[reduces_done])
            mask = score_out['mask']
            if not bool(mask[target_id].item()):
                raise ValueError(
                    f"train_scorer_step: target rule_id={target_id} is "
                    "inadmissible during cleanup pass.")
            masked_logits = score_out['masked_logits']
            log_probs = F.log_softmax(masked_logits, dim=-1)
            loss = -log_probs[target_id]
            losses.append(loss)
            sig = self.rule_signatures[target_id]
            arity = len(sig.get('rhs_categories', ()))
            d_now = int(word_subspace._depth[0].item())
            if arity == 2 and d_now >= 2:
                right = word_subspace.pop(0)
                left = word_subspace.pop(0)
                parent_payload = (
                    left['payload'] + right['payload']) / 2.0
            elif arity == 1 and d_now >= 1:
                only = word_subspace.pop(0)
                parent_payload = only['payload']
            else:
                break
            word_subspace.push(
                0, parent_payload,
                category_id_str=str(sig.get('lhs_category', 'UNK')),
                order=int(sig.get('lhs_order', 0)), ref_id=-1)
            reduces_done += 1
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        total = torch.stack(losses).sum()
        if optimizer is not None:
            total.backward()
            optimizer.step()
        return total

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

    # ---- Evaluation chart (MeronomySpec §3; math kernel, not a grammar
    # op -- permitted on frozen Ops per the class-docstring layout note).

    @staticmethod
    def eval_chart(a):
        """χ(a) = (1 + a) / 2: epistemic scalar → assumed membership.

        The evaluation map of MeronomySpec §3 -- beliefs cash into
        extents when they become membership-fold operands. χ is
        EVALUATION, not injection: no coordinate-wise map from PS
        memberships into a signed cube exists anywhere in the
        architecture. Corners are exact: ``+1 ↦ 1`` (certain-true),
        ``0 ↦ ½`` (Boole's maximal-vagueness point), ``−1 ↦ 0``
        (certain-false).
        """
        t = torch.as_tensor(a)
        return (1 + t) / 2

    @staticmethod
    def eval_chart_inv(m):
        """χ⁻¹(m) = 2m − 1: assumed membership → epistemic scalar.

        Exact inverse of :func:`eval_chart`; affine, total, no
        rectification -- the old ReLU-injection law ``max(0, 2m−1)``
        is retired (the Stage-4 factored interface replaces it; see
        test_meronomy_laws.py's guard).
        """
        t = torch.as_tensor(m)
        return 2 * t - 1

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

    # ---- Soft (LogSumExp) RadMin / RadMax -------------------------------
    # 2026-05-29: LSE is the canonical smooth approximation of ``max`` --
    # ``max(x) <= tau * LSE(x/tau) <= max(x) + tau * log(n)``; the
    # gradient is the softmax (every input receives non-zero gradient
    # proportional to its softmax weight). As ``tau -> 0`` the soft
    # form approaches the hard ``_radmax`` / ``_radmin`` exactly; at
    # training-meaningful ``tau`` (default 0.1) both operands receive
    # gradient through the magnitude pool, which the hard form denies
    # to the per-cell loser.

    @staticmethod
    def _soft_radmax(x, y, tau=0.1):
        """LSE smooth same-sign maximum magnitude. See ``_radmax`` for
        the hard form; this is the gradient-friendly drop-in.

        ``soft_max_mag = tau * LSE([|x|/tau, |y|/tau])``. The
        zero-passthrough tail of the hard form is preserved so the
        function still degenerates correctly when one operand is
        exactly zero.
        """
        same_sign = (x * y > 0).float()
        sx, sy = torch.abs(x), torch.abs(y)
        stacked = torch.stack([sx / tau, sy / tau], dim=-1)
        soft_max_mag = tau * torch.logsumexp(stacked, dim=-1)
        core = same_sign * torch.sign(x) * soft_max_mag
        x_zero = (x == 0).float()
        y_zero = (y == 0).float()
        return core + x_zero * y + y_zero * x

    @staticmethod
    def _soft_radmin(x, y, tau=0.1):
        """LSE smooth same-sign minimum magnitude. Mirror of
        ``_soft_radmax`` for the conjunction kernel.

        ``soft_min_mag = -tau * LSE([-|x|/tau, -|y|/tau])`` (the
        canonical soft-min via negation of LSE on negatives).
        """
        same_sign = (x * y > 0).float()
        sx, sy = torch.abs(x), torch.abs(y)
        stacked = torch.stack([-sx / tau, -sy / tau], dim=-1)
        soft_min_mag = -tau * torch.logsumexp(stacked, dim=-1)
        return same_sign * torch.sign(x) * soft_min_mag

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
            monotonic=False → kind='soft'   (LSE-smoothed RadMin)
        2026-05-29: the non-monotonic kind was flipped from 'radial'
        (hard RadMin -- gradient routes only to the winning operand
        per cell) to 'soft' (LSE smooth min; both operands receive
        softmax-weighted gradient). LSE is the canonical smooth-max
        approximation in the optimization literature; the soft form
        approaches the hard one as ``tau -> 0`` while remaining
        differentiable everywhere at finite ``tau``.
        """
        kind = 'strict' if monotonic else 'soft'
        return Ops._lower_kernel(x, y, mode='AND', kind=kind)

    @staticmethod
    def _disjunction_kernel(x, y, monotonic=False):
        """Disjunction (union). Domain/range [-1, 1].

        Thin forwarder to Ops._lift_kernel(mode='OR'):
            monotonic=True  → kind='strict' (lattice max)
            monotonic=False → kind='soft'   (LSE-smoothed RadMax)
        2026-05-29: the non-monotonic kind was flipped from 'radial'
        (hard RadMax -- gradient routes only to the winning operand
        per cell, leaving the other operand with ``.grad == 0``) to
        'soft' (LSE smooth max; both operands receive softmax-weighted
        gradient). The failing
        ``test_union_grad_flows_to_both_children`` regression was the
        gating reason for the flip.
        """
        kind = 'strict' if monotonic else 'soft'
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
            Completes the S-space_role trinity partition of unity:
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
    def conjunctionReverse(result, y, W, monotonic=False, unit_ball=False,
                           left_rows=None, right_rows=None,
                           left_priming=None, right_priming=None):
        """Inverse-recommend ``(x1, x2)`` for ``conjunction(x1, x2) ≈ result``.

        Replaces the prior brute-force codebook search (which returned
        only the best left operand) with the mereology-guided
        recommender in :func:`Ops._binary_op_recommend`. Returns the
        pair drawn from the augmented codebook (learned ``W`` plus the
        synthetic ⊥ / ⊤ sentinels) — see that function for the
        algorithm. ``y`` is ignored; kept for backward-compatible
        signature. Falls back to ``(result, result)`` if ``W`` is None
        or empty.

        ``left_rows`` / ``right_rows`` (optional ``LongTensor``):
            restricts which learned-codebook rows are eligible for
            ``x1`` / ``x2`` selection. Forwarded to
            ``_binary_op_recommend``. Sentinels remain feasible
            regardless.

        ``left_priming`` / ``right_priming`` (optional ``FloatTensor``):
            soft boost-above-unity weights per W-row, biasing argmin /
            argmax toward primed candidates. Default 1.0 (multiplicative
            identity, no bias). See ``_binary_op_recommend``.
        """
        return Ops._binary_op_inverse_impl(
            result, W, 'intersection', monotonic, unit_ball=unit_ball,
            left_rows=left_rows, right_rows=right_rows,
            left_priming=left_priming, right_priming=right_priming)

    @staticmethod
    def disjunctionReverse(result, y, W, monotonic=False, unit_ball=False,
                           left_rows=None, right_rows=None,
                           left_priming=None, right_priming=None):
        """Inverse-recommend ``(x1, x2)`` for ``disjunction(x1, x2) ≈ result``.

        Replaces the prior brute-force codebook search (which returned
        only the best left operand) with the mereology-guided
        recommender in :func:`Ops._binary_op_recommend`. Returns the
        pair drawn from the augmented codebook (learned ``W`` plus the
        synthetic ⊥ / ⊤ sentinels) — see that function for the
        algorithm. ``y`` is ignored; kept for backward-compatible
        signature. Falls back to ``(result, result)`` if ``W`` is None
        or empty.

        ``left_rows`` / ``right_rows`` (optional ``LongTensor``):
            restricts which learned-codebook rows are eligible for
            ``x1`` / ``x2`` selection. Forwarded to
            ``_binary_op_recommend``.

        ``left_priming`` / ``right_priming`` (optional ``FloatTensor``):
            soft boost-above-unity weights per W-row, biasing argmin /
            argmax toward primed candidates. Default 1.0 (multiplicative
            identity, no bias). See ``_binary_op_recommend``.
        """
        return Ops._binary_op_inverse_impl(
            result, W, 'union', monotonic, unit_ball=unit_ball,
            left_rows=left_rows, right_rows=right_rows,
            left_priming=left_priming, right_priming=right_priming)

    @staticmethod
    def _binary_op_inverse_impl(result, W, op, monotonic, unit_ball=False,
                                left_rows=None, right_rows=None,
                                left_priming=None, right_priming=None):
        """Backward-compatible shim for the union / intersection inverse.

        Delegates to :func:`Ops._binary_op_recommend`. Returns the pair
        ``(x1, x2)`` drawn from the augmented codebook. Falls back to
        ``(result, result)`` when ``W`` is None / empty.

        ``op`` may be the string ``'union'`` / ``'intersection'``, or one
        of the legacy callable handles ``Ops.conjunction`` (→ intersection)
        / ``Ops.disjunction`` (→ union). ``unit_ball`` is accepted for
        signature parity with the prior implementation but the new
        mereology-guided recommender is metric-free; the flag is
        currently ignored.

        ``left_rows`` / ``right_rows`` (optional ``LongTensor``):
            forwarded to ``_binary_op_recommend`` to restrict the W-row
            candidate set for each operand. See that function's
            docstring for the typed restricted-candidate inverse story.
        """
        if W is None or (hasattr(W, 'shape') and W.shape[0] == 0):
            return result, result
        op_name = Ops._normalize_lattice_op(op)
        return Ops._binary_op_recommend(
            result, W, op_name, monotonic=monotonic,
            left_rows=left_rows, right_rows=right_rows,
            left_priming=left_priming, right_priming=right_priming)

    @staticmethod
    def _normalize_lattice_op(op):
        """Resolve ``op`` to ``'union'`` or ``'intersection'``.

        Accepts the string names directly, the public callables
        ``Ops.conjunction`` / ``Ops.disjunction`` (or their kernels), and
        the symmetric ``Ops.intersection`` / ``Ops.union`` aliases. Any
        other handle raises ``ValueError`` — the recommender is monotonic
        lattice-only.
        """
        if isinstance(op, str):
            name = op
        else:
            name = getattr(op, 'rule_name', None)
            if name is None:
                name = getattr(op, '__name__', '') or ''
            name = name.replace('_kernel', '').replace('_', '')
        if name in ('conjunction', 'intersection'):
            return 'intersection'
        if name in ('disjunction', 'union'):
            return 'union'
        raise ValueError(
            f"Ops._binary_op_inverse_impl: unsupported op {op!r} "
            "(expected union / intersection / conjunction / disjunction)")

    @staticmethod
    def _binary_op_recommend(result, W, op_name, monotonic=True,
                             left_rows=None, right_rows=None,
                             left_priming=None, right_priming=None):
        """Inverse-recommend a pair ``(x1, x2)`` from an augmented codebook.

        ``op_name``:
          * ``'union'``        — ``y = max(x1, x2)``; pick ``x1`` as the
            largest part ``≤ y`` then ``x2`` as the smallest part
            ``≥ y − x1``.
          * ``'intersection'`` — ``y = min(x1, x2)``; pick ``x1`` as the
            smallest part ``≥ y``, then ``x2`` as the smallest part
            ``≥ y`` distinct from ``x1`` and minimizing
            ``Ops.overlapOf(x2, x1)`` (tie-break: smaller ``norm(x2)``).

        ``W`` is the learned codebook (``[K, D]``); we augment it
        non-mutatingly with sentinels ⊥ = zeros and ⊤ = ones so the
        ordering filters are never empty. Result shape ``(..., D)`` is
        preserved; the two returned tensors are shaped like ``result``.
        Size metric is the canonical last-dim L2 ``Ops.norm``.

        ``monotonic`` is accepted for signature symmetry; the algorithm
        is monotonic-only by construction (the codebase exposes no
        bitonic inverse for these ops).

        ``left_rows`` / ``right_rows`` (optional ``LongTensor``):
            indices into ``W`` (0..K-1) restricting which learned
            codebook rows may be selected for ``x1`` / ``x2``
            respectively. The ⊥ / ⊤ sentinels remain feasible
            regardless of restriction (so the algorithm cannot fail to
            return a pair). When both are ``None`` (default) behavior
            matches the unrestricted algorithm byte-for-byte.

            Used by the typed restricted-candidate inverse for
            lift/lower: the call site computes the intersection
            ``refs_by_category[cat] ∩ refs_by_order[k]`` from the
            attached ``KnowledgeView`` and passes the result, masking
            argmin/argmax to admissible refs only. Plan: §Lift/Lower
            Restricted-Candidate Inverse.

        ``left_priming`` / ``right_priming`` (optional ``FloatTensor``,
        length ``K``):
            soft boost-above-unity weights per W-row (default 1.0 =
            multiplicative identity). For argmax steps, scores are
            multiplied by the weight (primed rows preferred). For
            argmin steps, scores are divided by the weight (primed
            rows look smaller, hence preferred). The ⊥ / ⊤ sentinels
            are pinned to 1.0 regardless of input. When both are
            ``None`` (default), behavior matches the un-primed
            algorithm byte-for-byte.

            Plan: doc/plans/2026-05-20-primed-reverse-generation.md
            §Application: multiplied onto .activation (A. selection).
        """
        if W is None or W.shape[0] == 0:
            return result, result

        K, D = W.shape
        bottom = torch.zeros(1, D, dtype=W.dtype, device=W.device)
        top = torch.ones(1, D, dtype=W.dtype, device=W.device)
        C = torch.cat([bottom, W, top], dim=0)         # [K+2, D]
        K_aug = C.shape[0]

        out_shape = result.shape
        flat = result.reshape(-1, D)                    # [N, D]
        N = flat.shape[0]

        # Broadcast each result position against the augmented codebook.
        y = flat.unsqueeze(1)                           # [N, 1, D]
        S = C.unsqueeze(0).expand(N, K_aug, D)          # [N, K, D]

        sizes = Ops.norm(C)                             # [K]
        NEG_INF = torch.tensor(
            float('-inf'), dtype=W.dtype, device=W.device)
        POS_INF = torch.tensor(
            float('inf'), dtype=W.dtype, device=W.device)

        # Build per-operand candidate masks over the augmented C-rows.
        # ⊥ (index 0) and ⊤ (index K_aug-1) are always feasible. W-row
        # indices from ``left_rows`` / ``right_rows`` shift by +1 to
        # account for the ⊥ prefix.
        def _row_mask(rows):
            if rows is None:
                return torch.ones(K_aug, dtype=torch.bool, device=W.device)
            m = torch.zeros(K_aug, dtype=torch.bool, device=W.device)
            m[0] = True
            m[K_aug - 1] = True
            if rows.numel() > 0:
                c_idx = rows.long().to(W.device) + 1
                c_idx = c_idx[(c_idx >= 1) & (c_idx <= K)]
                if c_idx.numel() > 0:
                    m[c_idx] = True
            return m
        left_mask = _row_mask(left_rows)
        right_mask = _row_mask(right_rows)

        # Build per-operand priming weights over the augmented C-rows.
        # Default = 1.0 everywhere (multiplicative identity, no bias).
        # ⊥ (index 0) and ⊤ (index K_aug-1) are pinned to 1.0 regardless
        # of caller-supplied weights. W-row weights at index i land at
        # C-row index i+1 to account for the ⊥ prefix.
        def _row_weights(priming):
            w = torch.ones(K_aug, dtype=W.dtype, device=W.device)
            if priming is None:
                return w
            p = torch.as_tensor(priming, dtype=W.dtype, device=W.device)
            if p.ndim != 1:
                p = p.reshape(-1)
            # Truncate or pad to K (W's row count); ignore extras.
            k_in = min(int(p.shape[0]), K)
            if k_in > 0:
                w[1:1 + k_in] = p[:k_in]
            return w
        left_w = _row_weights(left_priming)
        right_w = _row_weights(right_priming)

        if op_name == 'union':
            # 1) x1 = argmax_{S ≤ y, S in left_rows ∪ sentinels}
            #        norm(S) * priming(S)
            #    (argmax: multiply by priming so primed rows preferred)
            le_y = (S <= y).all(dim=-1) & left_mask.unsqueeze(0)  # [N, K]
            primed_sizes = sizes * left_w               # [K_aug]
            scores = torch.where(
                le_y, primed_sizes.unsqueeze(0).expand(N, K_aug), NEG_INF)
            idx1 = scores.argmax(dim=-1)                # [N]
            x1 = C[idx1]                                # [N, D]

            # 2) residual r = y − x1
            r = flat - x1                               # [N, D]

            # 3) x2 = argmin_{S ≥ r, S in right_rows ∪ sentinels}
            #        norm(S) / priming(S)
            #    (argmin: divide by priming so primed rows look smaller)
            r_b = r.unsqueeze(1)                        # [N, 1, D]
            ge_r = (S >= r_b).all(dim=-1) & right_mask.unsqueeze(0)
            primed_sizes_r = sizes / right_w            # [K_aug]
            scores = torch.where(
                ge_r, primed_sizes_r.unsqueeze(0).expand(N, K_aug), POS_INF)
            idx2 = scores.argmin(dim=-1)                # [N]
            x2 = C[idx2]                                # [N, D]

            return x1.reshape(out_shape), x2.reshape(out_shape)

        if op_name == 'intersection':
            # 1) x1 = argmin_{S ≥ y, S in left_rows ∪ sentinels}
            #        norm(S) / priming(S)
            ge_y = (S >= y).all(dim=-1)                 # [N, K]
            ge_y_left = ge_y & left_mask.unsqueeze(0)
            primed_sizes_l = sizes / left_w             # [K_aug]
            scores = torch.where(
                ge_y_left, primed_sizes_l.unsqueeze(0).expand(N, K_aug), POS_INF)
            idx1 = scores.argmin(dim=-1)                # [N]
            x1 = C[idx1]                                # [N, D]

            # 2) x2 = argmin_{S ≥ y, S ≠ x1, S in right_rows ∪ sentinels}
            #    (overlapOf(S, x1).norm + tie) / priming(S)
            same_as_x1 = (
                torch.arange(K_aug, device=W.device).unsqueeze(0)
                == idx1.unsqueeze(1))                   # [N, K]
            feasible = ge_y & ~same_as_x1 & right_mask.unsqueeze(0)
            x1_b = x1.unsqueeze(1).expand(N, K_aug, D)  # [N, K, D]
            overlap = Ops._radmin(S, x1_b)              # [N, K, D]
            overlap_size = Ops.norm(overlap)            # [N, K]
            tie_weight = sizes.unsqueeze(0).expand(N, K_aug)
            primary = overlap_size + 1e-9 * tie_weight
            primary = primary / right_w.unsqueeze(0)    # primed → smaller
            scores = torch.where(feasible, primary, POS_INF)
            # If no S ≠ x1 is feasible (degenerate codebook), fall back
            # to x1 itself for x2 (still satisfies S ≥ y).
            any_feasible = feasible.any(dim=-1)         # [N]
            idx2 = scores.argmin(dim=-1)                # [N]
            idx2 = torch.where(any_feasible, idx2, idx1)
            x2 = C[idx2]                                # [N, D]

            return x1.reshape(out_shape), x2.reshape(out_shape)

        raise ValueError(
            f"Ops._binary_op_recommend: unsupported op_name {op_name!r} "
            "(expected 'union' or 'intersection')")

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
            if kind == 'soft':
                # LSE smooth same-sign max magnitude. Gradient-friendly
                # drop-in for 'radial'; both operands receive non-zero
                # softmax-weighted gradient per cell.
                return Ops._soft_radmax(X1, X2)
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
    def liftReverseAll(Y, W=None, monotonic=False,
                       left_rows=None, right_rows=None,
                       left_priming=None, right_priming=None):
        """Multi-return reverse for the lift dispatcher (Step 6).

        Pairs with the parent plan's Layer-2.5 grammar convention
        ``X1, X2 = liftReverse(Y)``. For mode='OR' (the analysis-of-
        synthesis direction) the inverse is the mereology-guided
        recommender ``Ops.disjunctionReverse``; with W supplied, returns
        the recommended pair ``(x1, x2)`` such that
        ``disjunction(x1, x2) ≈ Y`` (drawn from the augmented codebook;
        see ``Ops._binary_op_recommend``). Without W, returns ``(Y, Y)``
        so callers get a tuple of the expected shape.

        This is the new multi-return convention; the existing 2-arg
        ``Ops.liftReverse(result, right)`` remains for analytic-inverse
        callers (it returns a single tensor, not a tuple).

        ``left_rows`` / ``right_rows`` and ``left_priming`` /
        ``right_priming`` are forwarded to ``disjunctionReverse``; see
        ``Ops._binary_op_recommend`` for semantics.
        """
        if W is None or (hasattr(W, 'shape') and W.shape[0] == 0):
            return (Y, Y)
        return Ops.disjunctionReverse(
            Y, Y, W, monotonic=monotonic,
            left_rows=left_rows, right_rows=right_rows,
            left_priming=left_priming, right_priming=right_priming)

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
            if kind == 'soft':
                # LSE smooth same-sign min magnitude. Gradient-friendly
                # drop-in for 'radial'; both operands receive non-zero
                # softmax-weighted gradient per cell.
                return Ops._soft_radmin(X1, X2)
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
    def lowerReverseAll(Y, W=None, monotonic=False,
                        left_rows=None, right_rows=None,
                        left_priming=None, right_priming=None):
        """Multi-return reverse for the lower dispatcher (Step 6).

        Pairs with the parent plan's Layer-2.5 grammar convention
        ``X1, X2 = lowerReverse(Y)``. For mode='AND' (the synthesis-of-
        analysis direction) the inverse is the mereology-guided
        recommender ``Ops.conjunctionReverse``; with W supplied, returns
        the recommended pair ``(x1, x2)`` such that
        ``conjunction(x1, x2) ≈ Y`` (drawn from the augmented codebook;
        see ``Ops._binary_op_recommend``). Without W, returns ``(Y, Y)``
        so callers get a tuple of the expected shape.

        See ``Ops.liftReverseAll`` for the dual.

        ``left_rows`` / ``right_rows`` and ``left_priming`` /
        ``right_priming`` are forwarded to ``conjunctionReverse``; see
        ``Ops._binary_op_recommend`` for semantics.
        """
        if W is None or (hasattr(W, 'shape') and W.shape[0] == 0):
            return (Y, Y)
        return Ops.conjunctionReverse(
            Y, Y, W, monotonic=monotonic,
            left_rows=left_rows, right_rows=right_rows,
            left_priming=left_priming, right_priming=right_priming)

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

    @staticmethod
    def _order_align(a, b):
        """Broadcast ``a`` / ``b`` to a common last-dim shape.

        This codebase carries no formal multivector grade ladder; the only
        "orders" present are post-codebook scalar ``[...]`` vs. bivector
        ``[..., 2]``. Align by upcasting a scalar (last dim 1 or missing)
        to a bivector via the ``[x, x]`` lift, else broadcast on the last
        dim. Used by ``partOf`` / ``wholeOf`` / ``overlapOf`` so callers
        don't have to harmonise shapes manually.
        """
        D_a = a.shape[-1]
        D_b = b.shape[-1]
        if D_a == D_b:
            return a, b
        if D_a == 1 and D_b > 1:
            return a.expand(*a.shape[:-1], D_b), b
        if D_b == 1 and D_a > 1:
            return a, b.expand(*b.shape[:-1], D_a)
        raise ValueError(
            f"Ops._order_align: incompatible last dims {D_a} vs {D_b}")

    @staticmethod
    def partOf(S1, S2):
        """Elementwise ``S1 ≤ S2`` reduced over the last dim with ``all``.

        Mereological "is part of": True iff every component of S1 is
        bounded above by the matching component of S2. Order-aligned
        first so scalar and bivector operands compose. Returns a boolean
        tensor shaped like ``S1.shape[:-1]`` (broadcast over S2).
        """
        a, b = Ops._order_align(S1, S2)
        return (a <= b).all(dim=-1)

    @staticmethod
    def wholeOf(S1, S2):
        """Elementwise ``S1 ≥ S2`` reduced over the last dim with ``all``.

        Mereological "contains": True iff every component of S1 is
        bounded below by the matching component of S2. Order-aligned
        first.
        """
        a, b = Ops._order_align(S1, S2)
        return (a >= b).all(dim=-1)

    @staticmethod
    def overlapOf(S1, S2):
        """Elementwise zero-directed min — same semantics as ``Ops._radmin``.

        Returns a tensor shaped like the broadcast of S1 and S2, holding
        the same-sign minimum magnitude (zero collapse on sign
        disagreement). Used by the binary-op inverse recommender to score
        candidate operand pairs by their mutual overlap.
        """
        a, b = Ops._order_align(S1, S2)
        return Ops._radmin(a, b)
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
                 subsymbolicOrder=0,
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
        # Loss operates on the OUTPUT space_role, which carries no where/when in the
        # converged modality architecture; source the what/where/when split
        # widths from canonical_shape("OutputSpace"). An explicit positive arg
        # still wins (e.g. for unit tests that drive the where/when loss terms).
        from architecture import canonical_shape
        _os_where, _os_when = canonical_shape("OutputSpace")
        self.nWhere = nWhere if (nWhere is not None and nWhere > 0) else _os_where
        self.nWhen = nWhen if (nWhen is not None and nWhen > 0) else _os_when

        if certainty:
            self.output_criterion = CertaintyWeightedCrossEntropy()
        elif nOutput <= 2:
            self.output_criterion = nn.MSELoss()
        elif subsymbolicOrder > 0:
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
            # guarantee without that host sync on CUDA: the assertion is
            # enqueued on the stream (no DtoH) and a non-finite loss
            # aborts the process at the next kernel launch -- unmissable,
            # never a silent success; on CPU it asserts immediately
            # (tests still catch divergence).
            #
            # MPS exception: ``torch._assert_async`` is not implemented
            # for the MPS backend as of 2026-05-27 (PyTorch issue #141287).
            # Reduce on CPU, not MPS: radix byte-mode can corrupt MPS
            # scalar reductions, causing finite fp32 losses to report
            # non-finite. This remains syncful on Apple Silicon, but
            # preserves the fail-loud contract without trusting the
            # affected backend reduction. The compiled / CUDA paths remain
            # sync-free.
            if lossIn.device.type == "mps":
                if not bool(torch.isfinite(lossIn.detach().cpu()).all()):
                    raise RuntimeError(
                        "ModelLoss.forward: non-finite reconstruction "
                        "loss (Inf/NaN) -- divergence. Fail-loud per "
                        "project numerical contract; do NOT nan_to_num.")
            else:
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
    ``WholeSpace.accumulate_symbol_objective``, ``SymbolSpace.truth_modulated_loss``).
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
                      space="WholeSpace", category="symbol")
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
#     differ. Right metric for PartSpace and WholeSpace.
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
        learnable_codebook=False,
        ema_update=True,
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
        # Learnable-codebook mode (Phase 1A.1 -- perceptual codebook).
        # When True the in-call EMA Parameter mutation is suppressed (the
        # codebook is no longer rewritten in ``forward``) and the STE is
        # switched to the codebook-attached form so the codebook
        # ``nn.Parameter`` receives the downstream task-loss gradient and
        # is trained purely by ``optimizer.step`` (eager, OUTSIDE the
        # traced/compiled forward). This makes ``forward`` idempotent:
        # no persistent buffer/Parameter is mutated in-call. Default
        # False preserves the byte-identical EMA path used by the
        # Conceptual/Symbolic codebooks (single-writer invariant).
        self.learnable_codebook = bool(learnable_codebook)
        # Asymmetric VQ (asymmetric-vq plan sec.3, rev. 2026-06-09): master
        # gate for the in-forward EMA codebook update. Constructor parameter,
        # default True (the standard VQ-VAE behavior). The WholeSpace and
        # category VQs pass False -- EMA is a single-objective crutch replaced
        # by the reconstruction gradient on the codebook (the input->codebook
        # leg of the asymmetric routing). Distinct from ``learnable_codebook``
        # (which also flips the in-forward gradient estimator).
        self.ema_update = bool(ema_update)
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

        # Intent-priming selection boost (GrammarOpsPass §5; author
        # 2026-06-11). An installed ``selection_boost_fn`` returns
        # boost-above-unity row weights ``[V]`` (or None = identity),
        # produced from the single current intent's graded similarity
        # against this tower's rows. The boost enters the row selection
        # as an additive LOG-boost on the scores — sign-safe in both
        # score modes, monotone in the boost, with 1.0 the exact
        # multiplicative identity. Byte-identical when no producer is
        # installed (or it returns None): primed RECOGNITION is a
        # multiplicative factor on the soft selection, never a guard.
        log_boost = None
        boost_fn = getattr(self, 'selection_boost_fn', None)
        if boost_fn is not None:
            boosts = boost_fn(V, flat.device)
            if boosts is not None:
                log_boost = torch.log(
                    boosts.to(dtype=flat.dtype, device=flat.device)
                    .clamp_min(1e-12))

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
                        scores = flat @ codebook.T
                        if log_boost is not None:
                            scores = scores + log_boost
                        indices = scores.argmax(dim=-1)
                    else:
                        parts = []
                        for s in range(0, N, chunk):
                            scores = flat[s:s+chunk] @ codebook.T
                            if log_boost is not None:
                                scores = scores + log_boost
                            parts.append(scores.argmax(dim=-1))
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
                        if log_boost is not None:
                            scores = scores + log_boost
                        indices = scores.argmax(dim=-1)
                    else:
                        parts = []
                        for s in range(0, N, chunk):
                            scores = flat[s:s+chunk] @ codebook.T
                            scores = scores - half_b_norms_sq
                            if log_boost is not None:
                                scores = scores + log_boost
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
        #
        # ``learnable_codebook`` (Phase 1A.1) suppresses this in-call
        # Parameter mutation entirely -- the exact same gate as
        # ``freeze_codebook`` -- because in that mode the codebook is
        # trained by the downstream task-loss gradient (via the
        # codebook-attached STE below) in the eager ``optimizer.step``,
        # not by an in-forward EMA write. This is what makes ``forward``
        # idempotent for the perceptual codebook.
        if (self.training and not freeze_codebook
                and not self.learnable_codebook and self.ema_update):
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
                # Reference-partitioned update law (GrammarOpsPass §6d;
                # author 2026-06-11). An installed ``update_mask_fn``
                # returns the rows ALLOWED to move this step (BoolTensor
                # [V]) or None (= all rows; the legacy path below is
                # byte-identical when no mask is installed). Frozen rows
                # keep their codebook value AND their EMA accumulators
                # (no decay, no batch mass — they simply do not
                # participate this step) and are exempt from dead-code
                # expiry, so a later mask change cannot make them jump
                # from stale statistics.
                allowed = None
                mask_fn = getattr(self, 'update_mask_fn', None)
                if mask_fn is not None:
                    allowed = mask_fn(V, flat_f.device)
                if allowed is not None:
                    allowed = allowed.to(device=self.cluster_size.device,
                                         dtype=torch.bool)
                    allow_f = allowed.to(self.cluster_size.dtype)
                    decay_vec = self.decay * allow_f + (1.0 - allow_f)
                    self.cluster_size.mul_(decay_vec).add_(
                        cluster_size_batch.to(self.cluster_size.dtype)
                        * allow_f,
                        alpha=1.0 - self.decay,
                    )
                    self.embed_avg.mul_(decay_vec.unsqueeze(-1)).add_(
                        embed_sum.to(self.embed_avg.dtype)
                        * allow_f.unsqueeze(-1),
                        alpha=1.0 - self.decay,
                    )
                else:
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
                if allowed is not None:
                    # §6d: frozen rows keep their current value exactly.
                    new_embed = torch.where(
                        allowed.unsqueeze(-1),
                        new_embed.to(self.codebook.dtype),
                        self.codebook,
                    )
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
                    if allowed is not None:
                        # §6d: frozen rows never expire or get revived.
                        expired = expired & allowed
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
        # requested and the input carries gradient, otherwise STE.
        #
        # Two STE forms, identical in FORWARD VALUE (both equal
        # ``quantized_raw``) and identical in the encoder gradient
        # (``d/dx = identity``); they differ only in whether the
        # codebook ``nn.Parameter`` receives gradient:
        #
        #   * vanilla (EMA codebooks): ``x + (quantized_raw - x).detach()``
        #     -- ``quantized_raw`` is detached, so the codebook gets NO
        #     gradient from this output (it is trained by the EMA write
        #     above instead).
        #   * learnable (Phase 1A.1, perceptual codebook):
        #     ``quantized_raw + (x - x.detach())`` -- ``quantized_raw``
        #     stays attached, so the downstream task-loss gradient flows
        #     into the selected codebook rows. With EMA suppressed this
        #     is the ONLY codebook training signal; it lands in the eager
        #     ``optimizer.step`` (outside the traced forward), so the
        #     forward performs no in-call persistent mutation.
        if self.rotation_trick and self.training and x.requires_grad:
            quantized = self._rotate_to(x, quantized_raw)
        elif self.learnable_codebook:
            quantized = quantized_raw + (x - x.detach())
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
    #
    # Post-2026-05-29 grammar-file-refactor (\xa75): the negation / non /
    # conjunction / disjunction / lift / lower / part bindings now live
    # in Language.py because those layer classes moved there. The
    # ``equal`` binding stays here because EqualLayer stays here.
    bindings = (
        ('equal',       Ops._equal_kernel,       EqualLayer),
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


# Backward-compat: grammar rule operator classes have been moved
# to Language.py. Provide module-level lazy attribute lookup so old
# `from Layers import IntersectionLayer` (and similar) continue to
# work. GRAMMAR_LAYER_CLASSES is delegated the same way. See
# doc/plans/2026-05-29-grammar-file-refactor.md §5.
_MOVED_TO_LANGUAGE = frozenset({
    'NotLayer',
    'NonLayer',
    'IntersectionLayer',
    'UnionLayer',
    'LiftLayer',
    'VerbLayer',
    'AdverbLayer',
    'LowerLayer',
    'SymbolizeLayer',
    'ConjunctionLayer',
    'DisjunctionLayer',
    'IsEqualLayer',
    'PartLayer',
    'QueryLayer',
    'GRAMMAR_LAYER_CLASSES',
})

def __getattr__(name):
    if name in _MOVED_TO_LANGUAGE:
        import Language
        return getattr(Language, name)
    raise AttributeError(
        f"module 'Layers' has no attribute {name!r}")
