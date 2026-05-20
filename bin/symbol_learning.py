"""Symbol-learning statistics + promotion policy.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 6 — scaffold; deferred §Phase 2 → Phase 3 closeout —
policy + ``extend_artifact`` integration.

Two layers, both disabled-by-default and detached from autograd:

  * ``SymbolLearningStats`` — accumulates two hook signals:
      - **Zero-order / QE** via ``observe_qe(activation, snapped)``:
        squared quantization error, plus online leader-clustering that
        produces per-cluster (centroid, qe_ema, stability) summaries.
      - **Higher-order / PMI** via ``observe_reduce(left_ref, right_ref,
        parent_ref, *, parent_scalar=None, parent_category=None,
        parent_order=None)``: pair counts + the rule's LHS metadata so
        the policy can promote a pair into a real ref at the right
        category and order.
    ``flush()`` returns a snapshot and resets state.

  * ``SymbolLearningPolicy`` — MDL-flavored trigger logic. Reads a
    flushed snapshot and emits ``embed.NewRef`` candidates:
      - **Zero-order** when a cluster has stability ≥ ``stability_n``
        and ``qe_ema ≥ qe_promote_threshold``.
      - **Higher-order** when a pair has ``log(P(a,b) / (P(a) P(b))) >
        pmi_threshold`` and ``count ≥ count_threshold``.

  * ``flush_and_promote(stats, policy, path)`` — orchestrator. Flushes
    the stats, runs the policy, and calls ``embed.extend_artifact`` if
    there are candidates. The mutation lives outside of any autograd
    forward by construction.

Numerical-divergence policy (per project memory): NaN / Inf inputs to
``observe_qe`` raise ``ValueError`` rather than getting silently
``nan_to_num``-cleaned. Drift should be loud so it can be debugged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from util import TheXMLConfig


def symbol_learning_enabled_from_config() -> bool:
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


@dataclass
class _Cluster:
    """One leader-clustering cluster: centroid + EMA of squared QE +
    stability count. Centroid is a detached 1-D tensor."""
    centroid: torch.Tensor
    qe_ema: float
    stability: int


class SymbolLearningStats:
    """Detached accumulator for symbol-learning trigger signals.

    All accumulation is plain Python state (floats, ints, dicts, +
    detached tensors for cluster centroids) so nothing leaks back into
    autograd from the hook points.

    ``enabled=False`` (the default) makes every ``observe_*`` method a
    cheap no-op.
    """

    def __init__(
        self,
        enabled: bool = False,
        *,
        leader_radius: float = 0.5,
        ema_alpha: float = 0.1,
    ):
        self.enabled = bool(enabled)
        self.leader_radius = float(leader_radius)
        self.ema_alpha = float(ema_alpha)
        self.qe_count: int = 0
        self.qe_sum_squared: float = 0.0
        self.pair_counts: Dict[Tuple[int, int], int] = {}
        self.pair_info: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._clusters: List[_Cluster] = []

    def observe_qe(self, activation: torch.Tensor,
                   snapped: torch.Tensor) -> None:
        """Record one quantization-error measurement.

        Computes squared distance between the pre-snap activation and
        the snapped prototype, updates the aggregate QE counter, and
        runs online leader clustering on the un-snapped activation.

        NaN / Inf inputs raise ``ValueError`` rather than being silently
        cleaned, per the project's "fail loud on numerical divergence"
        rule.
        """
        if not self.enabled:
            return
        if torch.isnan(activation).any() or torch.isinf(activation).any():
            raise ValueError(
                "SymbolLearningStats.observe_qe: NaN/Inf in activation")
        if torch.isnan(snapped).any() or torch.isinf(snapped).any():
            raise ValueError(
                "SymbolLearningStats.observe_qe: NaN/Inf in snapped")
        with torch.no_grad():
            act = activation.detach()
            snap = snapped.detach()
            diff = act - snap
            sq = float(diff.pow(2).sum().item())
            flat = act.reshape(-1)
        self.qe_count += 1
        self.qe_sum_squared += sq
        self._update_clusters(flat, sq)

    def _update_clusters(self, vec: torch.Tensor, sq_qe: float) -> None:
        """Online leader clustering: find the nearest existing cluster
        within ``leader_radius`` (Euclidean), update its EMA + centroid
        + stability. If none qualifies, start a new cluster."""
        if not self._clusters:
            self._clusters.append(_Cluster(
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
            self._clusters.append(_Cluster(
                centroid=vec.clone(), qe_ema=sq_qe, stability=1))
            return
        a = self.ema_alpha
        nearest.qe_ema = (1.0 - a) * nearest.qe_ema + a * sq_qe
        nearest.centroid = (1.0 - a) * nearest.centroid + a * vec
        nearest.stability += 1

    def observe_reduce(
        self,
        left_ref: int,
        right_ref: int,
        parent_ref: int,
        *,
        parent_scalar: Optional[float] = None,
        parent_category: Optional[str] = None,
        parent_order: Optional[int] = None,
    ) -> None:
        """Record one REDUCE outcome.

        ``left_ref`` / ``right_ref`` are the operand ref_ids;
        ``parent_ref`` is the result's ref_id. The optional kwargs
        carry the rule's LHS metadata so the higher-order policy can
        promote a co-occurring pair at the right category and order.
        """
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

    def flush(self) -> Dict[str, Any]:
        """Return a snapshot of accumulated stats and reset state.

        Snapshot keys:
          ``qe_count``       — aggregate observation count
          ``qe_sum_squared`` — aggregate Σ‖act − snap‖²
          ``qe_mean``        — convenience: sum_sq / count
          ``pair_counts``    — ``{(left,right) -> count}`` (shallow copy)
          ``pair_info``      — ``{(left,right) -> {count, parent_scalar_sum,
                                                   parent_category, parent_order,
                                                   parent_ref}}``
          ``clusters``       — ``[{centroid, qe_ema, stability}]``
                              detached / Python-floats
        """
        snap: Dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# Policy: turn flushed snapshots into NewRef candidates
# ---------------------------------------------------------------------------


@dataclass
class SymbolLearningPolicy:
    """MDL-flavored trigger thresholds + a ``propose_candidates`` step
    that turns a flushed ``SymbolLearningStats`` snapshot into a list of
    ``embed.NewRef`` records.

    Zero-order:
      Promote a cluster with ``stability >= stability_n`` AND
      ``qe_ema >= qe_promote_threshold`` to a new order-0 ref at
      ``default_zero_order_category`` / ``default_zero_order_parent_ref_id``.
      The cluster's centroid's scalar mean becomes the new ref's
      learned scalar.

    Higher-order:
      Promote a pair ``(a, b)`` with ``count >= count_threshold`` AND
      ``log(P(a,b) / (P(a) P(b))) >= pmi_threshold`` to a new ref at
      the recorded ``parent_category`` / ``parent_order`` (from the
      rule's LHS metadata). The new ref's scalar is the mean of the
      reduce-time parent activations.

    Promotion is "candidate-only" — ``propose_candidates`` does NOT
    call ``extend_artifact``. Use ``flush_and_promote`` to combine
    the stages.
    """

    qe_promote_threshold: float = 0.5
    stability_n: int = 10
    pmi_threshold: float = 1.0
    count_threshold: int = 5
    default_zero_order_category: str = 'NEW'
    default_zero_order_parent_ref_id: int = 0
    default_higher_order_category: str = 'NEW'
    default_higher_order_order: int = 1

    def propose_candidates(self, snapshot: Dict[str, Any]) -> List[Any]:
        """Apply MDL-flavored thresholds to ``snapshot`` and return a
        list of ``embed.NewRef`` candidates. May be empty."""
        from embed import NewRef
        candidates: List[Any] = []
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
            left_marginal: Dict[int, int] = {}
            right_marginal: Dict[int, int] = {}
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


def flush_and_promote(
    stats: SymbolLearningStats,
    policy: SymbolLearningPolicy,
    artifact_path: str,
) -> List[Any]:
    """Flush ``stats``, run ``policy``, call ``extend_artifact`` on any
    candidates. Returns the candidate list (empty when nothing crossed
    the thresholds).

    The flush boundary is the caller's responsibility — e.g., end of
    training batch. By construction this never mutates the artifact
    inside an autograd forward.
    """
    from embed import extend_artifact
    snap = stats.flush()
    candidates = policy.propose_candidates(snap)
    if candidates:
        extend_artifact(artifact_path, candidates)
    return candidates
