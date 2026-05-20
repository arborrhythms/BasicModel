"""Symbol-learning statistics scaffold.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 6.

Disabled-by-default subsystem that collects detached statistics from
two hook points:

  * **Zero-order / QE.** After conceptual activation forms (before the
    codebook snap), ``observe_qe(activation, snapped)`` accumulates
    the squared quantization error. Future symbol-learning policies
    use this to detect activation regions whose QE is consistently
    high — candidates for promotion to a new ref.

  * **Higher-order / PMI.** After an STM reduce succeeds,
    ``observe_reduce(left_ref, right_ref, parent_ref)`` accumulates
    adjacency-pair counts. Used to compute PMI for composition
    promotion ("these two refs co-occur as a unit often enough to
    deserve a single ref of their own").

This module **does not** call ``extend_artifact``. It only records
detached statistics. A separate flush boundary — explicitly invoked
at training-step boundaries by orchestration code — decides whether
the accumulated stats justify a promotion, and only then mutates the
artifact. Keeping mutations out of the autograd forward keeps gradient
flow clean.
"""
from typing import Any, Dict, Tuple

import torch

from util import TheXMLConfig


def symbol_learning_enabled_from_config() -> bool:
    """Read ``<architecture><symbolLearning enabled="..."/>`` from XML.

    Defaults to ``False`` when the key is absent. The attribute parser
    accepts ``"true"`` / ``"false"`` (case-insensitive) and ``"1"`` /
    ``"0"``. Any other value falls back to ``False``.
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


class SymbolLearningStats:
    """Detached accumulator for symbol-learning trigger signals.

    All accumulation is done as Python ``float`` / ``int`` / ``dict``
    state — no tensors in the running totals so there's no gradient
    tape leaking out of the hook points.

    ``enabled=False`` (the default) makes every ``observe_*`` method a
    cheap no-op, so leaving the scaffold instantiated has near-zero
    cost in production.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.qe_count: int = 0
        self.qe_sum_squared: float = 0.0
        self.pair_counts: Dict[Tuple[int, int], int] = {}

    def observe_qe(self, activation: torch.Tensor,
                   snapped: torch.Tensor) -> None:
        """Record one quantization-error measurement: the squared
        distance between the pre-snap activation and the snapped
        prototype. No-op when disabled.

        ``activation`` / ``snapped`` may carry autograd tape; we
        ``.detach()`` before reducing so nothing leaks into the
        statistic.
        """
        if not self.enabled:
            return
        with torch.no_grad():
            diff = (activation.detach() - snapped.detach())
            sq = float(diff.pow(2).sum().item())
        self.qe_count += 1
        self.qe_sum_squared += sq

    def observe_reduce(self, left_ref: int, right_ref: int,
                       parent_ref: int) -> None:
        """Record one REDUCE outcome. ``left_ref`` / ``right_ref`` are
        the operand ref_ids; ``parent_ref`` is the result. Increments
        the ``(left, right)`` adjacency count for later PMI
        computation. No-op when disabled.

        ``parent_ref`` is accepted for future use (e.g., to track
        which parent refs each pair produces) but isn't currently
        used; the minimal scaffold tracks pair frequency only.
        """
        if not self.enabled:
            return
        key = (int(left_ref), int(right_ref))
        self.pair_counts[key] = self.pair_counts.get(key, 0) + 1

    def flush(self) -> Dict[str, Any]:
        """Return a snapshot of accumulated stats and reset internal
        state.

        Returned dict has keys ``qe_count``, ``qe_sum_squared``,
        ``qe_mean`` (computed convenience), ``pair_counts`` (a shallow
        copy). Caller decides whether the snapshot justifies any
        ``extend_artifact`` calls; this method itself never mutates
        the artifact.
        """
        snap = {
            'qe_count': self.qe_count,
            'qe_sum_squared': self.qe_sum_squared,
            'qe_mean': (self.qe_sum_squared / self.qe_count
                        if self.qe_count > 0 else 0.0),
            'pair_counts': dict(self.pair_counts),
        }
        self.qe_count = 0
        self.qe_sum_squared = 0.0
        self.pair_counts = {}
        return snap
