"""Parser-neutral ParseState.

Plan: path-to-complete §3 — "Introduce a parser-neutral ParseState. Do
not make STM fake _chart_score first."

Both the chart (CKY) and the STM (shift/reduce) backends populate the
same ``ParseState`` type. Downstream consumers (SVO extraction, syntax
dumps, reverse/generate, diagnostics) read from ``ParseState`` rather
than reaching into chart-only or STM-only internals. A legacy adapter
projects ``ParseState`` back to ``_chart_score`` / ``_chart_vec`` /
``_chart_pos`` / ``_derivation_trace`` for code that still depends on
those fields during the migration window.

Shapes:

  Frame
    payload        torch.Tensor  [D]
    category       str
    order          int
    ref_id         int           (-1 = unsnapped)
    span_start     int           (token-position index, inclusive)
    span_end       int           (token-position index, exclusive)

  Action
    rule_id        int
    operand_indices  tuple[int, ...]   indices into ``frames``
    parent_index   int                  index into ``frames``
    score          float                raw scorer logit (or chart cell score)
    probability    float                normalized over admissible rules

  ParseState
    frames                 list[Frame]
    actions                list[Action]
    trace                  list[Action]    Viterbi / best derivation
    row_traces             dict[int, list[Action]]
    current_rules          dict[str, list[list[int]]]   per-tier per-row
    generate_rules         dict[str, list[list[int]]]   reversed
    leaf_category_probs    Optional[torch.Tensor]  [N_tokens, N_categories]

The legacy adapter is :func:`project_to_chart_fields`; it sets the
named attributes on a target object (typically the chart) so callers
that still read ``chart._chart_score`` / ``chart._derivation_trace``
keep working.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class Frame:
    payload: torch.Tensor
    category: str
    order: int
    ref_id: int = -1
    span_start: int = -1
    span_end: int = -1


@dataclass
class Action:
    rule_id: int
    operand_indices: Tuple[int, ...]
    parent_index: int
    score: float = 0.0
    probability: float = 0.0


@dataclass
class ParseState:
    frames: List[Frame] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    trace: List[Action] = field(default_factory=list)
    row_traces: Dict[int, List[Action]] = field(default_factory=dict)
    current_rules: Dict[str, List[List[int]]] = field(default_factory=dict)
    generate_rules: Dict[str, List[List[int]]] = field(default_factory=dict)
    leaf_category_probs: Optional[torch.Tensor] = None

    def add_leaf(self, payload, category, order, ref_id, position):
        """Add a SHIFT-pushed leaf at the given token position. Returns
        the frame index."""
        idx = len(self.frames)
        self.frames.append(Frame(
            payload=payload,
            category=str(category),
            order=int(order),
            ref_id=int(ref_id),
            span_start=int(position),
            span_end=int(position) + 1,
        ))
        return idx

    def add_reduce(self, *, rule_id, operand_indices, parent_payload,
                   parent_category, parent_order, parent_ref_id=-1,
                   score=0.0, probability=0.0):
        """Record one REDUCE: create the parent frame from operand
        frames' spans, append the action, return the parent's frame
        index."""
        operands = tuple(int(i) for i in operand_indices)
        if not operands:
            raise ValueError("ParseState.add_reduce: empty operand_indices")
        spans = [(self.frames[i].span_start, self.frames[i].span_end)
                 for i in operands]
        parent_span_start = min(s for s, _ in spans)
        parent_span_end = max(e for _, e in spans)
        parent_idx = len(self.frames)
        self.frames.append(Frame(
            payload=parent_payload,
            category=str(parent_category),
            order=int(parent_order),
            ref_id=int(parent_ref_id),
            span_start=int(parent_span_start),
            span_end=int(parent_span_end),
        ))
        self.actions.append(Action(
            rule_id=int(rule_id),
            operand_indices=operands,
            parent_index=parent_idx,
            score=float(score),
            probability=float(probability),
        ))
        return parent_idx


def project_to_chart_fields(state: ParseState, target: Any,
                            *, n_tokens: Optional[int] = None,
                            n_categories: Optional[int] = None) -> None:
    """Legacy adapter: populate ``_chart_score`` / ``_chart_vec`` /
    ``_chart_pos`` / ``_derivation_trace`` on ``target`` from a
    ``ParseState``.

    This lets chart-only consumers keep working while their reads are
    migrated. The projections are best-effort summaries:

      * ``_derivation_trace[row]`` is a list of ``(rule_id, parent_span)``
        tuples drawn from ``state.trace``, mirroring the chart's
        existing trace shape.
      * ``_chart_score`` / ``_chart_vec`` are NOT reconstructed cell-by-
        cell — the STM doesn't have all-cells information. The adapter
        leaves them ``None`` so callers that don't gate on ``is None``
        can spot the difference (the migration step is to gate on
        ``getattr(target, '_chart_score', None) is None`` and fall back
        to ``ParseState`` reads).
      * ``_chart_pos`` is populated from ``leaf_category_probs`` when
        available.

    Plan: path-to-complete §3 (adapter) and §6 (consumer migration).
    """
    # Derivation trace: list[list[(rule_id, parent_span)]] indexed by
    # batch row.
    rows = getattr(state, 'row_traces', None) or {0: state.trace}
    if rows:
        out = []
        for b in sorted(rows):
            trace_row = [
                (a.rule_id,
                 state.frames[a.parent_index].span_start,
                 state.frames[a.parent_index].span_end)
                for a in rows[b]
            ]
            out.append(trace_row)
        target._derivation_trace = out
    else:
        target._derivation_trace = [[]]
    # Chart-cell tensors are stem-only when projecting from STM.
    target._chart_score = None
    target._chart_vec = None
    # POS distribution: defer to leaf_category_probs when present.
    target._chart_pos = state.leaf_category_probs
