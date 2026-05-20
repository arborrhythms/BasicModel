"""STM shift/reduce driver + small rule scorer.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 3 — new small module, intentionally NOT a port of the
chart's ``_rule_embed`` machinery. The STM scorer scores only
admissible reduce actions (after typed admissibility masking) so its
input doesn't need to be a full categorical mixing layer — it just
needs to pick *among the typed survivors*.

Scope: this module ships the data-flow scaffolding. Full STM-as-active-
parser activation (driving compose() / generate() under
``parser_backend='stm'``) is the next step; this driver is testable in
isolation against fixture rule signatures.
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from embed import admissibility_mask, mask_logits


class RuleScorer(nn.Module):
    """Small MLP scoring rules from top-of-stack payloads.

    Input: top operand payload(s) — ``left`` (required) and ``right``
    (optional, ``None`` for unary REDUCE). Concatenates and projects to
    a per-rule logit vector. Architecturally minimal — one hidden layer
    + linear head — because admissibility masking does the hard
    discrimination; the scorer just orders the survivors.
    """

    def __init__(self, payload_dim: int, n_rules: int,
                 hidden_dim: Optional[int] = None):
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

    def forward(self, left: torch.Tensor,
                right: Optional[torch.Tensor]) -> torch.Tensor:
        """Return ``[n_rules]`` logits for the given operand payload(s)."""
        if right is None:
            right = torch.zeros_like(left)
        x = torch.cat([left, right], dim=-1)
        return self.head(self.body(x))


class STMDriver:
    """Coordinates a ``TypedStack`` + rule signatures + a ``RuleScorer``.

    SHIFT pushes a token frame onto the stack with category / order /
    ref_id metadata. REDUCE builds the admissibility mask, masks the
    scorer's logits, and picks the highest-scoring admissible rule.

    REDUCE returns the chosen rule's index and signature; this initial
    version doesn't apply rule semantics to compute the parent payload
    (a separate concern — needs the actual op kernels from
    ``Layers.Ops``). Callers wire in payload computation as a follow-up.
    """

    def __init__(self, typed_stack, rule_signatures: List[Dict[str, Any]],
                 scorer: RuleScorer):
        self.typed_stack = typed_stack
        self.rule_signatures = list(rule_signatures)
        self.scorer = scorer

    def shift(self, b: int, payload: torch.Tensor,
              *, category: str, order: int, ref_id: int) -> None:
        """Push a new frame onto row ``b``'s stack."""
        self.typed_stack.push(
            b, payload,
            category_id_str=category,
            order=int(order),
            ref_id=int(ref_id))

    def reduce_step(self, b: int) -> Dict[str, Any]:
        """Pick the highest-scoring admissible rule for row ``b``'s stack
        top. Returns ``{'rule_index': int, 'rule_signature': dict}``.

        Raises ``RuntimeError`` when no rule is admissible (the parser
        is stuck — caller decides whether to backtrack, stall, or fall
        back to a chart oracle).
        """
        mask = self.typed_stack.reduce_admissibility(
            b, self.rule_signatures)
        if not mask.any():
            raise RuntimeError(
                "STMDriver.reduce_step: no admissible rule for current "
                "stack top. Stack-top frames may need a different SHIFT, "
                "or the rule set may need extension.")
        # Pull operand payloads for the scorer.
        d = int(self.typed_stack._depth[b].item())
        right = self.typed_stack.top(b, k=1)['payload'] if d >= 1 else None
        left = self.typed_stack.top(b, k=2)['payload'] if d >= 2 else right
        # For unary REDUCE (d == 1) treat left as the single operand.
        if d == 1:
            logits = self.scorer(left, None)
        else:
            logits = self.scorer(left, right)
        masked = mask_logits(logits, mask)
        rule_index = int(torch.argmax(masked).item())
        return {
            'rule_index': rule_index,
            'rule_signature': self.rule_signatures[rule_index],
        }
