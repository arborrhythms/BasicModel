"""Supervised training for the STMDriver's RuleScorer.

Plan: path-to-complete §7 closeout — train ``STMDriver.scorer``
against chart-Viterbi selections (or any oracle that emits per-REDUCE
target rule_ids) so ``compose(stm)`` matches the oracle's
``current_rules`` on representative grammars. Once trained, the
byte-for-byte equivalence test in
``test/test_cky_retirement_parity_gate.py`` passes and the chart-only
machinery can be removed.

API:

  ``train_step(driver, input_vectors, target_rule_ids, *, snap_fn,
              optimizer=None) -> torch.Tensor``
    Replays the SHIFT/REDUCE loop on ``input_vectors[0]`` (single-row,
    batch 1) and accumulates cross-entropy loss between
    ``softmax(masked_logits)`` and the next ``target_rule_ids[i]`` at
    each REDUCE position. When ``optimizer`` is given, steps it after
    backward. Returns the scalar loss tensor.

  ``targets_from_chart_compose(word_space, input_vectors) -> list[int]``
    Runs the chart-backed compose and projects its rule selections
    into a flat list of target rule_ids in the order an STM
    SHIFT/REDUCE loop would consume them. The chart's
    ``current_rules`` is per-tier per-row; we flatten by walking the
    derivation in trace order (chart's ``_collect_rule_selections``
    already gives trace-order per row).

``snap_fn`` is a callable ``(payload) -> (ref_id, category, order)``
that mirrors :py:meth:`WordSpace._stm_snap_token` so the trainer
doesn't depend on a fully-constructed WordSpace. Pass
``word_space._make_snap_fn()`` to get the same snap the live STM
uses.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def train_step(
    driver,
    input_vectors: torch.Tensor,
    target_rule_ids: List[int],
    *,
    snap_fn: Callable,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> torch.Tensor:
    """Run one supervised training step on the STM driver.

    Replays the SHIFT/REDUCE loop greedily (matching ``_stm_drive``'s
    left-corner shift-reduce order). At each REDUCE position, the
    target rule_id is taken from ``target_rule_ids`` in order. Loss
    is cross-entropy of ``softmax(masked_logits)`` against the
    target. Inadmissible target rule_ids (which shouldn't happen with
    valid supervision) trigger a ValueError.

    Single-row contract (input_vectors[0]) — the STM driver is
    inherently per-row; batching across rows is a follow-up.
    """
    if input_vectors.ndim != 3 or input_vectors.shape[0] < 1:
        raise ValueError(
            "train_step: expected input_vectors of shape [B, N, D]")
    if optimizer is not None:
        optimizer.zero_grad()
    typed_stack = driver.typed_stack
    # Reset row 0.
    while int(typed_stack._depth[0].item()) > 0:
        typed_stack.pop(0)
    N = int(input_vectors.shape[1])
    reduces_done = 0
    losses: List[torch.Tensor] = []
    max_reduces = max(1, N * 3 + 4)
    for n in range(N):
        payload = input_vectors[0, n]
        ref_id, category, order = snap_fn(payload)
        driver.shift(0, payload,
                     category=category, order=order, ref_id=ref_id)
        # Greedy reduce while admissible.
        while reduces_done < max_reduces:
            try:
                score_out = driver._score_reduce(0)
            except RuntimeError:
                break
            if reduces_done >= len(target_rule_ids):
                break
            target_id = int(target_rule_ids[reduces_done])
            mask = score_out['mask']
            if not bool(mask[target_id].item()):
                raise ValueError(
                    f"train_step: target rule_id={target_id} is "
                    "inadmissible at REDUCE position "
                    f"{reduces_done}. Supervision is inconsistent "
                    "with the typed admissibility mask.")
            masked_logits = score_out['masked_logits']
            log_probs = F.log_softmax(masked_logits, dim=-1)
            loss = -log_probs[target_id]
            losses.append(loss)
            # Apply the target rule to advance the stack (not the
            # argmax — supervision drives the trajectory).
            sig = driver.rule_signatures[target_id]
            arity = len(sig.get('rhs_categories', ()))
            if arity == 2 and int(typed_stack._depth[0].item()) >= 2:
                right = typed_stack.pop(0)
                left = typed_stack.pop(0)
                parent_payload = (
                    left['payload'] + right['payload']) / 2.0
            elif arity == 1 and int(typed_stack._depth[0].item()) >= 1:
                only = typed_stack.pop(0)
                parent_payload = only['payload']
            else:
                break
            typed_stack.push(
                0, parent_payload,
                category_id_str=str(sig.get('lhs_category', 'UNK')),
                order=int(sig.get('lhs_order', 0)), ref_id=-1)
            reduces_done += 1
    # Final cleanup: REDUCE until depth==1 or out of targets.
    while reduces_done < min(max_reduces, len(target_rule_ids)):
        if int(typed_stack._depth[0].item()) <= 1:
            break
        try:
            score_out = driver._score_reduce(0)
        except RuntimeError:
            break
        target_id = int(target_rule_ids[reduces_done])
        mask = score_out['mask']
        if not bool(mask[target_id].item()):
            raise ValueError(
                f"train_step: target rule_id={target_id} is "
                "inadmissible during cleanup pass.")
        masked_logits = score_out['masked_logits']
        log_probs = F.log_softmax(masked_logits, dim=-1)
        loss = -log_probs[target_id]
        losses.append(loss)
        sig = driver.rule_signatures[target_id]
        arity = len(sig.get('rhs_categories', ()))
        if arity == 2 and int(typed_stack._depth[0].item()) >= 2:
            right = typed_stack.pop(0)
            left = typed_stack.pop(0)
            parent_payload = (
                left['payload'] + right['payload']) / 2.0
        elif arity == 1 and int(typed_stack._depth[0].item()) >= 1:
            only = typed_stack.pop(0)
            parent_payload = only['payload']
        else:
            break
        typed_stack.push(
            0, parent_payload,
            category_id_str=str(sig.get('lhs_category', 'UNK')),
            order=int(sig.get('lhs_order', 0)), ref_id=-1)
        reduces_done += 1
    if not losses:
        # No REDUCEs happened; nothing to train on this input.
        return torch.tensor(0.0, requires_grad=True)
    total = torch.stack(losses).sum()
    if optimizer is not None:
        total.backward()
        optimizer.step()
    return total
