"""TypedStack — STM stack data structure with per-frame typed metadata.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§STM Shift/Reduce Runtime — STM stack item metadata.

The plan calls for parallel tensors carrying ``category``, ``order``,
``ref_id`` alongside the vector payload, so REDUCE can build a typed
admissibility mask from the top-of-stack state and gate inadmissible
rules before the reducer softmax. This module provides the data
structure; the parser-side wiring (SHIFT / REDUCE dispatch, actually
calling this from the inference loop) is the Phase-2.5 closeout work.

Parallel tensors:

    _buffer    [B, cap, D]   float  — vector payload per slot
    _category  [B, cap]      long   — int category id (-1 = empty)
    _order     [B, cap]      long   — conceptual order per slot
    _ref_id    [B, cap]      long   — taxonomy ref_id (-1 = unsnapped)
    _depth     [B]           long   — number of live slots per row

A parallel ``_category_names`` Python list-of-lists carries the
string form of the category for slots whose ``category_id`` hasn't
been assigned yet (the bootstrap path: STM SHIFT pushes by category
name; the integer assignment from the codebook comes later).
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from embed import admissibility_mask


class TypedStack(nn.Module):
    """Per-row stack with parallel metadata tensors.

    Push / pop / top operate per-row (host-side index ``b``). All five
    parallel structures stay in sync.

    2026-05-20 substrate fix: ``TypedStack`` is an ``nn.Module`` and
    its parallel tensors are registered via ``register_buffer`` so
    ``.to(device)`` / ``.cuda()`` move them together and ``state_dict``
    sees them as a coherent group. The Python list of string-form
    category names stays as a regular attribute (no device concept).
    """

    def __init__(self, batch: int, max_depth: int, dim: int):
        super().__init__()
        self.batch = int(batch)
        self.max_depth = int(max_depth)
        self.dim = int(dim)
        self.register_buffer(
            '_buffer',
            torch.zeros(self.batch, self.max_depth, self.dim))
        self.register_buffer(
            '_category',
            torch.full((self.batch, self.max_depth), -1, dtype=torch.long))
        self.register_buffer(
            '_order',
            torch.zeros((self.batch, self.max_depth), dtype=torch.long))
        self.register_buffer(
            '_ref_id',
            torch.full((self.batch, self.max_depth), -1, dtype=torch.long))
        self.register_buffer(
            '_depth',
            torch.zeros(self.batch, dtype=torch.long))
        # Parallel string-form category names. Populated when ``push``
        # is given ``category_id_str``; left ``None`` when the int id
        # is the primary form.
        self._category_names: List[List[Optional[str]]] = [
            [None] * self.max_depth for _ in range(self.batch)
        ]

    def push(
        self,
        b: int,
        vec: torch.Tensor,
        *,
        category_id: Optional[int] = None,
        category_id_str: Optional[str] = None,
        order: int = 0,
        ref_id: int = -1,
    ) -> None:
        """Push one frame onto row ``b``'s stack.

        ``category_id`` (int) and / or ``category_id_str`` (str) may be
        provided. At least one must be set. When only the string form
        is given, the int slot defaults to -1 and the integer-keyed
        admissibility paths can't be used until a codebook lookup
        fills in the id.
        """
        if category_id is None and category_id_str is None:
            raise ValueError(
                "TypedStack.push: provide category_id or category_id_str")
        d = int(self._depth[b].item())
        assert d < self.max_depth, (
            f"TypedStack overflow at row {b}: max_depth={self.max_depth}")
        self._buffer[b, d] = vec
        self._category[b, d] = int(category_id) if category_id is not None else -1
        self._order[b, d] = int(order)
        self._ref_id[b, d] = int(ref_id)
        self._category_names[b][d] = category_id_str
        self._depth[b] = d + 1

    def pop(self, b: int) -> Dict[str, Any]:
        """Pop the top frame from row ``b`` and return its metadata."""
        d = int(self._depth[b].item())
        assert d > 0, (
            f"TypedStack underflow at row {b}: stack is empty")
        top_slot = d - 1
        out = {
            'payload':  self._buffer[b, top_slot].clone(),
            'category': int(self._category[b, top_slot].item()),
            'category_str': self._category_names[b][top_slot],
            'order':    int(self._order[b, top_slot].item()),
            'ref_id':   int(self._ref_id[b, top_slot].item()),
        }
        # Clear the slot so subsequent invariant checks behave.
        self._buffer[b, top_slot] = 0
        self._category[b, top_slot] = -1
        self._order[b, top_slot] = 0
        self._ref_id[b, top_slot] = -1
        self._category_names[b][top_slot] = None
        self._depth[b] = top_slot
        return out

    def top(self, b: int, k: int = 1) -> Dict[str, Any]:
        """Peek at the top frame on row ``b`` (k=1) without popping.

        ``k > 1`` returns the k-th frame from the top (k=1 is the
        most recent; k=2 is the one beneath it; etc.).
        """
        d = int(self._depth[b].item())
        assert d >= k, (
            f"TypedStack.top: row {b} has {d} items, asked for k={k}")
        slot = d - k
        return {
            'payload':  self._buffer[b, slot].clone(),
            'category': int(self._category[b, slot].item()),
            'category_str': self._category_names[b][slot],
            'order':    int(self._order[b, slot].item()),
            'ref_id':   int(self._ref_id[b, slot].item()),
        }

    def reduce_admissibility(
        self,
        b: int,
        rule_signatures: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Build the admissibility mask for row ``b``'s current stack top.

        Reads the top two items (or top one for unary REDUCEs at
        depth==1) and matches against each rule signature via
        :func:`embed.admissibility_mask`. Returns a length-``len(rule_signatures)``
        ``BoolTensor``.

        Convention: with depth ``d``, the "left operand" is slot
        ``d-2`` (second from top) and the "right operand" is slot
        ``d-1`` (top). When ``d == 1``, only ``left`` is set —
        producing a unary admissibility check.
        """
        d = int(self._depth[b].item())
        if d == 0:
            return torch.zeros(len(rule_signatures), dtype=torch.bool)
        right_slot = d - 1
        left_slot = d - 2 if d >= 2 else d - 1
        if d == 1:
            return admissibility_mask(
                rule_signatures,
                left_cat=self._category_names[b][left_slot]
                or str(int(self._category[b, left_slot].item())),
                left_order=int(self._order[b, left_slot].item()),
            )
        return admissibility_mask(
            rule_signatures,
            left_cat=self._category_names[b][left_slot]
            or str(int(self._category[b, left_slot].item())),
            left_order=int(self._order[b, left_slot].item()),
            right_cat=self._category_names[b][right_slot]
            or str(int(self._category[b, right_slot].item())),
            right_order=int(self._order[b, right_slot].item()),
        )
