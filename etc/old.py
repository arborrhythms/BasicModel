# Deprecated code moved here for reference.
# These classes are no longer used in the active codebase.

import torch
import torch.nn as nn
from Space import SubSpace, WhereEncoding, WhenEncoding
from util import TheDevice


class StackSpace(SubSpace):
    """Preallocated SubSpace with stack semantics (push/pop/peek).

    Used by SymbolicSpace to hold the symbol stack during shift/reduce.
    Each entry has what (codebook vector), where (position), and when
    (derivation order). The stack is a contiguous region of a preallocated
    SubSpace buffer with a position pointer.
    """

    def __init__(self, maxSize, nDim, nWhere=0, nWhen=0):
        inputShape = [maxSize, nDim + nWhere + nWhen]
        outputShape = [maxSize, nDim + nWhere + nWhen]
        whereEncoding = WhereEncoding(maxSize, nWhere)
        whenEncoding = WhenEncoding(10000, nWhen)
        super().__init__(
            inputShape=inputShape,
            outputShape=outputShape,
            whereEncoding=whereEncoding,
            whenEncoding=whenEncoding,
        )
        self.maxSize = maxSize
        self.nStackDim = nDim
        self.pos = 0
        self._consumed = set()
        # Preallocate tensors for stack entries
        self._what_buf = None   # [B, maxSize, nDim]
        self._where_buf = None  # [B, maxSize, nWhere]
        self._when_buf = None   # [B, maxSize, nWhen]
        self._batch_size = 0

    def reset(self, batch_size=1):
        """Clear the stack for a new derivation."""
        device = TheDevice.get()
        self.pos = 0
        self._consumed = set()
        self._batch_size = batch_size
        self._what_buf = torch.zeros(batch_size, self.maxSize, self.nStackDim, device=device)
        if self.nWhere > 0:
            self._where_buf = torch.zeros(batch_size, self.maxSize, self.nWhere, device=device)
        if self.nWhen > 0:
            self._when_buf = torch.zeros(batch_size, self.maxSize, self.nWhen, device=device)

    def push(self, what, where=None, when=None):
        """Push an entry onto the stack.

        Args:
            what: [B, nDim] concept vector or codebook vector
            where: [B, nWhere] positional encoding (optional)
            when: [B, nWhen] temporal encoding (optional)

        Returns:
            int: stack index of the pushed entry
        """
        idx = self.pos
        self._what_buf[:, idx, :] = what
        if where is not None and self._where_buf is not None:
            self._where_buf[:, idx, :] = where
        if when is not None and self._when_buf is not None:
            self._when_buf[:, idx, :] = when
        self.pos += 1
        return idx

    def pop(self):
        """Pop the top entry from the stack.

        Returns:
            tuple: (what, where, when) tensors for the popped entry
        """
        self.pos -= 1
        idx = self.pos
        what = self._what_buf[:, idx, :]
        where = self._where_buf[:, idx, :] if self._where_buf is not None else None
        when = self._when_buf[:, idx, :] if self._when_buf is not None else None
        return what, where, when

    def peek(self, n=1):
        """Return the top n entries without popping.

        Returns:
            tuple: (what, where, when) tensors [B, n, dim]
        """
        start = max(0, self.pos - n)
        what = self._what_buf[:, start:self.pos, :]
        where = self._where_buf[:, start:self.pos, :] if self._where_buf is not None else None
        when = self._when_buf[:, start:self.pos, :] if self._when_buf is not None else None
        return what, where, when

    def mark_consumed(self, idx):
        """Flag an entry as consumed by a reduce (still accessible)."""
        self._consumed.add(idx)

    def get_stack(self):
        """Return all entries up to current position.

        Returns:
            tuple: (what, where, when) tensors [B, pos, dim]
        """
        what = self._what_buf[:, :self.pos, :]
        where = self._where_buf[:, :self.pos, :] if self._where_buf is not None else None
        when = self._when_buf[:, :self.pos, :] if self._when_buf is not None else None
        return what, where, when

    def get_active(self):
        """Return unconsumed entries only.

        Returns:
            tuple: (what, where, when, indices) — indices lists active positions
        """
        active = [i for i in range(self.pos) if i not in self._consumed]
        if not active:
            return (torch.zeros(self._batch_size, 0, self.nStackDim, device=TheDevice.get()),
                    None, None, [])
        indices = torch.tensor(active, device=TheDevice.get())
        what = self._what_buf[:, indices, :]
        where = self._where_buf[:, indices, :] if self._where_buf is not None else None
        when = self._when_buf[:, indices, :] if self._when_buf is not None else None
        return what, where, when, active
