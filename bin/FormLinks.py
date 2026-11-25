"""The table of forms: a form of length n is one row linking its first and last n-1 bytes."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LEAVES = 256
# Sorts after every real pair key (a key is below capacity squared).
UNUSED_KEY = torch.iinfo(torch.int64).max


class FormLinks(nn.Module):
    """Rows 0..255 are the byte leaves; each later row links two rows one byte shorter.

    Rows are numbered as admitted, parts before wholes, so both links of a
    row are below it. ``pair_keys`` holds ``left * capacity + right`` of the
    rows in use, sorted, and ``pair_rows`` the row of each key; unused
    entries hold ``UNUSED_KEY``. Buffers are allocated once at ``capacity``.
    """

    def __init__(self, capacity, longest):
        super().__init__()
        self.capacity, self.longest = int(capacity), int(longest)
        if self.capacity <= LEAVES:
            raise ValueError(f'FormLinks capacity must exceed {LEAVES}, got {self.capacity}')
        if self.longest < 1:
            raise ValueError(f'FormLinks longest must be positive, got {self.longest}')
        self.register_buffer('left', torch.full((self.capacity,), -1, dtype=torch.int32))
        self.register_buffer('right', torch.full((self.capacity,), -1, dtype=torch.int32))
        length = torch.zeros(self.capacity, dtype=torch.int32)
        length[:LEAVES] = 1
        self.register_buffer('length', length)
        self.register_buffer('count', torch.zeros(self.capacity, dtype=torch.int32))
        self.register_buffer('used', torch.tensor(LEAVES, dtype=torch.int64))
        self.register_buffer('pair_keys', torch.full((self.capacity,), UNUSED_KEY, dtype=torch.int64))
        self.register_buffer('pair_rows', torch.arange(self.capacity, dtype=torch.int64))

    # -- find ---------------------------------------------------------------

    def find(self, left, right):
        """Row of the form with links ``left``, ``right`` (any shape); -1 where unknown or a link is -1."""
        valid = (left >= 0) & (right >= 0)
        key = torch.where(valid, left.long() * self.capacity + right.long(), 0)
        place = torch.searchsorted(self.pair_keys, key).clamp_max(self.capacity - 1)
        hit = valid & (self.pair_keys[place] == key)
        return torch.where(hit, self.pair_rows[place], -1)

    def _leaves(self, data, lengths):
        """Byte rows [B, L] to leaf rows, -1 past each extent."""
        places = torch.arange(data.shape[1], device=data.device)
        return torch.where(places < lengths[:, None], data.long(), -1)

    def _level(self, rows):
        """Rows of the forms one byte longer at each start, from rows of forms at that length."""
        return F.pad(self.find(rows[:, :-1], rows[:, 1:]), (0, 1), value=-1)[:, :rows.shape[1]]

    # -- recognise ----------------------------------------------------------

    def recognise(self, data, lengths, longest=None):
        """Rows of every substring and the longest known form at each start.

        ``data`` [B, L] bytes, ``lengths`` [B] valid extents. Returns
        ``rows`` [B, longest, L] (``rows[b, n-1, i]`` is the form of length n
        at byte i, -1 where unknown) and ``best`` [B, L] (-1 past the extent).
        """
        longest = self.longest if longest is None else int(longest)
        cur = self._leaves(data, lengths)
        best, levels = cur, [cur]
        for _ in range(1, longest):
            cur = self._level(cur)
            best = torch.where(cur >= 0, cur, best)
            levels.append(cur)
        return torch.stack(levels, 1), best

    # -- spell --------------------------------------------------------------

    def spell(self, rows, longest=None):
        """Bytes [n, longest] (-1 padding) and lengths [n] of ``rows`` [n], by their links.

        Byte i of a form of length m is reached by m-1-i left links, then i right links.
        """
        longest = self.longest if longest is None else int(longest)
        known = rows >= 0
        start = rows.long().clamp_min(0)
        m = torch.where(known, self.length[start].long(), 0)
        places = torch.arange(longest, device=rows.device)
        lefts = (m[:, None] - 1 - places).clamp_min(0)
        steps = (m[:, None] - 1).clamp_min(0)
        cur = start[:, None].expand(-1, longest)
        for s in range(longest - 1):
            go_left, go_right = s < lefts, (s >= lefts) & (s < steps)
            cur = torch.where(go_left, self.left[cur].long(),
                              torch.where(go_right, self.right[cur].long(), cur))
        return torch.where(places < m[:, None], cur, -1), m

    # -- count and admit (post-sentence reset) --------------------------------

    def rows_of(self, data, lengths):
        """Row of each whole unit [U] (-1 where unknown or empty); one level per byte of the longest."""
        cur = self._leaves(data, lengths)
        unit = torch.where(lengths == 1, cur[:, 0], -1)
        for n in range(2, data.shape[1] + 1):
            cur = self._level(cur)
            unit = torch.where(lengths == n, cur[:, 0], unit)
        return unit

    def tally(self, data, lengths):
        """Raise ``count`` of each known unit by one; returns the unit rows (-1 for unknown)."""
        unit = self.rows_of(data, lengths)
        known = unit >= 0
        self.count.index_add_(0, unit.clamp_min(0), known.to(self.count.dtype))
        return unit

    def admit_forms(self, data, lengths):
        """Admit every unit with the substrings it lacks, shortest first; returns the unit rows.

        A full table raises before the level that would overflow is written;
        levels already admitted stay (each is a complete set of parts).
        """
        cur = self._leaves(data, lengths)
        unit = torch.where(lengths == 1, cur[:, 0], -1)
        for n in range(2, data.shape[1] + 1):
            a, b = cur[:, :-1], cur[:, 1:]
            missing = (a >= 0) & (b >= 0) & (self.find(a, b) < 0)
            keys = torch.unique(a[missing] * self.capacity + b[missing])
            new = int(keys.numel())
            if new:
                first = int(self.used)
                if first + new > self.capacity:
                    raise RuntimeError(
                        f'FormLinks full: {first} rows in use, {new} more of length {n}, '
                        f'capacity {self.capacity}. Raise PartSpace <formCapacity>.')
                self.left[first:first + new] = (keys // self.capacity).to(self.left.dtype)
                self.right[first:first + new] = (keys % self.capacity).to(self.right.dtype)
                self.length[first:first + new] = n
                self.used.fill_(first + new)
                self.sort_pairs()
            cur = self._level(cur)
            unit = torch.where(lengths == n, cur[:, 0], unit)
        return unit

    @torch.no_grad()
    def sort_pairs(self):
        """Rebuild the sorted pair index from the rows in use."""
        rows = torch.arange(self.capacity, device=self.left.device)
        live = (rows >= LEAVES) & (rows < self.used)
        keys = torch.where(live, self.left.long() * self.capacity + self.right.long(), UNUSED_KEY)
        order = torch.argsort(keys, stable=True)
        self.pair_keys.copy_(keys[order])
        self.pair_rows.copy_(order)
