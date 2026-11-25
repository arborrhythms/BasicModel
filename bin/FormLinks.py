"""The table of forms: a form of length n is one row linking its first and last n-1 bytes."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LEAVES = 256
# Sorts after every real pair key (a key is below capacity squared).
UNUSED_KEY = torch.iinfo(torch.int64).max
HALF = 64  # width of each half of a code: parts half, then wholes half
# Odd multiplier spreading row numbers over the increment subsets (made coprime at init).
SPREAD = 2654435761


class FormLinks(nn.Module):
    """Rows 0..255 are the byte leaves; each later row links two rows one byte shorter.

    Rows are numbered as admitted, parts before wholes, so both links of a
    row are below it. ``pair_keys`` holds ``left * capacity + right`` of the
    rows in use, sorted, and ``pair_rows`` the row of each key; unused
    entries hold ``UNUSED_KEY``. Buffers are allocated once at ``capacity``.

    ``count`` is a row's readings as a whole unit; recognise returns a row
    only once its count reaches ``threshold`` (leaves always), else its parts.

    ``code`` [capacity, 2*HALF]: the parts half of a leaf is its eight bits
    as (bit, not bit) pairs; of a form, the coordinate maximum of its two
    parts' plus its increment. The wholes half of a part is the coordinate
    maximum over the wholes linking it of their wholes half plus their
    increment (zero when held by none). A row's increment is one on the
    ``spread`` coordinates of a subset unique to the row, so every whole
    exceeds its parts strictly there, and every part its wholes.
    """

    def __init__(self, capacity, longest, threshold):
        super().__init__()
        self.capacity, self.longest = int(capacity), int(longest)
        self.threshold = int(threshold)
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
        if self.threshold < 0:
            raise ValueError(f'FormLinks threshold must not be negative, got {self.threshold}')
        # Smallest subset size giving every row its own subset of HALF coordinates.
        self.spread = next(k for k in range(1, HALF + 1) if math.comb(HALF, k) >= self.capacity)
        self.subsets = math.comb(HALF, self.spread)
        self.multiplier = SPREAD % self.subsets
        while math.gcd(self.multiplier, self.subsets) != 1:
            self.multiplier += 1
        binom = torch.tensor([[math.comb(i, j) for j in range(self.spread + 1)] for i in range(HALF)])
        self.register_buffer('binom', binom, persistent=False)
        code = torch.zeros(self.capacity, 2 * HALF)
        bits = (torch.arange(LEAVES)[:, None] >> torch.arange(8)) & 1
        code[:LEAVES, :16] = torch.stack((bits, 1 - bits), -1).reshape(LEAVES, 16).float()
        self.register_buffer('code', code)

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

    def _promoted(self, rows):
        """``rows`` where a leaf or counted to the threshold, else -1."""
        counted = self.count[rows.clamp_min(0)] >= self.threshold
        return torch.where((rows >= 0) & ((rows < LEAVES) | counted), rows, -1)

    # -- recognise ----------------------------------------------------------

    def recognise(self, data, lengths, longest=None):
        """Rows of every substring and the longest known form at each start.

        ``data`` [B, L] bytes, ``lengths`` [B] valid extents. Returns
        ``rows`` [B, longest, L] (``rows[b, n-1, i]`` is the form of length n
        at byte i, -1 where unknown or below the threshold) and ``best``
        [B, L] (-1 past the extent). Links are followed through uncounted rows.
        """
        longest = self.longest if longest is None else int(longest)
        cur = self._leaves(data, lengths)
        best, levels = cur, [cur]
        for _ in range(1, longest):
            cur = self._level(cur)
            shown = self._promoted(cur)
            best = torch.where(shown >= 0, shown, best)
            levels.append(shown)
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

    def _unit_rows(self, data, lengths):
        """Row of each whole unit [U], counted or not (-1 where unknown or empty)."""
        cur = self._leaves(data, lengths)
        unit = torch.where(lengths == 1, cur[:, 0], -1)
        for n in range(2, data.shape[1] + 1):
            cur = self._level(cur)
            unit = torch.where(lengths == n, cur[:, 0], unit)
        return unit

    def rows_of(self, data, lengths):
        """Row of each whole unit [U]; -1 where unknown, empty or below the threshold."""
        return self._promoted(self._unit_rows(data, lengths))

    def tally(self, data, lengths):
        """Raise ``count`` of each known unit by one; returns the unit rows (-1 for unknown)."""
        unit = self._unit_rows(data, lengths)
        self._raise(unit)
        return unit

    def _raise(self, unit):
        known = unit >= 0
        self.count.index_add_(0, unit.clamp_min(0), known.to(self.count.dtype))

    def close(self, data, lengths):
        """The post-sentence reset: admit what the units lack, then raise their counts."""
        unit = self.admit_forms(data, lengths)
        self._raise(unit)
        return unit

    def _increment(self, rows):
        """[n, HALF] ones on the coordinate subset of each row (colex unranking of a spread rank)."""
        rank = rows.long() * self.multiplier % self.subsets
        left = torch.full_like(rank, self.spread)
        out = []
        for i in range(HALF - 1, -1, -1):
            c = self.binom[i][left.clamp_min(0)]
            take = (left > 0) & (c <= rank)
            rank = rank - torch.where(take, c, 0)
            left = left - take.long()
            out.append(take)
        return torch.stack(out[::-1], 1).to(self.code.dtype)

    @torch.no_grad()
    def admit_forms(self, data, lengths):
        """Admit every unit with the substrings it lacks, shortest first; returns the unit rows.

        A new row's parts half is made as it is admitted; the wholes half is
        then pushed from each unit's longest substrings down to its leaves.
        A full table raises before the level that would overflow is written;
        levels already admitted stay (each is a complete set of parts).
        """
        cur = self._leaves(data, lengths)
        unit = torch.where(lengths == 1, cur[:, 0], -1)
        levels, grown = [], False
        try:
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
                    lo, hi = keys // self.capacity, keys % self.capacity
                    self.left[first:first + new] = lo.to(self.left.dtype)
                    self.right[first:first + new] = hi.to(self.right.dtype)
                    self.length[first:first + new] = n
                    rows = torch.arange(first, first + new, device=keys.device)
                    self.code[first:first + new, :HALF] = torch.maximum(
                        self.code[lo, :HALF], self.code[hi, :HALF]) + self._increment(rows)
                    self.used.fill_(first + new)
                    self.sort_pairs()
                    grown = True
                cur = self._level(cur)
                levels.append(cur)
                unit = torch.where(lengths == n, cur[:, 0], unit)
        finally:
            if grown:
                self._push_wholes(levels)
        return unit

    def _push_wholes(self, levels):
        """Raise each part's wholes half to its wholes' plus their increment, longest first."""
        for rows in reversed(levels):
            rows = rows[rows >= 0].unique()
            src = self.code[rows, HALF:] + self._increment(rows)
            for link in (self.left, self.right):
                index = link[rows].long()[:, None].expand(-1, HALF)
                self.code[:, HALF:].scatter_reduce_(0, index, src, 'amax')

    @torch.no_grad()
    def sort_pairs(self):
        """Rebuild the sorted pair index from the rows in use."""
        rows = torch.arange(self.capacity, device=self.left.device)
        live = (rows >= LEAVES) & (rows < self.used)
        keys = torch.where(live, self.left.long() * self.capacity + self.right.long(), UNUSED_KEY)
        order = torch.argsort(keys, stable=True)
        self.pair_keys.copy_(keys[order])
        self.pair_rows.copy_(order)
