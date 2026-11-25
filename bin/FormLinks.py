"""The table of forms: a form of length n is one row linking its first and last n-1 bytes."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LEAVES = 256
# Sorts after every real pair key (a key is below capacity squared).
UNUSED_KEY = torch.iinfo(torch.int64).max
HALF = 64  # width of each half of a code: parts half, then wholes half
LEAF_BITS = 8  # coordinates set in a byte's direction
LEAF_SEED = 1  # fixes the byte directions
SCALE = 2.0 ** 32  # fixed point of the running whole sums: integer adds are exact in any order


class FormLinks(nn.Module):
    """Rows 0..255 are the byte leaves; each later row links two rows one byte shorter.

    Rows are numbered as admitted, parts before wholes, so both links of a
    row are below it. ``pair_keys`` holds ``left * capacity + right`` of the
    rows in use, sorted, and ``pair_rows`` the row of each key; unused
    entries hold ``UNUSED_KEY``. Buffers are allocated once at ``capacity``.

    ``count`` is a row's readings as a whole unit; recognise returns a row
    only once its count reaches ``threshold`` (leaves always), else its parts.

    ``code`` [capacity, 2*HALF]: the parts half of a leaf is ``step`` on
    ``LEAF_BITS`` coordinates of its own; of a form, halfway between the
    mean of its distinct proper substrings and the mean of the distinct
    forms holding it (``1 - step`` everywhere when none holds it). Placed so
    at admission, recentred when read by ``close``. The wholes half is not
    written here.

    ``whole_sum`` [capacity, HALF] (int64, units of 1/SCALE) and
    ``whole_count`` [capacity] keep, for each row, the sum and number of the
    forms holding it: a form adds its code to every distinct substring when
    admitted, and its change whenever it moves.
    """

    def __init__(self, capacity, longest, threshold, step=1 / 16):
        super().__init__()
        self.capacity, self.longest = int(capacity), int(longest)
        self.threshold, self.step = int(threshold), float(step)
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
        if not 0 < self.step < .5:
            raise ValueError(f'FormLinks step must lie in (0, 1/2), got {self.step}')
        code = torch.zeros(self.capacity, 2 * HALF)
        noise = torch.rand(LEAVES, HALF, generator=torch.Generator().manual_seed(LEAF_SEED))
        code[:LEAVES, :HALF].scatter_(1, noise.topk(LEAF_BITS, 1).indices, self.step)
        self.register_buffer('code', code)
        self.register_buffer('whole_sum', torch.zeros(self.capacity, HALF, dtype=torch.int64))
        self.register_buffer('whole_count', torch.zeros(self.capacity, dtype=torch.int32))

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

    @torch.no_grad()
    def close(self, data, lengths):
        """The post-sentence reset: admit what the units lack, recentre their forms, raise their counts."""
        unit = self.admit_forms(data, lengths)
        self.recentre(data, lengths)
        self._raise(unit)
        return unit

    # -- the parts half of the code -------------------------------------------

    def _parts(self, rows, n):
        """Distinct proper substrings of forms ``rows`` [F] of length n: rows [F, E] and weights [F, E].

        E = n(n+1)/2 - 1, level by level down to the leaves; a repeat within a
        level, and every entry of a row < 0, weighs zero.
        """
        valid = rows >= 0
        cur = rows.long().clamp_min(0)[:, None]
        out, first = [], []
        for size in range(2, n + 1):
            cur = torch.cat((self.left[cur[:, :1]], self.right[cur]), 1).long().clamp_min(0)
            earlier = torch.ones(size, size, dtype=torch.bool, device=rows.device).tril(-1)
            seen = ((cur[:, :, None] == cur[:, None, :]) & earlier).any(-1)
            out.append(cur)
            first.append(~seen & valid[:, None])
        return torch.cat(out, 1), torch.cat(first, 1)

    def _quantise(self, code):
        """Code in units of 1/SCALE; a power-of-two scale keeps float32 values exact."""
        return (code * SCALE).round().long()

    def _whole_mean(self, rows):
        """Mean code [F, HALF] of the forms holding ``rows``; ``1 - step`` where none."""
        count = self.whole_count[rows].to(self.code.dtype)[:, None]
        mean = self.whole_sum[rows].to(self.code.dtype) / (count.clamp_min(1) * SCALE)
        return torch.where(count > 0, mean, 1 - self.step)

    def _place(self, rows, n, admitted):
        """Put forms ``rows`` [F] of length n halfway between their parts' and wholes' means.

        Each distinct part's whole sum takes the form's code when ``admitted``,
        else its change; ``rows`` < 0 are skipped.
        """
        parts, weight = self._parts(rows, n)
        w = weight.to(self.code.dtype)
        part_mean = (self.code[parts, :HALF] * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1)
        target = rows.long().clamp_min(0)
        new = (part_mean + self._whole_mean(target)) / 2
        change = self._quantise(new)
        if not admitted:
            change = change - self._quantise(self.code[target, :HALF])
        keep = rows >= 0
        new = torch.where(keep[:, None], new, self.code[target, :HALF])
        self.code[:, :HALF].index_copy_(0, target, new)
        flat = parts.reshape(-1)
        self.whole_sum.index_add_(0, flat, (change[:, None, :] * weight[..., None]).reshape(-1, HALF))
        if admitted:
            self.whole_count.index_add_(0, flat, weight.reshape(-1).to(self.whole_count.dtype))

    def _first_seen(self, rows):
        """``rows`` [P] with each repeat after its first place set to -1 (sort, fixed shape)."""
        order = torch.argsort(rows, stable=True)
        ranked = rows[order]
        repeat = F.pad(ranked[1:] == ranked[:-1], (1, 0), value=False)
        return torch.where(repeat.scatter(0, order, repeat), -1, rows)

    @torch.no_grad()
    def recentre(self, data, lengths):
        """Recentre every form of the units, level by level from length 2, each form once a level."""
        cur = self._leaves(data, lengths)
        for n in range(2, data.shape[1] + 1):
            cur = self._level(cur)
            self._place(self._first_seen(cur[:, :data.shape[1] - n + 1].reshape(-1)), n, False)

    @torch.no_grad()
    def admit_forms(self, data, lengths):
        """Admit every unit with the substrings it lacks, shortest first; returns the unit rows.

        A new row is placed by its parts and wholes as it is admitted, and
        added to the whole sums of its parts. A full table raises before the
        level that would overflow is written; levels already admitted stay
        (each is a complete set of parts).
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
                lo, hi = keys // self.capacity, keys % self.capacity
                self.left[first:first + new] = lo.to(self.left.dtype)
                self.right[first:first + new] = hi.to(self.right.dtype)
                self.length[first:first + new] = n
                self.used.fill_(first + new)
                self.sort_pairs()
                self._place(torch.arange(first, first + new, device=keys.device), n, True)
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
