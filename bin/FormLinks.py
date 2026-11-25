"""The table of forms: a form of length n is one row linking its first and last n-1 bytes."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LEAVES = 256
# Sorts after every real pair key (a key is below capacity squared).
UNUSED_KEY = torch.iinfo(torch.int64).max
HALF = 64  # coordinates of a code written here; the rest stay zero
SCALE = 2.0 ** 32  # fixed point of the holder sums: integer adds are exact in any order
MASK32 = 0xFFFFFFFF


class FormLinks(nn.Module):
    """Rows 0..255 are the byte leaves; each later row links two rows one byte shorter.

    Rows are numbered as inserted, parts before wholes, so both links of a
    row are below it. ``pair_keys`` holds ``left * capacity + right`` of the
    rows in use, sorted, and ``pair_rows`` the row of each key; unused
    entries hold ``UNUSED_KEY``. Buffers are allocated once at ``capacity``.

    ``count`` is a row's readings as a whole unit; recognise returns a row
    only once its count reaches ``threshold`` (leaves always), else its parts.

    ``code`` [capacity, 2*HALF], first HALF coordinates: a byte sits at
    ``step`` times its hashed point (``_points``) and never moves; a form
    starts at its hashed point. A pull step moves a form ``rate`` of the way
    to its target: with direct holders (forms it is a link of), the midpoint
    of its links' mean and its holders' mean; with none,
    ``clamp(1 - step + m - mean(m), 0, 1)`` for m its links' mean. Steps are
    taken by the rows read (``recentre_rows``) and over the whole table at
    a save (``sweep``, ``rounds`` steps).

    ``holder_sum`` [capacity, HALF] (int64, units of 1/SCALE) and
    ``holder_count`` [capacity]: the codes and number of a row's direct
    holders, a holder counted once when both its links are the row. Kept by
    adding each inserted form and each movement to its two links.
    """

    def __init__(self, capacity, longest, threshold=3, step=1 / 16, rate=.1, rounds=2000):
        super().__init__()
        self.capacity, self.longest = int(capacity), int(longest)
        self.threshold, self.step = int(threshold), float(step)
        self.rate, self.rounds = float(rate), int(rounds)
        if self.capacity <= LEAVES:
            raise ValueError(f'FormLinks capacity must exceed {LEAVES}, got {self.capacity}')
        if self.longest < 1:
            raise ValueError(f'FormLinks longest must be positive, got {self.longest}')
        if self.threshold < 0:
            raise ValueError(f'FormLinks threshold must not be negative, got {self.threshold}')
        if not 0 < self.step < .5:
            raise ValueError(f'FormLinks step must lie in (0, 1/2), got {self.step}')
        if not 0 < self.rate <= 1:
            raise ValueError(f'FormLinks rate must lie in (0, 1], got {self.rate}')
        if self.rounds < 0:
            raise ValueError(f'FormLinks rounds must not be negative, got {self.rounds}')
        self.register_buffer('left', torch.full((self.capacity,), -1, dtype=torch.int32))
        self.register_buffer('right', torch.full((self.capacity,), -1, dtype=torch.int32))
        length = torch.zeros(self.capacity, dtype=torch.int32)
        length[:LEAVES] = 1
        self.register_buffer('length', length)
        self.register_buffer('count', torch.zeros(self.capacity, dtype=torch.int32))
        self.register_buffer('used', torch.tensor(LEAVES, dtype=torch.int64))
        self.register_buffer('pair_keys', torch.full((self.capacity,), UNUSED_KEY, dtype=torch.int64))
        self.register_buffer('pair_rows', torch.arange(self.capacity, dtype=torch.int64))
        code = torch.zeros(self.capacity, 2 * HALF)
        code[:LEAVES, :HALF] = self.step * self._points(torch.arange(LEAVES))
        self.register_buffer('code', code)
        self.register_buffer('holder_sum', torch.zeros(self.capacity, HALF, dtype=torch.int64))
        self.register_buffer('holder_count', torch.zeros(self.capacity, dtype=torch.int32))

    @staticmethod
    def _mul32(x, c):
        """``x * c`` mod 2^32 for x below 2^32, in two 16-bit halves of c so no int64 product overflows."""
        low = x * (c & 0xFFFF)
        high = ((x * (c >> 16)) & 0xFFFF) << 16
        return (low + high) & MASK32

    @classmethod
    def _points(cls, rows):
        """Uniform point [N, HALF] in [0, 1) of each row: lowbias32 of ``row * HALF + coordinate``, top 24 bits."""
        x = (rows.long()[:, None] * HALF + torch.arange(HALF, device=rows.device)) & MASK32
        x = x ^ (x >> 16)
        x = cls._mul32(x, 0x7feb352d)
        x = x ^ (x >> 15)
        x = cls._mul32(x, 0x846ca68b)
        x = x ^ (x >> 16)
        return (x >> 8).float() * 2.0 ** -24

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

    # -- cover --------------------------------------------------------------

    def cover(self, data, unit_spans, parts=None, rows=None):
        """Tile each unit left to right by the longest promoted form that starts at the byte and ends inside it.

        ``data`` [B, L] bytes; ``unit_spans`` [B, W, 2] (start, end) byte
        extents, empty where start >= end; ``rows`` optionally ``recognise``'s
        rows over the whole width. Returns part rows [B, W, P], extents
        [B, W, P, 2] (-1 past the tiles) and ``counts`` [B, W], the number of
        tiles of the unit, which may exceed P (tiles past P are not listed).
        Bytes are always available. Tiles by pointer jumping, no host read.
        """
        B, L = data.shape
        P = self.longest if parts is None else int(parts)
        if rows is None:
            rows, _ = self.recognise(data, torch.full((B,), L, device=data.device))
        K = rows.shape[1]
        start, end = unit_spans[..., 0].long(), unit_spans[..., 1].long()
        pos = torch.arange(L, device=data.device)
        inside = (pos >= start[..., None]) & (pos < end[..., None])
        room = torch.where(inside, end[..., None], 0).amax(1) - pos
        n = torch.arange(1, K + 1, device=data.device)[None, :, None]
        ok = ((rows >= 0) | (n == 1)) & (n <= room[:, None, :])
        step = (ok * n).amax(1)
        nxt = torch.where((step > 0) & (step < room), pos + step, L)
        sentinel = torch.full((B, 1), L, dtype=torch.long, device=data.device)
        nxt = torch.cat((nxt, sentinel), 1)
        step = torch.cat((step, torch.zeros_like(sentinel)), 1)
        first = torch.where(start < end, start, L).clamp(0, L)
        seq, jump = first[..., None], nxt
        while seq.shape[-1] < P:
            seq = torch.cat((seq, torch.gather(jump, 1, seq.reshape(B, -1)).reshape(B, seq.shape[1], -1)), -1)
            jump = torch.gather(jump, 1, jump)
        seq = seq[..., :P]
        tiles, jump = (step > 0).long(), nxt
        for _ in range(max(1, (L + 1).bit_length())):
            tiles = tiles + torch.gather(tiles, 1, jump) * (jump < L)
            jump = torch.gather(jump, 1, jump)
        counts = torch.gather(tiles, 1, first)
        flat = seq.reshape(B, -1)
        size = torch.gather(step, 1, flat)
        real = (flat < L) & (size > 0)
        at = ((size - 1).clamp_min(0) * L + flat.clamp_max(L - 1))
        part = torch.where(real, torch.gather(rows.reshape(B, -1), 1, at), -1).reshape(B, -1, P)
        extent = torch.stack((torch.where(real, flat, -1), torch.where(real, flat + size, -1)), -1)
        return part, extent.reshape(B, -1, P, 2), counts

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

    # -- count ----------------------------------------------------------------

    def _unit_rows(self, data, lengths, longest=None):
        """Row of each whole unit [U], counted or not (-1 where unknown or empty).

        ``longest``: a static bound on the units' lengths; levels above it are not built.
        """
        cur = self._leaves(data, lengths)
        unit = torch.where(lengths == 1, cur[:, 0], -1)
        for n in range(2, self._top(data, longest) + 1):
            cur = self._level(cur)
            unit = torch.where(lengths == n, cur[:, 0], unit)
        return unit

    def _top(self, data, longest):
        """The longest level a call builds: the data's width, or the caller's static bound below it."""
        return data.shape[1] if longest is None else min(data.shape[1], int(longest))

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
    def read(self, data, lengths, admit=True, longest=None):
        """Row of each whole unit [U] (-1 where unknown or empty); with ``admit``, insert what the units lack and count them first.

        ``longest``: a static bound on the units' lengths (see ``insert``).
        """
        if not admit:
            return self._unit_rows(data, lengths, longest)
        unit = self._insert(data, lengths, longest)
        self._raise(unit)
        return unit

    # -- insert --------------------------------------------------------------

    def _append(self, keys, n):
        """Rows of length n at ``used`` for the distinct keys of ``keys`` [N] (``UNUSED_KEY`` for none).

        Returns the new keys sorted [N] (``UNUSED_KEY`` past them) and their rows.
        """
        size, cap = keys.shape[0], self.capacity
        ranked = torch.sort(keys).values
        first = (ranked != UNUSED_KEY) & F.pad(ranked[1:] != ranked[:-1], (1, 0), value=True)
        places = torch.arange(size, device=keys.device)
        new = torch.full((2 * size,), UNUSED_KEY, dtype=torch.int64, device=keys.device)
        new = new.index_copy(0, torch.where(first, torch.cumsum(first, 0) - 1, size + places), ranked)[:size]
        count = first.sum()
        torch._assert_async(self.used + count <= cap, 'FormLinks full: raise PartSpace <formCapacity>')
        valid = places < count
        rows = self.used + places
        target = torch.where(valid, rows, 0).clamp_max(cap - 1)
        self.left.index_add_(0, target, torch.where(valid, new // cap + 1, 0).to(self.left.dtype))
        self.right.index_add_(0, target, torch.where(valid, new % cap + 1, 0).to(self.right.dtype))
        self.length.index_add_(0, target, (valid * n).to(self.length.dtype))
        x = torch.where(valid[:, None], self._points(rows), 0.)
        self.code[:, :HALF].index_add_(0, target, x)
        self._register(torch.where(valid, new // cap, 0), torch.where(valid, new % cap, 0), self._quantise(x), valid)
        self.used.add_(count)
        return new, rows

    def _merge(self, keys, rows):
        """Merge keys [N] (any order, ``UNUSED_KEY`` for none) and their rows into the sorted pair index."""
        size, cap = keys.shape[0], self.capacity
        keys, order = torch.sort(keys)
        rows = rows[order]
        places = torch.arange(size, device=keys.device)
        old = self.pair_keys
        at_old = torch.arange(cap, device=keys.device) + torch.searchsorted(keys, old)
        at_new = torch.where(keys != UNUSED_KEY, places + torch.searchsorted(old, keys), cap + size + places)
        keys_out = torch.full((cap + 2 * size,), UNUSED_KEY, dtype=torch.int64, device=keys.device)
        rows_out = torch.zeros(cap + 2 * size, dtype=torch.int64, device=keys.device)
        keys_out.index_copy_(0, at_old, old).index_copy_(0, at_new, keys)
        rows_out.index_copy_(0, at_old, self.pair_rows).index_copy_(0, at_new, rows)
        self.pair_keys.copy_(keys_out[:cap])
        self.pair_rows.copy_(rows_out[:cap])

    @torch.no_grad()
    def insert(self, data, lengths, longest=None):
        """Insert every unit with the substrings it lacks, shortest first; returns the unit rows [B].

        Fixed shape, no host read. A new row takes its hashed point and is
        added to its links' holder sums. A full table raises by a device
        assertion. ``longest``, a static Python bound on the units' lengths,
        limits the levels built; a unit above it raises by a device assertion.
        """
        return self._insert(data, lengths, longest)

    def _insert(self, data, lengths, longest=None):
        total = self._top(data, longest)
        torch._assert_async((lengths <= total).all(), 'FormLinks: a unit is longer than its bound')
        cur = self._leaves(data, lengths)
        unit = torch.where(lengths == 1, cur[:, 0], -1)
        added = []
        for n in range(2, total + 1):
            width = data.shape[1] - n + 1
            a, b = cur[:, :width], cur[:, 1:width + 1]
            found = self.find(a, b)
            key = torch.where((a >= 0) & (b >= 0), a * self.capacity + b, UNUSED_KEY)
            new, new_rows = self._append(torch.where(found < 0, key, UNUSED_KEY).reshape(-1), n)
            added.append((new, new_rows))
            place = torch.searchsorted(new, key).clamp_max(new.shape[0] - 1)
            row = torch.where((found < 0) & (key != UNUSED_KEY) & (new[place] == key), new_rows[place], found)
            cur = F.pad(row, (0, n - 1), value=-1)
            unit = torch.where(lengths == n, cur[:, 0], unit)
        if added:
            self._merge(torch.cat([k for k, _ in added]), torch.cat([r for _, r in added]))
        return unit

    # -- the pull ---------------------------------------------------------------

    def _quantise(self, code):
        """Code in units of 1/SCALE; a power-of-two scale keeps float32 values exact."""
        return (code * SCALE).round().long()

    def _register(self, L, R, amount, counted=None):
        """Add ``amount`` [N, HALF] to the holder sums of links ``L`` and ``R`` (once where equal), and ``counted`` [N] to their counts."""
        other = R != L
        self.holder_sum.index_add_(0, L, amount)
        self.holder_sum.index_add_(0, R, torch.where(other[:, None], amount, 0))
        if counted is not None:
            self.holder_count.index_add_(0, L, counted.to(self.holder_count.dtype))
            self.holder_count.index_add_(0, R, (counted & other).to(self.holder_count.dtype))

    def _pull(self, rows, keep):
        """One step for ``rows`` [N] where ``keep`` (kept rows distinct, all forms); targets from the codes before the step."""
        L = self.left[rows].long().clamp_min(0)
        R = self.right[rows].long().clamp_min(0)
        x = self.code[rows, :HALF]
        m = (self.code[L, :HALF] + self.code[R, :HALF]) / 2
        n = self.holder_count[rows][:, None]
        held = self.holder_sum[rows].to(x.dtype) / (n.clamp_min(1).to(x.dtype) * SCALE)
        top = (1 - self.step + m - m.mean(1, keepdim=True)).clamp(0, 1)
        target = torch.where(n > 0, (m + held) / 2, top)
        moved = torch.where(keep[:, None], x + self.rate * (target - x), x)
        self.code[:, :HALF].index_copy_(0, rows, moved)
        self._register(L, R, torch.where(keep[:, None], self._quantise(moved) - self._quantise(x), 0))

    @torch.no_grad()
    def recentre_rows(self, rows):
        """One pull step for each form in ``rows`` (any shape; -1 and bytes stay) and for its two links, each once.

        The holder sums follow every movement. Fixed shape, no host read.
        """
        flat = rows.reshape(-1).long()
        real = flat >= LEAVES
        at = flat.clamp_min(0)
        links = torch.cat((flat, torch.where(real, self.left[at].long(), -1), torch.where(real, self.right[at].long(), -1)))
        ordered = torch.sort(links).values
        keep = F.pad(ordered[1:] != ordered[:-1], (1, 0), value=True) & (ordered >= LEAVES)
        self._pull(torch.where(keep, ordered, 0), keep)

    @torch.no_grad()
    def sweep(self, rounds=None):
        """Maintenance at a save: ``rounds`` (default ``self.rounds``) pull steps over every form in use. Host side."""
        rows = torch.arange(LEAVES, int(self.used), device=self.code.device)
        keep = torch.ones_like(rows, dtype=torch.bool)
        for _ in range(self.rounds if rounds is None else int(rounds)):
            self._pull(rows, keep)
