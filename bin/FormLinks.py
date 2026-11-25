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
SCALE = 2.0 ** 32  # fixed point of the running sums: integer adds are exact in any order


class FormLinks(nn.Module):
    """Rows 0..255 are the byte leaves; each later row links two rows one byte shorter.

    Rows are numbered as inserted, parts before wholes, so both links of a
    row are below it. ``pair_keys`` holds ``left * capacity + right`` of the
    rows in use, sorted, and ``pair_rows`` the row of each key; unused
    entries hold ``UNUSED_KEY``. Buffers are allocated once at ``capacity``.

    ``count`` is a row's readings as a whole unit; recognise returns a row
    only once its count reaches ``threshold`` (leaves always), else its parts.

    ``code`` [capacity, 2*HALF]: the parts half of a leaf is ``step`` on
    ``LEAF_BITS`` coordinates of its own; of a form, halfway between the
    mean of its distinct proper substrings and the mean of the distinct
    forms holding it (``1 - step`` everywhere when none holds it). Placed so
    at insertion, recentred when read by ``close``. The wholes half is not
    written here.

    Running sums in units of 1/SCALE (int64), by row:

    - ``part_sum`` [capacity, HALF], ``part_count`` [capacity]: the codes and
      number of the distinct proper substrings. With L, R the links, M the
      middle (``right[L]``, none at length 2) and u = ``border``,
      ``S(F) = S(L) + S(R) - S(M) + x(L) + x(R) - x(M) - x(u)``: the parts
      of L and of R overlap in those of M plus u, the longest stretch that
      begins and ends F and does not occur inside M (L itself when L == R).
      Refreshed for every form of the units passed to ``insert`` or
      ``close``, parts before wholes; a form not passed keeps the sum of
      its parts' codes as they were when it was last passed.
    - ``whole_sum`` [capacity, HALF], ``whole_count`` [capacity]: the codes
      and number of the forms holding the row, exact at all times; a moved
      or inserted form's change reaches its distinct parts through the same
      overlap rule, top-down (``_spread``).
    """

    def __init__(self, capacity, longest, threshold=3, step=1 / 16):
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
        self.register_buffer('part_sum', torch.zeros(self.capacity, HALF, dtype=torch.int64))
        self.register_buffer('part_count', torch.zeros(self.capacity, dtype=torch.int32))
        self.register_buffer('border', torch.full((self.capacity,), -1, dtype=torch.int32))
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

    # -- count (post-sentence reset) ------------------------------------------

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
    def read(self, data, lengths, admit=True):
        """Row of each whole unit [U] (-1 where unknown or empty); with ``admit``, insert what the units lack and count them first."""
        if not admit:
            return self._unit_rows(data, lengths)
        unit = self._insert(data, lengths)[0]
        self._raise(unit)
        return unit

    @torch.no_grad()
    def close(self, data, lengths):
        """The post-sentence reset: insert what the units lack, raise their counts, recentre their forms."""
        unit, levels = self._insert(data, lengths)
        self._raise(unit)
        self._recentre(levels)
        return unit

    # -- running sums -----------------------------------------------------------

    def _quantise(self, code):
        """Code in units of 1/SCALE; a power-of-two scale keeps float32 values exact."""
        return (code * SCALE).round().long()

    def _q(self, rows):
        return self._quantise(self.code[rows, :HALF])

    def _whole_mean(self, rows):
        """Mean code [F, HALF] of the forms holding ``rows``; ``1 - step`` where none."""
        count = self.whole_count[rows].to(self.code.dtype)[:, None]
        mean = self.whole_sum[rows].to(self.code.dtype) / (count.clamp_min(1) * SCALE)
        return torch.where(count > 0, mean, 1 - self.step)

    def _sums(self, rows, n, u):
        """Distinct-part sum [F, HALF] and count [F] of forms ``rows`` (>= 0) of length n with border ``u``."""
        L = self.left[rows].long().clamp_min(0)
        R = self.right[rows].long().clamp_min(0)
        M = self.right[L].long().clamp_min(0)
        has = u >= 0
        links = torch.stack((L, R, M, u.clamp_min(0)))
        sign = torch.tensor([1, 1, -1, -1], device=rows.device)[:, None]
        sign = sign * torch.stack((torch.ones_like(has), torch.ones_like(has), torch.full_like(has, n >= 3), has))
        q = (self._q(links) * sign[..., None]).sum(0)
        held = (self.part_sum[links[:3]] * sign[:3, :, None]).sum(0)
        C = (self.part_count[links[:3]].long() * sign[:3]).sum(0) + sign.sum(0)
        return held + q, C

    def _centre(self, rows, S, C):
        """Halfway between the parts' mean (``S``/``C``) and the wholes' mean."""
        part_mean = S.to(self.code.dtype) / (C.clamp_min(1).to(self.code.dtype)[:, None] * SCALE)
        return (part_mean + self._whole_mean(rows)) / 2

    def _flat(self, levels, n):
        """Rows of length n at each start that can hold one, flattened [B*(W-n+1)]."""
        width = levels[0].shape[1]
        return levels[n - 1][:, :width - n + 1].reshape(-1)

    def _claim(self, flat):
        """First place of each row in ``flat`` [P], as a [capacity] map, and the mask of first places."""
        places = torch.arange(flat.shape[0], device=flat.device)
        rows = flat.clamp_min(0)
        claim = torch.full((self.capacity,), flat.shape[0], dtype=torch.int64, device=flat.device)
        claim.scatter_reduce_(0, rows, torch.where(flat >= 0, places, flat.shape[0]), 'amin')
        return claim, (flat >= 0) & (claim[rows] == places)

    def _spread(self, levels, own):
        """Add each form's amount (``own[n]`` [P_n, HALF+1]: code sum, count) to the whole sums of its distinct parts.

        Top-down by the overlap rule: a form's amount goes to L and R and
        their parts, back from M and its parts, and back from u. An amount
        held for a row and its parts collects at the row's first place in
        its level, is added to the row, and goes on the same way.
        """
        claims, pend = {}, {}

        def claim(n):
            if n not in claims:
                claims[n] = self._claim(self._flat(levels, n))
            return claims[n]

        for n in range(len(levels), 0, -1):
            held = pend.pop(n, None)
            if held is None and n not in own:
                continue
            flat = self._flat(levels, n)
            _, first = claim(n)
            rows = flat.clamp_min(0)
            if held is not None:
                held = torch.where(first[:, None], held, 0)
                self.whole_sum.index_add_(0, rows, held[:, :HALF])
                self.whole_count.index_add_(0, rows, held[:, HALF].to(self.whole_count.dtype))
            if n == 1:
                break
            amount = own[n] if held is None else held if n not in own else held + own[n]
            amount = torch.where(first[:, None], amount, 0)
            L = self.left[rows].long().clamp_min(0)
            R = self.right[rows].long().clamp_min(0)
            below, _ = claim(n - 1)
            size = self._flat(levels, n - 1).shape[0]
            to = torch.where(first, torch.cat((below[L], below[R])).view(2, -1), 0).clamp_max(size - 1)
            add = torch.zeros(size, HALF + 1, dtype=torch.int64, device=flat.device) if n - 1 not in pend else pend[n - 1]
            pend[n - 1] = add.index_add_(0, to.reshape(-1), torch.cat((amount, amount)))
            if n >= 3:
                M = self.right[L].long().clamp_min(0)
                lower, _ = claim(n - 2)
                size = self._flat(levels, n - 2).shape[0]
                to = torch.where(first, lower[M], 0).clamp_max(size - 1)
                add = torch.zeros(size, HALF + 1, dtype=torch.int64, device=flat.device) if n - 2 not in pend else pend[n - 2]
                pend[n - 2] = add.index_add_(0, to, -amount)
            u = self.border[rows].long()
            back = torch.where((first & (u >= 0))[:, None], -amount, 0)
            self.whole_sum.index_add_(0, u.clamp_min(0), back[:, :HALF])
            self.whole_count.index_add_(0, u.clamp_min(0), back[:, HALF].to(self.whole_count.dtype))
            claims.pop(n)

    # -- insert --------------------------------------------------------------

    def _next_same(self, rows, k):
        """Place of the next equal row of length k within each unit [B, W]; W where none."""
        width = rows.shape[1]
        part = rows[:, :width - k + 1]
        places = torch.arange(part.shape[1], device=rows.device, dtype=torch.int32)
        later = places[None, :] > places[:, None]
        same = (part[:, :, None] == part[:, None, :]) & (part[:, :, None] >= 0) & later
        return F.pad(torch.where(same, places, width).amin(-1).long(), (0, k - 1), value=width)

    def _border(self, levels, nexts, rows, n):
        """Row of u for the forms ``rows`` [B, w] of length n at each start; -1 where none.

        A stretch of length k is a border where the rows of the prefix and of
        the suffix of length k agree, and lies outside M where its next
        occurrence after the start is past n-1-k. More than one raises.
        """
        width = rows.shape[1]
        k = torch.arange(1, n, device=rows.device)[:, None, None]
        places = torch.arange(width, device=rows.device)
        whole = torch.stack(levels[:n - 1])
        pre, nxt = whole[..., :width], torch.stack(nexts[:n - 1])[..., :width]
        suf = torch.gather(whole, 2, (places + n - k).expand_as(pre))
        found = (rows >= 0) & (pre >= 0) & (pre == suf) & (nxt > places + n - 1 - k)
        torch._assert_async((found.sum(0) <= 1).all(), 'FormLinks: a form needs two borders')
        return torch.where(found, pre, -1).amax(0)

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
    def insert(self, data, lengths):
        """Insert every unit with the substrings it lacks, shortest first; returns the unit rows [B].

        Fixed shape, no host read. A new row is placed at the centroid of its
        parts and its wholes, and added to the whole sums of its parts; the
        part sums of every form of the units are refreshed. A full table
        raises by a device assertion.
        """
        return self._insert(data, lengths)[0]

    def _insert(self, data, lengths):
        total = data.shape[1]
        cur = self._leaves(data, lengths)
        unit = torch.where(lengths == 1, cur[:, 0], -1)
        levels, nexts, own, added = [cur], [self._next_same(cur, 1)], {}, []
        before = self.used.clone()
        for n in range(2, total + 1):
            width = total - n + 1
            a, b = cur[:, :width], cur[:, 1:width + 1]
            found = self.find(a, b)
            key = torch.where((a >= 0) & (b >= 0), a * self.capacity + b, UNUSED_KEY)
            new, new_rows = self._append(torch.where(found < 0, key, UNUSED_KEY).reshape(-1), n)
            added.append((new, new_rows))
            place = torch.searchsorted(new, key).clamp_max(new.shape[0] - 1)
            row = torch.where((found < 0) & (key != UNUSED_KEY) & (new[place] == key), new_rows[place], found)
            u = self._border(levels, nexts, row, n)
            keep, rows = (row >= 0).reshape(-1), row.clamp_min(0).reshape(-1)
            S, C = self._sums(rows, n, u.reshape(-1))
            fresh = (row >= before).reshape(-1) & keep
            x = torch.where(fresh[:, None], self._centre(rows, S, C), self.code[rows, :HALF])
            self.part_sum.index_copy_(0, rows, torch.where(keep[:, None], S, self.part_sum[rows]))
            self.part_count.index_copy_(0, rows, torch.where(keep, C, self.part_count[rows].long()).to(torch.int32))
            self.border.index_copy_(0, rows, torch.where(keep, u.reshape(-1), self.border[rows].long()).to(torch.int32))
            self.code[:, :HALF].index_copy_(0, rows, x)
            own[n] = torch.cat((torch.where(fresh[:, None], self._quantise(x), 0), fresh[:, None].long()), 1)
            cur = F.pad(row, (0, n - 1), value=-1)
            levels.append(cur)
            nexts.append(self._next_same(cur, n))
            unit = torch.where(lengths == n, cur[:, 0], unit)
        if added:
            self._merge(torch.cat([k for k, _ in added]), torch.cat([r for _, r in added]))
        self._spread(levels, own)
        return unit, levels

    # -- recentre -------------------------------------------------------------

    @torch.no_grad()
    def recentre(self, data, lengths):
        """Recentre every form of the units, level by level from length 2."""
        cur = self._leaves(data, lengths)
        levels = [cur]
        for _ in range(2, data.shape[1] + 1):
            cur = self._level(cur)
            levels.append(cur)
        self._recentre(levels)

    @torch.no_grad()
    def recentre_rows(self, rows, data, lengths):
        """Recentre only ``rows`` (any shape; -1 and bytes stay), each once.

        A row goes halfway between its kept part centroid (``part_sum`` /
        ``part_count`` as stored) and its kept whole centroid. Every row must
        occur in ``data`` [B, L] (``lengths`` [B]); the move reaches the whole
        sums of the row's distinct parts by ``_spread`` over the level table
        of ``data``. Fixed shape, no host read.
        """
        ordered = torch.sort(torch.where(rows.reshape(-1) >= LEAVES, rows.reshape(-1).long(), -1)).values
        safe, keep = ordered.clamp_min(0), ordered >= 0
        count = self.part_count[safe].to(self.code.dtype)[:, None]
        part_mean = self.part_sum[safe].to(self.code.dtype) / (count.clamp_min(1) * SCALE)
        old = self.code[safe, :HALF]
        x = torch.where(keep[:, None] & (count > 0), (part_mean + self._whole_mean(safe)) / 2, old)
        delta = self._quantise(x) - self._quantise(old)
        self.code[:, :HALF].index_copy_(0, safe, x)
        levels = [self._leaves(data.long(), lengths)]
        for _ in range(1, min(self.longest, data.shape[1])):
            levels.append(self._level(levels[-1]))
        own = {}
        for n in range(2, len(levels) + 1):
            held = self._flat(levels, n)
            at = torch.searchsorted(ordered, held.contiguous()).clamp_max(ordered.shape[0] - 1)
            hit = (held >= 0) & (ordered[at] == held)
            own[n] = F.pad(torch.where(hit[:, None], delta[at], 0), (0, 1))
        self._spread(levels, own)

    @torch.no_grad()
    def sweep(self, chunk=4096):
        """Maintenance at a save: every form's part sum from the current codes, and the form recentred.

        Length by length from 2, parts before wholes, at most ``chunk`` rows
        at a time; each move reaches the whole sums of the form's parts by
        ``_spread`` over the level table of its own spelling, so the whole
        sums stay exact. Bytes stay. Host side; not captured.
        """
        used = int(self.used)
        longest = int(self.length[:used].max()) if used > LEAVES else 1
        for n in range(2, longest + 1):
            found = (self.length[:used] == n).nonzero()[:, 0]
            for start in range(0, found.shape[0], chunk):
                rows = found[start:start + chunk]
                S, C = self._sums(rows, n, self.border[rows].long())
                self.part_sum.index_copy_(0, rows, S)
                self.part_count.index_copy_(0, rows, C.to(self.part_count.dtype))
                old = self.code[rows, :HALF]
                x = self._centre(rows, S, C)
                self.code[:, :HALF].index_copy_(0, rows, x)
                data, lengths = self.spell(rows, n)
                levels = [self._leaves(data.clamp_min(0), lengths)]
                for _ in range(1, n):
                    levels.append(self._level(levels[-1]))
                self._spread(levels, {n: F.pad(self._quantise(x) - self._quantise(old), (0, 1))})

    def _recentre(self, levels):
        """Each level's forms from their refreshed part sums and their wholes; moves reach the whole sums after."""
        own = {}
        for n in range(2, len(levels) + 1):
            cur = self._flat(levels, n)
            keep, rows = cur >= 0, cur.clamp_min(0)
            S, C = self._sums(rows, n, self.border[rows].long())
            old = self.code[rows, :HALF]
            x = torch.where(keep[:, None], self._centre(rows, S, C), old)
            self.part_sum.index_copy_(0, rows, torch.where(keep[:, None], S, self.part_sum[rows]))
            self.code[:, :HALF].index_copy_(0, rows, x)
            own[n] = F.pad(self._quantise(x) - self._quantise(old), (0, 1))
        self._spread(levels, own)
