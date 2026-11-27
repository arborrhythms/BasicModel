"""WholeSpace observations used when a primitive property divides at reset."""
import torch
from torch import nn


def byte_counts(surface, spans):
    """Count byte occurrences per unit, including a masked byte inside a live unit."""
    lo, hi = spans.unbind(-1)
    batch, units = lo.shape
    ids = torch.arange(1, units + 1, device=surface.device).expand_as(lo)
    changes = surface.new_zeros((batch, surface.shape[1] + 1), dtype=torch.long)
    # The tiling is disjoint. A boundary can close one unit and open the
    # following unit at the same byte. Seat zero owns the uncovered bytes.
    changes.scatter_add_(1, lo, ids)
    changes.scatter_add_(1, hi, -ids)
    owner = changes[:, :-1].cumsum(1)
    address = owner * 256 + surface.long()
    counts = surface.new_zeros((batch, (units + 1) * 256), dtype=torch.int32)
    counts.scatter_add_(1, address, torch.ones_like(address, dtype=torch.int32))
    return counts.view(batch, units + 1, 256)[:, 1:]


class PropertyPulls(nn.Module):
    """Lifetime displacement statistics and the last 512 observations per row.

    Only byte multiplicities affect a split. A positive or negative projection
    casts one vote for every occurrence of the byte in its unit.

    The statistics are kept for every row. The histories are kept by seat.
    With ``SEATS`` rows or fewer a row's seat is itself: an observation reads
    every row, and nothing of it is read back to the host. With more (a wide
    whole space has 65,536 rows, whose histories would be tens of gigabytes)
    an observation reads only the rows some unit holds, at the cost of one
    read of how many those are, and a row takes a seat when it is observed:
    a free one, or else the seat of the row observed longest ago, which
    starts again from nothing (its count and its sums, as at
    :meth:`reset_row`). An observation that holds more rows than there are
    seats observes the first ``SEATS`` of them.
    """
    history = 512
    SEATS = 1024

    def __init__(self, weights):
        super().__init__()
        rows, width = weights.shape
        seats = min(rows, self.SEATS)
        bits = torch.int64 if weights.dtype == torch.float64 else torch.int32
        def column(name, shape, dtype):
            self.register_buffer(name, weights.new_zeros(shape, dtype=dtype), persistent=False)
        # Preserve the accumulator precision across model.half() and device moves.
        column('moments', (2, rows, width), bits)
        column('deltas', (seats, self.history, width), bits)
        column('letters', (seats, self.history, 256), torch.int32)
        column('count', (rows,), torch.long)
        column('first', (rows,), torch.long)
        column('clock', (), torch.long)
        # Each row's seat in the histories: the row itself where every row
        # has one. Otherwise -1 until the row is observed, with the row on
        # each seat (-1 while it is free) and the clock at which the seat
        # was last observed (-1 while it never was).
        wide = seats < rows
        unseated = torch.full((seats,), -1, dtype=torch.long, device=weights.device) if wide else None
        self.register_buffer('seat_of', torch.full((rows,), -1, dtype=torch.long, device=weights.device) if wide
                             else torch.arange(rows, device=weights.device), persistent=False)
        self.register_buffer('row_of', unseated, persistent=False)
        self.register_buffer('seen', None if unseated is None else unseated.clone(), persistent=False)

    @property
    def precision(self):
        return torch.float64 if self.moments.dtype == torch.int64 else torch.float32

    def seats(self, rows, now):
        """The seats of ``rows`` in the histories at the clock ``now``.

        ``rows`` are distinct, and no more of them than there are seats, so
        each without a seat is given one that no row of ``rows`` is on: the
        seats are taken longest unobserved first, the free ones before any.
        The row a seat is taken from, and the row that takes it, start again
        from nothing. Every operand keeps its shape, and nothing is read
        back to the host.
        """
        if self.row_of is None:
            return rows
        seat = self.seat_of[rows]
        new = seat < 0
        none, zero = torch.full_like(seat, -1), torch.zeros_like(seat)
        # The seats observed now are the latest, so none of them is taken.
        self.seen.scatter_reduce_(0, seat.clamp_min(0), torch.where(new, none, now), reduce='amax')
        pick = torch.argsort(self.seen)[(new.cumsum(0) - 1).clamp_min(0)]
        before = self.row_of[pick]
        evicted = new & (before >= 0)
        # Distinct addresses: a row whose seat is taken is not among ``rows``.
        index = torch.where(evicted, before, rows)
        self.seat_of[index] = torch.where(evicted, none, self.seat_of[index])
        for at, started in ((index, evicted), (rows, new)):
            self.count[at] = torch.where(started, zero, self.count[at])
            sums = self.moments[:, at]
            self.moments[:, at] = sums * (~started)[None, :, None].to(sums.dtype)
        seat = torch.where(new, pick, seat)
        self.seat_of[rows] = seat
        self.row_of[seat] = rows
        self.seen[seat] = now
        return seat

    @torch.no_grad()
    def observe(self, held, codes, weights, spellings):
        """Observe units in page and reading order, with all operands on device."""
        units, rows = held.shape
        if not units or not rows:
            return
        if weights.dtype == torch.float64 and self.precision != torch.float64:
            self.moments = self.moments.view(torch.float32).double().view(torch.int64)
            self.deltas = self.deltas.view(torch.float32).double().view(torch.int64)
        if self.row_of is None:
            # Every row has its own seat: all are read, as they stand.
            active, at = self.seat_of, slice(None)
        else:
            # Only the rows some unit holds are read or written: every other
            # row's statistics and history stand as they were.
            active = at = held.any(0).nonzero().flatten()[:self.deltas.shape[0]]
            if not len(active):
                self.clock.add_(units)
                return
            held = held[:, active]
        seat = self.seats(active, self.clock + units)[:, None]
        width = min(codes.shape[-1], weights.shape[-1])
        delta = codes[:, None, :width].to(weights) - weights[None, at, :width]
        squares = delta.square()
        moment = self.moments.view(self.precision)
        total, square = moment[:, at, :width].unbind(0)
        # The split threshold reads float sums accumulated in observation order.
        for value, squared, mask in zip(delta.unbind(), squares.unbind(), held.unbind()):
            total = torch.where(mask[:, None], total + value, total)
            square = torch.where(mask[:, None], square + squared, square)
        moment[:, at, :width] = torch.stack((total, square))
        prefix = held.T.cumsum(-1)
        amount = prefix[:, -1]
        unit = torch.arange(units, device=codes.device)
        first = torch.where(held, unit[:, None], units).amin(0)
        count = self.count[at]
        self.first[at] = torch.where((count == 0) & (amount > 0), self.clock + first, self.first[at])
        self.clock.add_(units)
        # Each destination occurs at most once even if a presentation exceeds
        # the entire history. Unoccupied seats retain their previous observations.
        keep = min(units, self.history)
        ordinal = (amount - self.history).clamp_min(0)[:, None] + unit[:keep]
        source = torch.searchsorted(prefix.contiguous(), (ordinal + 1).contiguous()).clamp_max(units - 1)
        slot = (count[:, None] + ordinal) % self.history
        row = torch.arange(held.shape[1], device=codes.device)[:, None]
        live = (ordinal < amount[:, None])[..., None]
        stored = self.deltas.view(self.precision)[..., :width]
        stored[seat, slot] = torch.where(live, delta[source, row], stored[seat, slot])
        self.letters[seat, slot] = torch.where(live, spellings[source], self.letters[seat, slot])
        self.count[at] = count + amount

    def restore_records(self, records):
        """Read the previous WholeSpace row records, including their insertion order."""
        moments = self.moments.view(self.precision)
        deltas = self.deltas.view(self.precision)
        for row, record in records.items():
            self.first[row] = self.clock
            self.clock.add_(1)
            recent = record['pulls'][-self.history:]
            # A row is seated before its statistics are written: taking a
            # seat starts a row again.
            seat = int(self.seats(self.seat_of.new_tensor([int(row)]), self.clock.clone())[0]) if recent else None
            n = int(record['n'])
            self.count[row] = n
            width = record['sum'].numel()
            moments[:, row, :width].copy_(torch.stack((record['sum'], record['sq'])))
            for index, (delta, spelling) in enumerate(recent, start=n - len(recent)):
                slot = index % self.history
                deltas[seat, slot, :width].copy_(delta)
                ids = torch.tensor(list(spelling), dtype=torch.long, device='cpu')
                self.letters[seat, slot].copy_(torch.bincount(ids, minlength=256))

    def pending(self, minimum, threshold=None):
        """At reset, offer eligible rows in their original insertion order."""
        live = self.count >= minimum
        if threshold is not None:
            total, squared = self.moments.view(self.precision).unbind(0)
            n = self.count.clamp_min(1)[:, None]
            variance = squared / n - (total / n).square()
            live = live & ~(variance.amax(-1) < threshold)
        ordered = torch.argsort(self.first, stable=True)
        return ordered[live[ordered]].tolist()

    def reset_row(self, row):
        # The next observation overwrites seat zero. Stale suffixes never vote.
        self.count[row] = 0
        self.moments[:, row].zero_()

    def split_bytes(self, row, count, direction):
        n = min(count, self.history)
        seat = int(self.seat_of[row])
        if seat < 0:
            # A row never observed has no history and votes for no byte.
            return torch.zeros(256, dtype=torch.bool, device=self.letters.device)
        delta = self.deltas.view(self.precision)[seat, :n]
        side = ((delta * direction).sum(-1) >= 0).int() * 2 - 1
        return (self.letters[seat, :n] * side[:, None]).sum(0) < 0
