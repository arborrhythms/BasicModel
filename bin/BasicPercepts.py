"""Basic-level occurrences and their promoted, fixed-capacity percept table."""
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


#: The where wires and the when wires of an attended percept.
WHERE_WIRES = WHEN_WIRES = 4


def attended_capacity():
    """The rows of the table a look presents, ``attendedPercepts``."""
    from util import TheXMLConfig
    return int(TheXMLConfig.get('architecture.attendedPercepts', default=256))


def extent_members(spans, apertures):
    """Located parts within each aperture and wholes containing it, both [B, K, O]."""
    start, end = spans[:, None].unbind(-1)
    lo, hi = apertures.unbind(-1)
    lo, hi = lo[..., None], hi[..., None]
    live = (end > start) & (hi > lo)
    return live & (start >= lo) & (end <= hi), live & (start <= lo) & (end >= hi)


def scoped_max(values, scope):
    """Pool located values [R,B,O,2] over each aperture without an R×B×K×O allocation."""
    if not scope.shape[1]:
        return values[:, :, :0]
    # Keep the row-by-occurrence Pi table independent of the reading axis.
    # The extra zero is the initial value of the original scatter reduction,
    # including its share of the gradient when its lane ties at zero.
    maxima = []
    for mask in scope.unbind(1):
        visible = values.masked_fill(~mask[None, ..., None], -torch.inf)
        maxima.append(F.pad(visible, (0, 0, 0, 1)).amax(2))
    return torch.stack(maxima, dim=2)


def reading_scope(bank, width):
    """Occurrences in each reading's aperture, with each matching extent counted once."""
    scope = getattr(bank, 'scope', None)
    if scope is None:
        if getattr(bank, 'where', None) is None or getattr(bank, 'span', None) is None:
            slots = torch.arange(width, device=bank.token.device)
            return bank.token[:, None] == slots[None, :, None]
        inside, around = extent_members(bank.span, bank.where)
        scope = inside | around
    if width > scope.shape[1]:
        return F.pad(scope, (0, 0, 0, width - scope.shape[1]))
    return scope[:, :width]


@dataclass(frozen=True)
class PerceptBank:
    """The location-bound percepts of a sentence, made once by its one
    lookup and held on the device: every occurrence at every level of the
    percept tower (a part's event, a known form of several units, a whole's
    span, a reading's bracket), each as its muxed row (``table``: its code's
    two lanes, the where wires and the when wires, each wire ``c`` on the
    lanes ``c`` and ``1 - c``), with its span, the reading that holds it and
    its time; and the code of each row of the first Pi the lookup names, on
    two lanes (``codes``: for, then against). A look reads a table of at
    most ``attendedPercepts`` of them (:func:`attended`)."""
    rows: tuple
    codes: torch.Tensor             # [rows, 2, width] each row's code: for, then against
    span: torch.Tensor              # [B, occurrences, 2]
    token: torch.Tensor             # [B, occurrences] the reading that holds it, or -1
    time: torch.Tensor              # [B, occurrences] the time of that reading
    table: torch.Tensor             # [B, occurrences, width + where + when, 2]
    activity: torch.Tensor          # [B, occurrences, 2] its own activity, on two lanes
    where: torch.Tensor             # [B, readings, 2] each reading's bracket
    when: torch.Tensor              # [B, readings, 2] each reading's interval
    width: int = 0                  # the what columns of ``table``
    identity_width: int = 0         # the parts half of the what columns
    when_band: torch.Tensor = None       # [B, readings, 4]
    registry: object = None              # the where wires (``WhereRegistry``)
    # [B, occurrences] whether an occurrence can tell against a row it
    # holds: a part's event, a stored form, or a run of a row's parts.
    # ``None``: every occurrence can.
    holder: torch.Tensor = None
    # Derived from this table’s spans. Replacing its geometry starts a fresh scope.
    scope: torch.Tensor = field(default=None, init=False, repr=False, compare=False)
    # Request-independent Pi reads, scoped to this sentence's opening:
    # part, whole, added, outside, live location factor, stored location factor.
    # replace(bank, ...) deliberately drops these: another table is another read.
    read_basis: tuple = field(default=None, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class Attended:
    """A look's table: each attended percept's occurrence ``index`` ``[B,
    A]``, whether the row is filled, its muxed row (the where columns
    gathered from the where wires), its span, reading and time."""
    index: torch.Tensor
    filled: torch.Tensor
    table: torch.Tensor       # [B, A, width + where + when, 2]
    span: torch.Tensor        # [B, A, 2]
    token: torch.Tensor       # [B, A]
    time: torch.Tensor        # [B, A]


def attended(bank, share=None, capacity=None):
    """The occurrences a look attends and their table (:class:`Attended`).

    All of them where they fit. Otherwise the ``capacity`` most salient by
    ``share`` ``[B, O]`` (rule 79: "the most salient are those that are
    returned"), a draw among equals; no order by extent."""
    capacity = attended_capacity() if capacity is None else int(capacity)
    span = bank.span
    B, O = bank.token.shape
    real = span[..., 1] > span[..., 0]
    if O <= capacity:
        index = torch.arange(O, device=span.device)[None].expand(B, -1)
        filled = real
    else:
        share = torch.zeros(B, O, device=span.device) if share is None else share.detach().to(span.device)
        key = torch.where(real, share.float(), -torch.inf)
        order = torch.rand(B, O, device=span.device).argsort(-1)
        order = order.gather(1, (-key.gather(1, order)).argsort(dim=-1, stable=True))
        index = order[:, :capacity]
        filled = real.gather(1, index)
    table = bank.table.gather(1, index[..., None, None].expand(-1, -1, *bank.table.shape[2:]))
    chosen = span.gather(1, index[..., None].expand(-1, -1, 2))
    if bank.registry is not None:
        start = torch.where(filled, chosen[..., 0], -1)
        wires = bank.registry.encode('text', start).to(table)
        lanes = torch.stack((wires, 1 - wires), -1) * filled[..., None, None]
        table = torch.cat((table[..., :bank.width, :], lanes, table[..., bank.width + wires.shape[-1]:, :]), -2)
    token = torch.where(filled, bank.token.gather(1, index), -1)
    return Attended(index, filled, table, chosen, token, bank.time.gather(1, index))


def held_entries(entries):
    """Entries read at zero or above, the derivative passing at zero."""
    return torch.where(entries >= 0, entries, 0.)


def lesser(a, b):
    """The lesser of ``a`` and ``b``; at a tie the derivative reaches both."""
    return torch.where(a < b, a, b) + torch.where(a == b, a - a.detach(), 0.)


def field_factor(lanes, entries):
    """How far an attended percept holds a row's entries on one field (its
    where, or its when), by shared size: ``sum min(lane, entry) / sum
    entry`` over the wires' two lanes, ``[R, B, A]``, for ``lanes`` ``[B,
    A, wires, 2]`` and ``entries`` ``[R, wires, 2]`` (at zero or above). One
    where a row's entries sum to nothing."""
    total = entries.sum((-1, -2))                                              # [R]
    R = entries.shape[0]
    out = lanes.new_ones(R, *lanes.shape[:2])
    live = (total > 0).nonzero().flatten()
    if len(live):
        e = entries.index_select(0, live)
        shared = lesser(lanes[None], e[:, None, None]).sum((-1, -2))          # [L, B, A]
        out = out.index_copy(0, live, shared / total.index_select(0, live)[:, None, None])
    return out


def field_share(lanes, entries):
    """The derivative of :func:`field_factor` by each of the percept's lanes
    times the lane, summed, ``[R, B, A]``: ``sum lane . [lane <= entry] /
    sum entry``; nothing where a row's entries sum to nothing."""
    total = entries.sum((-1, -2))
    out = lanes.new_zeros(entries.shape[0], *lanes.shape[:2])
    live = (total > 0).nonzero().flatten()
    if len(live):
        e = entries.index_select(0, live)
        x = lanes[None]
        held = (x <= e[:, None, None]).to(x)
        out = out.index_copy(0, live, (x * held).sum((-1, -2)) / total.index_select(0, live)[:, None, None])
    return out


def least_of_greater(x, c, chunk=1 << 24):
    """``min_k max(x[b, a, k], c[r, k])``, ``[R, B, A]``, for wires ``x``
    ``[B, A, K]`` and a constant ``c`` ``[R, K]``. The least is found
    without a gradient, a slab of rows at a time, and its value is then
    taken from ``x`` by index, so the gradient reaches ``x`` at the least
    wire alone and nothing ``[R, B, A, K]`` is kept for the backward."""
    B, A, K = x.shape
    R = c.shape[0]
    if not K:
        return x.new_ones(R, B, A)
    c = c.to(x)
    step = max(1, chunk // max(1, B * A * K))
    with torch.no_grad():
        held = x.detach()[None]
        least = torch.cat([torch.maximum(held, c[r:r + step, None, None, :]).argmin(-1)
                           for r in range(0, R, step)]) if R else x.new_zeros(0, B, A, dtype=torch.long)
    base = (torch.arange(B * A, device=x.device) * K).reshape(B, A)
    taken = x.reshape(-1)[base[None] + least]
    kept = c.gather(1, least.reshape(R, -1)).reshape(R, B, A)
    return torch.maximum(taken, kept)


def merge_occurrences(own, context):
    """Deduplicate exact intervals in nomination order, independently in each stream."""
    spans = torch.cat((own, context), dim=1)
    if not spans.shape[1]:
        return spans, spans[..., 0].bool(), spans.new_zeros(len(spans))
    start, end = spans.unbind(-1)
    # Stable lexicographic order groups equal intervals, earliest nomination first.
    order = end.argsort(dim=1, stable=True)
    order = order.gather(1, start.gather(1, order).argsort(dim=1, stable=True))
    ordered = spans.gather(1, order[..., None].expand_as(spans))
    distinct = F.pad((ordered[:, 1:] != ordered[:, :-1]).any(-1), (1, 0), value=True)
    first = torch.zeros_like(distinct).scatter(1, order, distinct) & (end > start)
    order = (~first).int().argsort(dim=1, stable=True)
    live = first.gather(1, order)
    extents = spans.gather(1, order[..., None].expand_as(spans))
    return torch.where(live[..., None], extents, 0), live & (order < own.shape[1]), first.sum(1)


def spelling_occurrences(surface, spellings, groups, nominations, known, first_part):
    """Exact overlapping matches, the nominated extents and all known extent keys."""
    from FieldReverse import _window_hits
    batch, width = surface.shape
    empty = surface.new_empty(batch, 0)
    extent_keys, priorities, lengths = [], [], []
    matches = _window_hits(surface, torch.ones_like(surface, dtype=torch.bool), spellings, groups)
    for size, rows, match in matches:
        start = torch.arange(rows.shape[1], device=surface.device)
        key = start * (width + 1) + start + size
        extent_keys.append(torch.where(match & known[rows], key, -1))
        lengths.append(torch.full_like(rows, size))
        rank = nominations[rows]
        priorities.append(torch.where(match & (rank >= 0), rank * (width + 1) + start, -1))
    if not priorities:
        return empty.reshape(batch, 0, 2), empty.reshape(batch, 0, 2), empty
    priority = torch.cat(priorities, dim=1)
    keys = torch.cat(extent_keys, dim=1)
    # Match the form-major nomination order. A length/position has only one
    # exact spelling, even if several hashes collide.
    order = priority.argsort(dim=1, stable=True)
    sizes = torch.cat(lengths, 1)
    starts = priority.remainder(width + 1)
    spans = torch.where((priority >= 0)[..., None], torch.stack((starts, starts + sizes), -1), 0)
    spans = spans.gather(1, order[..., None].expand_as(spans))
    learned = priority.gather(1, order) >= first_part * (width + 1)
    return torch.where(learned[..., None], 0, spans), torch.where(learned[..., None], spans, 0), keys.sort(1).values


def part_starts(raw, spans, spellings):
    """First occurrence of each learned part within each span, or the surface width."""
    raw = raw.to(torch.uint8)
    batch, width = raw.shape
    lo, hi = spans.clamp(0, width).unbind(-1)
    starts = []
    for group in spellings:
        size = group.shape[1]
        if size > width:
            starts.append(lo.new_full((*lo.shape, len(group)), width))
            continue
        windows = raw.unfold(1, size, 1)
        count = windows.shape[1]
        pos = torch.arange(count, device=raw.device)
        step = max(1, (1 << 22) // max(1, batch * count * size))
        for keys in group.split(step):
            hit = (windows[:, :, None] == keys).all(-1).transpose(1, 2)
            # The nearest hit at or after each byte. A span may start anywhere,
            # including inside another hit, and a match must end inside it.
            positions = torch.where(hit, pos, width)
            following = positions.flip(-1).cummin(-1).values.flip(-1)
            following = F.pad(following, (0, 1), value=width)
            first = following.gather(2, lo.clamp_max(count)[:, None].expand(-1, len(keys), -1))
            first = first.transpose(1, 2)
            starts.append(torch.where(first + size <= hi[..., None], first, width))
    return torch.cat(starts, -1) if starts else lo.new_empty((*lo.shape, 0))


def parts_for_units(raw, spans, spellings, rows, added_rank, padding):
    """Native addresses for the observed parts, in the ladder's constituent order."""
    first = part_starts(raw, spans, spellings)
    width = raw.shape[1]
    present = first < width
    # Characters follow surface order. Learned runs follow admission order.
    rank = torch.where(added_rank < 0, first, width + added_rank)
    order = torch.where(present, rank, torch.iinfo(rank.dtype).max).argsort(-1)
    return torch.where(present.gather(-1, order), rows[order], padding), present.sum(-1)


def _utf8_boundaries(raw, remaining):
    """A cut through a multibyte character exposes each truncated byte as an element."""
    width = raw.shape[1]
    padded = F.pad(raw, (0, 3))
    first, second, third, fourth = (padded[:, n:n + width] for n in range(4))
    continuation = lambda byte: (byte >= 0x80) & (byte <= 0xBF)
    two = (first >= 0xC2) & (first <= 0xDF) & (remaining >= 2)
    three = ((first >= 0xE0) & (first <= 0xEF) & (remaining >= 3) & continuation(third)
             & ((first != 0xE0) | (second >= 0xA0)) & ((first != 0xED) | (second < 0xA0)))
    four = ((first >= 0xF0) & (first <= 0xF4) & (remaining >= 4)
            & continuation(third) & continuation(fourth)
            & ((first != 0xF0) | (second >= 0x90)) & ((first != 0xF4) | (second < 0x90)))
    interior = torch.zeros_like(raw, dtype=torch.bool)
    for offset, starts in enumerate((two | three | four, three | four, four), start=1):
        interior |= F.pad(starts & continuation(second), (offset, 0))[:, :width]
    return F.pad(~interior, (0, 1), value=True)


def disjoint_occurrences(raw, spans, spellings, rows, ranks, padding, *, escaped_elements=False, form_rows=None):
    """Tile each ordered unit with its longest proper part at each surviving start."""
    batch, width = raw.shape
    units = spans.shape[1]
    recognized = None if form_rows is None else spans.new_full((batch, units), -1)
    if not units or not width:
        result = rows.new_full((batch, width), padding), spans.new_zeros(batch, units), spans.new_zeros(batch, width, 2)
        return result if form_rows is None else (*result, recognized)
    pos = torch.arange(width, device=raw.device).expand(batch, -1).contiguous()
    lo, hi = spans.unbind(-1)
    seat = torch.searchsorted(torch.where(hi > lo, lo, width).contiguous(), pos, right=True) - 1
    parent = seat.clamp_min(0)
    left, right = lo.gather(1, parent), hi.gather(1, parent)
    live = (seat >= 0) & (pos < right)
    boundaries = _utf8_boundaries(raw, torch.where(live, right - pos, 0)) if escaped_elements else None
    stride = torch.ones_like(pos)
    address = torch.full_like(pos, padding)
    base = 0
    for group in spellings:
        size = group.shape[1]
        if size <= width:
            windows = raw.unfold(1, size, 1)
            count = windows.shape[1]
            chunk = max(1, (1 << 22) // max(1, batch * count * size))
            for offset in range(0, len(group), chunk):
                keys = group[offset:offset + chunk]
                hit = (windows[:, :, None] == keys).all(-1)
                index = hit.to(torch.int32).argmax(-1) + base + offset
                proper = (size < right[:, :count] - left[:, :count]) | (ranks[index] < 0)
                matched = hit.any(-1)
                if form_rows is not None:
                    # A full-unit hit is already present in this comparison.
                    # Its native name can count an exposure without becoming
                    # one of the form's own reconstruction operands.
                    start = lo.clamp(0, count - 1)
                    whole = (hi - lo == size) & matched.gather(1, start)
                    recognized = torch.where(whole, form_rows[index.gather(1, start)], recognized)
                keep = live[:, :count] & (pos[:, :count] + size <= right[:, :count]) & proper & matched
                if boundaries is not None:
                    keep = keep & boundaries[:, :count] & boundaries[:, size:size+count]
                keep = F.pad(keep, (0, size - 1))
                stride = torch.where(keep, size, stride)
                address = torch.where(keep, F.pad(rows[index], (0, size - 1)), address)
        base += len(group)

    # Follow just the cuts reachable from each unit start. Redirect the last
    # step to a sentinel so adjacent units cannot affect each other's cover.
    sentinel = pos.new_full((batch, 1), width)
    successor = pos + stride
    jump = torch.cat((torch.where(successor < right, successor, width), sentinel), 1)
    reached = torch.cat((live & (pos == left), sentinel < width), 1)
    for _ in range(width.bit_length()):
        reached = reached | torch.zeros_like(jump).scatter_add(1, jump, reached.long()).bool()
        jump = jump.gather(1, jump)
    starts = reached[:, :width]
    counts = torch.zeros_like(lo).scatter_add(1, parent, starts.long())
    # Across disjoint units a sentence can have at most one part per byte.
    order = torch.where(starts, pos, width).sort(1).values
    valid = order < width
    take = order.clamp_max(width - 1)
    extents = torch.stack((order, successor.gather(1, take)), -1)
    result = torch.where(valid, address.gather(1, take), padding), counts, torch.where(valid[..., None], extents, 0)
    return result if form_rows is None else (*result, recognized)


def pack_disjoint_units(ids, counts, extents, padding, capacity):
    """Gather the sentence's occurrences into the word loop's part seats."""
    if not ids.shape[1]:
        shape = (*counts.shape, capacity)
        return ids.new_full(shape, padding), extents.new_zeros(*shape, 2)
    seats = torch.arange(capacity, device=ids.device)
    first = counts.cumsum(1) - counts
    take = (first[..., None] + seats).clamp_max(ids.shape[1] - 1).flatten(1)
    valid = seats < counts[..., None]
    rows = ids.gather(1, take).reshape_as(valid)
    spans = extents.gather(1, take[..., None].expand(-1, -1, 2)).reshape(*valid.shape, 2)
    return torch.where(valid, rows, padding), torch.where(valid[..., None], spans, 0)


def disjoint_parts(raw, spans, spellings, rows, ranks, padding):
    """Fixed-width view of a disjoint cover for callers without a tighter bound."""
    ids, counts, extents = disjoint_occurrences(raw, spans, spellings, rows, ranks, padding)
    parts, locations = pack_disjoint_units(ids, counts, extents, padding, raw.shape[1])
    return parts, counts, locations


def occurrence_parts(raw, spans, spellings, part_codes, catalog):
    """Join distinct located parts once, preserving the basis gradient at ties."""
    batch, occurrences = spans.shape[:2]
    if not spellings:
        return (part_codes.new_zeros(batch, occurrences, part_codes.shape[-1]),
                catalog.new_ones(batch, occurrences, catalog.shape[-1]))
    found = part_starts(raw, spans, spellings) < raw.shape[1]
    coordinates = torch.cat((part_codes, catalog.to(part_codes)), -1)
    step = max(1, (1 << 22) // max(1, batch * coordinates.numel()))
    joined = torch.cat([coordinates.masked_fill(~mask[..., None], -torch.inf).amax(-2)
                        for mask in found.split(step, dim=1)], dim=1)
    parts, tight = joined.split((part_codes.shape[-1], catalog.shape[-1]), dim=-1)
    present = found.any(-1, keepdim=True)
    return torch.where(present, parts, 0.), torch.where(present, tight, 1.)


def held_at(bank, table, codes=None):
    """Each row of ``bank`` at each attended percept of ``table`` ``[B, A,
    columns, 2]``, read from the graded what columns by the least and the
    greatest (each ``[rows, B, A]``): ``part``, the least of the percept's
    parts wires over the row's parts; ``whole``, the least, over the wires
    the row's wholes cut, of one less the percept's wholes wire; ``added``,
    the greatest of the percept's parts wires over the parts the row's
    against lane names (a part its nearer holders add). A row's own code
    is graded too: ``max(wire, 1 - code)`` stands for a part it holds to the
    degree ``code``. At codes of 0 and 1 these are the read of
    :meth:`BasicPercepts.read_points`."""
    codes = bank.codes if codes is None else codes
    iw, width = bank.identity_width, bank.width
    point = table[..., :width, 0].to(codes)
    row, against = codes[:, 0].detach(), codes[:, 1].detach()
    part = least_of_greater(point[..., :iw], 1 - row[:, :iw])
    whole = least_of_greater(1 - point[..., iw:], row[:, iw:])
    added = 1 - least_of_greater(1 - point[..., :iw], 1 - against[:, :iw])
    return part, whole, added


def fold_boundaries(table, addresses, rows, lengths, *, whole):
    """Join parts or meet wholes, including repeated members in a row."""
    values = table[addresses]
    bounds = table.new_full((lengths.shape[0], table.shape[1]), float('inf') if whole else -float('inf'))
    bounds = bounds.scatter_reduce(0, rows[:, None].expand_as(values), values, 'amin' if whole else 'amax')
    return torch.where(lengths[:, None] > 0, bounds, 1. if whole else 0.)


class PlaceEntries(nn.Module):
    """Each row's entries on the where wires and on the when wires of an
    attended percept: ``where_entries`` ``[capacity, 2 * WHERE_WIRES]`` and
    ``when_entries`` ``[capacity, 2 * WHEN_WIRES]``, each the row's value on
    every wire's first lane (against the wire ``c``) and then on its second
    (against ``1 - c``). Learned and read at zero or above; a row whose
    entries on a field are all at nothing reads one at every place on it
    (the field is dropped)."""
    def __init__(self, capacity):
        super().__init__()
        self.where_entries = nn.Parameter(torch.zeros(int(capacity), 2 * WHERE_WIRES))
        self.when_entries = nn.Parameter(torch.zeros(int(capacity), 2 * WHEN_WIRES))

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        for name in ('where_entries', 'when_entries'):
            state.setdefault(prefix + name, getattr(self, name).detach().clone())
        super()._load_from_state_dict(state, prefix, *args, **kwargs)


class BasicPercepts(nn.Module):
    #: Entries of the first Pi allocated for each row, on average.
    ENTRIES_PER_ROW = 16
    #: The size of the keyed draw that raises a conjunction above its members.
    INCREMENT = .1
    def __init__(self, native, identity_width, property_width, capacity):
        super().__init__()
        from OrderZero import SparsePiLayer3
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('basicPerceptCapacity must be positive')
        self.identity_width, self.property_width = int(identity_width), int(property_width)
        self.width = self.identity_width + self.property_width
        object.__setattr__(self, 'native', native)
        object.__setattr__(self, 'whole', None)
        # A percept's code is two lanes, each within 0 to 1 (:meth:`lanes`),
        # read from the codes of its parts and wholes whenever it is read: a
        # whole's code may change after the row is promoted, so no copy is
        # kept.
        self.register_buffer('promoted', torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))
        # A reserved address can read its occurrence's own parts/wholes before
        # recurrence promotes it. Only declare/promote writes a durable row.
        self.occurrences = [None] * self.capacity
        self.index = {}
        self.words = {}
        self.selectors = {}
        self.register_buffer('places', torch.full((self.capacity,), -1, dtype=torch.long))
        # The table of values: each value percept's felt value, exact. Its
        # code is its place among the bins of the felt value (rule 55);
        # the table is what the way back reads. Negative references below
        # the fixed-axis namespace address it.
        self.register_buffer('value_codes', torch.zeros(self.capacity))
        self.register_buffer('value_count', torch.zeros((), dtype=torch.long))
        self.revision = 0
        # Each basic percept is a row of the first Pi over its parts and
        # wholes; the rows are written in the order the percepts are reserved.
        self.pi = SparsePiLayer3(self.capacity, self.capacity * self.ENTRIES_PER_ROW)
        # A conjunction of basic percepts is a row whose members are rows.
        self.conjunctions = {}
        # The stored form each row of a form spells, and the rows of each
        # form; each row's key in ``index``.
        self._form_roots, self.form_rows, self._keys = {}, {}, {}
        # The run of characters of each row that spells no stored form and
        # whose parts are in order (:meth:`_run_of`).
        self.runs = {}
        # A place's row: its kept where wires (on their two lanes) and the
        # what it keeps (:mod:`Places`); the cones met at each band.
        self.place_rows, self.place_cones = {}, {}
        # Each row's kept value and weight on the where wires and on the when
        # wires of an attended percept. They are learned, so the module that
        # learns them registers them (``PartSpace.place_entries``); the
        # identity that holds this table keeps no parameter.
        object.__setattr__(self, 'entries', PlaceEntries(self.capacity))

    @property
    def where_entries(self):
        return self.entries.where_entries

    @property
    def when_entries(self):
        return self.entries.when_entries

    def part_codes(self, references):
        return self._boundary_codes(references, whole=False)

    def is_value(self, row):
        parts = self.occurrences[int(row)][0]
        return bool(parts) and min(parts) < -self.identity_width

    def whole_codes(self, references):
        return self._boundary_codes(references, whole=True)

    @staticmethod
    def value_bin(felt, width):
        """The bin of a felt value among ``width`` bins of the felt chart,
        ``floor(width * felt)``, the last holding a felt value of one."""
        felt = torch.as_tensor(felt)
        return (felt * width).floor().long().clamp(0, width - 1)

    @staticmethod
    def value_coordinates(width, place):
        """The coordinate of each bin of the catalog of one place's values:
        a keyed draw, as the parts of a form are drawn, with one coordinate a
        bin and none shared, so that the order of the line is read
        coordinate by coordinate. Each place has its own table of values (a
        variable of its own); a value with no place keeps the bins in order."""
        # Drawn on the host, where the keyed generator lives, whatever the
        # default device; the caller moves the table to its values.
        if place is None or int(place) < 0:
            return torch.arange(width, device='cpu')
        generator = torch.Generator(device='cpu').manual_seed(0x5ca1a7 + int(place))
        return torch.randperm(width, generator=generator, device='cpu')

    def value_code(self, felt, width, place=None):
        """A value's code in a half ``width`` wide (rule 55). The catalog is
        its place's bins of the felt value, each a part with one coordinate
        of its own (:meth:`value_coordinates`). Parts half: the OR of the
        bins at or below the value, built up from all zeros. Wholes half: all
        ones with the bins above it cut away. On a line these are one code."""
        bins = self.value_bin(felt, width).to(self.value_codes.device)
        ordered = (torch.arange(width, device=self.value_codes.device) <= bins[..., None]).to(self.value_codes.dtype)
        coordinates = self.value_coordinates(width, place).to(self.value_codes.device)
        result = torch.zeros_like(ordered)
        result[..., coordinates] = ordered
        return result

    def _boundary_codes(self, references, *, whole, place=None):
        ids = torch.as_tensor(references, device=self.value_codes.device, dtype=torch.long)
        width = self.property_width if whole else self.identity_width
        result = self.value_codes.new_zeros(*ids.shape, width)
        values = ids < -self.identity_width
        if bool(values.any()):
            # A value is its own tightest part and its own whole: its code in
            # either half is its cut of its place's sorted bins.
            result[values] = self.value_code(self.felt_of(ids[values]), width, place)
        if bool((~values).any()):
            bank = self.whole if whole else self.native._basis
            native = bank.lookup_rows(ids[~values])
            result[~values] = (native[..., self.identity_width:self.width] if whole
                               else native[..., :self.identity_width])
        return result

    def _identity(self):
        return getattr(self.native, 'identity', None) if self.native is not None else None

    def _part_text(self, part):
        """The characters a store row stands for in the part map: a part's
        run, a stored form, or the row's own bytes; ``None`` where it has
        none (no part map, or a row with no bytes)."""
        identity = self._identity()
        if identity is None or int(part) < 0:
            return None
        stamp = (len(identity.atoms), len(identity.word_rows))
        if self.__dict__.get('_text_stamp') != stamp:
            from WordIdentity import bytes_of
            texts = {row: bytes_of(atom[1]) for atom, row in identity.atoms.items()}
            texts.update({row: raw for raw, row in identity.word_rows.items()})
            self.__dict__['_texts'], self.__dict__['_text_stamp'] = texts, stamp
        held = self._texts.get(int(part))
        if held is not None:
            return held
        # A row's bytes are read once, when it is first read as a part, so
        # that its code does not move as the store grows.
        known = self.__dict__.setdefault('_raw_texts', {})
        if int(part) not in known:
            try:
                raw = self.native.bytes_for(int(part))
            except (IndexError, KeyError, AttributeError):
                raw = None
            known[int(part)] = bytes(raw) if raw and not raw.startswith(identity.PREFIX[:1]) else None
        return known[int(part)]

    def felt_of(self, references):
        """The exact felt value in the table of each value reference."""
        ids = torch.as_tensor(references, device=self.value_codes.device, dtype=torch.long)
        seats = -ids - self.identity_width - 1
        if bool(((seats < 0) | (seats >= self.value_count)).any()):
            raise ValueError('unavailable value percept reference')
        return self.value_codes[seats]

    def felt(self, rows):
        """The way back: the exact felt value of each value percept row,
        read from the table of values. A code gives only the bin, to 1/64 of
        the felt chart (:meth:`bin_of`)."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        references = [self.occurrences[int(row)][0][0] for row in rows.flatten().tolist()]
        if any(reference >= -self.identity_width for reference in references):
            raise ValueError('only a value percept has a felt value')
        return self.felt_of(references).reshape(rows.shape)

    def bin_of(self, codes, place=None):
        """The bin a value code of ``place`` cuts at, from its parts half: the
        last of its run of ones from the first bin. ``-1`` where a half is no
        such run, or where the two halves, being as wide, do not cut alike."""
        def run(half):
            half = half[..., self.value_coordinates(half.shape[-1], place).to(half.device)]
            count = half.eq(1).long().sum(-1)
            prefix = torch.arange(half.shape[-1], device=half.device) < count[..., None]
            return count, half.eq(prefix.to(half)).all(-1) & (count > 0)
        count, valid = run(codes[..., :self.identity_width])
        if self.property_width:
            other, whole = run(codes[..., self.identity_width:self.width])
            valid = valid & whole & ((other == count) | (self.property_width != self.identity_width))
        return torch.where(valid, count - 1, -1)

    @torch.no_grad()
    def perceive_value(self, value, *, place=None):
        """Admit the exact value as both its tightest part and its whole."""
        value = torch.as_tensor(value, device=self.value_codes.device, dtype=self.value_codes.dtype)
        if value.ndim or not bool((value > 0) & (value <= 1)):
            raise ValueError('a value percept requires one felt presence in (0, 1]')
        known = (self.value_codes[:int(self.value_count)] == value).nonzero().flatten()
        if len(known):
            seat = int(known[0])
        else:
            seat = int(self.value_count)
            if seat == self.capacity:
                raise OverflowError('value percept capacity exhausted')
            self.value_codes[seat] = value
            self.value_count += 1
        reference = -self.identity_width - 1 - seat
        row = self.reserve((reference,), (reference,), place=place)
        self.promote(row)
        return row

    def collection_code(self, rows):
        """Common part and covering whole of the observed basic percepts."""
        codes = self.lookup_rows(rows)
        return torch.cat((codes[..., :self.identity_width].amin(0),
                          codes[..., self.identity_width:].amax(0)))

    def read_points(self, points, rows):
        """Read the two native boundaries of each basic percept separately,
        each 0 or 1 (rule 55): the point holds the percept's parts, and the
        percept's wholes hold the point. A boundary that does not hold is
        witnessed against: the point lies outside it."""
        from IdentityRows import meet_ratios
        codes = self.lookup_rows(rows).to(points)
        lower = codes[:, :self.identity_width]
        upper = codes[:, self.identity_width:]
        part, _ = meet_ratios(points[None, ..., :self.identity_width].clamp_min(0), lower[:, None, None])
        whole, _ = meet_ratios(upper[:, None, None],
            points[None, ..., self.identity_width:self.width].clamp_min(0))
        tolerance = 1e-6
        part, whole = (part >= 1 - tolerance).to(points), (whole >= 1 - tolerance).to(points)
        return torch.stack((torch.stack((part, 1 - part), -1), torch.stack((whole, 1 - whole), -1)))

    @staticmethod
    def read_collection(boundaries, weights):
        """A collection generalizes its two boundaries, then requires both."""
        from FieldLogic import reduce
        # Each view offers a part and a whole. Join alternatives on each
        # boundary before meeting the boundaries, retaining both poles.
        alternatives = reduce(boundaries, operation='union', dim=1, weights=weights[None, :, None, None])
        return reduce(alternatives, operation='intersection', dim=0)

    def bind(self, whole):
        object.__setattr__(self, 'whole', whole)

    def reserve_conjunction(self, members):
        """A row of the first Pi whose members are basic percepts."""
        members = tuple(sorted(dict.fromkeys(map(int, members))))
        if len(members) < 2 or any(not 0 <= m < len(self.index) for m in members):
            raise ValueError('a conjunction joins two or more basic percepts')
        parts = tuple(dict.fromkeys(part for m in members for part in self.occurrences[m][0]))
        wholes = set.intersection(*(set(self.occurrences[m][1]) for m in members))
        return self.reserve(parts, tuple(sorted(wholes)), members=members)

    def reserve(self, parts, wholes, *, word=None, place=None, members=None):
        parts, wholes = tuple(map(int, parts)), tuple(dict.fromkeys(map(int, wholes)))
        if not parts:
            raise ValueError('a basic percept needs identity parts')
        key = (parts, wholes) if place is None else (parts, wholes, int(place))
        if members is not None:
            key = ('conjunction', tuple(members))
        row = self.index.get(key)
        if row is None:
            row = len(self.index)
            if row >= self.capacity:
                raise OverflowError('BasicPercepts capacity exhausted')
            self.index[key] = row
            self._keys[row] = key
            self.occurrences[row] = parts, wholes
            if members is not None:
                self.conjunctions[row] = tuple(members)
            else:
                self.form_root(row)
            if self.pi.append(self.member_keys(row), against=self.against_keys(row)) != row:
                raise RuntimeError('the first Pi disagrees with the basic percept rows')
            if place is not None:
                self.places[row] = int(place)
            self.revision += 1
        if word is not None:
            self.words[int(word)] = row
        return row

    def reserve_place(self, plus, minus, what=None):
        """A row of the first Pi whose entries are on the where wires: on
        each wire's first lane ``plus`` (its least for lane over the
        occurrences met), on its second ``minus`` (the least of its against
        lane); and the what it keeps, a code of the what columns (``None``
        for a place that keeps none)."""
        plus = torch.as_tensor(plus, dtype=torch.float32).detach().cpu()
        minus = torch.as_tensor(minus, dtype=torch.float32).detach().cpu()
        what = None if what is None else torch.as_tensor(what, dtype=torch.float32).detach().cpu()
        kept = () if what is None else tuple(what.nonzero().flatten().tolist())
        key = ('place', tuple(round(v, 6) for v in plus.tolist()),
               tuple(round(v, 6) for v in minus.tolist()), kept)
        row = self.index.get(key)
        if row is None:
            row = len(self.index)
            if row >= self.capacity:
                raise OverflowError('BasicPercepts capacity exhausted')
            self.index[key] = row
            self._keys[row] = key
            self.occurrences[row] = (), ()
            self.place_rows[row] = plus, minus, what
            self._form_roots[row] = None
            if self.pi.append(self.member_keys(row)) != row:
                raise RuntimeError('the first Pi disagrees with the basic percept rows')
            # The place's where entries start at the cone's code: the least
            # of each wire's two lanes over the occurrences met.
            with torch.no_grad():
                self.where_entries[row] = torch.cat((plus, minus)).to(self.where_entries)
            self.revision += 1
        return row

    def _entries(self, rows, like):
        """Each row's where and when entries, each ``[R, wires, 2]``, read at
        zero or above."""
        index = torch.as_tensor(list(rows), dtype=torch.long, device=self.where_entries.device)
        lanes = lambda e, n: held_entries(e.index_select(0, index)).to(like).reshape(-1, 2, n).transpose(1, 2)
        return lanes(self.where_entries, WHERE_WIRES), lanes(self.when_entries, WHEN_WIRES)

    def place_factor(self, table, rows):
        """Each row's entries on the where and the when wires read at each
        attended percept of ``table`` ``[B, A, columns, 2]``, ``[rows, B,
        A]``: the product of :func:`field_factor` on the where and on the
        when. One at the place whose wires are the row's entries, falling as
        they differ, and one everywhere on a field whose entries are at
        nothing."""
        where, when = self._entries(rows, table)
        w = table[..., self.width:self.width + WHERE_WIRES, :]
        t = table[..., self.width + WHERE_WIRES:self.width + WHERE_WIRES + WHEN_WIRES, :]
        return field_factor(w, where) * field_factor(t, when)

    def kept_percept(self, rows):
        """The first Pi's way back (rule 79; §19.121, 1): for each row, the
        percept it keeps, ``(what, where, when)``: its code on the ``.what``
        on two lanes, ``[R, 2, width]`` (:meth:`bank_codes`), and its
        entries on the ``.where`` and the ``.when`` wires, ``[R, wires, 2]``,
        each wire on its two lanes, read at zero or above as
        :meth:`place_factor` reads them. Matched against an occurrence of a
        sentence's field by shared size, these are the row's own read of
        that occurrence (:func:`held_at` and :meth:`place_factor`). Nothing
        is stored for it. A place's row recognises a place and gives back no
        ``.what`` (§19.133)."""
        rows = list(rows)
        what = self.bank_codes(rows)
        place = torch.tensor([row in self.place_rows for row in rows], dtype=torch.bool, device=what.device)
        what = torch.where(place[:, None, None], 0., what)
        where, when = self._entries(rows, what)
        return what, where, when

    def where_share(self, table, rows):
        """Each row's share on the where wires of each attended percept per
        unit of its share there, ``[rows, B, A]`` (:func:`field_share`)."""
        where, _when = self._entries(rows, table)
        return field_share(table[..., self.width:self.width + WHERE_WIRES, :], where)

    def place_code(self, rows, columns):
        """The coordinates appended to each row's what in the space of
        meanings, ``[rows, columns]``: its where entries' first lane, its
        when entries' first lane, then their second lanes, as many as
        ``columns`` holds (none where it holds none)."""
        index = torch.as_tensor(list(rows), dtype=torch.long, device=self.where_entries.device)
        where = held_entries(self.where_entries.detach().index_select(0, index))
        when = held_entries(self.when_entries.detach().index_select(0, index))
        full = torch.cat((where[:, :WHERE_WIRES], when[:, :WHEN_WIRES],
                          where[:, WHERE_WIRES:], when[:, WHEN_WIRES:]), -1)
        columns = int(columns)
        if columns > full.shape[-1]:
            full = F.pad(full, (0, columns - full.shape[-1]))
        return full[:, :columns].to(self.value_codes)

    def carrying(self, tower, features):
        features = tuple(features if isinstance(features, (tuple, list)) else (int(features),))
        if tower == 'ps':
            # Part literals are ordered runs, including repeated parts.
            # Set containment would identify (a,b) with (b,a) and (a,a)
            # with (a), erasing precisely the information reverse needs.
            def contains(parts):
                return any(parts[start:start + len(features)] == features
                           for start in range(len(parts) - len(features) + 1))
            return tuple(row for row in self.index.values()
                         if contains(self.occurrences[row][0]))
        return tuple(row for row in self.index.values()
                     if set(features) <= set(self.occurrences[row][1]))

    def _tightest(self, parts):
        """The form's own tightest whole, from the catalog of its parts, or
        ``None`` where its parts are no form's of the catalog."""
        identity = getattr(self.native, 'identity', None)
        if identity is None or min(parts) < -self.identity_width:
            return None
        return identity.tightest(parts)

    def _run_tightest(self, run):
        """The tightest whole of a run of characters: the OR of the catalog
        codes of its parts, as an occurrence of the run reads it."""
        identity = self._identity()
        if not identity.property_width:
            return None
        atoms = identity.words.get(run) or identity.atoms_of(run)
        return torch.stack([identity.catalog_code(atom) for atom in atoms]).amax(0)

    def _stored_boundary_codes(self, references, *, whole, row):
        # Occurrence addresses already live in the host inventory. Only value
        # references require a place-specific chart. Native codes stay live.
        if min(references) < -self.identity_width:
            return self._boundary_codes(references, whole=whole, place=int(self.places[row]))
        bank = self.whole if whole else self.native._basis
        ids = torch.tensor(references, dtype=torch.long, device=self.value_codes.device)
        native = bank.lookup_rows(ids)
        half = native[..., self.identity_width:self.width] if whole else native[..., :self.identity_width]
        return half.to(self.value_codes)

    def subspace_bounds(self, row):
        """The parts half, the greatest of its parts' codes; the wholes half,
        the least of its wholes' codes: the class of all characters (all
        ones), the classes it lies in, and the form itself as its tightest
        whole."""
        parts, wholes = self.occurrences[int(row)]
        run = self.runs.get(int(row))
        if run is None:
            identity = self._stored_boundary_codes(parts, whole=False, row=row).amax(0)
        else:
            identity = self._identity().form(run).to(self.value_codes)
        if not self.property_width:
            return identity, identity.new_empty(0)
        if wholes:
            if self.whole is None and any(reference >= -self.identity_width for reference in wholes):
                raise ValueError('basic percept property read requires its WholeSpace')
            properties = self._stored_boundary_codes(wholes, whole=True, row=row).amin(0)
        else:
            properties = identity.new_ones(self.property_width)
        tightest = self._tightest(parts) if run is None else self._run_tightest(run)
        if tightest is not None:
            properties = torch.minimum(properties, tightest.to(properties))
        return identity, properties

    def _bank_bounds(self, groups, *, whole):
        """Gather each occupied native address once for this set of row folds."""
        width = self.property_width if whole else self.identity_width
        references = sorted({ref for group in groups for ref in group})
        if not references or not width:
            return self.value_codes.new_ones(len(groups), width)
        bank = self.whole if whole else self.native._basis
        if bank is None:
            raise ValueError('basic percept property read requires its WholeSpace')
        position = {ref: i for i, ref in enumerate(references)}
        device = self.value_codes.device
        addresses = torch.tensor([position[ref] for group in groups for ref in group], dtype=torch.long, device=device)
        rows = torch.tensor([i for i, group in enumerate(groups) for _ in group], dtype=torch.long, device=device)
        lengths = torch.tensor(list(map(len, groups)), dtype=torch.long, device=device)
        refs = torch.tensor(references, device=device, dtype=torch.long)
        native = bank.lookup_rows(refs)
        table = native[:, self.identity_width:self.width] if whole else native[:, :width]
        return fold_boundaries(table.to(self.value_codes), addresses, rows, lengths, whole=whole)

    def _codes_for(self, rows):
        """Read ordinary stored rows together; keep the place-specific value charts."""
        ordinary, parts, wholes = [], [], []
        codes = [None] * len(rows)
        for n, row in enumerate(rows):
            if not 0 <= row < len(self.index):
                raise ValueError('unavailable basic percept reference')
            members = self.occurrences[row]
            if members is None:
                codes[n] = self.code(row)
                continue
            p, w = members
            if (row in self.place_rows or row in self.runs or row in self.conjunctions
                    or min((*p, *w), default=0) < -self.identity_width):
                codes[n] = self.code(row)
            else:
                ordinary.append(n)
                parts.append(p)
                wholes.append(w)
        if ordinary:
            lower = self._bank_bounds(parts, whole=False)
            upper = self._bank_bounds(wholes, whole=True)
            for i, n in enumerate(ordinary):
                tightest = self._tightest(parts[i]) if self.property_width else None
                bound = upper[i] if tightest is None else torch.minimum(upper[i], tightest.to(upper))
                codes[n] = torch.cat((lower[i], bound))
        return codes

    def form_root(self, row):
        """Resolve a percept's native form once, retaining its address as the inventory grows."""
        row = int(row)
        identity = self._identity()
        if row in self._form_roots:
            root = self._form_roots[row]
            if isinstance(root, bytes):
                # Earlier checkpoints name forms by spelling. The native
                # vocabulary may be restored after this module's extra state.
                root = None if identity is None else identity.word_rows.get(root)
                if root is not None:
                    self._form_roots[row] = root
            return root
        root = None
        parts = self.occurrences[row][0]
        if row not in self.conjunctions and identity is not None and min(parts) >= -self.identity_width:
            try:
                root = identity.spelling_row(parts)
            except (IndexError, KeyError):
                pass
        self._form_roots[row] = root
        if root is not None:
            self.form_rows.setdefault(self.native.bytes_for(root), []).append(row)
        elif row not in self.conjunctions:
            self._run_of(row)
        return root

    def form_of(self, row):
        """Canonical bytes for APIs which inspect a named form."""
        root = self.form_root(row)
        if root is not None:
            return self.native.bytes_for(root)
        return self._form_roots[int(row)]  # None or a legacy name awaiting vocabulary restore

    @property
    def forms(self):
        return {row: self.form_of(row) for row in self._form_roots}

    def _run_of(self, row):
        """A row of two or more parts in order that spells no stored form:
        the run of their characters is a part of its own in the part map
        (``WordIdentity.admit_part``), carried by every stored form and every
        occurrence that holds that run, so that the order of the parts is
        kept in the code."""
        identity = self._identity()
        parts = self.occurrences[row][0]
        if identity is None or len(parts) < 2 or min(parts) < -self.identity_width:
            return None
        texts = [self._part_text(part) for part in parts]
        if any(text is None for text in texts):
            return None
        from WordIdentity import text_of
        run = b''.join(texts)
        if len(text_of(run)) < 2:
            return None
        self.runs[row] = run
        from WordIdentity import base_atoms
        for atom in base_atoms(run):
            identity._atom(atom)
        if text_of(run) not in identity.added:
            identity.admit_part(run)
        return run

    def _extend(self, row, parts):
        """A row of a form takes the parts its form has gained, after its
        own, under a key of its new parts."""
        old, wholes = self.occurrences[row]
        extra = tuple(part for part in map(int, parts) if part not in old)
        if not extra:
            return False
        new = old + extra
        self.occurrences[row] = new, wholes
        key = self._keys[row]
        key = (new, *key[1:])
        # The count of rows is the size of the index: an old key is given up
        # only for a new key that no other row holds.
        if key not in self.index:
            del self.index[self._keys[row]]
            self.index[key] = row
            self._keys[row] = key
        return True

    @torch.no_grad()
    def follow_forms(self, changed, new=None):
        """What follows a change of stored forms' codes (rule 61). Each row
        of a changed form takes the parts its form gained, in its occurrence
        and in the first Pi. Every promoted row of a changed form, of a form
        that a changed or ``new`` form holds as a run (its against lane reads
        its nearest holders), and of a conjunction over one of them, has its
        code read again, as every code is read (:meth:`lanes`). Returns
        those promoted rows."""
        identity = getattr(self.native, 'identity', None)
        if identity is None:
            return []
        for row in range(len(self.index)):
            self.form_root(row)
        if not self.form_rows:
            return []
        changed = set(changed)
        touched = changed | ({bytes(new)} if new is not None else set())
        held = set().union(*(identity.inside.get(form, ()) for form in touched)) if touched else set()
        rows, extended = set(), False
        for form in changed | held:
            for row in self.form_rows.get(form, ()):
                if form in changed:
                    extended |= self._extend(row, identity.parts(form))
                rows.add(row)
        if not rows:
            return []
        for whole, members in self.conjunctions.items():
            if rows.intersection(members):
                rows.add(whole)
                rows.update(members)
        rows = [row for row in sorted(rows) if bool(self.promoted[row])]
        if rows or extended:
            self.revision += 1
        if extended:
            self._refresh_pi()
        return rows

    def against(self, row, *, code=None):
        """A percept's against lane, as wide as its code.

        Parts half: the parts that the form's nearest holders add, the
        stored forms nearest to holding it as a run (rule 51), where the for
        lane leaves room, ``min(a, 1 - for)``; a conjunction has none there.
        Wholes half: what its upper bound cuts away, one less its for lane
        (rule 52); for a value, the bins above it (rule 55)."""
        if not 0 <= int(row) < len(self.index):
            raise ValueError('unavailable basic percept reference')
        if int(row) in self.place_rows:
            return self.value_codes.new_zeros(self.width)
        code = self.code(row) if code is None else code
        parts, _wholes = self.occurrences[int(row)]
        held = code.new_zeros(self.identity_width)
        identity = getattr(self.native, 'identity', None)
        raw = None if int(row) in self.conjunctions else self.form_of(row)
        if identity is not None and raw is not None:
            atoms = identity.against_atoms(raw)
            if atoms:
                bits = torch.stack([identity.atom_code(atom) for atom in atoms]).amax(0)
                held = torch.minimum(bits.to(code), 1 - code[:self.identity_width].detach().clamp(0, 1))
        # A conjunction over this percept is a strict whole of it like any
        # other (rule 53): what its other members add tells against it.
        holders = [members for whole, members in self.conjunctions.items()
                   if int(row) in members and bool(self.promoted[whole])]
        if holders:
            added = torch.stack([self.code(member)[:self.identity_width] for members in holders
                                 for member in members if member != int(row)]).amax(0)
            held = torch.maximum(held, torch.minimum(added.to(code),
                1 - code[:self.identity_width].detach().clamp(0, 1)))
        return torch.cat((held, 1 - code[self.identity_width:].clamp(0, 1)))

    def follow(self, rows):
        """``rows``, whose holders may have changed: their against lanes are
        read again at the next read (:meth:`against`). Returns the promoted
        ones."""
        changed = [row for row in map(int, rows) if bool(self.promoted[row])]
        if changed:
            self.revision += 1
        return changed

    def against_keys(self, row):
        """A row's against entries in the first Pi: the parts that its
        nearest holders add, each as a run of one part (rule 53)."""
        if int(row) in self.conjunctions or int(row) in self.place_rows:
            return []
        parts, _wholes = self.occurrences[int(row)]
        identity = getattr(self.native, 'identity', None)
        if identity is None or min(parts) < -self.identity_width:
            return []
        raw = self.form_of(row)
        if raw is None:
            return []
        return [('ps', (identity.atoms[atom],)) for atom in identity.against_atoms(raw)
                if atom in identity.atoms]

    def _refresh_pi(self):
        """Write the first Pi again where a row's against entries changed: a
        form admitted later can be a nearer holder, a part added later a new
        thing it adds."""
        identity = getattr(self.native, 'identity', None)
        if identity is None:
            return
        stamp = (len(identity.words), len(identity.added), identity.revision, len(self.index))
        if getattr(self, '_pi_stamp', None) == stamp:
            return
        self._pi_stamp = stamp
        rows = range(len(self.index))
        if all(self.pi.row_members(r) == self.member_keys(r)
               and sorted(self.pi.row_against(r)) == sorted(self.against_keys(r)) for r in rows):
            return
        self.pi.clear()
        for row in rows:
            self.pi.append(self.member_keys(row), against=self.against_keys(row))

    def lanes(self, row):
        """A percept's two lanes, ``[2, width]``: for, then against."""
        code = self.code(row)
        return torch.stack((code, self.against(row, code=code)))

    def code(self, row):
        if not 0 <= int(row) < len(self.index):
            raise ValueError('unavailable basic percept reference')
        if int(row) in self.place_rows:
            # A place has no part and no whole: it is read on the where wires.
            return self.value_codes.new_zeros(self.width)
        members = self.conjunctions.get(int(row))
        if members is not None:
            # The greatest over its members' codes and a small draw keyed to
            # the row, so that it lies above each member and is not their sum.
            greatest = torch.stack([self.code(member) for member in members]).amax(0)
            draw = torch.rand(greatest.shape, device='cpu',
                              generator=torch.Generator(device='cpu').manual_seed(int(row)))
            return (greatest + self.INCREMENT * draw.to(greatest)).clamp(max=1.)
        return torch.cat(self.subspace_bounds(int(row)))

    @torch.no_grad()
    def promote(self, row):
        row = int(row)
        if not self.promoted[row]:
            self.promoted[row] = True
            self.count += 1
            self.revision += 1
        return row

    def _leaves(self):
        ancestry = getattr(self.native, 'part_groups', {})
        def leaves(part, visiting=()):
            if part in visiting:
                raise ValueError('cyclic native part references')
            return tuple(leaf for child in ancestry[part]
                         for leaf in leaves(child, (*visiting, part))) if part in ancestry else (part,)
        return leaves

    def member_keys(self, row):
        """The keys of a basic percept's members in the percept tower: its
        run of parts and each of its wholes. A value percept has none."""
        if int(row) in self.conjunctions:
            return [('bp', member) for member in self.conjunctions[int(row)]]
        if int(row) in self.place_rows:
            return []
        parts, wholes = self.occurrences[int(row)]
        if min(parts) < -self.identity_width:
            return []
        leaves = self._leaves()
        return [('ps', tuple(leaf for part in parts for leaf in leaves(part))),
                *(('ws', whole) for whole in wholes)]

    def read(self, percepts, brackets, rows, **options):
        """Tie both towers live, with the complementary pole from properties."""
        return self.read_bank(self.lookup(percepts, brackets, rows, **options))

    def _occurrence_what(self, percepts, spans, wholes):
        """Each occurrence's code ``[B, O, width]``, as a row's code is made
        (:meth:`subspace_bounds`). The parts half: the greatest of the codes
        of the identity atoms of the bytes it spans (of the part leaves it
        covers, where the store has no identity), each present to degree
        one. The wholes half: the least, over ``wholes``, of the greater of
        a whole's code and one less the degree it holds over the bytes
        spanned (its for lane, where it covers them all), and its tightest
        whole (all ones where none). Graded, with the gradient of each
        whole's degree."""
        from PerceptProperties import counts_in_spans
        part_ids, part_spans, primitive, raw, _ = percepts
        raw = raw[:, 0] if raw.ndim == 3 else raw
        device = raw.device
        B, O = spans.shape[:2]
        real = spans[..., 1] > spans[..., 0]
        codes = self.value_codes.new_zeros(B, O, self.width)
        held = codes.new_ones(B, O, self.property_width)
        outside = codes.new_zeros(B, O, self.property_width)
        identity = getattr(self.native, 'identity', None)
        bank = identity.occurrence_bank(device) if identity is not None and O else None
        if bank is not None:
            spellings, rows, catalog, _ = bank
            basis = self.native._basis.lookup_rows(rows)[..., :self.identity_width]
            parts, tight = occurrence_parts(raw, spans, spellings, basis, catalog)
            codes[..., :self.identity_width] = parts
            held = torch.minimum(held, tight)
        elif identity is not None and O:
            text = raw.tolist()
            coded, index = {}, []
            for b, row in enumerate(spans.tolist()):
                for a, z in row:
                    run = bytes(v & 0xff for v in text[b][a:z]) if z > a else None
                    if run not in coded:
                        coded[run] = len(coded)
                    index.append(coded[run])
            parts = self.value_codes.new_zeros(len(coded), self.identity_width)
            tight = self.value_codes.new_ones(len(coded), self.property_width)
            groups = []
            for run, n in coded.items():
                if run is None:
                    groups.append([])
                    continue
                atoms = identity.words.get(run) or identity.atoms_of(run)
                found = [identity.atoms[atom] for atom in atoms if atom in identity.atoms]
                groups.append(found)
                if found:
                    tightest = identity.tightest(found) if self.property_width else None
                    if tightest is not None:
                        tight[n] = tightest.to(tight)
            # One gather for all the sentence's runs, preserving each run's
            # exact max and its tie gradient. Padding must not join a zero tie.
            width = max(map(len, groups), default=0)
            if width:
                ids = torch.tensor([g + [-1] * (width - len(g)) for g in groups],
                                   dtype=torch.long, device=codes.device)
                values = self.part_codes(ids.clamp_min(0))
                parts = values.masked_fill((ids < 0)[..., None], -torch.inf).amax(1)
                parts = torch.where((ids >= 0).any(1)[:, None], parts, 0.)
            index = torch.tensor(index, dtype=torch.long, device=codes.device).reshape(B, O)
            codes[..., :self.identity_width] = parts[index]
            held = torch.minimum(held, tight[index])
        elif part_ids.shape[1] and O:
            leaves = part_ids.to(device)
            part = self.part_codes(leaves.clamp_min(0)).to(codes) * (leaves >= 0)[..., None]   # [B, P, iw]
            leaf_spans = part_spans.to(device)
            inside = ((leaf_spans[:, None, :, 0] >= spans[..., :1]) & (leaf_spans[:, None, :, 1] <= spans[..., 1:])
                      & (leaves >= 0)[:, None]).to(codes)                                     # [B, O, P]
            codes[..., :self.identity_width] = (inside[..., None] * part[:, None]).amax(2) if part.shape[1] else 0.
        if wholes and self.property_width and self.whole is not None and primitive is not None and O:
            # A whole held to the degree ``f`` (its for lane over the bytes
            # spanned, where it covers them all) bounds the wholes wires by
            # the greater of its code and ``1 - f``; nothing is cut.
            counts = counts_in_spans(raw, spans.to(device), observed=raw != 0)
            rows = torch.tensor(wholes, device=device)
            evidence = primitive.evidence_on_counts(counts, rows=rows)
            complete = real & (counts.sum(-1) == spans[..., 1] - spans[..., 0])
            degree, away = (evidence * complete[..., None, None]).to(codes).unbind(-1)
            whole_codes = self.whole.lookup_rows(rows)[..., self.identity_width:self.width].to(codes)
            bound = least_of_greater(1 - degree, whole_codes.T).permute(1, 2, 0)             # [B, O, pw]
            held = torch.minimum(held, bound)
            # A whole witnessed outside to the degree ``g`` over every byte
            # spanned bounds the against lane of each wire it cuts by ``g``:
            # a wire holds the least over the wholes that cut it, so that a
            # whole's own wires do not take another whole's degree.
            # A wire is cut by a whole wherever its code is below one.
            uncut = (whole_codes >= 1).to(codes)                                            # [W, pw]
            outside = least_of_greater(away, uncut.T).permute(1, 2, 0) * (1 - uncut).amax(0)
        codes[..., self.identity_width:] = held
        return codes * real[..., None], outside * real[..., None]

    def bank_codes(self, rows):
        """Each row's code on two lanes ``[rows, 2, width]``, read from the
        codes of its parts and wholes, once at the lookup of a sentence. A
        place's is the what it keeps (no parts and every whole where it
        keeps none), with nothing against."""
        result = self.value_codes.new_zeros(len(rows), 2, self.width)
        ordinary = [row for row in rows if row not in self.place_rows and row not in self.conjunctions]
        kept = iter(self._codes_for(ordinary))
        for n, row in enumerate(rows):
            if row in self.place_rows:
                what = self.place_rows[row][2]
                if what is None:
                    result[n, 0, self.identity_width:] = 1.
                else:
                    result[n, 0] = what.to(result)
            elif row not in self.conjunctions:
                code = next(kept)
                result[n] = torch.stack((code, self.against(row, code=code))).detach().to(result)
        return result.clamp(0., 1.)

    def _run_extents(self, part_ids, part_spans, keys):
        """Where each run of several parts among ``keys`` is found as an
        ordered, adjacent run of the part events: its span, for each
        stream."""
        identity = getattr(self.native, 'identity', None)
        atoms = (set(identity.atoms.values()) | set(identity.word_rows.values())) if identity is not None else set()
        valid = (part_ids >= 0) & (part_spans[..., 1] > part_spans[..., 0])
        starts, ends = [], []
        # A stored form's parts are a set; a row's run is in order.
        ordered = {tuple(self.occurrences[row][0]) for row in self.runs}
        for key in keys:
            run = key[1] if key[0] == 'ps' and isinstance(key[1], tuple) else ()
            count = part_ids.shape[1] - len(run) + 1
            if len(run) < 2 or count < 1 or (set(run) <= atoms and run not in ordered):
                continue
            match = valid[:, :count].clone()
            for offset, pid in enumerate(run):
                match &= valid[:, offset:offset + count] & (part_ids[:, offset:offset + count] == pid)
                if offset:
                    match &= part_spans[:, offset - 1:offset - 1 + count, 1] == part_spans[:, offset:offset + count, 0]
            starts.append(torch.where(match, part_spans[:, :count, 0], -1))
            ends.append(torch.where(match, part_spans[:, len(run) - 1:len(run) - 1 + count, 1], -1))
        if not starts:
            return part_spans[:, :0]
        return torch.stack((torch.cat(starts, 1), torch.cat(ends, 1)), -1)

    def _form_spans(self, part_ids, part_spans, keys):
        """Host inspection of the same native run extents used by the lookup."""
        spans = self._run_extents(part_ids, part_spans, keys)
        return [[tuple(span) for span in row if span[0] >= 0] for row in spans.tolist()]

    def _spelling_geometry(self, text, rows, identity, delivered):
        from FieldReverse import index_spellings
        nominated = {form for form in (self.form_of(row) for row in rows
                                       if row not in self.place_rows and row not in self.conjunctions)
                     if form is not None and len(form) > 1}
        words = () if identity is None else identity.words
        # Learned multibyte parts may overlap and can be present at several
        # locations before the corresponding whole has a promoted Pi row.
        from WordIdentity import bytes_of
        learned = () if identity is None else tuple(bytes_of(run) for run in identity.added)
        candidates = tuple(dict.fromkeys((*nominated, *learned)))
        forms = tuple(dict.fromkeys((*candidates, *words)))
        if delivered is not None and all(b'\x00' not in form for form in forms):
            text = text[:, :delivered]
        forms = [form for form in forms if 0 < len(form) <= text.shape[1]]
        ranks = {form: i for i, form in enumerate(candidates)}
        surface = text.to(torch.uint8).long()
        _, spellings, groups = index_spellings(list(enumerate(forms)), text.device)
        rank = surface.new_tensor([ranks.get(form, -1) for form in forms])
        known_forms = set(learned).union(words)
        known = torch.tensor([form in known_forms for form in forms], device=text.device, dtype=torch.bool)
        spelled, learned, keys = spelling_occurrences(surface, spellings, groups, rank, known, len(nominated))
        return spelled, learned, keys, text.shape[1]

    def lookup(self, percepts, brackets, rows, **options):
        """The one lookup of a sentence (:class:`PerceptBank`): its
        occurrences, each part's event, each known form of several units,
        each whole's span and each reading's bracket, each as its muxed row,
        and the codes of ``rows``. What a look reads of it is
        :meth:`read_bank` over a table of :func:`attended` ones."""
        rows = tuple(dict.fromkeys(map(int, rows)))
        # A nested native row retains its recorded part witnesses. Descend
        # those references, not its spelling or an alternative byte tiling.
        ancestry = getattr(self.native, 'part_groups', {})
        leaves = self._leaves()
        if ancestry and not options.get('element_input', False):
            ids, spans, primitive, raw, whole_spans = percepts
            expanded = []
            for row_ids, row_spans in zip(ids.tolist(), spans.tolist()):
                events = []
                for part, (start, end) in zip(row_ids, row_spans):
                    if part < 0 or end <= start:
                        continue
                    cursor = start
                    for leaf in leaves(part):
                        stop = cursor + len(self._part_text(leaf) or self.native.bytes_for(leaf))
                        events.append((leaf, cursor, stop)); cursor = stop
                    if cursor != end:
                        raise ValueError('native part ancestry disagrees with its extent')
                expanded.append(events)
            width = max(map(len, expanded), default=0)
            ids = ids.new_full((len(expanded), width), -1)
            spans = spans.new_zeros(len(expanded), width, 2)
            for batch, events in enumerate(expanded):
                if events:
                    values = ids.new_tensor(events)
                    ids[batch, :len(events)] = values[:, 0]
                    spans[batch, :len(events)] = values[:, 1:]
            percepts = ids, spans, primitive, raw, whole_spans
        ordered = {}
        def with_members(row, seen):
            for member in self.conjunctions.get(row, ()):
                with_members(member, seen)
            seen.setdefault(row, None)
            return seen
        for row in rows:
            with_members(row, ordered)
        rows = tuple(ordered)
        self._refresh_pi()
        # The runs of several parts that the rows hold: where one is found,
        # it is an occurrence of its own.
        keys = tuple(dict.fromkeys(key for row in rows if row not in self.conjunctions
                                   for key in self.member_keys(row)))
        # The occurrences: each part's event, each known form of several
        # units where it is found, each whole's span, each reading's
        # bracket, once each in a stream.
        part_ids, part_spans, _primitive, raw, whole_spans = percepts
        device = (raw[:, 0] if raw.ndim == 3 else raw).device
        brackets = brackets.to(device)
        parts = part_spans.to(device)
        runs = self._run_extents(part_ids.to(device), parts, keys)
        text = raw[:, 0] if raw.ndim == 3 else raw
        identity = getattr(self.native, 'identity', None)
        if not raw.is_floating_point():
            spelled, learned, known, width = self._spelling_geometry(text, rows, identity, options.get('delivered'))
        else:
            spelled = learned = parts[:, :0]
            known, width = None, text.shape[1]
        own = torch.cat((parts, runs, spelled), 1)
        context = torch.cat((whole_spans.to(device), brackets, learned), 1)
        spans, holder, counts = merge_occurrences(own, context)
        # The precise live width preserves attention's draw among equal
        # nominations, including the RNG consumed when the table is over capacity.
        if spans.numel():
            ends = torch.where(spans[..., 0] < 0, text.shape[1], spans[..., 1])
            seats, limit = torch.stack((counts.max(), ends.max())).tolist()
        else:
            seats = limit = 0
        spans, holder = spans[:, :seats], holder[:, :seats]
        if known is not None and known.shape[1]:
            # Native inspection accepts Python slice bounds on external intervals.
            bounds = torch.where(spans < 0, text.shape[1] + spans, spans).clamp(0, text.shape[1])
            lo, hi = bounds.unbind(-1)
            key = lo * (width + 1) + hi
            at = torch.searchsorted(known, key.contiguous()).clamp_max(known.shape[1] - 1)
            holder = holder | ((hi > lo) & (hi <= width) & (known.gather(1, at) == key))
        real = spans[..., 1] > spans[..., 0]
        holds, containing = extent_members(spans, brackets)
        token = (torch.where(holds.any(1), holds.int().argmax(1), -1) if holds.shape[1]
                 else torch.full_like(spans[..., 0], -1))
        when = options.get('when')
        when = torch.zeros_like(brackets) if when is None else when.to(device)
        time = (torch.where(token >= 0, when[..., 0].gather(1, token.clamp_min(0)), -1)
                if holds.shape[1] else token.clone())
        wholes = tuple(dict.fromkeys(whole for row in rows if row not in self.place_rows
                                     and row not in self.conjunctions for whole in self.occurrences[row][1]
                                     if whole >= -self.identity_width))
        # The occurrence fold reads only the bytes inside nominated
        # intervals. Preserve legacy negative slice semantics for unusual
        # external intervals, and leave the original source geometry intact.
        surface = raw[..., :limit]
        what, outside = self._occurrence_what((part_ids, part_spans, _primitive, surface, whole_spans), spans, wholes)
        dtype = options.get('dtype') or what.dtype
        # The against lane: nothing on the parts half; on the wholes half, the
        # degree to which each whole is witnessed outside over the occurrence.
        what = torch.stack((what, torch.cat((torch.zeros_like(what[..., :self.identity_width]),
                                             outside), -1)), -1)
        registry, when_encoding = options.get('registry'), options.get('when_encoding')
        place = torch.where(real, spans[..., 0], -1)
        wires = (registry.encode('text', place).detach().to(what) if registry is not None
                 else what.new_zeros(*place.shape, WHERE_WIRES))
        ticks = (when_encoding.encode(time.clamp_min(0)).to(what) * (time >= 0)[..., None]
                 if when_encoding is not None else what.new_zeros(*place.shape, WHEN_WIRES))
        lanes = lambda c: torch.stack((c, 1 - c), -1) * real[..., None, None]
        table = torch.cat((what * real[..., None, None], lanes(wires), lanes(ticks)), -2)
        # An occurrence is present where every byte it spans is observed: a
        # missing byte supplies neither lane of any row read there.
        text = raw[:, 0] if raw.ndim == 3 else raw
        seen = F.pad((text != 0).long().cumsum(-1), (1, 0))
        ends = spans.clamp(0, text.shape[-1])
        observed = (seen.gather(1, ends[..., 1]) - seen.gather(1, ends[..., 0])) == spans[..., 1] - spans[..., 0]
        activity = (real & observed)[..., None].expand(-1, -1, 2).to(dtype)
        when_band = None if when_encoding is None else when_encoding.encode(when[..., 0])
        bank = PerceptBank(rows, self.bank_codes(rows).to(device), spans, token, time, table.to(dtype),
                           activity, brackets.clone(), when.clone(), self.width, self.identity_width,
                           when_band, registry, holder)
        object.__setattr__(bank, 'scope', holds | containing)
        return bank

    def prepare(self, bank):
        """Read the fixed opening once for all looks and candidate paths.

        Only geometry and the Pi's local read are shared. Activity, requests,
        conjunctions and forecasts are evaluated on each path's own history.
        The bank expires at the sentence boundary before another update.
        """
        if bank.rows:
            stored = self.place_factor(bank.table, bank.rows)
            # The bank stores detached location wires. An attended read
            # reconnects those wires to the registry; salience deliberately
            # reads the stored ones. Share each graph only with its own readers.
            live = (stored if bank.registry is None else self.place_factor(
                attended(bank, capacity=bank.token.shape[1]).table, bank.rows))
            basis = (*held_at(bank, bank.table),
                     self.outside(bank.table, bank.rows, bank.width),
                     live, stored)
            object.__setattr__(bank, 'read_basis', basis)
        return bank

    @staticmethod
    def release(bank):
        """Release the prepared Pi reads when their opening ends."""
        object.__setattr__(bank, 'read_basis', None)

    def read_bank(self, bank, activity=None, look=None):
        """The basic percepts of a bank's readings over a look's table
        (``look``, or the :func:`attended` ones, chosen by ``activity``'s
        share where a sentence does not fit), each occurrence present to the
        degree of the for lane of ``activity`` ``[B, occurrences, 2]`` (its
        own where not given).

        A row's value at an attended percept is the product of its entries
        there: on the what, for where the percept holds the row's parts and
        the row's wholes hold the percept, against where it holds the parts
        and lies outside a whole or holds a part that a nearer holder adds
        (:func:`held_at`); on the where and the when, :meth:`place_factor`.
        A conjunction is the first Pi over its members' rows. A reading's
        value is the greatest over the attended percepts it holds."""
        from PerceptField import PerceptField
        basis = bank.read_basis if look is None else None
        if look is None:
            look = attended(bank, share=None if activity is None else activity.sum(-1))
        own = bank.activity if activity is None else activity
        index = look.index[..., None].expand(-1, -1, 2)
        presence = (own.to(look.table).gather(1, index) * look.filled[..., None])[..., 0]   # [B, A]
        rows = bank.rows
        K = bank.where.shape[1]
        slot = torch.where(look.token >= 0, look.token, K)
        # The wide-open reading and its reconstruction use the original
        # placements. A narrowed look can also see wholes around its aperture.
        scope = None if activity is None else reading_scope(bank, K)
        if scope is not None:
            scope = scope.gather(2, look.index[:, None].expand(-1, K, -1)) & look.filled[:, None]

        def by_reading(x):
            """``[n, B, A, 2]`` joined over each reading's attended percepts."""
            if scope is not None:
                return scoped_max(x, scope)
            out = x.new_zeros(x.shape[0], x.shape[1], K + 1, 2)
            return out.scatter_reduce(2, slot[None, ..., None].expand_as(x), x, 'amax')[:, :, :K]
        B, A = presence.shape
        if rows:
            if basis is None:
                part, whole, added = held_at(bank, look.table)
                outside = self.outside(look.table, rows, bank.width)
                factor = self.place_factor(look.table, rows)
            else:
                part, whole, added, outside, factor = (
                    value.gather(2, look.index[None].expand(len(rows), -1, -1)) for value in basis[:5])
            at = presence[None] * factor
            # Only an occurrence of its own tells against what it holds by
            # what it lacks; any occurrence that holds a row's parts tells
            # against it to the degree its wholes are witnessed outside there.
            told = (at if bank.holder is None
                    else at * (bank.holder.gather(1, look.index) & look.filled).to(at)[None])
            unheld = torch.maximum((1 - whole) * told, outside * at)
            lanes = list(torch.stack((part * whole * at, part * torch.maximum(unheld, added * told)), -1))
            # Members come before the conjunctions over them.
            members = [('bp', r) for r in rows]
            for n, row in enumerate(rows):
                if row in self.conjunctions:
                    lanes[n] = self.pi(torch.stack(lanes), (row,), members)[0]
            events = self.held(torch.stack(lanes), rows)
            values = by_reading(events)
            boundaries = torch.stack((by_reading(torch.stack((part * at, torch.zeros_like(at)), -1)),
                                      by_reading(torch.stack((part * whole * at, part * unheld), -1))))
        else:
            events = look.table.new_zeros(0, B, A, 2)
            values = look.table.new_zeros(0, B, K, 2)
            boundaries = values.new_zeros(2, *values.shape)
        return PerceptField(tuple(('bp', row) for row in rows), values, events, look.span,
            bank.where, bank.when, None, bank.when_band, boundaries, None, look.token, look.index, scope is not None)

    def outside(self, table, rows, width):
        """How far each row's own wholes are witnessed outside over each
        attended percept of ``table``, ``[rows, B, A]``: for each whole, the
        greatest over the wires it cuts of the percept's wholes wire on the
        against lane (each wire the least over the wholes that cut it); for a
        row, the greatest over its wholes. Nothing for a row with no whole of
        its own (a place, a conjunction)."""
        B, A = table.shape[:2]
        named = [() if row in self.place_rows or row in self.conjunctions or self.occurrences[row] is None
                 else tuple(w for w in self.occurrences[row][1] if w >= -self.identity_width) for row in rows]
        wholes = sorted({w for ws in named for w in ws})
        out = table.new_zeros(len(rows), B, A)
        if not wholes or not self.property_width:
            return out
        kept = (self.whole_codes(wholes).detach() >= 1).to(table)                     # [W, pw]
        away = table[..., self.identity_width:width, 1]                                # [B, A, pw]
        degree = 1 - least_of_greater(1 - away, kept)                                  # [W, B, A]
        position = {w: n for n, w in enumerate(wholes)}
        for r, ws in enumerate(named):
            if ws:
                out[r] = degree[[position[w] for w in ws]].amax(0)
        return out

    def held(self, lanes, rows):
        """Each row's lanes ``[K, ..., 2]`` with what its conjunctions add:
        a conjunction over a row is one of its holders (rule 53), so where
        the conjunction's other members are all found, what tells against
        the row is found, and the row is read both (its holder stands in
        its place). Joined by the union, as the first Pi's against entries."""
        rows = [int(row) for row in rows]
        position = {row: index for index, row in enumerate(rows)}
        holders = {row: [members for whole, members in self.conjunctions.items()
                         if row in members and bool(self.promoted[whole])
                         and all(m in position for m in members)] for row in rows}
        if not any(holders.values()):
            return lanes
        result = list(lanes)
        for row, found in holders.items():
            if not found:
                continue
            index = position[row]
            others = torch.stack([torch.stack([lanes[position[m]][..., 0] for m in members if m != row]).amin(0)
                                  for members in found])
            added = 1 - (1 - others.clamp(0, 1)).prod(0)
            against = 1 - (1 - lanes[index][..., 1]) * (1 - lanes[index][..., 0] * added)
            result[index] = torch.stack((lanes[index][..., 0], against), -1)
        return torch.stack(result)

    def lookup_rows(self, rows, *, valid=None):
        rows = torch.as_tensor(rows, device=self.value_codes.device, dtype=torch.long)
        valid = torch.ones_like(rows, dtype=torch.bool) if valid is None else valid
        if bool(((rows < 0) | (rows >= len(self.index)))[valid].any()):
            raise ValueError('unavailable basic percept row')
        selected, inverse = torch.unique(rows[valid], return_inverse=True)
        result = self.value_codes.new_zeros(*rows.shape, self.width)
        if len(selected):
            codes = torch.stack(self._codes_for(selected.tolist()))
            result[valid] = codes[inverse]
        return result

    def boundary_rows(self, rows, *, valid=None, role):
        code = self.lookup_rows(rows, valid=valid)
        if role == 0:
            return torch.cat((code[..., :self.identity_width],
                              torch.zeros_like(code[..., self.identity_width:])), -1)
        return torch.cat((torch.ones_like(code[..., :self.identity_width]),
                          code[..., self.identity_width:]), -1)

    def property_presence(self, counts, rows):
        """Live property meet for each addressed basic-level occurrence."""
        shape = rows.shape
        if counts.shape[:-2] != shape[:-1]:
            raise ValueError('basic percept presence must align to word locations')
        primitive = self.whole.primitive_properties.on_counts(counts)
        result = primitive.new_zeros(*primitive.shape[:-1], shape[-1])
        for row in torch.unique(rows).tolist():
            if not 0 <= row < len(self.index): continue
            wholes = self.occurrences[row][1]
            presence = primitive[..., list(wholes)].amin(-1) if wholes else primitive.new_ones(primitive.shape[:-1])
            result = result + presence[..., None] * (rows == row).unsqueeze(-2)
        return result

    def word_weights(self, counts, word_rows):
        # The declared word slab supplies one code seat per location in each
        # stream. Repeated words keep separate seats: deduplicating them would
        # give a compiled input an axis whose width changes between batches.
        valid = word_rows >= 0
        rows = word_rows[..., None]
        # The parts witness is the actual admitted basic occurrence at this
        # word; the property witness stays live through the native reader.
        presence = self.property_presence(counts.unsqueeze(-2), rows).squeeze(-2).squeeze(-1)
        weights = presence * valid
        return weights, self.lookup_rows(word_rows, valid=valid)

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(version=2, occurrences=tuple(self.occurrences[:len(self.index)]), words=self.words, selectors=tuple(self.selectors.items()), places=self.places[:len(self.index)].tolist(),
                                conjunctions=tuple(self.conjunctions.items()),
                                forms=tuple((row, raw) for row, raw in self.forms.items() if raw is not None),
                                place_rows=tuple((row, (plus.tolist(), minus.tolist(),
                                                        None if what is None else what.tolist()))
                                                 for row, (plus, minus, what) in self.place_rows.items())))

    def set_extra_state(self, state):
        from IndependentComponents import _unpack_state
        state = _unpack_state(state)
        if state.get('version') not in (1, 2) or len(state['occurrences']) > self.capacity:
            raise ValueError('basic percept checkpoint disagrees with construction')
        if state.get('numeric_axes'):
            raise ValueError('fixed numeric axes have no level witnesses; an old numeric checkpoint cannot be imported')
        self.occurrences[:] = [None] * self.capacity
        self.index = {}
        self.pi.clear()
        self._pi_stamp = None
        self.conjunctions = {}
        self._form_roots, self.form_rows, self._keys = {}, {}, {}
        self.runs = {}
        for row, raw in state.get('forms', ()):
            self._form_roots[int(row)] = bytes(raw)
            self.form_rows.setdefault(bytes(raw), []).append(int(row))
        conjunctions = dict(state.get('conjunctions', ()))
        places = state.get('places', [-1] * len(state['occurrences']))
        place_rows = dict(state.get('place_rows', ()))
        self.places.fill_(-1)
        self.place_rows, self.place_cones = {}, {}
        learned = self.where_entries.detach().clone()
        for row, ((parts, wholes), place) in enumerate(zip(state['occurrences'], places)):
            if row in place_rows:
                self.reserve_place(*place_rows[row])
                continue
            self.reserve(parts, wholes, place=None if place < 0 else place,
                         members=conjunctions.get(row))
        with torch.no_grad():
            self.where_entries.copy_(learned)
        self.words = dict(state['words'])
        self.selectors = dict(state['selectors'])
        if state.get('scalar_state') is not None:
            import warnings
            warnings.warn('retired scalar sidecar cannot supply perceptual witnesses; relearn its definitions')

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        # A checkpoint that stored the rows' codes: its width is checked, and
        # the codes are read again from the parts and wholes.
        codes = state.pop(prefix+'codes', None)
        state.pop(prefix+'against_codes', None)
        if codes is not None and codes.shape[-1] != self.width:
            raise ValueError(f'a basic percept code of this layout is {self.identity_width} parts and '
                             f'{self.property_width} wholes wide on each of two lanes; the checkpoint\'s '
                             f'is {codes.shape[-1]} wide on one lane')
        # Older checkpoints did not yet have value-bearing native endpoints.
        state.setdefault(prefix+'value_codes', torch.zeros_like(self.value_codes))
        state.setdefault(prefix+'value_count', torch.zeros_like(self.value_count))
        state.setdefault(prefix+'places', torch.full_like(self.places, -1))
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
