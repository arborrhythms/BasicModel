"""Basic-level occurrences and their promoted, fixed-capacity percept table."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


#: The where wires and the when wires of an attended percept.
WHERE_WIRES = WHEN_WIRES = 4


def attended_capacity():
    """The rows of the table a look presents, ``attendedPercepts``."""
    from util import TheXMLConfig
    return int(TheXMLConfig.get('architecture.attendedPercepts', default=256))


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


def _dense_rank(key, mask):
    """The rank of each masked value among the distinct masked values of its
    stream, ascending, ``[B, O]``."""
    big = key.abs().max() + 1 if key.numel() else key.new_ones(())
    held = torch.where(mask, key, big)
    ordered, order = held.sort(-1)
    new = torch.ones_like(ordered, dtype=torch.bool)
    new[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    ranks = new.long().cumsum(-1) - 1
    return torch.empty_like(ranks).scatter_(1, order, ranks)


def attended(bank, where=None, share=None, capacity=None):
    """The occurrences a look attends and their table (:class:`Attended`).

    All of them where they fit. Otherwise those inside ``where`` ``[B, 2]``
    (the sentence's extent where not given), the largest first, and those
    that contain it, the smallest first, a level down and a level up in
    turn, until the table is full; within a level that does not fit whole,
    the greater ``share`` ``[B, O]`` first, then a draw."""
    capacity = attended_capacity() if capacity is None else int(capacity)
    span = bank.span
    B, O = bank.token.shape
    real = span[..., 1] > span[..., 0]
    if O <= capacity:
        index = torch.arange(O, device=span.device)[None].expand(B, -1)
        filled = real
    else:
        if where is None:
            far = torch.iinfo(span.dtype).max
            where = torch.stack((torch.where(real, span[..., 0], far).amin(-1),
                                 torch.where(real, span[..., 1], -1).amax(-1)), -1)
        extent = span[..., 1] - span[..., 0]
        inside = real & (span[..., 0] >= where[:, None, 0]) & (span[..., 1] <= where[:, None, 1])
        around = real & ~inside & (span[..., 0] <= where[:, None, 0]) & (span[..., 1] >= where[:, None, 1])
        last = 2 * O + 2
        level = torch.where(inside, 2 * _dense_rank(-extent, inside),
                            torch.where(around, 2 * _dense_rank(extent, around) + 1, last))
        share = torch.zeros(B, O, device=span.device) if share is None else share.detach().to(span.device)
        order = torch.rand(B, O, device=span.device).argsort(-1)
        order = order.gather(1, (-share.float().gather(1, order)).argsort(dim=-1, stable=True))
        order = order.gather(1, level.gather(1, order).argsort(dim=-1, stable=True))
        index = order[:, :capacity]
        filled = level.gather(1, index) < last
    table = bank.table.gather(1, index[..., None, None].expand(-1, -1, *bank.table.shape[2:]))
    chosen = span.gather(1, index[..., None].expand(-1, -1, 2))
    if bank.registry is not None:
        start = torch.where(filled, chosen[..., 0], -1)
        wires = bank.registry.encode('text', start).to(table)
        lanes = torch.stack((wires, 1 - wires), -1) * filled[..., None, None]
        table = torch.cat((table[..., :bank.width, :], lanes, table[..., bank.width + wires.shape[-1]:, :]), -2)
    token = torch.where(filled, bank.token.gather(1, index), -1)
    return Attended(index, filled, table, chosen, token, bank.time.gather(1, index))


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


class PlaceEntries(nn.Module):
    """Each row's entries on the where wires and on the when wires of an
    attended percept: ``where_entries`` ``[capacity, 2 * WHERE_WIRES]`` and
    ``when_entries`` ``[capacity, 2 * WHEN_WIRES]``, each the kept value of every wire, then
    its weight. Learned; a weight is read at zero or above, and a row whose
    weights are at nothing reads one at every place (rule 31)."""
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
    #: The weight the cone of a place writes on each of its where wires.
    PLACE_WEIGHT = 10.

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
        self.forms, self.form_rows, self._keys = {}, {}, {}
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
        if place is None or int(place) < 0:
            return torch.arange(width)
        generator = torch.Generator().manual_seed(0x5ca1a7 + int(place))
        return torch.randperm(width, generator=generator)

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
                self.form_of(row)
            if self.pi.append(self.member_keys(row), against=self.against_keys(row)) != row:
                raise RuntimeError('the first Pi disagrees with the basic percept rows')
            if place is not None:
                self.places[row] = int(place)
            self.revision += 1
        if word is not None:
            self.words[int(word)] = row
        return row

    def reserve_place(self, plus, minus, what=None):
        """A row of the first Pi whose entries are on the where wires: the
        kept value ``plus`` of each wire (its least for lane over the
        occurrences met, ``minus`` the least of its against lane), each at
        :attr:`PLACE_WEIGHT`; and the what it keeps, a code of the what
        columns (``None`` for a place that keeps none)."""
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
            self.forms[row] = None
            if self.pi.append(self.member_keys(row)) != row:
                raise RuntimeError('the first Pi disagrees with the basic percept rows')
            with torch.no_grad():
                self.where_entries[row] = torch.cat((plus, torch.full_like(plus, self.PLACE_WEIGHT))).to(
                    self.where_entries)
            self.revision += 1
        return row

    def place_factor(self, table, rows):
        """Each row's entries on the where and the when wires read at each
        attended percept, ``[rows, B, A]``: ``exp(- sum w . (e . log(e / c)
        + (1 - e) . log((1 - e) / (1 - c))))`` over the wires ``c`` of
        ``table`` ``[B, A, columns, 2]``, ``e`` the row's kept value and
        ``w`` its weight on each. One where the wires are what the row
        keeps, falling as they differ, and one everywhere where every
        weight is at nothing."""
        index = torch.as_tensor(list(rows), dtype=torch.long, device=self.where_entries.device)
        where = self.where_entries.index_select(0, index).to(table)
        when = self.when_entries.index_select(0, index).to(table)
        kept = torch.cat((where[:, :WHERE_WIRES], when[:, :WHEN_WIRES]), -1).clamp(0., 1.)       # [R, n]
        weight = torch.cat((where[:, WHERE_WIRES:], when[:, WHEN_WIRES:]), -1)
        weight = torch.where(weight >= 0, weight, 0.)
        c = table[..., self.width:self.width + WHERE_WIRES + WHEN_WIRES, 0].clamp(1e-6, 1 - 1e-6)  # [B, A, n]
        own = (weight * (torch.xlogy(kept, kept) + torch.xlogy(1 - kept, 1 - kept))).sum(-1)       # [R]
        cross = (torch.einsum('bac,rc->rba', c.log(), weight * kept)
                 + torch.einsum('bac,rc->rba', (1 - c).log(), weight * (1 - kept)))
        # The divergence is never below zero but for rounding; at exactly
        # zero (every weight at nothing) its derivative by the weights passes,
        # so that an entry that starts at nothing can be taught.
        divergence = own[:, None, None] - cross
        return torch.exp(-torch.where(divergence >= 0, divergence, 0.))

    def where_weight(self, rows):
        """Each row's weights on the where wires summed, ``[rows]``: the
        share on the where wires of an attended percept is the row's share
        there times its value times this (the derivative of the factor by
        each wire's two lanes, times the lanes)."""
        index = torch.as_tensor(list(rows), dtype=torch.long, device=self.where_entries.device)
        weight = self.where_entries.index_select(0, index)[:, WHERE_WIRES:]
        return torch.where(weight >= 0, weight, 0.).sum(-1)

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

    def subspace_bounds(self, row):
        """The parts half, the greatest of its parts' codes; the wholes half,
        the least of its wholes' codes: the class of all characters (all
        ones), the classes it lies in, and the form itself as its tightest
        whole."""
        parts, wholes = self.occurrences[int(row)]
        ids = torch.tensor(parts, device=self.value_codes.device, dtype=torch.long)
        # A value's catalog is its place's (rule 55).
        place = int(self.places[int(row)])
        identity = self._boundary_codes(ids, whole=False, place=place).amax(0)
        run = self.runs.get(int(row))
        if run is not None:
            # In the part map: the run's own part, its characters and every
            # part found in it.
            identity = self._identity().form(run).to(identity)
        if not self.property_width:
            return identity, identity.new_empty(0)
        if wholes:
            if self.whole is None and any(reference >= -self.identity_width for reference in wholes):
                raise ValueError('basic percept property read requires its WholeSpace')
            ids = torch.tensor(wholes, device=self.value_codes.device, dtype=torch.long)
            properties = self._boundary_codes(ids, whole=True, place=place).amin(0)
        else:
            properties = identity.new_ones(self.property_width)
        tightest = self._tightest(parts) if run is None else self._run_tightest(run)
        if tightest is not None:
            properties = torch.minimum(properties, tightest.to(properties))
        return identity, properties

    def _form_of(self, parts):
        identity = getattr(self.native, 'identity', None)
        if identity is None or min(parts) < -self.identity_width:
            return None
        try:
            return identity.spelling(parts)
        except (IndexError, KeyError):
            return None

    def form_of(self, row):
        """The stored form a row spells, read from its parts when it is
        reserved and kept, or ``None`` (a conjunction, a value)."""
        row = int(row)
        if row in self.forms:
            return self.forms[row]
        raw = None if row in self.conjunctions else self._form_of(self.occurrences[row][0])
        self.forms[row] = raw
        if raw is not None:
            self.form_rows.setdefault(raw, []).append(row)
        elif row not in self.conjunctions:
            self._run_of(row)
        return raw

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
            self.form_of(row)
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

    def against(self, row):
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
        code = self.code(row)
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
        return torch.stack((self.code(row), self.against(row)))

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
            draw = torch.rand(greatest.shape, generator=torch.Generator().manual_seed(int(row)))
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
        whole's degree. The bytes are read on the host once, and each
        distinct run of bytes is coded once."""
        from PerceptProperties import counts_in_spans
        part_ids, part_spans, primitive, raw, _ = percepts
        raw = raw[:, 0] if raw.ndim == 3 else raw
        device = raw.device
        B, O = spans.shape[:2]
        real = spans[..., 1] > spans[..., 0]
        codes = self.value_codes.new_zeros(B, O, self.width)
        held = codes.new_ones(B, O, self.property_width)
        identity = getattr(self.native, 'identity', None)
        if identity is not None and O:
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
            for run, n in coded.items():
                if run is None:
                    continue
                atoms = identity.words.get(run) or identity.atoms_of(run)
                found = [identity.atoms[atom] for atom in atoms if atom in identity.atoms]
                if found:
                    parts[n] = self.part_codes(found).amax(0)
                    tightest = identity.tightest(found) if self.property_width else None
                    if tightest is not None:
                        tight[n] = tightest.to(tight)
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
            selected = counts[real]
            evidence = primitive.evidence_on_counts(selected, rows=torch.tensor(wholes, device=device))
            complete = selected.sum(-1) == (spans[..., 1] - spans[..., 0])[real]
            degree = codes.new_zeros(B, O, len(wholes))
            degree = degree.index_put((real.nonzero(as_tuple=True)),
                                      (evidence[..., 0] * complete[..., None]).to(codes))
            whole_codes = self.whole_codes(list(wholes)).to(codes)                          # [W, pw]
            bound = least_of_greater(1 - degree, whole_codes.T).permute(1, 2, 0)             # [B, O, pw]
            held = torch.minimum(held, bound)
        codes[..., self.identity_width:] = held
        return codes * real[..., None]

    def bank_codes(self, rows):
        """Each row's code on two lanes ``[rows, 2, width]``, read from the
        codes of its parts and wholes, once at the lookup of a sentence. A
        place's is the what it keeps (no parts and every whole where it
        keeps none), with nothing against."""
        result = self.value_codes.new_zeros(len(rows), 2, self.width)
        for n, row in enumerate(rows):
            if row in self.place_rows:
                what = self.place_rows[row][2]
                if what is None:
                    result[n, 0, self.identity_width:] = 1.
                else:
                    result[n, 0] = what.to(result)
            elif row not in self.conjunctions:
                result[n] = self.lanes(row).detach().to(result)
        return result.clamp(0., 1.)

    def _form_spans(self, part_ids, part_spans, keys):
        """Where each run of several parts among ``keys`` is found as an
        ordered, adjacent run of the part events: its span, for each
        stream."""
        identity = getattr(self.native, 'identity', None)
        atoms = (set(identity.atoms.values()) | set(identity.word_rows.values())) if identity is not None else set()
        valid = (part_ids >= 0) & (part_spans[..., 1] > part_spans[..., 0])
        found = [[] for _ in range(part_ids.shape[0])]
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
        if starts:
            spans = torch.stack((torch.cat(starts, 1), torch.cat(ends, 1)), -1)
            for b, row in enumerate(spans.tolist()):
                found[b] = [tuple(span) for span in row if span[0] >= 0]
        return found

    def _spelled_spans(self, raw, rows):
        """Where the stored form of each of ``rows`` of two or more bytes is
        found in the sentence's bytes: its span, for each stream. The bytes
        are read on the host once."""
        raw = raw[:, 0] if raw.ndim == 3 else raw
        spellings = {form for form in (self.form_of(row) for row in rows
                                       if row not in self.place_rows and row not in self.conjunctions)
                     if form is not None and len(form) > 1}
        found = []
        for line in raw.tolist():
            text = bytes(v & 0xff for v in line)
            spans = []
            for form in spellings:
                start = text.find(form)
                while start >= 0:
                    spans.append((start, start + len(form)))
                    start = text.find(form, start + 1)
            found.append(spans)
        return found

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
        if ancestry:
            ids, spans, primitive, raw, whole_spans = percepts
            expanded = []
            for row_ids, row_spans in zip(ids.tolist(), spans.tolist()):
                events = []
                for part, (start, end) in zip(row_ids, row_spans):
                    if part < 0 or end <= start:
                        continue
                    cursor = start
                    for leaf in leaves(part):
                        stop = cursor + len(self.native.bytes_for(leaf))
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
        forms = self._form_spans(part_ids.to(device), part_spans.to(device), keys)
        spelled = self._spelled_spans(raw, rows)
        found, holders = [], []
        identity = getattr(self.native, 'identity', None)
        lines = (raw[:, 0] if raw.ndim == 3 else raw).tolist()
        for b in range(brackets.shape[0]):
            held = dict.fromkeys(tuple(span) for source in (part_spans[b].tolist(), forms[b], spelled[b],
                                                            whole_spans[b].tolist(), brackets[b].tolist())
                                 for span in source if span[1] > span[0])
            found.append(list(held))
            # The occurrences that can tell against a row they hold: a part's
            # event, a run of a row's parts, a stored form. A bracket or a
            # whole's span that is none of these tells only for.
            own = {tuple(span) for source in (part_spans[b].tolist(), forms[b], spelled[b])
                   for span in source if span[1] > span[0]}
            if identity is not None:
                own.update(span for span in held if span not in own
                           and bytes(v & 0xff for v in lines[b][span[0]:span[1]]) in identity.words)
            holders.append([span in own for span in held])
        O = max(map(len, found), default=0)
        spans = torch.zeros(brackets.shape[0], O, 2, dtype=torch.long, device=device)
        holder = torch.zeros(brackets.shape[0], O, dtype=torch.bool, device=device)
        for b, held in enumerate(found):
            if held:
                spans[b, :len(held)] = torch.tensor(held, dtype=torch.long, device=device)
                holder[b, :len(held)] = torch.tensor(holders[b], dtype=torch.bool, device=device)
        real = spans[..., 1] > spans[..., 0]
        holds = ((spans[:, None, :, 0] >= brackets[..., :1]) & (spans[:, None, :, 1] <= brackets[..., 1:])
                 & real[:, None])
        token = (torch.where(holds.any(1), holds.int().argmax(1), -1) if holds.shape[1]
                 else torch.full_like(spans[..., 0], -1))
        when = options.get('when')
        when = torch.zeros_like(brackets) if when is None else when.to(device)
        time = (torch.where(token >= 0, when[..., 0].gather(1, token.clamp_min(0)), -1)
                if holds.shape[1] else token.clone())
        wholes = tuple(dict.fromkeys(whole for row in rows if row not in self.place_rows
                                     and row not in self.conjunctions for whole in self.occurrences[row][1]
                                     if whole >= -self.identity_width))
        what = self._occurrence_what(percepts, spans, wholes)
        dtype = options.get('dtype') or what.dtype
        what = torch.stack((what, torch.cat((torch.zeros_like(what[..., :self.identity_width]),
                                             1 - what[..., self.identity_width:]), -1)), -1)
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
        return PerceptBank(rows, self.bank_codes(rows).to(device), spans, token, time, table.to(dtype),
                           activity, brackets.clone(), when.clone(), self.width, self.identity_width,
                           when_band, registry, holder)

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
        if look is None:
            look = attended(bank, share=None if activity is None else activity.sum(-1))
        own = bank.activity if activity is None else activity
        index = look.index[..., None].expand(-1, -1, 2)
        presence = (own.to(look.table).gather(1, index) * look.filled[..., None])[..., 0]   # [B, A]
        rows = bank.rows
        K = bank.where.shape[1]
        slot = torch.where(look.token >= 0, look.token, K)

        def by_reading(x):
            """``[n, B, A, 2]`` joined over each reading's attended percepts."""
            out = x.new_zeros(x.shape[0], x.shape[1], K + 1, 2)
            return out.scatter_reduce(2, slot[None, ..., None].expand_as(x), x, 'amax')[:, :, :K]
        B, A = presence.shape
        if rows:
            part, whole, added = held_at(bank, look.table)
            factor = self.place_factor(look.table, rows)
            at = presence[None] * factor
            # Only an occurrence of its own tells against what it holds.
            told = (at if bank.holder is None
                    else at * (bank.holder.gather(1, look.index) & look.filled).to(at)[None])
            lanes = list(torch.stack((part * whole * at, part * torch.maximum(1 - whole, added) * told), -1))
            # Members come before the conjunctions over them.
            members = [('bp', r) for r in rows]
            for n, row in enumerate(rows):
                if row in self.conjunctions:
                    lanes[n] = self.pi(torch.stack(lanes), (row,), members)[0]
            events = self.held(torch.stack(lanes), rows)
            values = by_reading(events)
            boundaries = torch.stack((by_reading(torch.stack((part * at, torch.zeros_like(at)), -1)),
                                      by_reading(torch.stack((part * whole * at, part * (1 - whole) * told), -1))))
        else:
            events = look.table.new_zeros(0, B, A, 2)
            values = look.table.new_zeros(0, B, K, 2)
            boundaries = values.new_zeros(2, *values.shape)
        return PerceptField(tuple(('bp', row) for row in rows), values, events, look.span,
            bank.where, bank.when, None, bank.when_band, boundaries, None, look.token, look.index)

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
        selected = torch.unique(rows[valid])
        result = self.value_codes.new_zeros(*rows.shape, self.width)
        for row in selected.tolist():
            result = result + (valid & (rows == row))[..., None] * self.code(row)
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
        self.forms, self.form_rows, self._keys = {}, {}, {}
        self.runs = {}
        for row, raw in state.get('forms', ()):
            self.forms[int(row)] = bytes(raw)
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
