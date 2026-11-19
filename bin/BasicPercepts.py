"""Basic-level occurrences and their promoted, fixed-capacity percept table."""
import torch
from torch import nn


class BasicPercepts(nn.Module):
    #: Entries of the first Pi allocated for each row, on average.
    ENTRIES_PER_ROW = 16
    #: The size of the keyed draw that raises a conjunction above its members.
    INCREMENT = .1

    def __init__(self, native, identity_width, property_width, capacity):
        super().__init__()
        from OrderZero import PerceptPi
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('basicPerceptCapacity must be positive')
        self.identity_width, self.property_width = int(identity_width), int(property_width)
        self.width = self.identity_width + self.property_width
        object.__setattr__(self, 'native', native)
        object.__setattr__(self, 'whole', None)
        # A percept's code is two lanes, each within 0 to 1: the for lane in
        # ``codes`` and the against lane in ``against_codes``.
        self.register_buffer('codes', torch.zeros(self.capacity, self.width))
        self.register_buffer('against_codes', torch.zeros(self.capacity, self.width))
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
        self.pi = PerceptPi(self.capacity, self.capacity * self.ENTRIES_PER_ROW)
        # A conjunction of basic percepts is a row whose members are rows.
        self.conjunctions = {}

    def part_codes(self, references):
        return self._boundary_codes(references, whole=False)

    def is_value(self, row):
        return min(self.occurrences[int(row)][0]) < -self.identity_width

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
        bins = self.value_bin(felt, width).to(self.codes.device)
        ordered = (torch.arange(width, device=self.codes.device) <= bins[..., None]).to(self.codes.dtype)
        coordinates = self.value_coordinates(width, place).to(self.codes.device)
        result = torch.zeros_like(ordered)
        result[..., coordinates] = ordered
        return result

    def _boundary_codes(self, references, *, whole, place=None):
        ids = torch.as_tensor(references, device=self.codes.device, dtype=torch.long)
        width = self.property_width if whole else self.identity_width
        result = self.codes.new_zeros(*ids.shape, width)
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

    def felt_of(self, references):
        """The exact felt value in the table of each value reference."""
        ids = torch.as_tensor(references, device=self.codes.device, dtype=torch.long)
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
        value = torch.as_tensor(value, device=self.codes.device, dtype=self.codes.dtype)
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
            self.occurrences[row] = parts, wholes
            if members is not None:
                self.conjunctions[row] = tuple(members)
            if self.pi.append(self.member_keys(row), against=self.against_keys(row)) != row:
                raise RuntimeError('the first Pi disagrees with the basic percept rows')
            if place is not None:
                self.places[row] = int(place)
            self.revision += 1
        if word is not None:
            self.words[int(word)] = row
        return row

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

    def subspace_bounds(self, row):
        """The parts half, the greatest of its parts' codes; the wholes half,
        the least of its wholes' codes: the class of all characters (all
        ones), the classes it lies in, and the form itself as its tightest
        whole."""
        parts, wholes = self.occurrences[int(row)]
        ids = torch.tensor(parts, device=self.codes.device, dtype=torch.long)
        # A value's catalog is its place's (rule 55).
        place = int(self.places[int(row)])
        identity = self._boundary_codes(ids, whole=False, place=place).amax(0)
        if not self.property_width:
            return identity, identity.new_empty(0)
        if wholes:
            if self.whole is None and any(reference >= -self.identity_width for reference in wholes):
                raise ValueError('basic percept property read requires its WholeSpace')
            ids = torch.tensor(wholes, device=self.codes.device, dtype=torch.long)
            properties = self._boundary_codes(ids, whole=True, place=place).amin(0)
        else:
            properties = identity.new_ones(self.property_width)
        tightest = self._tightest(parts)
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

    def against(self, row):
        """A percept's against lane, as wide as its code.

        Parts half: the parts that the form's nearest holders add, the
        stored forms nearest to holding it as a run (rule 51), where the for
        lane leaves room, ``min(a, 1 - for)``; a conjunction has none there.
        Wholes half: what its upper bound cuts away, one less its for lane
        (rule 52); for a value, the bins above it (rule 55)."""
        if not 0 <= int(row) < len(self.index):
            raise ValueError('unavailable basic percept reference')
        code = self.code(row)
        parts, _wholes = self.occurrences[int(row)]
        held = code.new_zeros(self.identity_width)
        identity = getattr(self.native, 'identity', None)
        raw = None if int(row) in self.conjunctions else self._form_of(parts)
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

    @torch.no_grad()
    def follow(self, rows):
        """Write again the stored against lane of ``rows``, whose holders may
        have changed, and return each changed row's read pair before and
        after, for the concepts' columns to follow (rule 55)."""
        changed = {}
        for row in map(int, rows):
            if not bool(self.promoted[row]):
                continue
            before = self.read_codes([row])[0].clone()
            self.against_codes[row].copy_(self.against(row))
            after = self.read_codes([row])[0]
            if not torch.equal(before, after):
                changed[row] = before, after.clone()
        return changed

    def against_keys(self, row):
        """A row's against entries in the first Pi: the parts that its
        nearest holders add, each as a run of one part (rule 53)."""
        if int(row) in self.conjunctions:
            return []
        parts, _wholes = self.occurrences[int(row)]
        identity = getattr(self.native, 'identity', None)
        if identity is None or min(parts) < -self.identity_width:
            return []
        raw = self._form_of(parts)
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
        if all(sorted(self.pi.row_against(r)) == sorted(self.against_keys(r)) for r in rows):
            return
        self.pi.clear()
        for row in rows:
            self.pi.append(self.member_keys(row), against=self.against_keys(row))

    def lanes(self, row):
        """A percept's two lanes, ``[2, width]``: for, then against."""
        return torch.stack((self.code(row), self.against(row)))

    def read_pair(self, lanes):
        """What a concept reads of a percept, ``[..., 2, width]`` from its
        lanes ``[..., 2, width]``: first what tells for it, then what tells
        against it. In the parts half these are its two lanes. The wholes
        half is read on its complement (rule 54): what the upper bound cuts
        away tells for the percept, and nothing is read against it there.
        A value percept is read the same way (rule 55)."""
        cut = lanes[..., 1, self.identity_width:]
        positive = torch.cat((lanes[..., 0, :self.identity_width], cut), -1)
        negative = torch.cat((lanes[..., 1, :self.identity_width], torch.zeros_like(cut)), -1)
        return torch.stack((positive, negative), -2)

    def read_codes(self, rows):
        """The stored read pair of each promoted row, ``[..., 2, width]``."""
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.codes.device)
        flat = rows.flatten()
        lanes = torch.stack((self.codes.index_select(0, flat), self.against_codes.index_select(0, flat)), -2)
        return self.read_pair(lanes).reshape(*rows.shape, 2, self.width)

    def column(self, row):
        """A member's share of a concept's starting column (rule 54), from
        its live lanes (``OrderZeroSigma.member_column``)."""
        from OrderZero import OrderZeroSigma
        positive, negative = self.read_pair(self.lanes(row))
        return OrderZeroSigma.member_column(positive, negative)

    def code(self, row):
        if not 0 <= int(row) < len(self.index):
            raise ValueError('unavailable basic percept reference')
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
            self.codes[row].copy_(self.code(row))
            self.against_codes[row].copy_(self.against(row))
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
        parts, wholes = self.occurrences[int(row)]
        if min(parts) < -self.identity_width:
            return []
        leaves = self._leaves()
        return [('ps', tuple(leaf for part in parts for leaf in leaves(part))),
                *(('ws', whole) for whole in wholes)]

    def read(self, percepts, brackets, rows, **options):
        """Tie both towers live, with the complementary pole from properties."""
        from PerceptField import read_percepts, PerceptField
        from Spaces import PartSpace
        rows = tuple(dict.fromkeys(map(int, rows)))
        # A nested native row retains its recorded part witnesses. Descend
        # those references, not its spelling or an alternative byte tiling.
        # Concept columns and basic-percept identities remain unchanged when
        # perception promotes an ordered run to one native row.
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
        requested = rows
        def with_members(row, seen):
            for member in self.conjunctions.get(row, ()):
                with_members(member, seen)
            seen.setdefault(row, None)
            return seen
        ordered = {}
        for row in rows:
            with_members(row, ordered)
        rows = tuple(ordered)
        self._refresh_pi()
        keys, groups, opposed = [], [], []
        for row in rows:
            group = [] if row in self.conjunctions else self.member_keys(row)
            keys.extend(group); groups.append(group)
            opposed.append(self.pi.row_against(row))
            keys.extend(opposed[-1])
        native = read_percepts(percepts, brackets, keys, part_reader=PartSpace.part_memberships,
            native=self.native, **options)
        values, events, boundaries = [], [], []
        if rows:
            order = list(dict.fromkeys(keys))
            selected = native.selected(order)
            members = native.values.index_select(0, selected)
            happened = native.events.index_select(0, selected)
            # A run of parts is the percept's presence: present wherever it
            # lies in the extent, and at an event wherever it lies inside
            # that event's span; the same on both lanes (it has no against
            # of its own). A reading's value reads the wholes where the run
            # lies; an event reads them over the event's span.
            run = torch.tensor([key[0] == 'ps' for key in order], dtype=torch.bool, device=members.device)
            spans = native.spans.to(happened.device)
            inside = ((spans[:, :, None, 0] <= spans[:, None, :, 0])
                      & (spans[:, None, :, 1] <= spans[:, :, None, 1])
                      & (spans[:, None, :, 1] > spans[:, None, :, 0])).to(happened)
            lies = happened[..., None, :, 0] * inside[None, :, None]
            # A reading with no event (a value alone, no word) holds no run.
            lies = lies.amax(-1) if lies.shape[-1] else lies.sum(-1)
            value = self._row_values(members, happened, rows, groups, order, opposed)
            event = self._row_lanes(happened, run, rows, order, presence=lies)
            values, events = list(value), list(event)
            # Members come before the conjunctions over them.
            members = [('bp', r) for r in rows]
            for index, row in enumerate(rows):
                if row in self.conjunctions:
                    values[index] = self.pi(torch.stack(values), (row,), members)[0]
                    events[index] = self.pi(torch.stack(events), (row,), members)[0]
        for group in groups:
            if not group:
                zero = native.values.new_zeros(native.values.shape[1:])
                boundaries.append(torch.stack((zero, zero)))
                continue
            value = native.values.index_select(0, native.selected(group))
            # The boundaries of a collection: the run of parts, and the
            # least of its wholes on the for lane with the least on the
            # against lane over the run and the wholes.
            def least(v, channel):
                known = v[..., channel] > 0 if channel else v.any(-1)
                bound = torch.where(known, v[..., channel], torch.inf).amin(0)
                return torch.where(known.any(0), bound, 0.)
            whole = (torch.stack((least(value[1:], 0), least(value[1:], 1)), -1) if len(value) > 1
                     else torch.stack((torch.ones_like(value[0, ..., 0]), torch.zeros_like(value[0, ..., 0])), -1))
            boundaries.append(torch.stack((value[0], whole)))
        shape = (0, *native.values.shape[1:])
        value = torch.stack(values) if values else native.values.new_zeros(shape)
        event = torch.stack(events) if events else native.events.new_zeros((0, *native.events.shape[1:]))
        bands = None
        if native.percept_where is not None:
            grouped = []
            for group in groups:
                selected = native.percept_where.index_select(0, native.selected(group)).flatten(0, 1)
                grouped.append(selected[selected.any(-1)])
            width = max((len(group) for group in grouped), default=0)
            bands = (torch.stack([torch.nn.functional.pad(group, (0, 0, 0, width-len(group)))
                                  for group in grouped]) if grouped else native.percept_where[:0])
        return PerceptField(tuple(('bp', row) for row in rows), value, event, native.spans,
            native.where, native.when, bands, native.when_band,
            torch.stack(boundaries, 1) if boundaries else value.new_zeros(2, *value.shape))

    def _row_values(self, members, happened, rows, groups, order, opposed=None):
        """Each row's two lanes for a reading, ``[rows, ..., 2]``.

        The presence is the run's for lane over the reading. Each whole, and
        each against entry, is read where the run lies: its lanes at the
        events of the run's own parts, the greatest over them. The for lane
        is the presence times the product of the wholes' for lanes. The
        against lane is the presence times the union, ``1 - prod (1 - a)``,
        of the wholes' against lanes and the against entries' presence (rule
        53: a part that a nearer holder adds, found where the run lies). A
        row with neither reads ``[presence, 0]``; a row with no member reads
        nothing.
        """
        if not len(order):
            return members.new_zeros(len(rows), *members.shape[1:])
        opposed = [[] for _ in groups] if opposed is None else opposed
        position = {key: index for index, key in enumerate(order)}
        runs = [position[group[0]] if group else -1 for group in groups]
        device = members.device
        run = torch.tensor(runs, dtype=torch.long, device=device)
        held = (run >= 0).reshape(-1, *([1] * (members.ndim - 2))).to(members)
        safe = run.clamp_min(0)
        presence = members.index_select(0, safe)[..., 0] * held
        at_run = happened.index_select(0, safe)[..., 0]

        def where_run_lies(lists):
            width = max((len(w) for w in lists), default=0)
            if not width:
                return None, None
            index = torch.tensor([w + [0] * (width - len(w)) for w in lists], dtype=torch.long, device=device)
            mask = torch.tensor([[True] * len(w) + [False] * (width - len(w)) for w in lists], device=device)
            lanes = happened.index_select(0, index.flatten()).reshape(len(rows), width, *happened.shape[1:])
            lanes = at_run[:, None, ..., None] * lanes
            # With no event in the reading there is nowhere for the run to lie.
            lanes = lanes.amax(-2) if lanes.shape[-2] else lanes.sum(-2)
            return lanes, mask.reshape(len(rows), width, *([1] * (lanes.ndim - 2)))

        lanes, mask = where_run_lies([[position[key] for key in group[1:]] for group in groups])
        if lanes is None:
            for_lane, kept = torch.ones_like(presence), torch.ones_like(presence)
        else:
            for_lane = torch.where(mask[..., 0], lanes[..., 0], 1.).prod(1)
            kept = torch.where(mask[..., 0], 1. - lanes[..., 1], 1.).prod(1)
        entries, mask = where_run_lies([[position[key] for key in keys if key in position] for keys in opposed])
        if entries is not None:
            kept = kept * torch.where(mask[..., 0], 1. - entries[..., 0].clamp(0, 1), 1.).prod(1)
        return torch.stack((presence * for_lane, presence * (1. - kept)), -1)

    def _row_lanes(self, members, run, rows, order, *, presence=None):
        """Each row's two lanes from its members' memberships, ``[K, ..., 2]``.

        The run of parts is the presence (its for lane, or ``presence``
        where given), on both lanes. The for lane is the presence times the
        product of the wholes' for lanes. The against lane is the presence
        times the union, ``1 - prod (1 - a)``, of the wholes' against lanes
        and of the row's against entries in the first Pi (their presence,
        or ``presence`` where given): the percept is against where any of
        its wholes is witnessed not to hold, or where a part that a nearer
        holder adds is found. A row with neither reads ``[presence, 0]``.
        """
        shape = (-1, *([1] * (members.ndim - 1)))
        run = run.reshape(shape)
        held = members[..., 0] if presence is None else presence
        for_lane = torch.where(run[..., 0], held, members[..., 0])
        present = torch.where(run[..., 0], held, torch.ones_like(held))
        kept = torch.where(run[..., 0], held, 1. - members[..., 1])
        lane = lambda m: self.pi(torch.stack((m, m), -1), rows, order)[..., 0]
        present = lane(present)
        opposed = self.pi.opposed(torch.stack((held, held), -1), rows, order)
        wholes = present - lane(kept)
        against = wholes + (present - wholes) * opposed
        return torch.stack((lane(for_lane), against), -1)

    def lookup_rows(self, rows, *, valid=None):
        rows = torch.as_tensor(rows, device=self.codes.device, dtype=torch.long)
        valid = torch.ones_like(rows, dtype=torch.bool) if valid is None else valid
        if bool(((rows < 0) | (rows >= len(self.index)))[valid].any()):
            raise ValueError('unavailable basic percept row')
        selected = torch.unique(rows[valid])
        result = self.codes.new_zeros(*rows.shape, self.width)
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
                                conjunctions=tuple(self.conjunctions.items())))

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
        conjunctions = dict(state.get('conjunctions', ()))
        places = state.get('places', [-1] * len(state['occurrences']))
        self.places.fill_(-1)
        for row, ((parts, wholes), place) in enumerate(zip(state['occurrences'], places)):
            self.reserve(parts, wholes, place=None if place < 0 else place,
                         members=conjunctions.get(row))
        self.words = dict(state['words'])
        self.selectors = dict(state['selectors'])
        if state.get('scalar_state') is not None:
            import warnings
            warnings.warn('retired scalar sidecar cannot supply perceptual witnesses; relearn its definitions')

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        codes = state.get(prefix+'codes')
        if codes is not None and codes.shape[-1] != self.width:
            raise ValueError(f'a basic percept code of this layout is {self.identity_width} parts and '
                             f'{self.property_width} wholes wide on each of two lanes; the checkpoint\'s '
                             f'is {codes.shape[-1]} wide on one lane')
        # Older checkpoints did not yet have value-bearing native endpoints.
        state.setdefault(prefix+'value_codes', torch.zeros_like(self.value_codes))
        state.setdefault(prefix+'value_count', torch.zeros_like(self.value_count))
        state.setdefault(prefix+'places', torch.full_like(self.places, -1))
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
