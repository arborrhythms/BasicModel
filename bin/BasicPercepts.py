"""Basic-level occurrences and their promoted, fixed-capacity percept table."""
import torch
from torch import nn


class BasicPercepts(nn.Module):
    def __init__(self, native, identity_width, property_width, capacity):
        super().__init__()
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('basicPerceptCapacity must be positive')
        self.identity_width, self.property_width = int(identity_width), int(property_width)
        self.width = self.identity_width + self.property_width
        object.__setattr__(self, 'native', native)
        object.__setattr__(self, 'whole', None)
        self.register_buffer('codes', torch.zeros(self.capacity, self.width))
        self.register_buffer('promoted', torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))
        # A reserved address can read its occurrence's own parts/wholes before
        # recurrence promotes it. Only declare/promote writes a durable row.
        self.occurrences = [None] * self.capacity
        self.index = {}
        self.words = {}
        self.selectors = {}
        self.numeric_axes = {}
        # Native values are endpoints, not concepts or learned codes. Negative
        # references below the fixed-axis namespace address this bounded table.
        self.register_buffer('value_codes', torch.zeros(self.capacity))
        self.register_buffer('value_count', torch.zeros((), dtype=torch.long))
        self.revision = 0

    def part_codes(self, references):
        return self._boundary_codes(references, whole=False)

    def is_value(self, row):
        return min(self.occurrences[int(row)][0]) < -self.identity_width

    def whole_codes(self, references):
        return self._boundary_codes(references, whole=True)

    def _boundary_codes(self, references, *, whole):
        ids = torch.as_tensor(references, device=self.codes.device, dtype=torch.long)
        width = self.property_width if whole else self.identity_width
        result = self.codes.new_zeros(*ids.shape, width)
        values = ids < -self.identity_width
        if bool(values.any()):
            seats = -ids[values] - self.identity_width - 1
            if bool((seats >= self.value_count).any()):
                raise ValueError('unavailable value percept reference')
            result[values, -1] = self.value_codes[seats]
        if bool((~values).any()):
            bank = self.whole if whole else self.native._basis
            native = bank.lookup_rows(ids[~values])
            result[~values] = (native[..., self.identity_width:self.width] if whole
                               else native[..., :self.identity_width])
        return result

    @torch.no_grad()
    def perceive_value(self, value):
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
        row = self.reserve((reference,), (reference,))
        self.promote(row)
        return row

    def collection_code(self, rows):
        """Common part and covering whole of the observed basic percepts."""
        codes = self.lookup_rows(rows)
        return torch.cat((codes[..., :self.identity_width].amin(0),
                          codes[..., self.identity_width:].amax(0)))

    def read_points(self, points, rows):
        """Read the two native boundaries of each basic percept separately."""
        from IdentityRows import meet_ratios
        codes = self.lookup_rows(rows).to(points)
        lower = codes[:, :self.identity_width]
        upper = codes[:, self.identity_width:]
        part, _ = meet_ratios(points[None, ..., :self.identity_width].clamp_min(0), lower[:, None, None])
        whole, _ = meet_ratios(upper[:, None, None],
            points[None, ..., self.identity_width:self.width].clamp_min(0))
        zero = torch.zeros_like(part)
        return torch.stack((torch.stack((part, zero), -1), torch.stack((whole, zero), -1)))

    @staticmethod
    def read_collection(boundaries, weights):
        """A collection generalizes its two boundaries, then requires both."""
        from FieldLogic import reduce
        # Each view offers a part and a whole. Join alternatives on each
        # boundary before meeting the boundaries, retaining both poles.
        alternatives = reduce(boundaries, operation='union', dim=1, weights=weights[None, :, None, None])
        return reduce(alternatives, operation='intersection', dim=0)

    def numeric_dimension(self, axis):
        """One fixed native atom per numeric dimension, independent of values."""
        axis = int(axis)
        if not 0 <= axis < self.identity_width:
            raise ValueError('numeric dimension exceeds the declared percept cube')
        row = self.reserve((-axis-1,), ())
        self.numeric_axes[row] = axis
        self.promote(row)
        return row

    def bind(self, whole):
        object.__setattr__(self, 'whole', whole)

    def reserve(self, parts, wholes, *, word=None):
        parts, wholes = tuple(map(int, parts)), tuple(dict.fromkeys(map(int, wholes)))
        if not parts:
            raise ValueError('a basic percept needs identity parts')
        key = parts, wholes
        row = self.index.get(key)
        if row is None:
            row = len(self.index)
            if row >= self.capacity:
                raise OverflowError('BasicPercepts capacity exhausted')
            self.index[key] = row
            self.occurrences[row] = key
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

    def subspace_bounds(self, row):
        if int(row) in self.numeric_axes:
            code = self.codes.new_zeros(self.width)
            code[self.numeric_axes[int(row)]] = 1.
            return code[:self.identity_width], code[self.identity_width:]
        parts, wholes = self.occurrences[int(row)]
        ids = torch.tensor(parts, device=self.codes.device, dtype=torch.long)
        identity = self.part_codes(ids).amax(0)
        if not self.property_width:
            return identity, identity.new_empty(0)
        if wholes:
            if self.whole is None:
                raise ValueError('basic percept property read requires its WholeSpace')
            ids = torch.tensor(wholes, device=self.codes.device, dtype=torch.long)
            properties = self.whole_codes(ids).amin(0)
        else:
            properties = identity.new_ones(self.property_width)
        return identity, properties

    def code(self, row):
        if not 0 <= int(row) < len(self.index):
            raise ValueError('unavailable basic percept reference')
        return torch.cat(self.subspace_bounds(int(row)))

    @torch.no_grad()
    def promote(self, row, *, declared=False):
        row = int(row)
        if not self.promoted[row]:
            self.codes[row].copy_(self.code(row))
            self.promoted[row] = True
            self.count += 1
            self.revision += 1
        return row

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
        def leaves(part, visiting=()):
            if part in visiting:
                raise ValueError('cyclic native part references')
            return tuple(leaf for child in ancestry[part]
                         for leaf in leaves(child, (*visiting, part))) if part in ancestry else (part,)
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
        keys, groups = [], []
        for row in rows:
            parts, wholes = self.occurrences[row]
            group = ([] if min(parts) < -self.identity_width else
                     [('ps', tuple(leaf for part in parts for leaf in leaves(part))),
                      *(('ws', part) for part in wholes)])
            keys.extend(group); groups.append(group)
        native = read_percepts(percepts, brackets, keys, part_reader=PartSpace.part_memberships,
            native=self.native, **options)
        values, events, boundaries = [], [], []
        for group in groups:
            if not group:
                values.append(native.values.new_zeros(native.values.shape[1:]))
                events.append(native.events.new_zeros(native.events.shape[1:]))
                boundaries.append(torch.stack((values[-1], values[-1])))
                continue
            selected = native.selected(group)
            value = native.values.index_select(0, selected)
            event = native.events.index_select(0, selected)
            def minimum(v):
                def pole(rows, channel):
                    known = rows[..., channel] > 0 if channel else rows.any(-1)
                    bound = torch.where(known, rows[..., channel], torch.inf).amin(0)
                    return torch.where(known.any(0), bound, 0.)
                positive = pole(v, 0)
                negative = pole(v[1:], 1) if len(v) > 1 else torch.zeros_like(positive)
                return torch.stack((positive, negative), -1)
            # A property of a word does not assert the word's presence.
            values.append(minimum(value) * value[0, ..., :1].gt(0))
            events.append(minimum(event) * value[0, ..., None, :1].gt(0))
            whole = torch.stack((minimum(value[1:])[..., 0], minimum(value)[..., 1]), -1) if len(value) > 1 else torch.stack(
                (torch.ones_like(value[0, ..., 0]), torch.zeros_like(value[0, ..., 0])), -1)
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
        return _pack_state(dict(version=1, occurrences=tuple(self.occurrences[:len(self.index)]), words=self.words, selectors=tuple(self.selectors.items()), numeric_axes=self.numeric_axes))

    def set_extra_state(self, state):
        from IndependentComponents import _unpack_state
        state = _unpack_state(state)
        if state.get('version') != 1 or len(state['occurrences']) > self.capacity:
            raise ValueError('basic percept checkpoint disagrees with construction')
        self.occurrences[:] = [None] * self.capacity
        self.index = {}
        for parts, wholes in state['occurrences']:
            self.reserve(parts, wholes)
        self.words = dict(state['words'])
        self.selectors = dict(state['selectors'])
        self.numeric_axes = dict(state.get('numeric_axes', {}))
        if state.get('scalar_state') is not None:
            import warnings
            warnings.warn('retired scalar sidecar cannot supply perceptual witnesses; relearn its definitions')

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        # Older checkpoints did not yet have value-bearing native endpoints.
        state.setdefault(prefix+'value_codes', torch.zeros_like(self.value_codes))
        state.setdefault(prefix+'value_count', torch.zeros_like(self.value_count))
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
