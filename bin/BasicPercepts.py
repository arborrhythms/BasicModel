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
        self.revision = 0

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
        parts, wholes = self.occurrences[int(row)]
        ids = torch.tensor(parts, device=self.codes.device, dtype=torch.long)
        identity = self.native._basis.lookup_rows(ids)[..., :self.identity_width].amax(0)
        if not self.property_width:
            return identity, identity.new_empty(0)
        if wholes:
            if self.whole is None:
                raise ValueError('basic percept property read requires its WholeSpace')
            ids = torch.tensor(wholes, device=self.codes.device, dtype=torch.long)
            properties = self.whole.lookup_rows(ids)[..., self.identity_width:self.width].amin(0)
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
            group = [('ps', tuple(leaf for part in parts for leaf in leaves(part))),
                     *(('ws', part) for part in wholes)]
            keys.extend(group); groups.append(group)
        native = read_percepts(percepts, brackets, keys, part_reader=PartSpace.part_memberships,
            native=self.native, **options)
        values, events = [], []
        for group in groups:
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
            native.where, native.when, bands, native.when_band)

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
        return _pack_state(dict(version=1, occurrences=tuple(self.occurrences[:len(self.index)]), words=self.words, selectors=tuple(self.selectors.items())))

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
