"""Bounded identities, current values and mention readings between kept closes."""
from dataclasses import dataclass, field, replace
import copy
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class Name:
    token: tuple
    reading: int | None


@dataclass(frozen=True)
class Event:
    occurrence: tuple
    verb: int
    role: int
    when: int
    before: torch.Tensor | None = None
    basis: torch.Tensor | int | None = None
    taught: bool = False
    after: torch.Tensor | None = None
    modifiers: tuple = ()
    modal_taught: bool = False
    case_value: torch.Tensor | None = None


class Timeline(NamedTuple):
    events: tuple
    value: torch.Tensor


@dataclass(frozen=True)
class Token:
    occurrences: tuple
    kind: int | None
    count: int
    when: int
    timelines: tuple
    native: int | None = None
    pending: bool = False
    symbol: int | None = None
    transient: int | None = None
    where: torch.Tensor | None = None
    order: int | None = None
    support: dict = field(default_factory=dict)
    mass: float = 0.
    readings: tuple = ()
    refinement: dict = field(default_factory=dict)

    @property
    def value(self):
        return self.timelines[0].value

    @property
    def events(self):
        return self.timelines[0].events

    def with_actual(self, *, value=None, events=None, **metadata):
        actual = Timeline(self.events if events is None else events,
                          self.value if value is None else value)
        return replace(self, timelines=(actual, *self.timelines[1:]), **metadata)


class MediumTermMemory:
    """An introducing occurrence owns one value; names share that identity.

    Mutation belongs to a kept close. The pool ends the sentence's graph;
    trial-local values live in the shared STM chunks.
    Mention membership, used by the settled-join guard, excludes events.
    """

    def __init__(self, capacity=64, horizon=16, *, n_vectors=1,
                 field_width=None, match_cos=.8, mint_count=3):
        if capacity < 1:
            raise ValueError('mtmCapacity must be positive')
        if type(horizon) is not int or horizon < 1:
            raise ValueError('mtmHorizon must be positive')
        self.capacity = int(capacity)
        self.horizon = horizon
        self.closes, self.accesses = {}, {}
        self.entries = {}
        self.names = {}
        self.field_width = min(256, int(n_vectors)) if field_width is None else int(field_width)
        self.match_cos, self.mint_count = float(match_cos), int(mint_count)
        if self.field_width < 1 or not 0 < self.match_cos <= 1 or self.mint_count < 1:
            raise ValueError('invalid familiar-tier construction bounds')
        self.candidate_counter = 0
        # One reserved id for each possible unpromoted MTM token. These lie
        # beyond nVectors, never in the codebook, and are reused only after
        # their owner expires or acquires a durable symbol.
        self.transient_ids = torch.arange(int(n_vectors), int(n_vectors)+self.capacity, dtype=torch.long)
        self.transient_used = torch.zeros(self.capacity, dtype=torch.bool)

    def _allocate_id(self, requested=None):
        available = ~self.transient_used
        if requested is not None:
            available &= self.transient_ids == int(requested)
        seats = available.nonzero().flatten()
        if not len(seats):
            raise RuntimeError('identity transient id pool exhausted or requested id already in use')
        index = int(seats[0])
        self.transient_used[index] = True
        return int(self.transient_ids[index])

    def _release_id(self, entry):
        if entry.transient is not None:
            self.transient_used[self.transient_ids == entry.transient] = False

    @staticmethod
    def _where(where, previous=None):
        if where is None:
            return torch.zeros(4) if previous is None else previous.detach().clone()
        if not torch.is_tensor(where) or where.shape != (4,) or not torch.isfinite(where).all():
            raise ValueError('identity where requires four finite percept coordinates')
        return where.detach().clone()

    def promote(self, token, symbol):
        key = self.token_of(token)
        if key is None:
            raise ValueError('promotion requires a retained token')
        entry = self.entries[key]
        self._release_id(entry)
        self.entries[key] = replace(entry, symbol=int(symbol), transient=None, pending=False)
        reference = getattr(self, '_owner', None)
        owner = None if reference is None else reference()
        if owner is not None:
            owner._priming_edge_revision = None
            from FamiliarTier import class_members
            from LTM1 import OCCURRENCE, SYMBOL
            alloc = owner._concept_allocator
            affected = {parent for occurrence in entry.occurrences
                        for kind, parent in alloc.parts.wholes((OCCURRENCE, occurrence[-1]))
                        if kind == SYMBOL and parent in alloc.raised}
            for parent in affected:
                members = class_members(owner, parent)
                if members:
                    owner.cover_members(parent, members)

    @staticmethod
    def _value(value):
        if (not torch.is_tensor(value) or value.ndim != 1 or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())):
            raise ValueError('an identity requires one finite working value')
        return value.detach().clone()

    def token_of(self, occurrence):
        if occurrence in self.entries:
            return occurrence
        name = self.names.get(occurrence)
        if name is not None:
            return name.token
        if isinstance(occurrence, int):
            candidates = [key for key, entry in self.entries.items() if occurrence in (entry.native, entry.symbol, entry.transient)]
            return candidates[0] if len(candidates) == 1 else None
        return next((key for key, entry in self.entries.items()
                     if any(event.occurrence == occurrence and event.role == 0
                            for timeline in entry.timelines for event in timeline.events)), None)

    def _room(self):
        if len(self.entries) >= self.capacity:
            # The common access clock governs both kinds of entry.
            oldest = min(self.entries, key=lambda key: (
                max(self.accesses.get(key, {}).values(), default=-1), self.entries[key].when))
            self._detach(oldest)

    def _support(self, support):
        import math
        if any(type(row) is not int or not 0 <= row < int(self.transient_ids[0])
               or not math.isfinite(float(weight)) or float(weight) < 0 for row, weight in support.items()):
            raise ValueError('candidate support requires nonnegative finite inventory evidence')
        return dict(sorted(((row, float(weight)) for row, weight in support.items() if weight > 0),
                           key=lambda item: (-item[1], item[0]))[:self.field_width])

    def open_candidate(self, key=None, *, order, when, value, support):
        value, support = self._value(value), self._support(support)
        if type(order) is not int or order < 0 or not support:
            raise ValueError('a candidate requires a witnessed order and support')
        expected = ('candidate', self.candidate_counter)
        if key is not None and key != expected:
            raise ValueError('candidate key must use the pool counter')
        self._room()
        self.candidate_counter += 1
        self.entries[expected] = Token((), None, 1, int(when), (Timeline((), value),),
            transient=self._allocate_id(), pending=True, where=self._where(None), order=order,
            support=support, mass=sum(support.values()))
        return expected

    def match_candidate(self, order, value):
        value = self._value(value)
        scored = [(float(torch.nn.functional.cosine_similarity(entry.value.to(value), value, dim=0)), key)
                  for key, entry in self.entries.items() if entry.order == order]
        score, key = max(scored, key=lambda item: item[0], default=(-1., None))
        return key if score >= self.match_cos else None

    def update_candidate(self, key, *, when, value, support):
        entry = self.entries[key]
        if entry.order is None:
            raise ValueError('candidate update requires a candidate')
        value, support = self._value(value), self._support(support)
        if value.shape != entry.value.shape or not support:
            raise ValueError('candidate update differs from its construction')
        mass = sum(support.values())
        merged = dict(entry.support)
        for row, weight in support.items():
            merged[row] = merged.get(row, 0.) + weight
        self.entries[key] = entry.with_actual(value=torch.lerp(entry.value.to(value), value,
            mass / (entry.mass + mass)), count=entry.count+1, when=int(when),
            mass=entry.mass+mass, support=self._support(merged))
        return key

    def read_candidates(self, owner):
        """A differentiable, sparse read through the current member codes."""
        result = {}
        for key, entry in self.entries.items():
            if entry.order is not None:
                rows = list(entry.support)
                codes = owner.similarity_codebook.lookup_rows(rows)
                weights = codes.new_tensor(list(entry.support.values()))
                result[key] = (codes * weights[:, None]).sum(0) / weights.sum()
        return result

    def record_reading(self, key, stream, pair):
        """One two-lane witness per kept-close clock, retained within the horizon."""
        entry = self.entries[key]
        clock = self.closes.get(stream, 0)+1
        import math
        pair = tuple(float(x) for x in pair)
        if len(pair) != 2 or any(not math.isfinite(x) or x < 0 for x in pair):
            raise ValueError('candidate reading requires two finite nonnegative lanes')
        # Several focal rows can identify one candidate in the same field.
        # Retain both lanes independently; the last focal must not erase one.
        prior = next((item[2:] for item in entry.readings if item[:2] == (stream, clock)), (0., 0.))
        pair = tuple(max(a, b) for a, b in zip(pair, prior))
        retained = tuple(item for item in entry.readings
            if item[:2] != (stream, clock) and
            self.closes.get(item[0], 0)+1-item[1] <= self.horizon)
        readings = (*retained, (stream, clock, *pair))
        self.entries[key] = replace(entry, readings=readings,
            count=len(readings) if entry.order is not None else entry.count)

    def promote_candidates(self, owner):
        from FamiliarTier import promote_candidates
        return promote_candidates(owner)

    def reading_of(self, occurrence):
        name = self.names.get(occurrence)
        return None if name is None else name.reading

    def value_of(self, reference):
        token = self.token_of(reference)
        return None if token is None else self.entries[token].value

    def open(self, occurrence, *, kind, when, value, reading=None, native=None, pending=None, where=None, transient=None):
        value = self._value(value)
        pending = kind is None if pending is None else pending
        if type(pending) is not bool or (pending and kind is not None):
            raise ValueError('a pending identity has no resolved kind')
        if self.token_of(occurrence) != occurrence:
            self._detach(occurrence)
        if occurrence in self.entries:
            entry = self.entries[occurrence]
            self.entries[occurrence] = entry.with_actual(count=entry.count + 1,
                when=when, value=torch.maximum(entry.value.to(value), value),
                pending=entry.pending and pending, where=self._where(where, entry.where))
            self.names[occurrence] = Name(occurrence, reading)
            return occurrence
        self._room()
        identity = self._allocate_id(transient)
        self.entries[occurrence] = Token((occurrence,), kind, 1, when, (Timeline((), value),),
            native=native, pending=pending, transient=identity, where=self._where(where))
        self.names[occurrence] = Name(occurrence, reading)
        return occurrence

    def named(self, word, *, value, when=0):
        """Read a name's pending identity; first sight has no kept occurrence."""
        word = int(word)
        named = self.names.get(('name', word))
        if named is not None and named.token in self.entries:
            return named.token
        key = next((name.token for name in reversed(tuple(self.names.values()))
                    if name.reading == word and name.token in self.entries), None)
        if key is not None:
            self.names['name', word] = Name(key, word)
            return key
        key = ('name', word)
        self._room()
        self.entries[key] = Token((), None, 0, int(when),
            (Timeline((), self._value(value)),), pending=True,
            transient=self._allocate_id(), where=self._where(None))
        self.names[key] = Name(key, word)
        return key

    def bind(self, antecedent, occurrence, *, kind, when, value=None, reading=None, where=None):
        token = self.token_of(antecedent)
        if token is None:
            # Losing track does not resurrect an evicted introduction.
            return self.open(occurrence, kind=kind, when=when, value=value, reading=reading, where=where)
        if self.token_of(occurrence) != token:
            self._detach(occurrence)
        entry = self.entries[token]
        uses = entry.occurrences
        if occurrence not in uses:
            uses += (occurrence,)
        joined = entry.value if value is None else torch.maximum(entry.value.to(value), self._value(value))
        self.entries[token] = entry.with_actual(occurrences=uses,
            kind=kind if entry.kind is None else entry.kind,
            pending=entry.pending and kind is None,
            count=entry.count + 1, when=when, value=joined, where=self._where(where, entry.where))
        previous = self.names.get(occurrence)
        self.names[occurrence] = Name(token, previous.reading if reading is None and previous else reading)
        return token

    def alternative(self, token, *, when, before=None):
        """Continue the latest branch, opening it from the actual prefix once."""
        key = self.token_of(token)
        if key is None:
            raise ValueError('an alternative requires a retained identity')
        entry = self.entries[key]
        if len(entry.timelines) == 1:
            value = entry.value.clone() if before is None else self._value(before)
            if value.shape != entry.value.shape:
                raise ValueError('alternative value has a different width')
            self.entries[key] = replace(entry, when=when,
                timelines=(*entry.timelines, Timeline(entry.events, value)))
        return len(self.entries[key].timelines) - 1

    def update(self, token, value, *, when, timeline=0):
        key = self.token_of(token)
        if key is None:
            raise ValueError('cannot update an unretained identity')
        entry = self.entries[key]
        if type(timeline) is not int or not 0 <= timeline < len(entry.timelines):
            raise ValueError('cannot update an absent timeline')
        value = self._value(value)
        if value.shape != entry.value.shape:
            raise ValueError('timeline value has a different width')
        self.entries[key] = replace(entry, when=when, timelines=tuple(
            Timeline(item.events, value) if index == timeline else item
            for index, item in enumerate(entry.timelines)))

    def event(self, token, occurrence, *, verb, role, when, before=None, basis=None,
              after=None, modifiers=(), timeline=0, case_value=None):
        key = self.token_of(token)
        if key is None or role not in (0, 2) or type(verb) is not int:
            raise ValueError('an event requires a retained identity, verb and participant role')
        entry = self.entries[key]
        if type(timeline) is not int or not 0 <= timeline < len(entry.timelines):
            raise ValueError('an event requires a retained timeline')
        old = entry.timelines[timeline]
        if case_value is not None and (self._value(case_value).shape != old.value.shape or
                                     bool(((case_value < 0) | (case_value > 1)).any())):
            raise ValueError('an event case test requires its native content shape and chart')
        event = Event(occurrence, verb, role, when,
            None if before is None else self._value(before),
            self._value(basis) if torch.is_tensor(basis) else basis,
            after=None if after is None else self._value(after),
            modifiers=tuple(self._value(value) for value in modifiers),
            case_value=None if case_value is None else self._value(case_value))
        previous = next((item for item in old.events
                         if (item.occurrence, item.role) == (occurrence, role)), None)
        if previous is not None:
            event = replace(event, taught=previous.taught, modal_taught=previous.modal_taught)
            events = tuple(event if item is previous else item for item in old.events)
        else:
            events = (*old.events, event)
        self.entries[key] = replace(entry, when=when, timelines=tuple(
            Timeline(events, item.value) if index == timeline else item
            for index, item in enumerate(entry.timelines)))

    def _detach(self, occurrence):
        """Detach a mention; an event with this address has its own owner."""
        if occurrence in self.entries and self.entries[occurrence].order is not None:
            self._release_id(self.entries.pop(occurrence))
            self.accesses.pop(occurrence, None)
            return
        name = self.names.pop(occurrence, None)
        if name is not None:
            token, entry = name.token, self.entries[name.token]
            if token == occurrence:
                self._release_id(entry)
                del self.entries[token]
                self.accesses.pop(token, None)
                self.names = {address: name for address, name in self.names.items() if name.token != token}
            else:
                self.entries[token] = replace(entry, occurrences=tuple(
                    address for address in entry.occurrences if address != occurrence))
    def forget(self, occurrence):
        self._detach(occurrence)
        # One event can witness a subject and an object in different identities.
        for token, entry in tuple(self.entries.items()):
            self.entries[token] = replace(entry, timelines=tuple(
                Timeline(tuple(event for event in timeline.events if event.occurrence != occurrence), timeline.value)
                for timeline in entry.timelines))

    def merge(self, token, into, *, when):
        """Join a pending object without moving the antecedent's existing uses."""
        source, target = self.token_of(token), self.token_of(into)
        if source is None or target is None:
            raise ValueError('MTM merge requires two retained tokens')
        if source == target:
            return target
        previous, antecedent = self.entries[source], self.entries[target]
        self.entries[target] = antecedent.with_actual(
            occurrences=antecedent.occurrences + previous.occurrences,
            count=antecedent.count + previous.count, when=when,
            kind=previous.kind if antecedent.kind is None else antecedent.kind,
            pending=antecedent.pending and previous.pending,
            value=torch.maximum(antecedent.value, previous.value.to(antecedent.value)),
            events=antecedent.events + tuple(event for event in previous.events
                if not any((other.occurrence, other.role) == (event.occurrence, event.role)
                           for other in antecedent.events)))
        self.entries[target] = replace(self.entries[target], timelines=(
            *self.entries[target].timelines, *previous.timelines[1:]))
        for occurrence, name in tuple(self.names.items()):
            if name.token == source:
                self.names[occurrence] = replace(name, token=target)
        refinement = dict(self.entries[target].refinement)
        refinement['mentions'] = {**previous.refinement.get('mentions', {}),
                                  **refinement.get('mentions', {})}
        self.entries[target] = replace(self.entries[target], refinement=refinement)
        for stream, clock in self.accesses.pop(source, {}).items():
            retained = self.accesses.setdefault(target, {})
            retained[stream] = max(clock, retained.get(stream, 0))
        self._release_id(previous)
        del self.entries[source]
        return target

    def close(self, store, *, stream=None, references=()):
        """Advance once after a kept sentence; staging and reads never call this."""
        clock = self.closes.get(stream, 0) + 1
        self.closes[stream] = clock
        from DocumentCover import latest, segments
        current = None if stream is None or store is None else latest(store, stream)
        membership = {} if store is None else segments(store, stream)
        for key in self.entries:
            if self.entries[key].order is not None:
                continue  # A candidate belongs only to streams that read it.
            row = None if store is None else store.location(key)
            if current is not None and current not in membership.get(row, ()):
                continue
            # Uncovered legacy/fixture rows retain the original stream aging.
            if (stream is not None and isinstance(row, int) and bool(store.stream[row].any())
                    and (stream >= store.stream.shape[1] or not bool(store.stream[row, stream]))):
                continue
            self.accesses.setdefault(key, {stream: clock})
        for reference in references:
            key = self.token_of(reference)
            if key is not None:
                self.accesses.setdefault(key, {})[stream] = clock
                entry = self.entries[key]
                if entry.order is None and reference in self.names:
                    refinement = dict(entry.refinement)
                    mentions = {address: seen for address, seen in refinement.get('mentions', {}).items()
                                if self.closes.get(seen[0], 0)-seen[1] <= self.horizon}
                    mentions.setdefault(reference, (stream, clock))
                    refinement['mentions'] = mentions
                    self.entries[key] = replace(entry, refinement=refinement)
        for key, accesses in tuple(self.accesses.items()):
            if stream in accesses and clock - accesses[stream] > self.horizon:
                del accesses[stream]
                if not accesses:
                    self._detach(key)
        for key, entry in tuple(self.entries.items()):
            readings = tuple(item for item in entry.readings
                if self.closes.get(item[0], 0)-item[1] <= self.horizon)
            if readings != entry.readings:
                self.entries[key] = replace(entry, readings=readings,
                    count=len(readings) if entry.order is not None else entry.count)

    def end_document(self, cover, *, store):
        """A document ends its cover, not the epoch's familiarity horizon."""
        return None

    def state(self):
        """Checkpoint only committed values; export owns every tensor copy."""
        def snapshot(value):
            return value.detach().cpu().clone() if torch.is_tensor(value) else value
        def event_state(event):
            return (event.occurrence, event.verb, event.role, event.when,
                snapshot(event.before), snapshot(event.basis), event.taught,
                snapshot(event.after), tuple(map(snapshot, event.modifiers)), event.modal_taught,
                snapshot(event.case_value))
        return dict(version=10, candidate_counter=self.candidate_counter, field_width=self.field_width,
            transient_ids=snapshot(self.transient_ids), transient_used=snapshot(self.transient_used),
            closes=tuple(self.closes.items()),
            accesses=tuple((key, tuple(value.items())) for key, value in self.accesses.items()),
            entries=tuple(dict(key=key, occurrences=entry.occurrences,
            kind=entry.kind, count=entry.count, when=entry.when,
            native=entry.native, pending=entry.pending, symbol=entry.symbol, transient=entry.transient, where=snapshot(entry.where),
            order=entry.order, support=dict(entry.support), mass=entry.mass,
            readings=entry.readings, refinement=copy.deepcopy(entry.refinement),
            timelines=tuple((tuple(map(event_state, timeline.events)), snapshot(timeline.value))
                            for timeline in entry.timelines)) for key, entry in self.entries.items()),
            names=tuple((mention, name.token, name.reading) for mention, name in self.names.items()))

    def restore(self, state, *, device=None):
        """Validate a complete pool before replacing its current context."""
        if state is None:
            self.entries, self.names = {}, {}
            self.closes, self.accesses = {}, {}
            self.transient_used.zero_()
            self.candidate_counter = 0
            return
        if not isinstance(state, dict) or state.get('version') not in (8, 9, 10):
            raise ValueError('MTM checkpoint requires transient pool version 8, 9 or 10')
        counter = state.get('candidate_counter', 0)
        if type(counter) is not int or counter < 0 or state.get('field_width', self.field_width) != self.field_width:
            raise ValueError('candidate checkpoint disagrees with construction')
        ids, used = state.get('transient_ids'), state.get('transient_used')
        if (not torch.is_tensor(ids) or ids.dtype != torch.long or not torch.equal(ids.cpu(), self.transient_ids)
                or not torch.is_tensor(used) or used.dtype != torch.bool or used.shape != self.transient_used.shape):
            raise ValueError('identity transient pool disagrees with construction')
        if state['version'] >= 6 and 'documents' in state:
            raise ValueError('identity checkpoint contains retired document provenance')
        def address(value):
            return (isinstance(value, tuple) and len(value) == 3 and value[0] == 'ltm'
                    and isinstance(value[1], str) and type(value[2]) is int)
        def named(value):
            return (state['version'] >= 10 and isinstance(value, tuple) and len(value) == 2
                    and value[0] == 'name' and type(value[1]) is int and value[1] > 0)
        def symbol(value):
            return value is None or type(value) is int and value > 0
        def sentence(value):
            return address(value) or (isinstance(value, tuple) and len(value) == 2
                and value[0] == 'sym' and type(value[1]) is int and value[1] > 0)
        if not all(isinstance(state.get(field), (tuple, list)) for field in ('entries', 'names')):
            raise ValueError('identity checkpoint requires entries and names')
        entries, names = {}, {}
        for record in state['entries']:
            required = ('timelines',) if state['version'] >= 5 else ('value', 'events')
            if not isinstance(record, dict) or not set(('key', 'occurrences', 'kind', 'count',
                    'when', 'native', *required)) <= record.keys():
                raise ValueError('incomplete identity checkpoint entry')
            key, uses = record['key'], tuple(record['occurrences'])
            candidate = (isinstance(key, tuple) and len(key) == 2 and key[0] == 'candidate'
                         and type(key[1]) is int and 0 <= key[1] < counter)
            if (key in entries or (candidate and uses) or (not candidate and
                    (not (address(key) or named(key)) or not all(map(address, uses))
                     or (not named(key) and (not uses or uses[0] != key))
                     or len(set(uses)) != len(uses)))):
                raise ValueError('invalid identity checkpoint membership')
            order, support = record.get('order'), self._support(record.get('support', {}))
            if (candidate and (type(order) is not int or order < 0 or not support)
                    or not candidate and (order is not None or support)
                    or len(support) != len(record.get('support', {}))):
                raise ValueError('invalid candidate checkpoint support')
            if (not symbol(record['kind']) or not symbol(record['native']) or
                    type(record['count']) is not int or record['count'] < len(uses) or
                    type(record['when']) is not int):
                raise ValueError('invalid identity checkpoint metadata')
            pending = record['kind'] is None if state['version'] == 1 else record.get('pending')
            if type(pending) is not bool or (pending and record['kind'] is not None):
                raise ValueError('invalid identity checkpoint pending state')
            saved = record['timelines'] if state['version'] >= 5 else ((record['events'], record['value']),)
            if not isinstance(saved, (tuple, list)) or not saved:
                raise ValueError('identity checkpoint requires an actual timeline')
            timelines = []
            for timeline in saved:
                if not isinstance(timeline, (tuple, list)) or len(timeline) != 2:
                    raise ValueError('invalid identity checkpoint timeline')
                history, value = timeline
                value = self._value(value)
                expected = (timelines[0].value if timelines else
                            next(iter(entries.values())).value if entries else value)
                if value.shape != expected.shape:
                    raise ValueError('identity checkpoint values have different widths')
                if device is not None:
                    value = value.to(device)
                size = 11 if state['version'] >= 7 else 10 if state['version'] >= 5 else 7 if state['version'] == 4 else 4
                if not isinstance(history, (tuple, list)) or any(
                        not isinstance(event, (tuple, list)) or len(event) != size for event in history):
                    raise ValueError('invalid identity checkpoint event')
                events = tuple(Event(*event) for event in history)
                for event in events:
                    if event.case_value is not None and bool(((event.case_value < 0) | (event.case_value > 1)).any()):
                        raise ValueError('invalid identity event case test')
                    if (type(event.taught) is not bool or type(event.modal_taught) is not bool
                            or not isinstance(event.modifiers, (tuple, list))
                            or (event.basis is not None and not torch.is_tensor(event.basis)
                                and (type(event.basis) is not int or event.basis < -1))):
                        raise ValueError('invalid identity event basis or teacher metadata')
                    for snapshot in (event.before, event.after, event.case_value, *event.modifiers):
                        if snapshot is not None and self._value(snapshot).shape != value.shape:
                            raise ValueError('invalid identity event snapshot')
                    if torch.is_tensor(event.basis) and self._value(event.basis).shape != value.shape:
                        raise ValueError('invalid identity event snapshot')
                events = tuple(replace(event,
                    before=None if event.before is None else self._value(event.before).to(value),
                    after=None if event.after is None else self._value(event.after).to(value),
                    case_value=None if event.case_value is None else self._value(event.case_value).to(value),
                    modifiers=tuple(self._value(item).to(value) for item in event.modifiers),
                    basis=self._value(event.basis).to(value) if torch.is_tensor(event.basis) else event.basis)
                    for event in events)
                if (any(not sentence(event.occurrence) or event.role not in (0, 2) or
                        type(event.verb) is not int or event.verb <= 0 or type(event.when) is not int
                        for event in events) or len({(event.occurrence, event.role) for event in events}) != len(events)):
                    raise ValueError('invalid identity checkpoint event')
                timelines.append(Timeline(events, value))
            entries[key] = Token(uses, record['kind'], record['count'], record['when'],
                                 tuple(timelines), record['native'], pending, record.get('symbol'),
                                 record.get('transient'), self._where(record.get('where')),
                                 order, support, float(record.get('mass', 0.)),
                                 tuple(record.get('readings', ())), copy.deepcopy(record.get('refinement', {})))
            if not symbol(record.get('symbol')):
                raise ValueError('invalid identity checkpoint promoted symbol')
        for mention, token, reading in state['names']:
            if (not (address(mention) or named(mention)) or not (address(token) or named(token))
                    or not symbol(reading) or mention in names or token not in entries
                    or (mention not in entries[token].occurrences and not (named(mention) and reading == mention[1]))):
                raise ValueError('invalid identity checkpoint name')
            names[mention] = Name(token, reading)
        if (len(entries) > self.capacity or len(names) != sum(len(entry.occurrences)
                for entry in entries.values()) + sum(named(mention) for mention in names)):
            raise ValueError('identity checkpoint exceeds capacity or omits a name')
        clocks, accesses = {}, {}
        if state['version'] >= 3:
            try:
                clocks = dict(state['closes'])
                accesses = {key: dict(value) for key, value in state['accesses']}
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('invalid identity checkpoint expiry metadata') from error
            if (any((stream is not None and (type(stream) is not int or stream < 0)) or
                    type(clock) is not int or clock < 0 for stream, clock in clocks.items())
                    or any(key not in entries or not value or any(stream not in clocks or
                        type(clock) is not int or not 0 <= clock <= clocks[stream]
                        for stream, clock in value.items()) for key, value in accesses.items())):
                raise ValueError('invalid identity checkpoint expiry metadata')
        claimed = [entry.transient for entry in entries.values() if entry.symbol is None]
        if (any(type(identity) is not int or identity not in self.transient_ids.tolist() for identity in claimed)
                or len(claimed) != len(set(claimed))
                or any(entry.transient is not None for entry in entries.values() if entry.symbol is not None)
                or not torch.equal(used.cpu(), torch.tensor([int(identity) in claimed for identity in self.transient_ids]))):
            raise ValueError('identity checkpoint transient assignments are inconsistent')
        self.transient_used.copy_(used.cpu())
        self.entries, self.names = entries, names
        self.closes, self.accesses = clocks, accesses
        self.candidate_counter = counter
