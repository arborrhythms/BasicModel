"""Bounded identities, current values and mention readings between kept closes."""
from dataclasses import dataclass, replace
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
    trial-local values live in the numerical identity overlay instead.
    Mention membership, used by the settled-join guard, excludes events.
    """

    def __init__(self, capacity=64, horizon=16):
        if capacity < 1:
            raise ValueError('mtmCapacity must be positive')
        if type(horizon) is not int or horizon < 1:
            raise ValueError('mtmHorizon must be positive')
        self.capacity = int(capacity)
        self.horizon = horizon
        self.closes, self.accesses = {}, {}
        self.entries = {}
        self.names = {}

    @staticmethod
    def _value(value):
        if (not torch.is_tensor(value) or value.ndim != 1 or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())):
            raise ValueError('an identity requires one finite working value')
        return value.detach().clone()

    def token_of(self, occurrence):
        name = self.names.get(occurrence)
        if name is not None:
            return name.token
        if isinstance(occurrence, int):
            candidates = [key for key, entry in self.entries.items() if occurrence in (entry.native, entry.symbol)]
            return candidates[0] if len(candidates) == 1 else None
        return next((key for key, entry in self.entries.items()
                     if any(event.occurrence == occurrence and event.role == 0
                            for timeline in entry.timelines for event in timeline.events)), None)

    def reading_of(self, occurrence):
        name = self.names.get(occurrence)
        return None if name is None else name.reading

    def value_of(self, reference):
        token = self.token_of(reference)
        return None if token is None else self.entries[token].value

    def open(self, occurrence, *, kind, when, value, reading=None, native=None, pending=None):
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
                pending=entry.pending and pending)
            self.names[occurrence] = Name(occurrence, reading)
            return occurrence
        if len(self.entries) >= self.capacity:
            # Capacity retires an identity, not the occurrence row that may
            # still witness an event owned by another retained identity.
            self._detach(min(self.entries, key=lambda key: self.entries[key].when))
        self.entries[occurrence] = Token((occurrence,), kind, 1, when, (Timeline((), value),), native=native, pending=pending)
        self.names[occurrence] = Name(occurrence, reading)
        return occurrence

    def bind(self, antecedent, occurrence, *, kind, when, value=None, reading=None):
        token = self.token_of(antecedent)
        if token is None:
            # Losing track does not resurrect an evicted introduction.
            return self.open(occurrence, kind=kind, when=when, value=value, reading=reading)
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
            count=entry.count + 1, when=when, value=joined)
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
        name = self.names.pop(occurrence, None)
        if name is not None:
            token, entry = name.token, self.entries[name.token]
            if token == occurrence:
                del self.entries[token]
                self.accesses.pop(token, None)
                for address in entry.occurrences:
                    self.names.pop(address, None)
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
        for occurrence in previous.occurrences:
            self.names[occurrence] = replace(self.names[occurrence], token=target)
        for stream, clock in self.accesses.pop(source, {}).items():
            retained = self.accesses.setdefault(target, {})
            retained[stream] = max(clock, retained.get(stream, 0))
        del self.entries[source]
        return target

    def close(self, store, *, stream=None, references=()):
        """Advance once after a kept sentence; staging and reads never call this."""
        clock = self.closes.get(stream, 0) + 1
        self.closes[stream] = clock
        from DocumentCover import latest, segments
        current = None if stream is None else latest(store, stream)
        membership = segments(store, stream)
        for key in self.entries:
            row = store.location(key)
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
        for key, accesses in tuple(self.accesses.items()):
            if stream in accesses and clock - accesses[stream] > self.horizon:
                del accesses[stream]
                if not accesses:
                    self._detach(key)

    def end_document(self, cover, *, store):
        """Retire entries introduced in this retained document segment."""
        if cover is None:
            return
        from DocumentCover import segments
        membership = segments(store)
        for key in tuple(self.entries):
            if cover in membership.get(store.location(key), ()):
                self._detach(key)

    def state(self):
        """Checkpoint only committed values; export owns every tensor copy."""
        def snapshot(value):
            return value.detach().cpu().clone() if torch.is_tensor(value) else value
        def event_state(event):
            return (event.occurrence, event.verb, event.role, event.when,
                snapshot(event.before), snapshot(event.basis), event.taught,
                snapshot(event.after), tuple(map(snapshot, event.modifiers)), event.modal_taught,
                snapshot(event.case_value))
        return dict(version=7, closes=tuple(self.closes.items()),
            accesses=tuple((key, tuple(value.items())) for key, value in self.accesses.items()),
            entries=tuple(dict(key=key, occurrences=entry.occurrences,
            kind=entry.kind, count=entry.count, when=entry.when,
            native=entry.native, pending=entry.pending, symbol=entry.symbol,
            timelines=tuple((tuple(map(event_state, timeline.events)), snapshot(timeline.value))
                            for timeline in entry.timelines)) for key, entry in self.entries.items()),
            names=tuple((mention, name.token, name.reading) for mention, name in self.names.items()))

    def restore(self, state, *, device=None):
        """Validate a complete pool before replacing its current context."""
        if state is None:
            self.entries, self.names = {}, {}
            self.closes, self.accesses = {}, {}
            return
        if not isinstance(state, dict) or state.get('version') not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError('unsupported identity checkpoint')
        if state['version'] >= 6 and 'documents' in state:
            raise ValueError('identity checkpoint contains retired document provenance')
        def address(value):
            return (isinstance(value, tuple) and len(value) == 3 and value[0] == 'ltm'
                    and isinstance(value[1], str) and type(value[2]) is int)
        def symbol(value):
            return value is None or type(value) is int and value > 0
        if not all(isinstance(state.get(field), (tuple, list)) for field in ('entries', 'names')):
            raise ValueError('identity checkpoint requires entries and names')
        entries, names = {}, {}
        for record in state['entries']:
            required = ('timelines',) if state['version'] >= 5 else ('value', 'events')
            if not isinstance(record, dict) or not set(('key', 'occurrences', 'kind', 'count',
                    'when', 'native', *required)) <= record.keys():
                raise ValueError('incomplete identity checkpoint entry')
            key, uses = record['key'], tuple(record['occurrences'])
            if (not address(key) or not all(map(address, uses)) or key in entries or
                    not uses or uses[0] != key or len(set(uses)) != len(uses)):
                raise ValueError('invalid identity checkpoint membership')
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
                if (any(not address(event.occurrence) or event.role not in (0, 2) or
                        type(event.verb) is not int or event.verb <= 0 or type(event.when) is not int
                        for event in events) or len({(event.occurrence, event.role) for event in events}) != len(events)):
                    raise ValueError('invalid identity checkpoint event')
                timelines.append(Timeline(events, value))
            entries[key] = Token(uses, record['kind'], record['count'], record['when'],
                                 tuple(timelines), record['native'], pending, record.get('symbol'))
            if not symbol(record.get('symbol')):
                raise ValueError('invalid identity checkpoint promoted symbol')
        for mention, token, reading in state['names']:
            if (not address(mention) or not address(token) or not symbol(reading) or
                    mention in names or token not in entries or mention not in entries[token].occurrences):
                raise ValueError('invalid identity checkpoint name')
            names[mention] = Name(token, reading)
        if (len(entries) > self.capacity or len(names) != sum(len(entry.occurrences)
                for entry in entries.values())):
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
        self.entries, self.names = entries, names
        self.closes, self.accesses = clocks, accesses
