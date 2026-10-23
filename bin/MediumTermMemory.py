"""Bounded identities, current values and mention readings between kept closes."""
from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class Token:
    occurrences: tuple
    kind: int | None
    count: int
    when: int
    value: torch.Tensor
    events: tuple = ()
    native: int | None = None


class MediumTermMemory:
    """An introducing occurrence owns one value; names share that identity.

    Mutation belongs to a kept close. The pool ends the sentence's graph;
    trial-local values live in the numerical identity overlay instead.
    Mention membership, used by the settled-join guard, excludes events.
    """

    def __init__(self, capacity=64):
        if capacity < 1:
            raise ValueError('mtmCapacity must be positive')
        self.capacity = int(capacity)
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
            candidates = [key for key, entry in self.entries.items() if entry.native == occurrence]
            return candidates[0] if len(candidates) == 1 else None
        return next((key for key, entry in self.entries.items()
                     if any(event.occurrence == occurrence and event.role == 0
                            for event in entry.events)), None)

    def reading_of(self, occurrence):
        name = self.names.get(occurrence)
        return None if name is None else name.reading

    def value_of(self, reference):
        token = self.token_of(reference)
        return None if token is None else self.entries[token].value

    def open(self, occurrence, *, kind, when, value, reading=None, native=None):
        value = self._value(value)
        if self.token_of(occurrence) != occurrence:
            self.forget(occurrence)
        if occurrence in self.entries:
            entry = self.entries[occurrence]
            self.entries[occurrence] = replace(entry, count=entry.count + 1,
                when=when, value=torch.maximum(entry.value.to(value), value))
            self.names[occurrence] = Name(occurrence, reading)
            return occurrence
        if len(self.entries) >= self.capacity:
            self.forget(min(self.entries, key=lambda key: self.entries[key].when))
        self.entries[occurrence] = Token((occurrence,), kind, 1, when, value, native=native)
        self.names[occurrence] = Name(occurrence, reading)
        return occurrence

    def bind(self, antecedent, occurrence, *, kind, when, value=None, reading=None):
        token = self.token_of(antecedent)
        if token is None:
            # Losing track does not resurrect an evicted introduction.
            return self.open(occurrence, kind=kind, when=when, value=value, reading=reading)
        if self.token_of(occurrence) != token:
            self.forget(occurrence)
        entry = self.entries[token]
        uses = entry.occurrences
        if occurrence not in uses:
            uses += (occurrence,)
        joined = entry.value if value is None else torch.maximum(entry.value.to(value), self._value(value))
        self.entries[token] = replace(entry, occurrences=uses,
            kind=kind if entry.kind is None else entry.kind,
            count=entry.count + 1, when=when, value=joined)
        previous = self.names.get(occurrence)
        self.names[occurrence] = Name(token, previous.reading if reading is None and previous else reading)
        return token

    def update(self, token, value, *, when):
        key = self.token_of(token)
        if key is None:
            raise ValueError('cannot update an unretained identity')
        self.entries[key] = replace(self.entries[key], value=self._value(value), when=when)

    def event(self, token, occurrence, *, verb, role, when):
        key = self.token_of(token)
        if key is None or role not in (0, 2) or type(verb) is not int:
            raise ValueError('an event requires a retained identity, verb and participant role')
        entry = self.entries[key]
        event = Event(occurrence, verb, role, when)
        previous = tuple(item for item in entry.events
                         if (item.occurrence, item.role) != (occurrence, role))
        if previous != entry.events:
            # Re-observation updates evidence at its existing history position.
            events = tuple(event if (item.occurrence, item.role) == (occurrence, role)
                           else item for item in entry.events)
        else:
            events = (*previous, event)
        self.entries[key] = replace(entry, events=events, when=when)

    def forget(self, occurrence):
        name = self.names.pop(occurrence, None)
        if name is not None:
            token, entry = name.token, self.entries[name.token]
            if token == occurrence:
                del self.entries[token]
                for address in entry.occurrences:
                    self.names.pop(address, None)
            else:
                self.entries[token] = replace(entry, occurrences=tuple(
                    address for address in entry.occurrences if address != occurrence))
        # One event can witness a subject and an object in different identities.
        for token, entry in tuple(self.entries.items()):
            events = tuple(event for event in entry.events if event.occurrence != occurrence)
            if events != entry.events:
                self.entries[token] = replace(entry, events=events)

    def merge(self, token, into, *, when):
        """Join a pending object without moving the antecedent's existing uses."""
        source, target = self.token_of(token), self.token_of(into)
        if source is None or target is None:
            raise ValueError('MTM merge requires two retained tokens')
        if source == target:
            return target
        previous, antecedent = self.entries[source], self.entries[target]
        self.entries[target] = replace(antecedent,
            occurrences=antecedent.occurrences + previous.occurrences,
            count=antecedent.count + previous.count, when=when,
            kind=previous.kind if antecedent.kind is None else antecedent.kind,
            value=torch.maximum(antecedent.value, previous.value.to(antecedent.value)),
            events=antecedent.events + tuple(event for event in previous.events
                if not any((other.occurrence, other.role) == (event.occurrence, event.role)
                           for other in antecedent.events)))
        for occurrence in previous.occurrences:
            self.names[occurrence] = replace(self.names[occurrence], token=target)
        del self.entries[source]
        return target

    def state(self):
        """Checkpoint only committed values; export owns every tensor copy."""
        return dict(version=1, entries=tuple(dict(key=key, occurrences=entry.occurrences,
            kind=entry.kind, count=entry.count, when=entry.when, value=entry.value.detach().cpu().clone(),
            native=entry.native, events=tuple((event.occurrence, event.verb, event.role, event.when)
                                             for event in entry.events))
            for key, entry in self.entries.items()),
            names=tuple((mention, name.token, name.reading) for mention, name in self.names.items()))

    def restore(self, state, *, device=None):
        """Validate a complete pool before replacing its current context."""
        if state is None:
            self.entries, self.names = {}, {}
            return
        if not isinstance(state, dict) or state.get('version') != 1:
            raise ValueError('unsupported identity checkpoint')
        def address(value):
            return (isinstance(value, tuple) and len(value) == 3 and value[0] == 'ltm'
                    and isinstance(value[1], str) and type(value[2]) is int)
        def symbol(value):
            return value is None or type(value) is int and value > 0
        if not all(isinstance(state.get(field), (tuple, list)) for field in ('entries', 'names')):
            raise ValueError('identity checkpoint requires entries and names')
        entries, names = {}, {}
        for record in state['entries']:
            if not isinstance(record, dict) or not set(('key', 'occurrences', 'kind', 'count',
                    'when', 'value', 'native', 'events')) <= record.keys():
                raise ValueError('incomplete identity checkpoint entry')
            key, uses = record['key'], tuple(record['occurrences'])
            if (not address(key) or not all(map(address, uses)) or key in entries or
                    not uses or uses[0] != key or len(set(uses)) != len(uses)):
                raise ValueError('invalid identity checkpoint membership')
            if (not symbol(record['kind']) or not symbol(record['native']) or
                    type(record['count']) is not int or record['count'] < len(uses) or
                    type(record['when']) is not int):
                raise ValueError('invalid identity checkpoint metadata')
            value = self._value(record['value'])
            if entries and value.shape != next(iter(entries.values())).value.shape:
                raise ValueError('identity checkpoint values have different widths')
            if device is not None:
                value = value.to(device)
            if any(not isinstance(event, (tuple, list)) or len(event) != 4 for event in record['events']):
                raise ValueError('invalid identity checkpoint event')
            events = tuple(Event(*event) for event in record['events'])
            if (any(not address(event.occurrence) or event.role not in (0, 2) or
                    type(event.verb) is not int or event.verb <= 0 or type(event.when) is not int
                    for event in events) or len({(event.occurrence, event.role) for event in events}) != len(events)):
                raise ValueError('invalid identity checkpoint event')
            entries[key] = Token(uses, record['kind'], record['count'], record['when'], value,
                                 events, record['native'])
        for mention, token, reading in state['names']:
            if (not address(mention) or not address(token) or not symbol(reading) or
                    mention in names or token not in entries or mention not in entries[token].occurrences):
                raise ValueError('invalid identity checkpoint name')
            names[mention] = Name(token, reading)
        if (len(entries) > self.capacity or len(names) != sum(len(entry.occurrences)
                for entry in entries.values())):
            raise ValueError('identity checkpoint exceeds capacity or omits a name')
        self.entries, self.names = entries, names
