"""Bounded identity pointers between kept closes; no stored working content."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Token:
    occurrences: tuple
    kind: int | None
    count: int
    when: int


class MediumTermMemory:
    """Tokens are keyed by their introduction, independently of LTM and seats.

    ``when`` is the model tick decoded from RowTime.stamp at the kept close.
    All retained values are Python data. Reads neither refresh recency nor
    retain the working split/merge interpretation.
    """

    def __init__(self, capacity=64):
        if capacity < 1:
            raise ValueError('mtmCapacity must be positive')
        self.capacity = int(capacity)
        self.entries = {}
        self._tokens = {}

    def token_of(self, occurrence):
        return self._tokens.get(occurrence)

    def open(self, occurrence, *, kind, when):
        if self.token_of(occurrence) != occurrence:
            self.forget(occurrence)
        if occurrence in self.entries:
            entry = self.entries[occurrence]
            self.entries[occurrence] = replace(entry, count=entry.count + 1, when=when)
            return occurrence
        if len(self.entries) >= self.capacity:
            oldest = min(self.entries, key=lambda key: self.entries[key].when)
            self.forget(oldest)
        self.entries[occurrence] = Token((occurrence,), kind, 1, when)
        self._tokens[occurrence] = occurrence
        return occurrence

    def bind(self, antecedent, occurrence, *, kind, when):
        token = self.token_of(antecedent)
        if token is None:
            # Losing track does not resurrect an evicted introduction.
            return self.open(occurrence, kind=kind, when=when)
        if self.token_of(occurrence) != token:
            self.forget(occurrence)
        entry = self.entries[token]
        uses = entry.occurrences
        if occurrence not in uses:
            uses += (occurrence,)
        self.entries[token] = replace(entry, occurrences=uses,
                                     kind=kind if entry.kind is None else entry.kind,
                                     count=entry.count + 1, when=when)
        self._tokens[occurrence] = token
        return token

    def forget(self, occurrence):
        token = self._tokens.pop(occurrence, None)
        if token is None:
            return
        entry = self.entries[token]
        if token == occurrence:
            del self.entries[token]
            for address in entry.occurrences:
                self._tokens.pop(address, None)
        else:
            self.entries[token] = replace(entry, occurrences=tuple(
                address for address in entry.occurrences if address != occurrence))

    def merge(self, token, into, *, when):
        """A kept bind joins the pending object to its chosen antecedent."""
        source, target = self.token_of(token), self.token_of(into)
        if source is None or target is None:
            raise ValueError('MTM merge requires two retained tokens')
        if source == target:
            return target
        previous, antecedent = self.entries[source], self.entries[target]
        self.entries[target] = replace(antecedent,
            occurrences=antecedent.occurrences + previous.occurrences,
            count=antecedent.count + previous.count, when=when,
            kind=previous.kind if antecedent.kind is None else antecedent.kind)
        for occurrence in previous.occurrences:
            self._tokens[occurrence] = target
        del self.entries[source]
        return target
