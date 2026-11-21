"""Sparse percept activations and the attended percepts that carry them.

Concept definitions read only ``values``. Locations belong to ``events``, one
for each attended percept, and travel to attention and reconstruction by
percept identity, never concept row. Each independent reading owns one convex
where bracket and one when interval.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PerceptField:
    keys: tuple
    values: torch.Tensor       # [percept, batch, reading, pole]
    events: torch.Tensor       # [percept, batch, attended, pole]
    spans: torch.Tensor        # [batch, attended, 2], perception's coordinates
    where: torch.Tensor        # [batch, reading, 2], one bracket per reading
    when: torch.Tensor         # [batch, reading, 2], one interval per reading
    percept_where: torch.Tensor | None = None  # [feature, member, 4], native percept ladders
    when_band: torch.Tensor | None = None      # [batch, reading, 4], shared field time
    boundaries: torch.Tensor | None = None     # [part/whole, percept, batch, reading, pole]
    parts: tuple | None = None                 # (member keys, [member, batch, reading, pole]) read at this field
    reading: torch.Tensor | None = None        # [batch, attended] the reading that holds each, or -1
    index: torch.Tensor | None = None          # [batch, attended] each one's occurrence in the bank

    def clone(self):
        return type(self)(self.keys, *(None if getattr(self, name) is None else
            getattr(self, name).clone() for name in
            ('values', 'events', 'spans', 'where', 'when', 'percept_where', 'when_band', 'boundaries')),
            self.parts, None if self.reading is None else self.reading.clone(),
            None if self.index is None else self.index.clone())

    def select(self, batch, reading):
        """One stream's one reading: its attended percepts are those the
        reading holds, renumbered as reading 0."""
        b, e = slice(batch, batch + 1), slice(reading, reading + 1)
        held = self.reading[b] == reading
        return type(self)(self.keys, self.values[:, b, e], self.events[:, b] * held[None, ..., None],
                          self.spans[b], self.where[b, e], self.when[b, e],
                          self.percept_where,
                          None if self.when_band is None else self.when_band[b, e],
                          None if self.boundaries is None else self.boundaries[:, :, b, e],
                          None, torch.where(held, 0, -1), None if self.index is None else self.index[b])

    def selected(self, keys):
        lookup = {key: i for i, key in enumerate(self.keys)}
        return torch.tensor([lookup[key] for key in keys],
                            dtype=torch.long, device=self.values.device)

    def event_evidence(self, keys):
        """Each of ``keys`` at each attended percept of each reading, ``[keys,
        batch, reading, attended, pole]``, nothing outside the reading that
        holds it. Later definitions cannot invent an event in an earlier
        snapshot. Read on a selected field (one stream, one reading)."""
        lookup = {key: i for i, key in enumerate(self.keys)}
        unknown = len(self.keys)
        indices = torch.tensor([lookup.get(key, unknown) for key in keys],
                               dtype=torch.long, device=self.events.device)
        padded = torch.cat((self.events, self.events.new_zeros(1, *self.events.shape[1:])))
        chosen = padded.index_select(0, indices)                                # [k, B, A, 2]
        slots = torch.arange(self.values.shape[2], device=chosen.device)
        held = (self.reading[:, None, :] == slots[None, :, None]).to(chosen)    # [B, K, A]
        return chosen[:, :, None] * held[None, ..., None]
