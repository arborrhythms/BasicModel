"""Sparse percept activations and their native occurrences inside attention.

Concept definitions read only ``values``. Locations belong to ``events`` and
travel to attention and reconstruction by percept identity, never concept row.
Each independent reading owns one convex where bracket and one when interval.
"""
from dataclasses import dataclass

import torch

from PerceptProperties import counts_in_spans


@dataclass(frozen=True)
class PerceptField:
    keys: tuple
    values: torch.Tensor       # [percept, batch, reading, pole]
    events: torch.Tensor       # [percept, batch, reading, event, pole]
    spans: torch.Tensor        # [batch, event, 2], perception's coordinates
    where: torch.Tensor        # [batch, reading, 2], one bracket per reading
    when: torch.Tensor         # [batch, reading, 2], one interval per reading
    percept_where: torch.Tensor | None = None  # [feature, member, 4], native percept ladders
    when_band: torch.Tensor | None = None      # [batch, reading, 4], shared field time

    def clone(self):
        return type(self)(self.keys, *(None if getattr(self, name) is None else
            getattr(self, name).clone() for name in
            ('values', 'events', 'spans', 'where', 'when', 'percept_where', 'when_band')))

    def select(self, batch, reading):
        b, e = slice(batch, batch + 1), slice(reading, reading + 1)
        return type(self)(self.keys, self.values[:, b, e], self.events[:, b, e],
                          self.spans[b], self.where[b, e], self.when[b, e],
                          self.percept_where,
                          None if self.when_band is None else self.when_band[b, e])

    def selected(self, keys):
        lookup = {key: i for i, key in enumerate(self.keys)}
        return torch.tensor([lookup[key] for key in keys],
                            dtype=torch.long, device=self.values.device)

    def event_evidence(self, keys):
        """Later definitions cannot invent an event in an earlier snapshot."""
        lookup = {key: i for i, key in enumerate(self.keys)}
        unknown = len(self.keys)
        indices = torch.tensor([lookup.get(key, unknown) for key in keys],
                               dtype=torch.long, device=self.events.device)
        padded = torch.cat((self.events, self.events.new_zeros(1, *self.events.shape[1:])))
        return padded.index_select(0, indices)


def read_percepts(percepts, brackets, keys, *, part_reader, native=None,
                  dtype=None, when=None, registry=None, when_encoding=None):
    """Read each referenced native percept once, then pool before any fold."""
    part_ids, part_spans, primitive, raw, whole_spans = percepts
    raw = raw[:, 0] if raw.ndim == 3 else raw
    keys = tuple(dict.fromkeys(keys))
    B, E = raw.shape[0], brackets.shape[1]
    spans = torch.cat((part_spans, whole_spans), 1).to(raw.device)
    P = spans.shape[1]
    contained = ((spans[:, None, :, 0] >= brackets[..., 0, None])
                 & (spans[:, None, :, 1] <= brackets[..., 1, None])
                 & (spans[:, None, :, 1] > spans[:, None, :, 0]))
    events = torch.zeros(len(keys), B, E, P, 2, device=raw.device, dtype=dtype)
    ps = [i for i, key in enumerate(keys) if key[0] == 'ps']
    if ps:
        _, support = part_reader(native, [keys[i][1] for i in ps],
            part_ids, part_spans, brackets, raw.shape[1], return_support=True)
        # Containment is positive evidence at the matching part events. Other
        # identities are not observations of this part's complement.
        pair = torch.stack((support, torch.zeros_like(support)), -1).to(events)
        pair = torch.nn.functional.pad(pair, (0, 0, 0, P - part_ids.shape[1]))
        events = events.index_copy(0, torch.tensor(ps, device=raw.device), pair)
    ws = [i for i, key in enumerate(keys) if key[0] == 'ws']
    if ws and primitive is not None:
        counts = counts_in_spans(raw, spans, observed=raw != 0)
        pair = primitive.evidence_on_counts(counts)
        complete = counts.sum(-1) == spans[..., 1] - spans[..., 0]
        pair = pair * complete[..., None, None]
        ids = torch.tensor([keys[i][1] for i in ws], device=raw.device)
        pair = pair.index_select(2, ids).permute(2, 0, 1, 3)
        events = events.index_copy(0, torch.tensor(ws, device=raw.device),
            pair[:, :, None].expand(-1, -1, E, -1, -1).to(events))
    events = events * contained[None, ..., None]
    values = events.amax(-2) if P else events.sum(-2)
    if when is None:
        when = torch.zeros_like(brackets)
    bands = None
    if registry is not None:
        # A PS definition may reference a native group: retain the band's
        # address for EVERY member instead of inventing an address for the
        # conjunction. Padding makes no location claim.
        groups = [[registry.slices['parts' if role == 'ps' else 'wholes'][0] + member
                   for member in (row if isinstance(row, tuple) else (row,))]
                  for role, row in keys]
        width = max((len(group) for group in groups), default=1)
        addresses = torch.tensor([group + [-1] * (width - len(group)) for group in groups],
                                 dtype=torch.long, device=raw.device).reshape(len(keys), width)
        bands = torch.where((addresses >= 0)[..., None], registry.encoding.encode(addresses), 0.)
    when_band = None if when_encoding is None else when_encoding.encode(when[..., 0])
    return PerceptField(keys, values, events, spans, brackets.clone(), when.clone(), bands, when_band)
