"""Small detached percept witnesses beside the sentence's durable idea row."""
import torch
from torch.nn import functional as F

from PerceptField import PerceptField


@torch.no_grad()
def select(field, capacity, present):
    """Keep the strongest reading of each percept, preserving both poles."""
    B = field.values.shape[1]
    ids = torch.full((B, capacity), -1, device=field.values.device, dtype=torch.long)
    poles = field.values.new_zeros(B, capacity, 2)
    where = field.where.new_zeros(B, capacity, 2)
    if not capacity or not field.values.shape[2] or not len(field.keys):
        return ids, poles, where
    ordered = sorted(range(len(field.keys)), key=lambda i: field.keys[i][1])
    if any(field.keys[i][0] != 'bp' for i in ordered):
        raise ValueError('episodic entries require basic percept identities')
    strength, reading = field.values.sum(-1).max(-1)
    for b in range(B):
        if not bool(present[b]):
            continue
        chosen = sorted(ordered, key=lambda i: -float(strength[i, b]))[:capacity]
        for slot, i in enumerate(chosen):
            if not bool(strength[i, b] > 0):
                break
            r = int(reading[i, b])
            ids[b, slot] = field.keys[i][1]
            poles[b, slot] = field.values[i, b, r]
            where[b, slot] = field.where[b, r]
    return ids, poles, where


def stage(model, sid, present):
    store = model.symbolSpace.ltm_store
    model._sentence_episodes[int(sid)] = select(
        model._concept_owner()._percept_field, store.episodic_capacity, present)


@torch.no_grad()
def write(store, index, entry, *, new):
    """A re-witness keeps the first episode, including an empty first entry."""
    if not new or entry is None or not isinstance(index, int) or not store.episodic_capacity:
        return
    for name, value in zip(('episodic_percepts', 'episodic_poles', 'episodic_where'), entry):
        getattr(store, name)[index].copy_(value.detach())
    count(store, 'written', int(bool((entry[0] >= 0).any())))


def count(store, name, amount=1):
    """Passive counts of committed input closes and their episode rows."""
    ref = store.__dict__.get('_concept_owner')
    owner = None if ref is None else ref()
    model = None if owner is None else owner._model
    if model is not None:
        model._episode_counts[name] += amount


def plan_anchor(store, clause, entry, stream):
    """Reserve a sentence address separate from every semantic root address."""
    if entry is None or not store.episodic_capacity or not bool((entry[0] >= 0).any()):
        return None
    from Occurrence import address_key, document_digest, native_content_key
    context = store.__dict__.get('_write_address')
    if context is None:
        document, position, content = ('episode', 'direct', stream), 0, native_content_key(clause.meaning)
    else:
        document, position, content = ('episode', context[0].hex()), context[1], context[2]
    address = dict(document_key=document, sentence_index=position, content_key=content)
    existing = store._index_occurrences.get(address_key(document_digest(document), position, content))
    return address, existing


def anchor(store, plan, entry, *, stream, when):
    """An episode has only its time band; it cannot supply an idea root."""
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    address, existing = plan
    meaning = replace(ConceptualMeaning.from_description(store.slots.new_zeros(store.nDim)),
                      bindings={'_episode': True})
    row = store.append_meaning(meaning, kind='episode', evidence=(0., 0.),
        stream=stream, when=when, **address)
    write(store, row, entry, new=existing is None)
    count(store, 'anchors', int(existing is None))
    return row


def _field(owner, store, indices, stream):
    """Rebuild located readings, replacing only the querying stream."""
    current = owner._percept_field
    # STM can retain its constructed stream slab when a partial batch has
    # fewer live rows. The attended field owns this pass's batch dimension.
    B = current.values.shape[1] if current is not None else owner.stm.state[0].shape[0]
    entries = [(i, k) for i in indices for k in range(store.episodic_capacity)
               if int(store.episodic_percepts[i, k]) >= 0]
    matrix = owner._concept_allocator.layer(0).features
    kept_ids = {int(store.episodic_percepts[i, k]) for i, k in entries}
    required = {col // 2 for col in matrix._cols}
    others = B > 1 and current is not None
    keys = tuple(sorted({('bp', p) for p in kept_ids | required} |
                        (set(current.keys) if others else set())))
    key_rows = {key: row for row, key in enumerate(keys)}
    R = max(len(entries), current.values.shape[2] if others else 0)
    E = max(len(entries), current.spans.shape[1] if others else 0)
    values = store.slots.new_zeros(len(keys), B, R, 2)
    events = values.new_zeros(len(keys), B, R, E, 2)
    where = values.new_zeros(B, R, 2)
    when = torch.zeros(B, R, 2, device=values.device, dtype=torch.long)
    spans = where.new_zeros(B, E, 2)
    bands = values.new_zeros(B, R, 4)
    if others:
        # The other streams keep their present field. They did not query.
        r, e = current.values.shape[2], current.spans.shape[1]
        for b in range(B):
            if b == stream:
                continue
            for old, key in enumerate(current.keys):
                row = key_rows[key]
                values[row, b, :r] = current.values[old, b]
                events[row, b, :r, :e] = current.events[old, b]
            where[b, :r], when[b, :r] = current.where[b], current.when[b]
            spans[b, :e] = current.spans[b]
            bands[b, :r] = current.when_band[b]
    from RowTime import encoder
    clock = encoder()
    for reading, (i, k) in enumerate(entries):
        row = key_rows['bp', int(store.episodic_percepts[i, k])]
        pair = store.episodic_poles[i, k]
        values[row, stream, reading] = pair
        events[row, stream, reading, reading] = pair
        where[stream, reading] = spans[stream, reading] = store.episodic_where[i, k]
        band = store.entry_when(i)
        tick = clock.decode_index(band)
        when[stream, reading] = torch.stack((tick, tick + 1))
        bands[stream, reading] = band
    return PerceptField(keys, values.detach(), events.detach(), spans, where, when,
                        when_band=bands.detach())


def reinstate(store, frames, stream):
    """Review only episodes reached by the existing, metered time query."""
    ref = store.__dict__.get('_concept_owner')
    owner = None if ref is None else ref()
    if owner is None or owner._model is None:
        return False
    model = owner._model
    model._episode_counts['time_only_recalls'] += 1
    indices = [frame['index'] for frame in frames if isinstance(frame['index'], int)
               and 0 <= stream < store.stream.shape[1] and bool(store.stream[frame['index'], stream])
               and bool((store.episodic_percepts[frame['index']] >= 0).any())]
    if not indices:
        return False
    perception = _field(owner, store, indices, stream)
    a0 = owner.cs_read_percept_field(perception)
    from SentenceField import publish
    from Understanding import ConceptualField
    field = ConceptualField(owner._cs_field_concept_ids,
        F.pad(a0, (0, 0, 0, 0, 0, 0, 0, sum(owner._field_caps()) - len(a0))),
        perception.where, perception.when, None)
    owner._episode_review_field = field
    publish(model, field)
    model._episode_counts['reinstatements'] += 1
    return True
