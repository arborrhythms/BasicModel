"""Detached order-zero readings beside the sentence's durable idea row."""
from dataclasses import replace
import torch
from Understanding import ConceptualField


@torch.no_grad()
def select(field, capacity, present, orders):
    """Keep the strongest signed basic readings, in their original order."""
    B = field.evidence.shape[1]
    ids = torch.full((B, capacity), -1, device=field.evidence.device, dtype=torch.long)
    values = field.evidence.new_zeros(B, capacity)
    if not capacity or not field.evidence.shape[2] or not len(field.concept_ids):
        return ids, values
    concept_ids = (field.concept_ids[:, None].expand(-1, B)
                   if field.concept_ids.ndim == 1 else field.concept_ids)
    orders = torch.as_tensor(orders, device=concept_ids.device)
    orders = orders[:, None].expand(-1, B) if orders.ndim == 1 else orders
    signed = field.evidence[..., 0] - field.evidence[..., 1]
    for b in range(B):
        if not bool(present[b]):
            continue
        active = ((concept_ids[:, b] >= 0) & (orders[:, b] == 0))[:, None] & (signed[:, b] != 0)
        candidates = [(r, int(concept_ids[i, b]), i) for i, r in active.nonzero().tolist()]
        chosen = sorted(candidates, key=lambda item:
            (-abs(float(signed[item[2], b, item[0]])), item[0], item[1]))[:capacity]
        for slot, (r, _, i) in enumerate(sorted(chosen)):
            ids[b, slot] = concept_ids[i, b]
            values[b, slot] = signed[i, b, r]
    return ids, values


def stage(model, sid, present, field):
    store = model.symbolSpace.ltm_store
    owner = model._concept_owner()
    orders = field.concept_ids.new_tensor([-1 if int(cid) < 0 else
        owner._row_order(owner._csw_row_of(int(cid))) for cid in field.concept_ids.flatten()]
        ).reshape(field.concept_ids.shape)
    model._sentence_episodes[int(sid)] = select(
        field, store.episodic_capacity, present, orders)


@torch.no_grad()
def write(store, index, entry, *, new):
    """A re-witness keeps the first episode, including an empty first entry."""
    if not new or entry is None or not isinstance(index, int) or not store.episodic_capacity:
        return
    for name, value in zip(('episodic_concepts', 'episodic_values'), entry):
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


@torch.no_grad()
def reinstate(store, frames, stream):
    """Publish the remembered concepts, without reinterpreting old percepts."""
    ref = store.__dict__.get('_concept_owner')
    owner = None if ref is None else ref()
    if owner is None or owner._model is None:
        return False
    model = owner._model
    model._episode_counts['time_only_recalls'] += 1
    indices = [frame['index'] for frame in frames if isinstance(frame['index'], int)
               and 0 <= stream < store.stream.shape[1] and bool(store.stream[frame['index'], stream])
               and bool((store.episodic_concepts[frame['index']] >= 0).any())]
    if not indices:
        return False
    entries = []
    for i in indices:
        for k, cid in enumerate(store.episodic_concepts[i].tolist()):
            if cid < 0:
                continue
            if owner._csw_row_of(cid) is None:
                model._episode_counts['ids_skipped'] += 1
            else:
                entries.append((i, k, cid))
    from SentenceField import knowing, publish
    current = knowing(model)
    B = current.evidence.shape[1]
    ids = sorted(set(current.concept_ids.flatten().tolist()) | {cid for _, _, cid in entries})
    lookup = {cid: i for i, cid in enumerate(ids)}
    R = max(current.evidence.shape[2], len(entries))
    evidence = current.evidence.new_zeros(len(ids), B, R, 2)
    where = current.where.new_zeros(B, R, 2)
    when = current.when.new_zeros(B, R, 2)
    for b in range(B):
        if b == stream:
            continue
        column = current.concept_ids if current.concept_ids.ndim == 1 else current.concept_ids[:, b]
        for old, cid in enumerate(column.tolist()):
            evidence[lookup[cid], b, :current.evidence.shape[2]] = current.evidence[old, b]
        where[b, :current.where.shape[1]] = current.where[b]
        when[b, :current.when.shape[1]] = current.when[b]
    from RowTime import encoder
    cursor = 0
    for reading, (i, k, cid) in enumerate(entries):
        value = store.episodic_values[i, k]
        evidence[lookup[cid], stream, reading] = torch.stack((value.clamp_min(0), (-value).clamp_min(0)))
        raw = owner.word_surface_for_row(owner._csw_row_of(cid))
        if raw is None:
            model._episode_counts['surfaces_missing'] += 1
        length = 0 if raw is None else len(raw)
        where[stream, reading] = where.new_tensor((cursor, cursor + length))
        cursor += length + 1
        tick = encoder().decode_index(store.entry_when(i))
        when[stream, reading] = torch.stack((tick, tick + 1))
    field = ConceptualField(current.concept_ids.new_tensor(ids), evidence, where, when, None)
    owner._episode_review_field = field
    owner._cs_field_concept_ids = field.concept_ids
    owner._cs_field_where, owner._cs_field_when = where, when
    publish(model, field)
    perception = owner._percept_field
    if perception is not None:
        def emptied(value, axis):
            if value is None:
                return None
            result = value.clone()
            result.select(axis, stream).zero_()
            return result
        owner._percept_field = replace(perception,
            values=emptied(perception.values, 1), events=emptied(perception.events, 1),
            spans=emptied(perception.spans, 0), where=emptied(perception.where, 0),
            when=emptied(perception.when, 0), when_band=emptied(perception.when_band, 0))
    model._episode_counts['reinstatements'] += 1
    return True
