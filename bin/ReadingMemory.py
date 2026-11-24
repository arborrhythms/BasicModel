"""Located ideas from attention, committed as one sequence in the ordinary pair store."""
import torch
from Memory import OCCURRENCE, SYMBOL, RELATION


def brackets(model, percepts):
    """Name lexical occurrences from complete, visible native percepts. MTM owns nomination."""
    from WordSurfaces import is_lexical_form
    ps, owner = model.perceptualSpace, model._concept_owner()
    ids, spans = percepts[:2]
    intervals = ps._forward_input.get('word_intervals') or ()
    words = []
    for b, limits in enumerate(intervals):
        parts = [(pid, lo, hi) for pid, (lo, hi) in zip(ids[b].tolist(), spans[b].tolist(), strict=True)
                 if pid >= 0 and lo < hi]
        row = []
        for start, stop in limits:
            members = [(pid, lo, hi) for pid, lo, hi in parts if start <= lo < hi <= stop]
            if not members or members[0][1] != start or members[-1][2] != stop:
                continue
            if any(left[2] != right[1] for left, right in zip(members, members[1:])):
                continue
            native = [pid for pid, _, _ in members]
            form = b''.join(ps.percept_store.bytes_for(pid) for pid in native)
            if not is_lexical_form(form):
                continue
            word = owner.interpret.word_for_form(form)
            if word is None:
                word = owner.interpret.lookup_word(native, model.wholeSpace.property_rows_for_bytes(form),
                    form=form, word_reading=True, reading=owner.reading(b))
            row.append((start, stop, -1 if word is None else word))
        words.append(row)
    width = max(map(len, words), default=0)
    if not width:
        return None
    table = ids.new_tensor([row + [(0, 0, -1)] * (width - len(row)) for row in words])
    return table[..., :2], table[..., 2]


def _name_lookup(names, values, missing):
    """Resolve arbitrary occurrence names, with one sentinel for a missing row."""
    keys, order = names.sort(stable=True)
    positions = torch.searchsorted(keys, values.contiguous())
    size = names.shape[0]
    keys = torch.cat((keys, names.new_full((1,), -1)))
    rows = torch.cat((order, order.new_full((1,), missing)))
    return torch.where((positions < size) & (keys[positions] == values), rows[positions], missing)


def retained_masks(head, handles, kinds, endpoints, names, referents):
    """Reachable physical rows for a typed head. Handles are in dependency order."""
    relation_live = (head[0] == RELATION) & (handles == head[1])
    # Each local child precedes its parent. The loop bound is shape metadata;
    # the root, both child addresses and every branch mask stay on device.
    for i in reversed(range(handles.shape[0])):
        child = (kinds[i, :, None] == RELATION) & (handles == endpoints[i, :, None])
        relation_live = relation_live | (relation_live[i] & child.any(0))
    # Map an occurrence address to its potentially non-contiguous physical seat.
    # The final seat absorbs symbols and already-durable references.
    n = names.shape[0]
    targets = _name_lookup(names, torch.cat((head[1:].reshape(1), endpoints.flatten())), n)
    selected = torch.cat(((head[:1] == OCCURRENCE),
                          ((kinds == OCCURRENCE) & relation_live[:, None]).flatten()))
    live = names.new_zeros(n + 1).scatter_add(0, targets, selected.long()).bool()
    # Each idea has at most one occurrence referent. Pointer doubling covers
    # the entire chain in logarithmic rounds and also handles shared referents.
    following = torch.where(referents[:, 0] == OCCURRENCE, _name_lookup(names, referents[:, 1], n), n)
    following = torch.cat((following, names.new_full((1,), n)))
    for _ in range(n.bit_length()):
        live = live | torch.zeros_like(following).scatter_add(0, following, live.long()).bool()
        following = following[following]
    return live[:n], relation_live


def _retention_plan(store, head, durable, first_name):
    parts = store.parts
    parts._sort()
    handles = parts.relation_order[:len(parts)]
    names = store.row_names[:len(store)]
    root = names.new_tensor(parts.reference(head))
    keep, links = retained_masks(root, handles, parts.ref_kinds[handles], parts.refs[handles],
                                 names, store.referent[:len(store)])
    rows = keep.nonzero().flatten()
    handles = handles[links]
    source_names = names[rows]
    new_names = first_name + torch.arange(rows.numel(), device=names.device)
    # The same translation covers roots, operands and occurrence-to-occurrence
    # references. A durable name is legal only in its own address domain.
    references = torch.cat((root[None], store.referent[rows],
                            torch.stack((parts.ref_kinds[handles], parts.refs[handles]), -1).reshape(-1, 2)))
    kinds, values = references.unbind(-1)
    local = _name_lookup(source_names, values, len(rows))
    occurrence = kinds == OCCURRENCE
    translated = torch.cat((new_names, new_names.new_zeros(1)))[local]
    external = store.__dict__.get('_external_memory')
    shared = external is not None and external() is durable
    valid = ~occurrence | (local < len(rows))
    local_link = _name_lookup(handles, values, -1) >= 0
    valid &= (kinds != RELATION) | local_link
    if shared:
        valid |= occurrence & (_name_lookup(durable.row_names[:len(durable)], values, -1) >= 0)
        address = values.clamp(0, durable.parts.capacity - 1)
        valid |= ((kinds == RELATION) & (address == values) & (durable.parts.ref_kinds[address, 0] != 0))
    if not bool(valid.all()):
        raise ValueError('reading names an unavailable occurrence or relation')
    values = torch.where(occurrence & (local < len(rows)), translated, values)
    refs = torch.stack((kinds, values), -1)
    idea_refs = refs[1:1 + len(rows)]
    endpoints = refs[1 + len(rows):].reshape(-1, 2, 2)
    return rows, new_names, idea_refs, handles, endpoints, refs[0]


def _reserve_symbols(blocked, values):
    """A standalone relation allocator skips names held by symbols or episodic concepts."""
    legal = (values > 0) & (values < blocked.shape[0])
    blocked.index_fill_(0, torch.where(legal, values, 0).flatten(), True)


def translate_pairs(handles, endpoints, head, refs, tags, blocked, next_id, *, sequential):
    """Intern the incoming topology into private copies of the LTM address arrays.

    Existing pairs keep their handles, including pairs encountered on a prior
    stream. Local children are translated before their parent. Zero is the
    reserved sink when capacity runs out. The caller checks the batch before
    publishing any payload.
    """
    n = handles.shape[0]
    seats = torch.arange(refs.shape[0], device=refs.device)
    names = handles.new_zeros(n + 1)
    translated = torch.empty_like(endpoints)
    fresh = handles.new_zeros(n, dtype=torch.bool)
    ok = handles.new_ones((), dtype=torch.bool)
    local = _name_lookup(handles, endpoints[..., 1], n)
    root_at = _name_lookup(handles, head[1:2], n)
    for i in range(n):
        kinds, original = endpoints[i].unbind(-1)
        values = torch.where((kinds == RELATION) & (local[i] < n), names[local[i]], original)
        matches = (tags == kinds).all(1) & (refs == values).all(1)
        found = matches.any()
        if sequential:
            free = next_id
        else:
            free = torch.where(~blocked, seats, len(seats)).amin()
        available = tags[free.clamp(0, len(seats)-1).reshape(1), 0][0] == 0
        can_allocate = torch.logical_and((free > 0) & (free < len(seats)), available)
        ok = torch.logical_and(ok, torch.logical_or(found, can_allocate))
        name = torch.where(found, matches.long().argmax(), torch.where(can_allocate, free, 0))
        index = name.reshape(1)
        refs.index_copy_(0, index, values[None])
        tags.index_copy_(0, index, kinds[None].to(tags.dtype))
        blocked.index_fill_(0, index, True)
        if not sequential:
            _reserve_symbols(blocked, torch.where(kinds == SYMBOL, values, 0))
        names[i] = name
        translated[i] = torch.stack((kinds, values), -1)
        fresh[i] = ~found
        next_id = next_id + (~found).long()
    root_value = torch.where((head[0] == RELATION) & (root_at[0] < n), names[root_at][0], head[1])
    return names[:n], translated, fresh, torch.stack((head[0], root_value)), next_id, ok


def commit(stores, heads, durable):
    """Retain just the reachable idea payload and translate its ordered pairs."""
    plans = []
    n_ideas = n_pairs = 0
    for store, head in zip(stores, heads, strict=True):
        if store.nDim != durable.nDim or store.episodic_capacity != durable.episodic_capacity:
            raise ValueError('incompatible STM and LTM reading rows')
        if head is None:
            plans.append(None)
            continue
        plan = _retention_plan(store, head, durable, durable.names_issued + (n_ideas + 1))
        n_ideas += len(plan[0])
        n_pairs += len(plan[3])
        plans.append(plan)
    if len(durable) + n_ideas > durable.capacity:
        raise OverflowError('reading exceeds remaining long-term idea capacity')
    parts = durable.parts
    planned = []
    if n_pairs:
        refs, tags = parts.refs.clone(), parts.ref_kinds.clone()
        blocked = tags[:, 0] != 0
        blocked[0] = True
        allocator = getattr(parts.allocate, '__self__', None)
        sequential = allocator is not None
        if not sequential:
            _reserve_symbols(blocked, torch.where(tags == SYMBOL, refs, 0))
            _reserve_symbols(blocked, durable.episodic_concepts[:len(durable)])
            if parts.handle_reserved is not None:
                blocked |= blocked.new_tensor([parts.handle_reserved(row) for row in range(parts.capacity)])
        next_id = refs.new_tensor(allocator.next_id if sequential else 0)
        valid = blocked.new_ones(())
    for store, plan in zip(stores, plans, strict=True):
        if plan is None:
            planned.append(None)
            continue
        rows, names, referents, handles, endpoints, root = plan
        links = []
        if n_pairs and not sequential:
            _reserve_symbols(blocked, store.episodic_concepts[rows])
        if len(handles):
            assigned, endpoints, fresh, root, next_id, ok = translate_pairs(
                handles, endpoints, root, refs, tags, blocked, next_id, sequential=sequential)
            valid = valid & ok
            links = torch.cat((assigned[:, None], endpoints.flatten(1), fresh[:, None]), 1).tolist()
        planned.append((links, tuple(root.tolist())))
    if n_pairs and not bool(valid):
        raise OverflowError('reading exceeds remaining long-term relation capacity')

    committed = []
    for store, plan, allocation in zip(stores, plans, planned, strict=True):
        if allocation is None:
            committed.append(None)
            continue
        rows, names, referents, handles = plan[:4]
        durable.retain_ideas(store, rows, names, referents)
        links, root = allocation
        evidence = torch.stack((store.parts.truth_plus[handles], store.parts.truth_minus[handles],
                                store.parts.activity[handles]), 1).tolist()
        for (name, ka, a, kb, b, fresh), (positive, negative, activity) in zip(links, evidence, strict=True):
            if fresh and parts.allocate is not None and parts.allocate() != name:
                raise RuntimeError('the relation allocator changed during the reading close')
            parts.add((ka, a), (kb, b), symbol=name, truth_plus=positive, truth_minus=negative,
                      surprise=activity)
        committed.append(root)
    return tuple(committed)


def read(model, bank, concepts):
    """Write the narrowed looks to memory and return the caller its open field."""
    owner = model._concept_owner()
    # Each look binds its own packed rows. Keep their addresses and evidence
    # together when returning to the parallel input that triggered the read.
    names = ('_percept_field', '_cs_last_a0', '_cs_order0_raw', '_cs_field_rows',
             '_cs_field_concept_ids', '_cs_field_caps', '_cs_field_where', '_cs_field_when',
             '_cs_field_wires', '_cs_level_act_tensor', '_cs_level_rows', '_cs_wave_qe')
    opened = {name: vars(owner)[name] for name in names if name in vars(owner)}
    try:
        return _read(model, bank, concepts)
    finally:
        for name in names:
            if name in opened:
                object.__setattr__(owner, name, opened[name])
            else:
                vars(owner).pop(name, None)


def _read(model, bank, concepts):
    """Narrow until the request stops, STM fills or nothing is found."""
    from Places import readings
    from Attention import occurrence_probabilities, select_occurrence
    from LookForecast import OccurrenceCredit
    from SentenceLooks import attend, narrow, lower
    from dataclasses import replace
    from EpisodicMemory import select, unkept, uncertain
    from SentenceField import inventory
    from RequestRead import at_rows, spread_once
    owner, reader = model._concept_owner(), model.request_read
    B = len(concepts)
    strata = model.conceptualSpace.stm.strata
    stores = [strata.idea_stores[b * strata.trials] for b in range(B)]
    heads = [None] * B
    for store in stores:
        store.expire()
        store.to(bank.table)
    current = owner._cs_open_field
    def located(field, index=None):
        return at_rows(inventory(owner, field), field, reader.rows, B, index)
    opened = located(current)
    memory, own = torch.zeros_like(opened), bank.activity.clone()
    left = bank.where[..., 1] > bank.where[..., 0]
    known = concepts >= 0
    _, _, percept_codes = readings(bank)
    observation = model.conceptualSpace.stm.model_state
    observation.begin_looks(bank.where, left.any(-1))
    credit = OccurrenceCredit(owner, bank) if model.training and torch.is_grad_enabled() else None
    excluded, ambiguous = unkept(owner), uncertain(owner)
    for _ in range(min(model.attention_budget, model.conceptualSpace.stm.capacity)):
        units = reader.units(reader.given(opened, memory, left.any(-1)),
                             observation.request_state(model.where_registry))
        active = left.any(-1) & (reader.stopping(units).detach() <= 0)
        if not active.any():
            break
        request = attend(model, units)
        if credit is not None:
            anticipated = credit.forecast(request)
        field, _, scores = narrow(model, bank, request, own, concepts,
                                  known, left & active[:, None])
        probabilities = occurrence_probabilities(scores)
        index = select_occurrence(probabilities, sample=model.training)
        index = torch.where(active, index, -1)
        observation.observe(bank.where, index)
        found = index >= 0
        if not found.any():
            break
        if credit is not None:
            credit.observe(anticipated, index)
        batch = torch.arange(B, device=index.device)
        evidence = field.evidence[:, batch, index.clamp_min(0)].unsqueeze(2)
        episode = select(replace(field, evidence=evidence), stores[0].episodic_capacity,
                         found, unkept=excluded, uncertain=ambiguous)
        for b in found.nonzero().flatten().tolist():
            w, store = int(index[b]), stores[b]
            cid = int(concepts[b, w])
            code = percept_codes[b, w]
            if cid >= 0:
                row = owner._concept_row_of(cid)
                code = owner.similarity_codebook.lookup_rows(index.new_tensor([row]), stream=b)[0]
            point = store._fit(code).clone()
            point[store.when_start-4:store.when_start] = field.wires[b, w]
            point[store.when_start:store.when_start+4] = bank.when_band[b, w]
            row = store.append_idea(point, referent=None if cid < 0 else ('sym', cid),
                trust=float(scores[b, w].detach().clamp(0, 1)), entry=(episode[0][b], episode[1][b]))
            heads[b] = store.parts.append_occurrence(heads[b], store.row_name(row))
            left[b, w] = False
        memory = torch.where(found[:, None, None], spread_once(owner, located(field, index)), memory)
        own = lower(own, bank, index, found)
    if credit is not None:
        credit.register(model.errors)
    durable = model.symbolSpace.ltm_store
    committed = commit(stores, heads, durable)
    for b, head in enumerate(committed):
        if head is not None:
            owner.mtm.close(durable, stream=b)
    return committed
