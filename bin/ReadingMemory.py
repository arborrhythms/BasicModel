"""Located ideas from attention, committed as one sequence in the ordinary pair store."""
import torch
from architecture import WHERE_WIRES
from Memory import OCCURRENCE, SYMBOL, RELATION
from MemoryActivity import written_complete


def covered_units(ids, spans, units):
    """Find complete contiguous native covers of the delivered word intervals."""
    if not ids.shape[1]:
        return torch.zeros_like(units[..., 0], dtype=torch.bool)
    lo, hi = spans.unbind(-1)
    start, stop = units.unbind(-1)
    members = ((ids >= 0) & (lo < hi))[:, None] & (start[..., None] <= lo[:, None]) & (hi[:, None] <= stop[..., None])
    # A padding gap in the native row bank is legal. Compare each member
    # with the end of its preceding selected member in native order.
    locations = torch.arange(ids.shape[1], device=ids.device) + 1
    last = torch.where(members, locations, 0).cummax(-1).values
    previous = torch.nn.functional.pad(last[..., :-1], (1, 0))
    ends = torch.nn.functional.pad(hi, (1, 0))[:, None].expand(-1, units.shape[1], -1)
    required = torch.where(previous > 0, ends.gather(-1, previous), start[..., None])
    return ((stop > start) & (ends.gather(-1, last[..., -1:]).squeeze(-1) == stop)
            & (~members | (lo[:, None] == required)).all(-1) & members.any(-1))


def named_extents(model, percepts):
    """Annotate complete, visible native forms with their MTM names."""
    from WordSurfaces import is_lexical_form
    ps, owner = model.perceptualSpace, model._concept_owner()
    ids, spans = percepts[:2]
    units = ps._forward_input.get('unit_spans')
    pool_words = ps._forward_input.get('pool_word_ids')
    if pool_words is not None:
        # Preserve the field's compact extent axis, including unnamed visible
        # readings. The MTM decision is final for this occurrence.
        live = covered_units(ids, spans, units) & ps._forward_input['pool_visible']
        width = int(live.sum(1).amax()) if len(ids) else 0
        columns = (~live).long().argsort(dim=1, stable=True)[:, :width]
        keep = live.gather(1, columns)
        selected = units.gather(1, columns[..., None].expand(-1, -1, 2))
        return torch.where(keep[..., None], selected, 0), torch.where(keep, pool_words.gather(1, columns), -1)
    words = []
    # The existing nomination interface consumes native row names. Bring
    # the whole bank (and the word-major stem's unit extents) to this boundary in one read.
    events = torch.cat((ids[..., None], spans.to(ids)), -1)
    if torch.is_tensor(units):
        rows = torch.cat((events.flatten(1), units.to(ids).flatten(1)), 1).tolist()
        n = events.shape[1] * 3
        events = [[row[i:i + 3] for i in range(0, n, 3)] for row in rows]
        intervals = [[(s, e) for s, e in zip(row[n::2], row[n + 1::2]) if e > s] for row in rows]
    else:
        intervals = ps._forward_input.get('word_intervals') or ()
        events = events.tolist() if intervals else ()
    for b, limits in enumerate(intervals):
        parts = [(pid, lo, hi) for pid, lo, hi in events[b] if pid >= 0 and lo < hi]
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
        return spans[:, :0], ids[:, :0]
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


def _stream_addresses(names, live, values):
    """Look up names in each stream, using the padded width as its missing seat."""
    width = names.shape[1]
    keys, order = torch.where(live, names, torch.iinfo(names.dtype).max).sort(dim=1, stable=True)
    at = torch.searchsorted(keys, values.contiguous())
    order = torch.nn.functional.pad(order, (0, 1), value=width)
    seats = order.gather(1, at)
    padded = torch.nn.functional.pad(names, (0, 1))
    active = torch.nn.functional.pad(live, (0, 1))
    return torch.where(active.gather(1, seats) & (padded.gather(1, seats) == values), seats, width)


def retained_rows(heads, handles, pair_mask, endpoints, names, idea_mask, referents,
                  issued, shared, durable_names, durable_tags):
    """Trace all winning heads and translate occurrence names in stream order."""
    batch, pairs = handles.shape
    ideas = names.shape[1]
    relation = pair_mask & (heads[:, :1] == RELATION) & (handles == heads[:, 1:])
    kinds, values = endpoints.unbind(-1)
    children = _stream_addresses(handles, pair_mask, values.reshape(batch, -1)).reshape(batch, pairs, 2)
    relation = torch.nn.functional.pad(relation.long(), (0, 1))
    for i in reversed(range(pairs)):
        chosen = (kinds[:, i] == RELATION) & relation[:, i:i + 1].bool()
        relation = relation.scatter_reduce(1, children[:, i], chosen.long(), reduce='amax')
    relation = relation[:, :pairs].bool()

    all_refs = torch.cat((heads[:, None], referents, endpoints.reshape(batch, -1, 2)), 1)
    tags, values = all_refs.unbind(-1)
    local = _stream_addresses(names, idea_mask, values)
    selected = torch.cat((torch.ones_like(heads[:, :1], dtype=torch.bool),
                          torch.zeros_like(idea_mask), relation.repeat_interleave(2, dim=1)), 1)
    kept = names.new_zeros(batch, ideas + 1).scatter_add(1, local, ((tags == OCCURRENCE) & selected).long()).bool()
    following = torch.where(idea_mask & (referents[..., 0] == OCCURRENCE), local[:, 1:ideas + 1], ideas)
    following = torch.nn.functional.pad(following, (0, 1), value=ideas)
    for _ in range(ideas.bit_length()):
        kept = kept | torch.zeros_like(following).scatter_add(1, following, kept.long()).bool()
        following = following.gather(1, following)
    kept = kept[:, :ideas]
    new_names = issued + kept.flatten().long().cumsum(0).reshape_as(names)
    selected = torch.cat((selected[:, :1], kept, selected[:, ideas + 1:]), 1)
    local_link = _stream_addresses(handles, relation, values)
    durable_at = values.clamp(0, len(durable_tags) - 1)
    durable_link = (durable_at == values) & (durable_tags[durable_at] != 0)
    durable_idea = _name_lookup(durable_names, values.flatten(), -1).reshape_as(values) >= 0
    available = ((tags != RELATION) | (local_link < pairs) | (shared[:, None] & durable_link))
    available &= ((tags != OCCURRENCE) | (local < ideas) | (shared[:, None] & durable_idea))
    moved = torch.nn.functional.pad(new_names, (0, 1)).gather(1, local)
    values = torch.where((tags == OCCURRENCE) & (local < ideas), moved, values)
    translated = torch.stack((tags, values), -1)
    return (kept, relation, new_names, translated[:, 1:ideas + 1],
            translated[:, ideas + 1:].reshape_as(endpoints), translated[:, 0], (~selected | available).all())


def _retention_plans(stores, heads, durable):
    from torch.nn.utils.rnn import pad_sequence
    entries, roots, shared, handles, endpoints, names, referents = [], [], [], [], [], [], []
    plans = [None] * len(stores)
    for i, (store, head) in enumerate(zip(stores, heads, strict=True)):
        if store.nDim != durable.nDim or store.episodic_capacity != durable.episodic_capacity:
            raise ValueError('incompatible STM and LTM reading rows')
        if head is None:
            continue
        entries.append(i)
        parts = store.parts
        parts._sort()
        rows = parts.relation_order[:len(parts)]
        roots.append(parts.reference(head))
        external = store.__dict__.get('_external_memory')
        shared.append(external is not None and external() is durable)
        handles.append(rows)
        endpoints.append(torch.stack((parts.ref_kinds[rows], parts.refs[rows]), -1))
        names.append(store.row_names[:len(store)])
        referents.append(store.referent[:len(store)])
    if not entries:
        return plans, 0, 0
    # -1 is not an occurrence name or a relation handle.
    def pad(columns):
        return pad_sequence(columns, batch_first=True, padding_value=-1)
    handles, endpoints, names, referents = map(pad, (handles, endpoints, names, referents))
    kept, links, new_names, ideas, pairs, roots, valid = retained_rows(
        names.new_tensor(roots), handles, handles != -1, endpoints, names, names != -1, referents,
        durable.names_issued, names.new_tensor(shared, dtype=torch.bool),
        durable.row_names[:len(durable)], durable.parts.ref_kinds[:, 0])
    if not bool(valid):
        raise ValueError('reading names an unavailable occurrence or relation')
    sizes = torch.stack((kept.sum(1), links.sum(1))).tolist()
    idea_counts, pair_counts = sizes
    seats = torch.arange(names.shape[1], device=names.device).expand_as(names)
    columns = (seats[kept].split(idea_counts), new_names[kept].split(idea_counts),
               ideas[kept].split(idea_counts), handles[links].split(pair_counts),
               pairs[links].split(pair_counts), roots.unbind())
    for i, plan in zip(entries, zip(*columns, strict=True), strict=True):
        plans[i] = plan
    return plans, sum(idea_counts), sum(pair_counts)


def _reserve_symbols(blocked, values):
    """A standalone relation allocator skips names held by symbols or episodic concepts."""
    legal = (values > 0) & (values < blocked.shape[0])
    blocked.index_fill_(0, torch.where(legal, values, 0).flatten(), True)


def translate_pairs(handles, endpoints, head, refs, tags, blocked, next_id, *, sequential,
                    reserve_symbols=True, active=None):
    """Intern the incoming topology into private copies of the LTM address arrays.

    Existing pairs keep their handles, including pairs encountered on a prior
    stream. Local children are translated before their parent. Zero is the
    reserved sink when capacity runs out. The caller checks the batch before
    publishing any payload.
    """
    n = handles.shape[0]
    seats = torch.arange(refs.shape[0], device=refs.device)
    names = handles.new_zeros(n + 1)
    translated = torch.zeros_like(endpoints)
    active = torch.ones(n, device=handles.device, dtype=torch.bool) if active is None else active
    fresh = handles.new_zeros(n, dtype=torch.bool)
    ok = handles.new_ones((), dtype=torch.bool)
    if not n:
        return names[:0], translated, fresh, head.clone(), next_id.clone(), ok
    local = _name_lookup(handles, endpoints[..., 1], n)
    root_at = _name_lookup(handles, head[1:2], n)
    def step(i, names, translated, fresh, current_refs, current_tags, occupied, next_id, ok):
        at = i.reshape(1)
        enabled = active.index_select(0, at)[0]
        kinds, original = endpoints.index_select(0, at)[0].unbind(-1)
        children = local.index_select(0, at)[0]
        values = torch.where((kinds == RELATION) & (children < n), names[children], original)
        matches = (current_tags == kinds).all(1) & (current_refs == values).all(1)
        found = matches.any()
        free = next_id if sequential else torch.where(~occupied, seats, len(seats)).amin()
        available = current_tags.index_select(0, free.clamp(0, len(seats)-1).reshape(1))[0, 0] == 0
        can_allocate = (free > 0) & (free < len(seats)) & available
        ok = ok & (~enabled | found | can_allocate)
        name = torch.where(enabled, torch.where(found, matches.long().argmax(), torch.where(can_allocate, free, 0)), 0)
        index = name.reshape(1)
        current_refs = current_refs.index_copy(0, index,
            torch.where(enabled, values[None], current_refs.index_select(0, index)))
        current_tags = current_tags.index_copy(0, index,
            torch.where(enabled, kinds[None].to(current_tags.dtype), current_tags.index_select(0, index)))
        occupied = occupied.index_fill(0, index, True)
        if not sequential and reserve_symbols:
            symbols = torch.where(kinds == SYMBOL, values, 0)
            legal = enabled & (symbols > 0) & (symbols < len(occupied))
            occupied = occupied.index_fill(0, torch.where(legal, symbols, 0), True)
        names = names.index_copy(0, at, name.reshape(1))
        translated = translated.index_copy(0, at, torch.where(enabled, torch.stack((kinds, values), -1)[None], 0))
        created = enabled & ~found
        fresh = fresh.index_copy(0, at, created.reshape(1))
        return i + 1, names, translated, fresh, current_refs, current_tags, occupied, next_id + created.long(), ok

    initial = (handles.new_zeros(()), names, translated, fresh, refs.clone(), tags.clone(),
               blocked.clone(), next_id, ok)
    if torch.compiler.is_compiling():
        def condition(i, names, translated, fresh, refs, tags, blocked, next_id, ok):
            return i < n
        result = torch.while_loop(condition, step, initial)
    else:
        result = initial
        for _ in range(n):
            result = step(*result)
    _, names, translated, fresh, written_refs, written_tags, occupied, next_id, ok = result
    refs.copy_(written_refs)
    tags.copy_(written_tags)
    blocked.copy_(occupied)
    root_value = torch.where((head[0] == RELATION) & (root_at[0] < n), names[root_at][0], head[1])
    return names[:n], translated, fresh, torch.stack((head[0], root_value)), next_id, ok


def translate_batch(handles, endpoints, heads, active, refs, tags, blocked, next_id, *, sequential):
    """Intern all streams in row order, with a private namespace for each trial.

    Equal temporary handles from different streams never alias. Once their
    children have durable names, equal pairs share the ordinary LTM handle.
    A captured loop keeps sequential allocation order without unrolling a
    larger graph when the batch grows.
    """
    batch, width = handles.shape
    if not width:
        return handles.clone(), endpoints.clone(), active.clone(), heads.clone(), next_id, blocked.new_ones(())
    references = torch.cat((endpoints.reshape(batch, -1, 2), heads[:, None]), 1)
    kinds, values = references.unbind(-1)
    local = _stream_addresses(handles, active, values)
    own = (kinds == RELATION) & (local < width)
    local = local.clamp_max(width - 1) + torch.arange(batch, device=handles.device)[:, None] * width
    translated = torch.stack((kinds, torch.where(own, -local-1, values)), -1)
    temporary = -torch.arange(batch*width, device=handles.device)-1
    names, pairs, fresh, _, next_id, ok = translate_pairs(temporary,
        translated[:, :-1].reshape(-1, 2, 2), heads.new_zeros(2), refs, tags, blocked, next_id,
        sequential=sequential, active=active.flatten())
    roots = torch.stack((heads[:, 0], torch.where(own[:, -1], names[local[:, -1]], heads[:, 1])), -1)
    return names.reshape_as(handles), pairs.reshape_as(endpoints), fresh.reshape_as(active), roots, next_id, ok


def _bind_allocated_handles(parts, plans):
    """Resolve provisional seats when a store supplies its own handle allocator."""
    allocated, used = {}, set()
    for plan in plans:
        if plan is None:
            continue
        for provisional, *_, fresh in plan[0]:
            if fresh:
                handle = int(parts.allocate())
                if not 0 < handle < parts.capacity:
                    raise OverflowError('allocated relation handle exceeds capacity')
                if handle in used or parts.contains(handle):
                    raise ValueError('a live relation cannot be reassigned')
                allocated[provisional] = handle
                used.add(handle)
    def endpoint(kind, name):
        return allocated.get(name, name) if kind == RELATION else name
    result = []
    for plan in plans:
        if plan is None:
            result.append(None)
            continue
        links, (kind, root) = plan
        links = [(allocated.get(name, name), ka, endpoint(ka, a), kb, endpoint(kb, b), fresh)
                 for name, ka, a, kb, b, fresh in links]
        result.append((links, (kind, endpoint(kind, root))))
    return result


@torch.no_grad()
def _publish_sequential_pairs(parts, allocator, refs, tags, next_id, names, fresh, evidence):
    """Write the translated topology and graded observations of the whole batch."""
    from Memory import NULL_CONCEPT
    from MemoryActivity import accessed
    first, stop = allocator.next_id, int(next_id)
    if not bool((evidence.isfinite() & (evidence >= 0) & (evidence <= 1)).all()):
        raise ValueError('invalid retained relation evidence or activity')
    if bool(((tags[first:stop] == SYMBOL) & (refs[first:stop] == NULL_CONCEPT)).any()):
        raise ValueError('a null-concept cannot be a relation endpoint')
    # Relation handles in this protocol are global concept names. Translation
    # has reserved one contiguous range, in stream and dependency order.
    # Existing pairs can be encountered repeatedly, including earlier streams'
    # writes. Retain the maximum observation on each independent trust lane.
    right_symbol = tags[names, 1] == SYMBOL
    classes = tuple(getattr(parts, '_class_rows', ()))
    symbolic = tags[names, 0] == SYMBOL
    if classes:
        symbolic |= torch.isin(refs[names, 1], names.new_tensor(classes))
    changed = fresh | (parts.truth_plus[names] < evidence[:, 0]) | (parts.truth_minus[names] < evidence[:, 1])
    invalidate = bool((right_symbol & symbolic & changed).any())
    if first != stop:
        parts._sort()
        start = len(parts)
        for _ in range(first, stop):
            allocator.new_concept()
        parts.refs[first:stop].copy_(refs[first:stop])
        parts.ref_kinds[first:stop].copy_(tags[first:stop])
        handles = parts.__dict__.get('_sentence_handles')
        if handles is not None:
            handles.update(range(first, stop))
        seats = torch.arange(first, stop, device=names.device)
        parts.relation_order[start:start + len(seats)].copy_(seats)
        parts.activity[first:stop].copy_(evidence[fresh, 2])
        parts._write_count(start + len(seats))
        parts._order_revision = parts._topology_revision
    parts.truth_plus.scatter_reduce_(0, names, evidence[:, 0], reduce='amax')
    parts.truth_minus.scatter_reduce_(0, names, evidence[:, 1], reduce='amax')
    parts._symbol_epoch += int(invalidate)
    # Preserve each repeated access in the current trial's activity journal.
    for name in names[~fresh].tolist():
        accessed(parts, name)


def commit(stores, heads, durable):
    """Retain just the reachable idea payload and translate its ordered pairs."""
    plans, n_ideas, n_pairs = _retention_plans(stores, heads, durable)
    if len(durable) + n_ideas > durable.capacity:
        raise OverflowError('reading exceeds remaining long-term idea capacity')
    parts = durable.parts
    planned = []
    sequential = False
    if n_pairs:
        refs, tags = parts.refs.clone(), parts.ref_kinds.clone()
        blocked = tags[:, 0] != 0
        blocked[0] = True
        allocator = getattr(parts.allocate, '__self__', None)
        sequential = allocator is not None and parts.allocate == getattr(allocator, '_allocate_part_symbol', None)
        if parts.allocate is None:
            _reserve_symbols(blocked, torch.where(tags == SYMBOL, refs, 0))
            _reserve_symbols(blocked, durable.episodic_concepts[:len(durable)])
            if parts.handle_reserved is not None:
                blocked |= blocked.new_tensor([parts.handle_reserved(row) for row in range(parts.capacity)])
        next_id = refs.new_tensor(allocator.next_id if sequential else 0)
        valid = blocked.new_ones(())
    if n_pairs and sequential:
        from torch.nn.functional import pad
        width = max(0 if plan is None else len(plan[3]) for plan in plans)
        counts = refs.new_tensor([0 if plan is None else len(plan[3]) for plan in plans])
        handles = torch.stack([refs.new_full((width,), -1) if plan is None else
            pad(plan[3], (0, width-len(plan[3])), value=-1) for plan in plans])
        endpoints = torch.stack([refs.new_zeros(width, 2, 2) if plan is None else
            pad(plan[4], (0, 0, 0, 0, 0, width-len(plan[3]))) for plan in plans])
        heads = torch.stack([refs.new_zeros(2) if plan is None else plan[5] for plan in plans])
        live = torch.arange(width, device=refs.device)[None] < counts[:, None]
        assigned, endpoints, fresh, roots, next_id, valid = translate_batch(
            handles, endpoints, heads, live, refs, tags, blocked, next_id, sequential=True)
        if not bool(valid):
            raise OverflowError('reading exceeds remaining long-term relation capacity')
        evidence = torch.cat([torch.stack((store.parts.truth_plus[plan[3]],
            store.parts.truth_minus[plan[3]], store.parts.activity[plan[3]]), 1)
            for store, plan in zip(stores, plans) if plan is not None])
        _publish_sequential_pairs(parts, allocator, refs, tags, next_id,
                                  assigned[live], fresh[live], evidence)
        written_complete(durable, durable.retain_idea_batch(
            [(store, *plan[:3]) for store, plan in zip(stores, plans) if plan is not None], written=True))
        return tuple(None if plan is None else tuple(root) for plan, root in zip(plans, roots.tolist()))
    else:
        for store, plan in zip(stores, plans, strict=True):
            if plan is None:
                planned.append(None)
                continue
            rows, names, referents, handles, endpoints, root = plan
            links = []
            if n_pairs and parts.allocate is None:
                _reserve_symbols(blocked, store.episodic_concepts[rows])
            if len(handles):
                assigned, endpoints, fresh, root, next_id, ok = translate_pairs(
                    handles, endpoints, root, refs, tags, blocked, next_id, sequential=sequential,
                    reserve_symbols=parts.allocate is None)
                valid = valid & ok
                links = torch.cat((assigned[:, None], endpoints.flatten(1), fresh[:, None]), 1).tolist()
            planned.append((links, tuple(root.tolist())))
    if n_pairs and not bool(valid):
        raise OverflowError('reading exceeds remaining long-term relation capacity')
    if n_pairs and parts.allocate is not None and not sequential:
        planned = _bind_allocated_handles(parts, planned)

    retained = durable.retain_idea_batch(
        [(store, *plan[:3]) for store, plan in zip(stores, plans) if plan is not None], written=True)
    committed = []
    for store, plan, allocation in zip(stores, plans, planned, strict=True):
        if allocation is None:
            committed.append(None)
            continue
        handles = plan[3]
        links, root = allocation
        evidence = torch.stack((store.parts.truth_plus[handles], store.parts.truth_minus[handles],
                                store.parts.activity[handles]), 1).tolist()
        for (name, ka, a, kb, b, fresh), (positive, negative, activity) in zip(links, evidence, strict=True):
            parts.add((ka, a), (kb, b), symbol=name, truth_plus=positive, truth_minus=negative,
                      surprise=activity)
        committed.append(root)
    # Offered to consolidation with their relations written, as in the
    # sequential branch.
    written_complete(durable, retained)
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
    from Attention import occurrence_probabilities, select_occurrence
    from LookForecast import OccurrenceCredit
    from SentenceLooks import attend, perceive, request_occurrences
    from BasicPercepts import extent_members
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
    if not bank.span.shape[1]:
        return tuple(heads)
    current = owner._cs_open_field
    def located(field, index=None):
        return at_rows(inventory(owner, field), field, reader.rows, B, index)
    opened = located(current)
    memory, own = torch.zeros_like(opened), bank.activity.clone()
    left = bank.activity[..., 0] > 0
    from WordSurfaces import grammar_leaves
    surface = model._staged_concepts_in
    surface = surface[:, 0] if surface.ndim == 3 else surface
    content = grammar_leaves(surface, bank.span.clamp(0, surface.shape[-1]), left)
    # A named whole can annotate its exact occurrence. A smaller fixation
    # at the same start keeps its own content and native address.
    exact = (bank.span[:, :, None] == bank.where[:, None]).all(-1) & left[:, :, None]
    names_at = (torch.where(exact, concepts[:, None], -1).amax(-1) if concepts.shape[1]
                else torch.full_like(bank.time, -1))
    observation = model.conceptualSpace.stm.model_state
    observation.begin_looks(bank.span, left.any(-1))
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
        scores, activity = request_occurrences(model, bank, request, own, left & active[:, None])
        probabilities = occurrence_probabilities(scores)
        index = select_occurrence(probabilities, sample=model.training)
        index = torch.where(active, index, -1)
        observation.observe(bank.span, index)
        found = index >= 0
        if not found.any():
            break
        if credit is not None:
            credit.observe(anticipated, torch.where(found[:, None], observation.last_where, -1))
        batch = torch.arange(B, device=index.device)
        field = perceive(model, activity, bank, fixation=index)
        episode = select(field, stores[0].episodic_capacity, found, unkept=excluded, uncertain=ambiguous)
        # The row writers still take native names. Gather the selected
        # metadata once per look before writing each stream's own store.
        seats = index.clamp_min(0)
        names = names_at[batch, seats].tolist()
        trust = scores[batch, seats].detach().clamp(0, 1).tolist()
        # A separator is a real fixation and updates the next request's
        # location. The existing grammar leaf rule excludes it from the head.
        leaves = torch.where(content[batch, seats], index, -1)
        for b, (w, cid) in enumerate(zip(leaves.tolist(), names, strict=True)):
            if w < 0:
                continue
            store = stores[b]
            code = bank.table[b, w, :bank.width, 0]
            if cid >= 0:
                row = owner._concept_row_of(cid)
                code = owner.similarity_codebook.lookup_rows(index.new_tensor([row]), stream=b)[0]
            point = store._fit(code).clone()
            point[store.when_start-WHERE_WIRES:store.when_start] = field.wires[b, 0]
            point[store.when_start:store.when_start+4] = bank.table[b, w, -4:, 0]
            row = store.append_idea(point, referent=None if cid < 0 else ('sym', cid),
                trust=trust[b], entry=(episode[0][b], episode[1][b]))
            heads[b] = store.parts.append_occurrence(heads[b], store.row_name(row))
        seen, _ = extent_members(bank.span, field.where)
        left = left & ~seen[:, 0]
        memory = torch.where(found[:, None, None], spread_once(owner, located(field)), memory)
        own = torch.where(left[..., None], own, 0.)
    if credit is not None:
        credit.register(model.errors)
    durable = model.symbolSpace.ltm_store
    committed = owner.mtm.consolidate_heads(stores, heads, durable)
    from Occurrence import operand_name
    observation.publish_heads(model.present(), [
        -1 if head is None else operand_name(head[1]) if head[0] == OCCURRENCE else head[1]
        for head in committed])
    for b, head in enumerate(committed):
        if head is not None:
            owner.mtm.close(durable, stream=b)
    return committed
