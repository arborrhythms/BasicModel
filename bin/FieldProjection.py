"""Content projection and definition closure over the known concept inventory."""
import torch
from FieldLogic import reduce
from ConceptualMatch import match
from Understanding import ConceptualField


def reachable_rows(active, rows, standing, sources, targets, rounds, size, delayed=None):
    """Propagate each stream's occupied rows through its definition edges.

    ``delayed`` (the unit delay, item 1.4) is ``(targets, back)``: the rows
    holding the entries on concepts one step back ``[D]``, and whether each
    entry's concept was active one step back in each stream ``[D, B]``. A
    row holding such an entry on a concept that was active is reached, as
    a row is reached from what is active now, and the rounds carry on from
    it."""
    compiling = torch.compiler.is_compiling()
    counts = torch.zeros(size + (1 if compiling else 0), active.shape[1], dtype=torch.int32, device=active.device)
    reached = counts.index_add(0, rows, active.int()).gt(0)
    reached = reached.index_fill(0, standing, True)
    if delayed is not None:
        reached = reached | counts.index_add(0, delayed[0], delayed[1].int()).gt(0)
    for _ in range(rounds):
        parents = counts.index_add(0, targets, reached.index_select(0, sources).int())
        reached = reached | parents.gt(0)
    return reached[:size] if compiling else reached


def seated_names(allocator, size, like):
    """Concept name and retirement status at every physical field seat."""
    cached = getattr(allocator, '_field_seated_names', None)
    if torch.compiler.is_compiling() and cached is not None:
        return cached[1:]
    stamp = (size, like.device, tuple((id(layer._tensor_row_keys), len(layer._tensor_row_keys))
                                     for layer in allocator._layers.values()), frozenset(allocator.retired))
    if cached is None or cached[0] != stamp:
        # Later layers have precedence, as in concept_id_at_row.
        seats = {}
        for layer in allocator._layers.values():
            seats.update((row, key[1] if isinstance(key, tuple) else key)
                         for row, key in layer._tensor_row_keys.items() if 0 <= row < size)
        names = like.new_full((size + 1,), -1, dtype=torch.long)
        live = torch.ones(size, dtype=torch.bool, device=like.device)
        if seats:
            rows, ids = like.new_tensor(list(seats.items()), dtype=torch.long).T
            names.index_copy_(0, rows, ids)
            retired = [row for row, name in seats.items() if name in allocator.retired]
            if retired:
                live.index_fill_(0, rows.new_tensor(retired), False)
        cached = allocator._field_seated_names = stamp, names, live
    return cached[1:]


def field_seats(reached, blocks, minimum, *, live_caps=None):
    """Pack each page's reachable seats in physical order within each block."""
    batch = reached.shape[1]
    offset = 0
    groups = []
    for width in blocks:
        groups.append(reached[offset:offset + width])
        offset += width
    if live_caps is not None:
        caps = tuple(live_caps)
        identical = False
    else:
        counts = [group.long().sum(0).amax() for group in groups] if batch else [
            reached.new_zeros((), dtype=torch.long) for _ in groups]
        counts[0] = counts[0].clamp_min(minimum)
        shared = (reached[:offset] == reached[:offset, :1]).all() if batch else reached.new_tensor(False)
        *caps, identical = torch.stack([*counts, shared.long()]).tolist()
    if not batch:
        return reached.new_empty(0, dtype=torch.long), tuple(caps)
    rows, offset = [], 0
    for group, cap in zip(groups, caps, strict=True):
        if cap:
            positions = torch.where(group, group.long().cumsum(0) - 1, cap)
            source = torch.arange(offset, offset + len(group), device=group.device)[:, None].expand_as(positions)
            packed = positions.new_full((cap + 1, batch), -1)
            packed.scatter_reduce_(0, positions, source, 'amax')
            rows.append(packed[:cap])
        offset += len(group)
    table = torch.cat(rows) if rows else reached.new_empty((0, batch), dtype=torch.long)
    return (table[:, 0] if identical else table), tuple(caps)


def reached_entries(reached, topology):
    """The stored edges and symbol columns reached by each page, in edge order."""
    _, targets, columns, members, _ = topology
    selected = reached.index_select(0, targets).T.reshape(-1)
    flat = torch.arange(len(selected), device=reached.device)[selected]
    width = max(1, len(targets))
    pages, edges = flat // width, flat % width
    symbolic = torch.arange(len(edges), device=reached.device)[columns.index_select(0, edges) < len(members)]
    return pages, edges, symbolic


def rung_entries(store, addresses, like):
    """Index sparse entries while the field writer assigns its page seats."""
    pages, entries, symbolic = [], [], []
    # The inventory view elides out-of-range rows before numbering entries.
    physical = [(row, column) for row, column in zip(store._rows, store._cols, strict=True)
                if 0 <= row < store.nOutput]
    if store.__dict__.get('_delayed_columns'):
        # It leaves out the entries on concepts one step back too: they are read apart.
        physical = [(row, column) for row, column in physical if column <= store._room]
    for page, field_rows in enumerate(addresses):
        seated = set(field_rows)
        for edge, (row, column) in enumerate(physical):
            if row in seated:
                if column != store.bias_column():
                    symbolic.append(len(entries))
                pages.append(page)
                entries.append(edge)
    columns = like.new_tensor(pages + entries + symbolic)
    return columns.split((len(pages), len(entries), len(symbolic)))


def seated_rung(field, addresses, weights, topology, inventory_size, start, end, entry_plan=None, delayed=None):
    """Read a Sigma rung for all pages from their own seats in the same inventory.

    ``delayed`` (the unit delay, item 1.4) is ``(rows, positions, lanes)``:
    the entries on concepts one step back, each of the row ``rows`` ``[D]``
    with its number at ``positions`` among ``weights``, reading the kept
    lanes ``[D, B, 2]`` through the Not as any other entry reads a symbol's,
    in the same Sigma."""
    from Layers import NotLayer3, _EntryRead, signed_lanes
    edges, targets, columns, members, complete = topology
    size, batch, places, _ = field.shape
    width = end - start
    if addresses.ndim == 1:
        addresses = addresses[:, None].expand(-1, batch)
    # In a repeated row the final seat is the one visible to the bound store.
    ids = addresses.T
    seats = torch.arange(size, device=ids.device).expand_as(ids)
    inverse = ids.new_full((batch, inventory_size + 1), -1).scatter_reduce(
        1, torch.where((ids >= 0) & (ids < inventory_size), ids, inventory_size), seats, 'amax')
    destinations = inverse.index_select(1, targets)
    if torch.compiler.is_compiling():
        if width <= 0 or targets.shape[0] == 0:
            return field.new_zeros(max(width, 0), batch, places, 2)
        E = targets.shape[0]
        seat = destinations
        in_rung = (seat >= start) & (seat < end)
        is_sym = columns < members.shape[0]
        bias_lanes = field.new_zeros(batch, E, places, 2)
        bias_lanes[..., 0] = 1.
        if members.shape[0] > 0:
            parents = columns.clamp(0, members.shape[0] - 1)
            source = members.index_select(0, parents)
            page_idx = torch.arange(batch, device=field.device)[:, None, None]
            located = inverse[:, source.clamp_min(0)]
            sym_active = (complete.index_select(0, parents)[None]
                          & ((source[None] < 0) | (located >= 0)).all(-1))
            operands = field[located.clamp_min(0), page_idx]
            operands = torch.where(source[None, :, :, None, None] >= 0, operands, 1.)
            sym_lanes = operands.prod(2) * sym_active[..., None, None]
            lanes = torch.where(is_sym[None, :, None, None], sym_lanes, bias_lanes)
            valid = in_rung & (~is_sym[None] | sym_active)
        else:
            lanes = bias_lanes
            valid = in_rung
        value = weights.to(field)[edges][None].expand(batch, E) * valid.to(field)
        terms = _EntryRead.apply(
            value.reshape(batch * E, 1).expand(-1, places),
            NotLayer3.options(lanes.reshape(batch * E, places, 2)))
        pages_grid = torch.arange(batch, device=field.device)[:, None].expand(batch, E)
        target = ((seat - start).clamp(0, width - 1) * batch + pages_grid).reshape(-1)
        total = field.new_zeros(width * batch, places).index_add(0, target, terms)
        total = _delayed_terms(total, delayed, field, inverse, weights, start, width)
        return signed_lanes(total.tanh()).reshape(width, batch, places, 2)
    if entry_plan is None:
        flat = (destinations >= 0).reshape(-1).nonzero().flatten()
        cols = destinations.shape[1]
        pages = torch.div(flat, cols, rounding_mode='floor')
        entries = flat % cols
        symbolic = (columns.index_select(0, entries) < len(members)).nonzero().flatten()
    else:
        pages, entries, symbolic = entry_plan
    if not len(entries) or width <= 0:
        return field.new_zeros(max(width, 0), batch, places, 2)
    slots = columns.index_select(0, entries)
    # A page with only standing biases has no gradient into its input field.
    # Preserve that distinction, including when the optimizer has momentum.
    lanes = field.new_zeros(len(entries), places, 2)
    lanes[..., 0] = 1.
    valid = torch.ones_like(entries, dtype=torch.bool)
    if len(symbolic):
        parents = slots.index_select(0, symbolic)
        source = members.index_select(0, parents)
        page = pages.index_select(0, symbolic)[:, None]
        located = inverse[page, source.clamp_min(0)]
        active = complete.index_select(0, parents) & ((source < 0) | (located >= 0)).all(1)
        operands = field[located.clamp_min(0), page]
        operands = torch.where((source >= 0)[..., None, None], operands, 1.)
        lanes = lanes.index_copy(0, symbolic, operands.prod(1) * active[:, None, None])
        valid = valid.index_copy(0, symbolic, active)
    seat = destinations[pages, entries]
    valid = valid & (seat >= start) & (seat < end)
    value = weights.to(field)[edges[entries]] * valid.to(field)
    terms = _EntryRead.apply(value[:, None].expand(-1, places), NotLayer3.options(lanes))
    target = (seat - start).clamp(0, width - 1) * batch + pages
    total = field.new_zeros(width * batch, places).index_add(0, target, terms)
    total = _delayed_terms(total, delayed, field, inverse, weights, start, width)
    return signed_lanes(total.tanh()).reshape(width, batch, places, 2)


def _delayed_terms(total, delayed, field, inverse, weights, start, width):
    """``total`` ``[width * B, L]`` with the terms of the entries on
    concepts one step back added, each on its row's seat of the rung."""
    if delayed is None or not len(delayed[0]) or width <= 0:
        return total
    from Layers import NotLayer3, _EntryRead
    rows, positions, lanes = delayed
    batch, places = field.shape[1], field.shape[2]
    D = len(rows)
    seat = inverse.index_select(1, rows.to(inverse.device))                       # [B, D]
    valid = (seat >= start) & (seat < start + width)
    value = weights.to(field)[positions.to(weights.device)][None].expand(batch, D) * valid.to(field)
    options = NotLayer3.options(lanes.to(field).transpose(0, 1)[:, :, None].expand(batch, D, places, 2))
    terms = _EntryRead.apply(value.reshape(batch * D, 1).expand(-1, places), options.reshape(batch * D, places, 2))
    pages = torch.arange(batch, device=field.device)[:, None]
    target = ((seat - start).clamp(0, width - 1) * batch + pages).reshape(-1)
    return total.index_add(0, target, terms)


def inventory(owner):
    store = owner._concept_allocator.layer(0)
    return sorted(row for row in store._tensor_row_keys
                  if owner.concept_id_at_row(row) is not None)


def close(owner, rows, evidence, percepts=None):
    from CandidateGrowth import read_streams
    return read_streams(owner, lambda: _close(owner, rows, evidence, percepts), axis=1)


def _close(owner, rows, evidence, percepts=None):
    """Read each definition once, in dependency order: each kind's entries,
    each on its symbol's value or not-value (the Not), summed by the
    store's Sigma and stored as one signed value on two lanes. Where the
    basic percepts' lanes are given (``percepts``, ``(lanes [P, B, E, 2],
    rows [P])``), a definition's entry on a basic percept's row
    (``store.percept_entries``) reads that row's value or not-value from the
    value (its for lane; nothing is read against it) in the same Sigma."""
    from Layers import NotLayer3, sigma_fold, signed_lanes
    from CandidateGrowth import visible_store
    store = visible_store(owner._concept_allocator.layer(0))
    positions = {int(row): i for i, row in enumerate(rows)}
    values = store.values
    definitions = {row: [] for row in positions}
    # The entries on concepts one step back (item 1.4): ``(source, index)``.
    delayed = {row: [] for row in positions}
    behind = store.delayed_of_column if store.__dict__.get('_delayed_columns') else None
    entries = [(row, column, index) for (row, column), index in store._index.items() if row in positions]
    if values is not None and entries:
        indices = torch.tensor([entry[2] for entry in entries], device=values.device)
        active = values.detach().index_select(0, indices).ne(0).tolist()
        for (row, column, index), enabled in zip(entries, active, strict=True):
            if enabled and behind is not None and behind(column) is not None:
                delayed[row].append((behind(column), index))
            elif enabled:
                sources = None if store.thing_of_column(column) is None else store.column_rows(column)
                definitions[row].append((sources, index))
    # Each basic percept's place among the lanes given.
    at = {} if percepts is None else {int(p): i for i, p in enumerate(percepts[1])}
    done, pending = {}, set(positions)
    while pending:
        ready = [row for row in sorted(pending) if all(
            source not in positions or source in done
            for sources, _ in definitions[row] for source in (sources or ()))]
        if not ready:
            raise ValueError('cyclic concept field definitions')
        for row in ready:
            value = evidence[positions[row]]
            edges = definitions[row]
            bound = [(at[percept], weight) for (target, percept), weight in store.percept_entries.items()
                     if target == row and weight and percept in at]
            if edges or bound or delayed[row]:
                operands = []
                for sources, _ in edges:
                    if sources is None:
                        item = torch.zeros_like(value)
                        item[..., 0] = 1.
                    else:
                        # A symbol is the product of the concepts it holds.
                        held = [done.get(source, torch.zeros_like(value)) for source in sources]
                        item = torch.stack(held).prod(0) if held else torch.zeros_like(value)
                    operands.append(item)
                if delayed[row]:
                    # A concept one step back: its kept lanes in each stream.
                    sources = torch.tensor([source for source, _ in delayed[row]], device=value.device)
                    kept = owner.delayed_concepts(sources, value.shape[0], value.device).to(value)
                    operands.extend(kept[:, :, None].expand(-1, -1, value.shape[1], -1))
                    edges = edges + [(None, index) for _, index in delayed[row]]
                options = NotLayer3.options(torch.stack(operands)) if operands else value.new_zeros(0, *value.shape)
                entry = values.index_select(0, torch.tensor([index for _, index in edges], dtype=torch.long,
                                                            device=values.device)).to(options)
                if bound:
                    # A basic percept's row: its options from the Not of order zero.
                    lanes = percepts[0].index_select(0, torch.tensor([i for i, _ in bound], device=percepts[0].device))
                    options = torch.cat((options, NotLayer3.options(lanes.to(options))))
                    entry = torch.cat((entry, entry.new_tensor([float(w) for _, w in bound])))
                read = sigma_fold(entry, torch.zeros(len(entry), dtype=torch.long, device=options.device),
                                              torch.arange(len(entry), device=options.device), options, 1)[0]
                # A row without direct evidence takes its definition. Direct
                # perceptual evidence and a definition are independent sources.
                value = torch.maximum(value, signed_lanes(read))
            done[row] = value
            pending.remove(row)
    return torch.stack([done[int(row)] for row in rows]) if rows else evidence


def close_bracket(owner, field, bracket):
    """Add the definition read over the union of a bracket's readings.

    Located readings stay intact. The derived reading belongs to the
    containing bracket; a definition does not move its members' locations.
    """
    from dataclasses import replace
    if field.concept_ids.ndim != 1:
        raise ValueError('bracket closure requires shared concept identities')
    bracket = torch.as_tensor(bracket, device=field.where.device, dtype=field.where.dtype)
    if bracket.shape != (field.evidence.shape[1], 2):
        raise ValueError('a bracket has two endpoints for each stream')
    inside = ((field.where[..., 0] >= bracket[:, None, 0]) &
              (field.where[..., 1] <= bracket[:, None, 1]) &
              (field.where[..., 1] > field.where[..., 0]))
    joined = reduce(field.evidence, operation='union', dim=2, mask=inside[None])
    rows = [owner._concept_row_of(int(cid)) for cid in field.concept_ids]
    folded = close(owner, rows, joined[:, :, None])
    start = torch.where(inside, field.when[..., 0], torch.iinfo(field.when.dtype).max).amin(1)
    end = torch.where(inside, field.when[..., 1], torch.iinfo(field.when.dtype).min).amax(1)
    when = torch.where(inside.any(1)[:, None], torch.stack((start, end), -1), 0)
    return replace(field, evidence=torch.cat((field.evidence, folded), 2),
        where=torch.cat((field.where, bracket[:, None]), 1),
        when=torch.cat((field.when, when[:, None]), 1))


def stimulate(owner, point, *, order=0, where=None, when=None):
    """Candidate coverage on each lane, then upward definition closure."""
    if point.ndim == 2:
        point = point[:, None]
    if point.ndim != 3:
        raise ValueError('stimulate expects batch by reading by code')
    rows = inventory(owner)
    requested = torch.as_tensor(order, device=point.device, dtype=torch.long)
    native = requested.new_tensor([owner._row_order(row) for row in rows])
    # A point is compared with the codes of the orders it is read at only.
    compared = torch.isin(native, requested.flatten().unique()).nonzero().flatten()
    direct = point.new_zeros(len(rows), *point.shape[:2], 2)
    if len(compared):
        codes = owner.similarity_codebook.lookup_rows([rows[i] for i in compared.tolist()]).to(point)
        positive, _ = match(point[None], codes[:, None, None],
                            codebook=owner.similarity_codebook, lane=0)
        negative, _ = match(point[None], codes[:, None, None],
                            codebook=owner.similarity_codebook, lane=1)
        direct[compared] = torch.stack((positive, negative), -1).to(direct)
    # Percept collections share the native boundary reader before closure.
    if hasattr(owner, 'basic_percepts'):
        from PerceptCollections import read
        collections = read(owner, point)
        if collections:
            direct = torch.stack([collections.get(row, direct[i]) for i, row in enumerate(rows)])
    direct = direct * (native[:, None, None] == requested)[..., None]
    evidence = close(owner, rows, direct)
    B, E = point.shape[:2]
    if where is None:
        where = torch.zeros(B, E, 2, device=point.device, dtype=torch.long)
    if when is None:
        when = torch.zeros_like(where)
    ids = torch.tensor([owner.concept_id_at_row(row) for row in rows], device=point.device, dtype=torch.long)
    return ConceptualField(ids, evidence, where, when, None)


def orders(owner, field, *, stream=None):
    """Orders represented by evidence, read from their allocated blocks."""
    ids = field.concept_ids
    if ids.ndim == 1:
        evidence = field.evidence if stream is None else field.evidence[:, stream]
        active = evidence.flatten(1).ne(0).any(-1)
    else:
        evidence = field.evidence.amax((-1, -2)).ne(0)
        ids, active = ((ids.flatten(), evidence.flatten()) if stream is None else
                       (ids[:, stream], evidence[:, stream]))
    return tuple(sorted({owner._row_order(row) for cid, live in zip(ids, active)
        if bool(live) and int(cid) >= 0 and (row := owner._concept_row_of(int(cid))) is not None}))


def key(owner, field, *, order):
    """A centroid at the requested order; the complete field stays intact."""
    ids = field.concept_ids
    if ids.ndim != 1:
        raise ValueError('field key requires shared concept identities')
    rows = [owner._concept_row_of(int(cid)) if int(cid) >= 0 else None for cid in ids]
    valid = torch.tensor([row is not None and owner._row_order(row) == order for row in rows], device=ids.device)
    codes = owner.similarity_codebook.lookup_rows([0 if row is None else row for row in rows]).to(field.evidence)
    weights = field.evidence.amax(-1) * valid[:, None, None]
    mass = weights.sum(0)[...,None]
    value = torch.einsum('cbe,cd->bed', weights, codes)
    centroid = value / torch.where(mass > 0, mass, 1.)
    if bool(valid.any()):
        # Matmul and sum accumulate in different orders. Roundoff must not
        # take a convex combination beyond its input coordinates (notably
        # the exact membership rails consumed by the identity chart).
        centroid = centroid.clamp(codes[valid].amin(0), codes[valid].amax(0))
    return torch.where(mass > 0, centroid, 0.)
