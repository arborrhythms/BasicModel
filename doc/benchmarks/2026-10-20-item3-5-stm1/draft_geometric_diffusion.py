"""Read Part and conceptual overlap into one detached diffusion budget.

Only the priming surface is written. The overlap graph is temporary and sparse;
its candidates come from the current field, recent STM addresses and Part.
"""
from collections import deque
from itertools import islice

import torch


def part_edges(owner):
    """Undirected native Part pairs, mapped to existing bank seats only."""
    allocator = getattr(owner, '_concept_allocator', None)
    parts = None if allocator is None else allocator.parts
    if parts is None:
        return None
    pairs = parts.refs[parts.indices()]
    if not pairs.numel():
        return None
    seats = {symbol: owner._csw_row_of(symbol) for symbol in pairs.unique().tolist()}
    rows = [(seats[a], seats[b]) for a, b in pairs.tolist()
            if seats[a] is not None and seats[b] is not None and seats[a] != seats[b]]
    if not rows:
        return None
    left, right = torch.tensor(rows, device=pairs.device, dtype=torch.long).T
    return (torch.cat((left, right)), torch.cat((right, left)),
            torch.ones(2 * len(left), device=pairs.device))


def _recent_symbols(owner, b):
    model = getattr(owner, '_model', None)
    symbol = getattr(model, 'symbolSpace', None)
    expectation = getattr(symbol, 'expectation', None)
    if expectation is None or b >= len(expectation._inter_context):
        return ()
    store = owner._closed_clause_store()
    symbols = []
    for frame in expectation.situation_references(b):
        symbols.extend(frame.symbol_ids)
        symbols.extend(identity for _, identity, _ in frame.activations)
        location = store.location(frame.row_id)
        if location is not None:
            meaning = store.row(location)['meaning']
            if meaning is not None:
                symbols.extend(ref[1] for ref in meaning.role_refs
                               if ref is not None and ref[0] == 'sym')
    return tuple(dict.fromkeys(symbols))


@torch.no_grad()
def overlap_edges(owner, surface, active, field):
    """Bounded meet reads; return sparse (batch, source, destination, weight).

    At most one STM capacity of energetic sources and positive-overlap
    neighbours per source. At most four capacities of native Part nodes
    are visited per stream; no inventory-wide code search is performed.
    """
    from IdentityRows import coordinates, meet_ratios
    cb = getattr(owner, 'similarity_codebook', None)
    allocator = getattr(owner, '_concept_allocator', None)
    empty = torch.empty(0, device=surface.device, dtype=torch.long)
    result = (empty, empty, empty, surface.new_empty(0))
    audits = []
    if cb is None or allocator is None:
        return result, audits
    capacity = max(1, int(getattr(getattr(owner, 'stm', None), 'capacity', 8)))
    parts, vocabulary = allocator.parts, surface.shape[1]
    batches, sources, targets, weights = [], [], [], []
    for b in active.nonzero().flatten().tolist():
        energy, hot = (surface[b] - 1).abs().topk(min(capacity, vocabulary))
        hot = hot[energy > 0]
        if not len(hot):
            continue
        current = [r for r in field[b].tolist() if 0 <= r < vocabulary]
        # The field is already bounded by perception. Its complete contents
        # remain eligible, even when the input contains more than eight words.
        seats = set(current) | set(hot.tolist())
        seeds = [symbol for row in sorted(seats)
                 if (symbol := owner.concept_id_at_row(row)) is not None]
        seeds.extend(_recent_symbols(owner, b))
        pending, visited = deque(dict.fromkeys(seeds)), set()
        truncated = False
        while pending and len(visited) < 4 * capacity:
            symbol = pending.popleft()
            if symbol in visited:
                continue
            visited.add(symbol)
            row = owner._csw_row_of(symbol)
            if row is not None and 0 <= row < vocabulary:
                seats.add(row)
            # Cap frontier growth as well as record reads, including hubs.
            room = max(0, 4 * capacity - len(visited) - len(pending))
            nearby = list(islice((s for s in parts.neighbours(symbol)
                                  if s not in visited and s not in pending), room + 1))
            truncated |= len(nearby) > room
            pending.extend(nearby[:room])
        candidates = sorted(r for r in seats if owner.concept_id_at_row(r) is not None)
        if not candidates:
            continue
        indices = torch.tensor(candidates, device=surface.device)
        points = coordinates(owner, cb.lookup_rows(indices))
        slots = {row: i for i, row in enumerate(candidates)}
        comparisons = kept = 0
        for source in hot.tolist():
            if source not in slots:
                continue
            # Directional overlap: the share of the energetic source present
            # in the destination, using the same meet owner as bind.
            ratio = meet_ratios(points[slots[source]], points)[1]
            ratio[slots[source]] = 0
            value, order = ratio.topk(min(capacity, len(candidates)))
            valid = value > 0
            order, value = order[valid], value[valid]
            count = len(order)
            comparisons += len(candidates) - 1
            kept += count
            batches.extend([b] * count)
            sources.extend([source] * count)
            targets.extend(indices[order].tolist())
            weights.extend(value.unbind())
        audits.append(dict(stream=b, sources=len(hot), candidates=len(candidates),
                           part_reads=len(visited), comparisons=comparisons,
                           edges=kept, top_k=capacity,
                           frontier_truncated=truncated or bool(pending)))
    if weights:
        result = (empty.new_tensor(batches), empty.new_tensor(sources),
                  empty.new_tensor(targets), torch.stack(weights).to(surface))
    return result, audits


@torch.no_grad()
def diffuse(owner, surface, edges, spread, active, field):
    """Recorded and computed containment share one source-normalized outflow."""
    if spread <= 0 or not bool(active.any()):
        return
    from util import TheXMLConfig
    from PartRead import occurrence_roots
    from MereologicalCodes import occurrence_memberships
    B, V = surface.shape
    get_store = getattr(owner, '_closed_clause_store', None)
    store = None if get_store is None else get_store()
    records, pairs = [], []
    if store is not None and bool(TheXMLConfig.space('ConceptualSpace', 'membershipPriming', True)):
        records = occurrence_roots(owner, store)
        pairs = sorted({(int(code), int(row)) for code, rows in
                        occurrence_memberships(owner, store, records=records).items()
                        if 0 <= code < V for row in rows})
    words, rows = (torch.tensor(pairs, device=surface.device, dtype=torch.long).T
                   if pairs else (torch.empty(0, device=surface.device, dtype=torch.long),) * 2)
    src = dst = words.new_empty(0)
    weight = surface.new_empty(0)
    if edges is not None:
        src, dst, weight = (value.to(surface.device) for value in edges)
        valid = (src >= 0) & (src < V) & (dst >= 0) & (dst < V)
        src, dst, weight = src[valid], dst[valid], weight[valid].to(surface)
    (batch, source, target, overlap), audit = overlap_edges(owner, surface, active, field)
    degree = surface.new_zeros(V).index_add_(0, src, weight)
    degree.index_add_(0, words, surface.new_ones(len(words)))
    degree = degree.expand(B, -1).clone()
    degree.index_put_((batch, source), overlap, accumulate=True)
    budget = spread * (surface - 1) * active[:, None]
    graph_flow = budget[:, src] * weight / degree[:, src].clamp_min(1e-12)
    overlap_flow = budget[batch, source] * overlap / degree[batch, source].clamp_min(1e-12)
    flow = budget[:, words] / degree[:, words].clamp_min(1e-12)
    row_energy = surface.new_zeros(B, len(records)).index_add_(1, rows, flow)
    row_degree = surface.new_zeros(len(records)).index_add_(0, rows, surface.new_ones(len(rows)))
    received = surface.new_zeros(B, V).index_add_(1, words, row_energy[:, rows] / row_degree[rows])
    change = surface.new_zeros(B, V)
    change.index_add_(1, src, -graph_flow)
    change.index_add_(1, dst, graph_flow)
    change.index_put_((batch, source), -overlap_flow, accumulate=True)
    change.index_put_((batch, target), overlap_flow, accumulate=True)
    change.index_add_(1, words, -flow)
    change.add_(received)
    # Several hated neighbours can otherwise overdraw a hub, after which the
    # standing floor would create energy. Spend only the feasible fraction of
    # this stream's shared budget; transfer itself remains conservative.
    fraction = torch.where(change < 0, surface / (-change).clamp_min(1e-12),
                           1.).amin(-1).clamp_max(1.)
    fraction = torch.where(fraction < 1,
        fraction * (1 - torch.finfo(surface.dtype).eps), fraction)
    # Sum in double: a large neutral inventory otherwise hides small transfer
    # roundoff behind float32 summation of thousands of ones.
    before = surface.double().sum(-1)
    surface.add_(change * fraction[:, None])
    object.__setattr__(owner, '_last_occurrence_priming', dict(
        rows=len(set(rows.tolist())), edges=len(pairs),
        addresses=tuple(records[row]['address'] for row in rows.unique().tolist()),
        received=received * fraction[:, None], sent=flow * fraction[:, None],
        degree=degree.detach(), row_energy=row_energy * fraction[:, None]))
    object.__setattr__(owner, '_last_priming_diffusion', dict(
        graph_edges=len(src), overlap=audit, overlap_flow=overlap_flow * fraction[batch],
        before=before, after=surface.double().sum(-1), degree=degree,
        source=source, target=target, batch=batch, budget_fraction=fraction))
