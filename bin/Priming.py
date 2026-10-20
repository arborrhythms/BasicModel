"""Read retained Part edges into one detached, conservative diffusion budget.

Word-to-sentence memberships are derived from the Part tree. Only the priming
surface is written; conceptual code placement is not read by this channel.
"""
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


@torch.no_grad()
def diffuse(owner, surface, edges, spread, active):
    """Part pairs and derived memberships share one normalized outflow."""
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
    degree = surface.new_zeros(V).index_add_(0, src, weight)
    degree.index_add_(0, words, surface.new_ones(len(words)))
    degree = degree.expand(B, -1)
    budget = spread * (surface - 1) * active[:, None]
    graph_flow = budget[:, src] * weight / degree[:, src].clamp_min(1e-12)
    flow = budget[:, words] / degree[:, words].clamp_min(1e-12)
    row_energy = surface.new_zeros(B, len(records)).index_add_(1, rows, flow)
    row_degree = surface.new_zeros(len(records)).index_add_(0, rows, surface.new_ones(len(rows)))
    received = surface.new_zeros(B, V).index_add_(1, words, row_energy[:, rows] / row_degree[rows])
    change = surface.new_zeros(B, V)
    change.index_add_(1, src, -graph_flow)
    change.index_add_(1, dst, graph_flow)
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
        graph_edges=len(src),
        before=before, after=surface.double().sum(-1), degree=degree,
        budget_fraction=fraction))
