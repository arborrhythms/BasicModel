"""Taxonomy diffusion and a read-only geometric consumer channel."""
import torch


@torch.no_grad()
def part_edges(owner):
    """Cached SYMBOL/SYMBOL taxonomy; sentence chains conduct no energy."""
    parts = owner._concept_allocator.parts
    revision = parts.symbol_revision
    if owner._priming_edge_revision != revision:
        src, dst = owner._priming_edge_src, owner._priming_edge_dst
        count = 0
        rows = parts.ref_kinds.eq(parts.SYMBOL).all(-1).nonzero().flatten().tolist()
        for relation in rows:
            left, right = (owner._csw_row_of(int(symbol)) for symbol in parts.refs[relation])
            if left is None or right is None or left == right:
                continue
            if count + 2 > len(src):
                raise RuntimeError('taxonomy priming exceeds the constructed edge capacity')
            src[count:count+2] = src.new_tensor((left, right))
            dst[count:count+2] = dst.new_tensor((right, left))
            count += 2
        owner._priming_edge_count = count
        owner._priming_edge_revision = revision
        owner._priming_edge_rebuilds += 1
    count = owner._priming_edge_count
    return (owner._priming_edge_src[:count], owner._priming_edge_dst[:count],
            owner._priming_edge_weight[:count]) if count else None


@torch.no_grad()
def geometric(owner, surface):
    """Heat-path matrix overlap; transfer excess on read, never store it."""
    from util import TheXMLConfig
    # Five percent of active excess, at most eight neighbours: the standing
    # heat-path budget and a bounded conceptual neighbourhood (§12.2).
    spread = float(TheXMLConfig.space('ConceptualSpace', 'geometricSpread', .05))
    neighbours = int(TheXMLConfig.space('ConceptualSpace', 'geometricNeighbours', 8))
    if not 0 <= spread <= 1 or neighbours < 0:
        raise ValueError('geometric spread must lie in [0, 1] and neighbours be nonnegative')
    derived = getattr(getattr(owner, 'similarity_codebook', None), 'mereology', None)
    change = torch.zeros_like(surface)
    if derived is None or not spread or not neighbours or not derived.meaning_pairs:
        return change
    A = derived.placement.detach().to(surface)
    placed = derived.placement_witnesses > 0
    r = (surface - 1).clamp_min(0)
    active = r > 0
    # As in build_semantic_heat: context = r_S @ A_S; overlap = A @ context.
    scores = (r @ A) @ A.T
    scores.masked_fill_(active | ~placed[None], 0)
    k = min(neighbours, scores.shape[1])
    values, indices = scores.topk(k, dim=-1)
    mass = values.sum(-1, keepdim=True)
    budget = spread * r.sum(-1, keepdim=True) * (mass > 0)
    change.scatter_add_(1, indices, budget * values / mass.clamp_min(1e-12))
    change.sub_(budget * r / r.sum(-1, keepdim=True).clamp_min(1e-12))
    return change


@torch.no_grad()
def diffuse(owner, surface, edges, spread, active):
    """Conservative diffusion on the surviving explicit edge channel."""
    if spread <= 0 or not bool(active.any()):
        return
    B, V = surface.shape
    src = dst = torch.empty(0, device=surface.device, dtype=torch.long)
    weight = surface.new_empty(0)
    if edges is not None:
        src, dst, weight = (value.to(surface.device) for value in edges)
        valid = (src >= 0) & (src < V) & (dst >= 0) & (dst < V)
        src, dst, weight = src[valid], dst[valid], weight[valid].to(surface)
    degree = surface.new_zeros(V).index_add_(0, src, weight)
    degree = degree.expand(B, -1)
    budget = spread * (surface - 1) * active[:, None]
    graph_flow = budget[:, src] * weight / degree[:, src].clamp_min(1e-12)
    change = surface.new_zeros(B, V)
    change.index_add_(1, src, -graph_flow)
    change.index_add_(1, dst, graph_flow)
    # Several hated neighbours can otherwise overdraw a hub, after which the
    # standing floor would create energy. Spend only the feasible fraction of
    # this stream's shared budget; transfer itself remains conservative.
    fraction = torch.where(change < 0, surface / (-change).clamp_min(1e-12),
                           1.).amin(-1).clamp_max(1.)
    fraction = torch.where(fraction < 1,
        fraction * (1 - torch.finfo(surface.dtype).eps), fraction)
    # Sum in double: a large neutral inventory otherwise hides small transfer
    # roundoff behind float32 summation of thousands of ones.
    # Scalar diagnostics retain double precision on hosts without device float64.
    before = surface.cpu().double().sum(-1)
    surface.add_(change * fraction[:, None])
    object.__setattr__(owner, '_last_priming_diffusion', dict(
        graph_edges=len(src),
        before=before, after=surface.cpu().double().sum(-1), degree=degree,
        budget_fraction=fraction))
