"""Stored location diffusion and the directed class-membership cache."""
import math

import torch


def configure(owner):
    """Construct fixed geometric controls; malformed configuration is an error."""
    from util import TheXMLConfig
    get = lambda name, default: TheXMLConfig.space('ConceptualSpace', name, default)
    spread = float(get('geometricSpread', .05))
    neighbours = int(get('geometricNeighbours', 8))
    radius, rate, low, high = (float(get(name, default)) for name, default in (
        ('geometricRadius', 15.), ('geometricRadiusRate', .02),
        ('geometricRadiusMin', 1.), ('geometricRadiusMax', 90.)))
    if (not all(map(math.isfinite, (spread, radius, rate, low, high)))
            or not 0 <= spread <= 1 or neighbours < 0
            or not 0 <= rate < 1 or not 0 < low <= radius <= high <= 90):
        raise ValueError('invalid geometric diffusion controls')
    owner._geometric_spread = spread
    owner._geometric_neighbours = neighbours
    owner._geometric_radius_rate = rate
    owner._geometric_radius_min = math.radians(low)
    owner._geometric_radius_max = math.radians(high)
    owner.register_buffer('_geometric_radius', torch.tensor([math.radians(radius)]))


@torch.no_grad()
def part_edges(owner):
    """Cache (member, raised class, net positive evidence), never definitions."""
    allocator = owner._concept_allocator
    parts = allocator.parts
    classes = {key[1] for key in allocator.relate_idx if key[0] == 'part-base'}
    parts._class_rows = classes
    revision = parts.symbol_revision
    if owner._priming_edge_revision != revision:
        src, dst, weights = (owner._priming_edge_src, owner._priming_edge_dst,
                             owner._priming_edge_weight)
        count = 0
        rows = parts.ref_kinds[:, 1].eq(parts.SYMBOL).nonzero().flatten().tolist()
        for relation in rows:
            member, parent = parts.refs[relation].tolist()
            if parent not in classes:
                continue
            from FamiliarTier import member_row
            left = member_row(owner, int(parts.ref_kinds[relation, 0]), member)
            right = owner._csw_row_of(parent)
            if left is None or right is None or left == right:
                continue
            if count >= len(src):
                raise RuntimeError('class membership exceeds the constructed edge capacity')
            src[count], dst[count] = left, right
            weights[count] = (parts.truth_plus[relation] - parts.truth_minus[relation]).clamp_min(0)
            count += 1
        # An order-zero located conjunction records its same-order form
        # members as field constituents, rather than a next-order raise.
        for parent in classes:
            right = owner._csw_row_of(parent)
            if right is None:
                continue
            for ref in allocator.refs(parent, 'part') if hasattr(allocator, 'refs') else ():
                if not isinstance(ref, tuple) or ref[0] != 'field':
                    continue
                left = owner._csw_row_of(ref[1])
                if left is None or left == right:
                    continue
                if count >= len(src):
                    raise RuntimeError('class membership exceeds the constructed edge capacity')
                src[count], dst[count], weights[count] = left, right, ref[2]
                count += 1
        owner._priming_edge_count = count
        owner._priming_edge_revision = revision
        owner._priming_edge_rebuilds += 1
    count = owner._priming_edge_count
    return (owner._priming_edge_src[:count], owner._priming_edge_dst[:count],
            owner._priming_edge_weight[:count]) if count else None


@torch.no_grad()
def propagate(owner, surface, active):
    """One conservative, bounded location step over the active streams."""
    codes = getattr(getattr(owner, 'similarity_codebook', None), 'mereology', None)
    if codes is None or not codes.meaning_pairs or not bool(active.any()):
        return
    if surface.ndim != 2 or surface.shape[1] != len(codes.placement) or active.shape != surface.shape[:1]:
        raise RuntimeError('diffusion differs from the constructed stream/inventory shape')
    A = codes.placement.detach().to(surface)
    ok = (codes.placement_witnesses > 0) & (~codes.placement_derived | codes.placement_valid)
    r = (surface - 1).clamp_min(0)
    z = (r * ok) @ A
    scores = z @ A.T
    scores.masked_fill_((r > 0) | ~ok[None], 0)
    cosines = scores / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    radius = owner._geometric_radius.detach().clone()
    # Positive cosines only: masked zeros must not become neighbours at 90 deg.
    mask = (cosines > 0) & (cosines >= radius.cos())
    count = mask.sum(-1)
    k = min(owner._geometric_neighbours, surface.shape[1])
    values, indices = (cosines * mask).topk(k, dim=-1)
    mass = values.sum(-1, keepdim=True)
    total = r.sum(-1, keepdim=True)
    budget = owner._geometric_spread * total * (mass > 0) * active[:, None]
    change = torch.zeros_like(surface)
    change.scatter_add_(1, indices, budget * values / mass.clamp_min(1e-12))
    change.sub_(budget * r / total.clamp_min(1e-12))
    fraction = torch.where(change < 0, surface / (-change).clamp_min(1e-12),
                           1.).amin(-1).clamp_max(1.)
    fraction = torch.where(fraction < 1,
        fraction * (1 - torch.finfo(surface.dtype).eps), fraction)
    before = surface.cpu().double().sum(-1)
    surface.add_(change * fraction[:, None])
    object.__setattr__(owner, '_last_priming_diffusion', dict(
        before=before, after=surface.cpu().double().sum(-1),
        budget_fraction=fraction, receivers=count, radius=radius))
    mean = (count * active).sum().to(surface) / active.sum()
    target = owner._geometric_neighbours
    factor = 1 + owner._geometric_radius_rate * ((mean < target).to(surface) - (mean > target).to(surface))
    owner._geometric_radius.mul_(factor).clamp_(
        owner._geometric_radius_min, owner._geometric_radius_max)
