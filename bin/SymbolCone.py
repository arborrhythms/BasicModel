"""A symbol's row as a cone over coordinates of the conceptual space of the
order beneath its own.

The concepts a symbol holds give its for side: on each coordinate, what all
of them hold (the least of their codes), so a coordinate on which they vary
keeps the least, and one that any of them lacks has no weight. Its against
side is what its nearest neighbours hold that it does not, as a basic
percept's against entries are the parts its holders add.

A code is read at one place, one concept's code, never a join of several.
The fold is the plain product: each weighed coordinate's share of what the
row asks, ``min(1, x / w)``, raised to the coordinate's share of the row's
weight, so the read depends on the code's direction among the weighed
coordinates and not on its size beyond them; the against side is joined by
the union, any one coordinate being enough. Codes are non-negative pole
pairs, so nothing here changes a concept's place.
"""
import torch


def for_side(codes):
    """The cone's weights over ``codes`` ``[m, D]`` of its members: the
    least on each coordinate."""
    return codes.clamp_min(0).amin(0)


def against_side(weights, neighbours):
    """What the nearest ``neighbours`` ``[k, D]`` hold where the cone has no
    weight, the greatest of them on each such coordinate."""
    if not len(neighbours):
        return torch.zeros_like(weights)
    held = neighbours.clamp_min(0).amax(0)
    return torch.where(weights > 0, torch.zeros_like(held), held)


def read(weights, against, codes):
    """Each code's two lanes in the cone, ``[n, 2]``: for, the weighted
    product of ``min(1, x / w)`` over the weighed coordinates; against, the
    union over the against coordinates of ``min(1, x / a)``."""
    codes = codes.clamp_min(0)
    if codes.ndim == 1:
        codes = codes[None]
    weighed = weights > 0
    if not bool(weighed.any()):
        return codes.new_zeros(len(codes), 2)
    share = weights[weighed] / weights[weighed].sum()
    ratio = (codes[:, weighed] / weights[weighed]).clamp(max=1.)
    held = torch.exp((share * torch.log(ratio.clamp_min(1e-30))).sum(-1))
    held = torch.where((ratio > 0).all(-1), held, torch.zeros_like(held))
    opposed = against > 0
    if bool(opposed.any()):
        found = (codes[:, opposed] / against[opposed]).clamp(0., 1.)
        against_lane = 1. - (1. - found).prod(-1)
    else:
        against_lane = torch.zeros_like(held)
    return torch.stack((held, against_lane), -1)


def holds(weights, against, codes, width=1.):
    """Whether each code falls in the cone: read for at ``width`` or more
    and for more than against."""
    lanes = read(weights, against, codes)
    return (lanes[:, 0] >= width - 1e-6) & (lanes[:, 0] > lanes[:, 1])


def codes_of(owner, concepts):
    """The conceptual codes of ``concepts``, ``[m, D]``, at zero or above."""
    rows = [owner._concept_row_of(int(c)) for c in concepts]
    W = owner.similarity_codebook.W
    if any(row is None for row in rows):
        raise ValueError('a cone reads seated concepts')
    return W.detach()[torch.tensor(rows, dtype=torch.long, device=W.device)].clamp_min(0)


def same_order(owner, left, right):
    order = owner._concept_allocator.order_of
    return order(int(left)) == order(int(right)) >= 0


def neighbours(owner, members, weights, k):
    """The codes of the ``k`` concepts of the members' order, not members,
    that hold the most of the cone's weight (by the meet ratio), ``[k, D]``."""
    from IdentityRows import meet_ratios
    alloc = owner._concept_allocator
    order = alloc.order_of(int(members[0]))
    others = [cid for cid in alloc.concepts if cid not in alloc.retired and int(cid) not in members
              and alloc.order_of(int(cid)) == order and owner._concept_row_of(int(cid)) is not None]
    if not others or not bool((weights > 0).any()):
        return weights.new_zeros(0, len(weights))
    codes = codes_of(owner, others)
    covered = meet_ratios(weights.expand_as(codes), codes)[1]
    return codes[covered.argsort(descending=True, stable=True)[:k]]
