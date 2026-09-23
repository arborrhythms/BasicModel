"""Paired conceptual evidence; the final axis is presence and counterevidence.

The live field is [concept, batch, extent, 2]. Extents retain their positions
and scope through composition; only a symbol's readout unions over extents.
One distributed code names both symbols. No signed scalar stores this field.
"""
import torch


def union(values, dim):
    """Probabilistic union in log-complement coordinates, with exact rails."""
    eps = torch.finfo(values.dtype).eps
    safe = values + (values.clamp(0, 1 - eps) - values).detach()
    result = -torch.expm1(torch.log1p(-safe).sum(dim=dim))
    exact = torch.where((values >= 1).any(dim=dim), torch.ones_like(result), result)
    return result + (exact - result).detach()


def symbols(field):
    """Read [concept, batch, 2] without cancelling heterogeneous evidence."""
    if field.ndim != 4 or field.shape[-1] != 2:
        raise ValueError('concept evidence requires [concept, batch, occurrence, 2]')
    return union(field, dim=2)


def corners(pair):
    """Derived true-only, false-only, both and neither; never storage."""
    positive, negative = pair.unbind(-1)
    return torch.stack((positive * (1 - negative), negative * (1 - positive),
                        positive * negative, (1 - positive) * (1 - negative)), -1)


def decode(field, codes):
    """Two symbol rows per concept, addressed 2*i and 2*i+1, one code."""
    pair = symbols(field).permute(1, 0, 2)
    rows = torch.arange(2 * len(codes), device=codes.device)
    # The symbol leg has two rows per concept; dictionary addresses are
    # row // 2. Serial word references already carry concept-row addresses.
    poles = codes.index_select(0, rows // 2)
    poles = poles * torch.where(rows % 2 == 0, 1., -1.)[:, None]
    return pair.flatten(1, 2).unsqueeze(-1) * poles.unsqueeze(0)


def in_extents(position_evidence, position_spans, extents):
    """Keep [C,B,E,L,2] position pairs and read [C,B,E,2] extent evidence.

    Inclusion follows half-open input spans. A position must be wholly
    inside its subject, so a tile crossing an extent boundary cannot leak
    its evidence into either neighbouring subject.
    """
    if position_evidence.ndim != 4 or position_evidence.shape[-1] != 2:
        raise ValueError('position evidence requires [concept,batch,position,2]')
    if (position_spans.shape != (*position_evidence.shape[1:3], 2)
            or extents.ndim != 3 or extents.shape[0] != position_evidence.shape[1]
            or extents.shape[-1] != 2):
        raise ValueError('position spans and extents must match the evidence batch')
    position_spans, extents = position_spans.to(position_evidence.device), extents.to(position_evidence.device)
    ps, pe = position_spans.unbind(-1)
    es, ee = extents.unbind(-1)
    mask = ((ps[:, None] >= es[:, :, None]) & (pe[:, None] <= ee[:, :, None])
            & (pe[:, None] > ps[:, None]) & (ee[:, :, None] > es[:, :, None]))
    positions = position_evidence.unsqueeze(2) * mask[None, ..., None]
    return union(positions, dim=3), positions
