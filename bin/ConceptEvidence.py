"""Paired conceptual evidence; the final axis is presence and counterevidence.

The live field is [concept, batch, occurrence, 2]. Occurrences retain their
scope through composition; only a symbol's readout unions over occurrences.
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
    poles = torch.stack((codes, -codes), dim=1)
    return (pair.unsqueeze(-1) * poles.unsqueeze(0)).flatten(1, 2)


def admit(projections, floor):
    """Sign routes a projection; a measured floor rejects unrelated evidence."""
    if not 0 <= float(floor) < 1:
        raise ValueError('conceptEvidenceFloor must be in [0, 1)')
    return torch.stack((torch.relu(projections - floor),
                        torch.relu(-projections - floor)), -1).clamp_max(1 - floor) / (1 - floor)
