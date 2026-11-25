"""Paired conceptual evidence; the final axis is presence and counterevidence.

The live field is [concept, batch, bracket, 2]. Each bracket owns one where
and when pair; concepts have no individual coordinates. Native percept events
retain the support used for inverse attribution. No signed scalar stores this
field.
"""
import torch


def union(values, dim):
    """Idempotent union: repeated support contributes its maximum once."""
    if values.shape[dim] == 0:
        return values.sum(dim=dim)
    return values.amax(dim=dim)


def symbols(field):
    """Read [concept, batch, 2] without cancelling heterogeneous evidence."""
    if field.ndim != 4 or field.shape[-1] != 2:
        raise ValueError('concept evidence requires [concept, batch, occurrence, 2]')
    return union(field, dim=2)


def corners(pair):
    """Derived true-only, false-only, both and neither; never storage."""
    positive, negative = pair.unbind(-1)
    return torch.stack((torch.minimum(positive, 1 - negative),
                        torch.minimum(negative, 1 - positive),
                        torch.minimum(positive, negative),
                        torch.minimum(1 - positive, 1 - negative)), -1)


def both_on_rows(lanes, inputs, entry_rows, entry_inputs, weights, rows, *, n_inputs, n_rows):
    """Join each occupied Sigma entry's evidence, then read both poles of each requested row."""
    positions = torch.arange(len(inputs), device=inputs.device)
    # An absent percept contributes the neutral factor to the product.
    source = inputs.new_full((n_inputs + 1,), -1)
    source.scatter_reduce_(0, torch.where((inputs >= 0) & (inputs < n_inputs), inputs, n_inputs),
                           positions, reduce='amax')
    at = source[entry_inputs]
    padded = torch.cat((lanes, lanes.new_zeros(1, *lanes.shape[1:])))
    evidence = padded.index_select(0, torch.where(at >= 0, at, len(inputs)))
    evidence = torch.where((weights < 0)[:, None, None, None], evidence.flip(-1), evidence)
    factors = torch.where((weights.abs() > 0)[:, None, None, None], (1. - evidence).clamp_min(0.), 1.)
    index = entry_rows[:, None, None, None].expand_as(factors)
    product = lanes.new_ones(n_rows + 1, *lanes.shape[1:]).scatter_reduce(0, index, factors, reduce='prod')
    both = (1. - product).amin(-1)
    seats = torch.where((rows >= 0) & (rows < n_rows), rows, n_rows)
    if rows.ndim == 1:
        return both.index_select(0, seats)
    return both.gather(0, seats[..., None].expand(-1, -1, both.shape[-1]))


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
