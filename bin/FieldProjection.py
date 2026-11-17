"""Content projection and definition closure over the known concept inventory."""
import torch
from FieldLogic import reduce
from IdentityRows import meet_ratios
from Understanding import ConceptualField


def inventory(owner):
    store = owner._concept_allocator.layer(0)
    return sorted(row for row in store._tensor_row_keys
                  if owner.concept_id_at_row(row) is not None)


def close(owner, rows, evidence):
    """Fold each definition once, in dependency order, retaining live edges."""
    store = owner._concept_allocator.layer(0)
    positions = {int(row): i for i, row in enumerate(rows)}
    definitions = {}
    for row in positions:
        definitions[row] = []
        for matrix, operation in ((store.conjunctive, 'intersection'), (store, 'union')):
            edges = [(column % (store.nOutput+1), column >= store.nOutput+1, index)
                     for (target, column), index in matrix._index.items()
                     if target == row and float(matrix.values[index].detach()) > 0]
            if edges:
                definitions[row].append((matrix, operation, edges))
    done, pending = {}, set(positions)
    while pending:
        ready = [row for row in sorted(pending) if all(
            source not in positions or source in done
            for _, _, edges in definitions[row] for source, _, _ in edges)]
        if not ready:
            raise ValueError('cyclic concept field definitions')
        for row in ready:
            value = evidence[positions[row]]
            for matrix, operation, edges in definitions[row]:
                operands, weights = [], []
                for source, negative, edge in edges:
                    item = done.get(source, torch.zeros_like(value))
                    if source == store.nOutput:
                        item = torch.zeros_like(value)
                        item[..., 0] = 1.
                    operands.append(item.flip(-1) if negative else item)
                    weights.append(matrix.values[edge].to(value))
                combined = reduce(torch.stack(operands, -2), operation=operation,
                                  weights=torch.stack(weights))
                # A row without direct evidence takes its definition. Direct
                # perceptual evidence and a definition are independent sources.
                value = torch.maximum(value, combined)
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
    rows = [owner._csw_row_of(int(cid)) for cid in field.concept_ids]
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
    codes = owner.similarity_codebook.lookup_rows(rows).to(point)
    width = min(codes.shape[-1], point.shape[-1])
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    form_width = width if derived is None else min(width, derived.percept_event_width)
    form = codes[:, None, None, :form_width].clamp_min(0)
    positive, _ = meet_ratios(point[None, ..., :form_width].clamp_min(0), form)
    negative, _ = meet_ratios((-point[None, ..., :form_width]).clamp_min(0), form)
    if form_width < width:
        if (width - form_width) % 2:
            raise ValueError('stimulate requires two equal meaning blocks')
        lane_width = (width - form_width) // 2
        # Native codes carry signless meaning in separate for/against blocks.
        # Form identifies a percept; it cannot supply conceptual entailment.
        meaning = codes[:, form_width:form_width + lane_width].clamp_min(0)
        known = meaning.sum(-1)[:, None, None] > 0
        lanes = point[..., form_width:width].split(lane_width, -1)
        for_lane, _ = meet_ratios(lanes[0][None].clamp_min(0), meaning[:, None, None])
        against_lane, _ = meet_ratios(lanes[1][None].clamp_min(0), meaning[:, None, None])
        positive = torch.where(known, for_lane, positive)
        negative = torch.where(known, against_lane, negative)
    direct = torch.stack((positive, negative), -1)
    # Percept collections share the native boundary reader before closure.
    if hasattr(owner, 'basic_percepts'):
        from PerceptCollections import read
        collections = read(owner, point)
        if collections:
            direct = torch.stack([collections.get(row, direct[i]) for i, row in enumerate(rows)])
    requested = torch.as_tensor(order, device=point.device, dtype=torch.long)
    native = requested.new_tensor([owner._row_order(row) for row in rows])
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
        if bool(live) and int(cid) >= 0 and (row := owner._csw_row_of(int(cid))) is not None}))


def key(owner, field, *, order):
    """A centroid at the requested order; the complete field stays intact."""
    ids = field.concept_ids
    if ids.ndim != 1:
        raise ValueError('field key requires shared concept identities')
    rows = [owner._csw_row_of(int(cid)) if int(cid) >= 0 else None for cid in ids]
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
