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


def stimulate(owner, point, *, where=None, when=None):
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
    evidence = close(owner, rows, direct)
    B, E = point.shape[:2]
    if where is None:
        where = torch.zeros(B, E, 2, device=point.device, dtype=torch.long)
    if when is None:
        when = torch.zeros_like(where)
    ids = torch.tensor([owner.concept_id_at_row(row) for row in rows], device=point.device, dtype=torch.long)
    return ConceptualField(ids, evidence, where, when, None)


def key(owner, field):
    """Evidence-weighted centroid; no inventory row is created for a pattern."""
    ids = field.concept_ids
    if ids.ndim != 1:
        raise ValueError('field key requires shared concept identities')
    rows = [owner._csw_row_of(int(cid)) if int(cid) >= 0 else None for cid in ids]
    valid = torch.tensor([row is not None for row in rows], device=ids.device)
    codes = owner.similarity_codebook.lookup_rows([0 if row is None else row for row in rows]).to(field.evidence)
    weights = field.evidence.amax(-1) * valid[:, None, None]
    mass = weights.sum(0)[...,None]
    value = torch.einsum('cbe,cd->bed', weights, codes)
    return value / torch.where(mass > 0, mass, 1.)
