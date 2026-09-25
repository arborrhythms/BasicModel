"""The ten accessible subsystems and the grammar's enforced effect contract."""
from enum import Enum


class Subsystem(Enum):
    PERCEPT = 'perceptual knowing'
    KNOWING = 'order-zero knowing'
    SYMBOLIC = 'higher-order knowing'
    SERIAL = 'serial thinking'
    PRIMING = 'priming'
    EXPECTATION = 'expectation'
    LTM = 'long-term memory'
    BUDGET = 'budget'
    MERONYMY = 'meronymic access'
    TAXONOMY = 'taxonomic access'


S = Subsystem
PERMISSIONS = {
    'compose': (frozenset((S.PERCEPT, S.KNOWING, S.SYMBOLIC, S.SERIAL, S.PRIMING, S.MERONYMY)),
                frozenset((S.KNOWING, S.SYMBOLIC, S.SERIAL))),
    'thought': (frozenset(S) - {S.PERCEPT},
                frozenset((S.KNOWING, S.SYMBOLIC, S.SERIAL, S.EXPECTATION, S.BUDGET))),
    'generate': (frozenset((S.KNOWING, S.SYMBOLIC, S.SERIAL, S.BUDGET)), frozenset((S.PERCEPT, S.BUDGET))),
    'chooser': (frozenset((S.KNOWING, S.SYMBOLIC, S.SERIAL, S.EXPECTATION, S.LTM, S.BUDGET)), frozenset()),
    'seal': (frozenset((S.SERIAL, S.EXPECTATION)), frozenset((S.LTM, S.EXPECTATION))),
}


def check_access(grammar, reads, writes):
    if grammar not in PERMISSIONS:
        raise ValueError('unknown grammar effect owner')
    if any(not isinstance(item, Subsystem) for item in (*reads, *writes)):
        raise ValueError('grammar scopes must be accessible-mind Subsystem members')
    allowed_read, allowed_write = PERMISSIONS[grammar]
    if not set(reads) <= allowed_read or not set(writes) <= allowed_write:
        raise ValueError(f'{grammar} effect exceeds its subsystem permissions')


def apply_thought_effect(model, result, *, row, work):
    """Commit parameter-free effects to the existing parallel field.

    Serial effects remain on the one ThoughtRecord owner. Code/set/what
    effects seed dictionary-indexed knowing; native higher-order membership
    unfolds through the same conceptual decomposition that builds the pyramid.
    No LTM row or nearest-vector search is performed here.
    """
    import torch
    from Queries import _existing_row, _basis
    if result is None:
        return
    if result.semantic_id == 'arma':
        discourse = getattr(getattr(model, 'symbolSpace', None), 'discourse', None)
        if discourse is not None and result.value is not None:
            # The sole predictor owns staging. Its live prior calculation is
            # numerically the same detached value retained on the thought.
            discourse.expect_next_meaning(row, record=True, refresh=True)
        return
    if getattr(model, '_anticipating_expectation_row', None) == row:
        # Anticipation can retrieve serial frames and form an estimate, but
        # cannot change the input's order-0 field or its reading attention.
        return
    if result.semantic_id not in ('quantize', 'lookup', 'what'):
        return
    space = getattr(model, 'conceptualSpace', None)
    carrier = getattr(space, 'subspace', None)
    if carrier is None:
        return
    seeds = []
    if result.semantic_id == 'quantize':
        reference = result.evidence.get('reference')
        if reference is not None:
            seeds.append(reference)
    else:
        members = result.evidence.get('frames', ()) if result.semantic_id == 'what' else result.value or ()
        for member in members:
            meaning = member.get('meaning')
            if meaning is not None:
                seeds.extend(ref for ref in meaning.role_refs if ref and ref[0] == 'sym')
            reference = member.get('reference')
            if reference is not None:
                seeds.append(reference)
            seeds.extend(('row', code) for role in member.get('leaf_codes', ()) for code in role)
    if not seeds:
        return
    spaces = list(getattr(model, 'conceptualSpaces', ()) or ())
    owner = getattr(carrier, '_concept_code_owner', None)
    if owner is not None and 0 <= owner < len(spaces):
        # The cutover field and its definitions use the same dictionary,
        # even when the terminal carrier belongs to a later tower stage.
        space = spaces[owner]
    elif spaces:
        owner = next(i for i, candidate in enumerate(spaces) if candidate is space)
    basis = _basis(space)
    existing = getattr(carrier, '_concept_activations', None)
    batch = max(row + 1, int(existing.shape[1]) if torch.is_tensor(existing) else 1)
    sparse = callable(getattr(space, '_sparse_active', None)) and space._sparse_active()
    count = sum(space._order_caps()) if sparse else len(basis)
    location = getattr(carrier, '_thought_occurrence', None)
    previous_locations = int(existing.shape[2]) if torch.is_tensor(existing) else 0
    if location is None or location >= previous_locations:
        location = previous_locations
    field = basis.new_zeros(count, batch, max(previous_locations, location + 1), 2)
    if torch.is_tensor(existing):
        n = min(len(field), len(existing))
        field[:n, :existing.shape[1], :previous_locations] = existing[:n].detach()
    object.__setattr__(carrier, '_thought_occurrence', location)
    # An inferred subject has no input byte extent. Keep a sentinel extent
    # and empty position evidence beside its independently retained pair.
    extents = getattr(carrier, '_concept_extents', None)
    positions = getattr(carrier, '_concept_position_evidence', None)
    if torch.is_tensor(extents) and extents.shape[1] < field.shape[2]:
        added = field.shape[2] - extents.shape[1]
        extents = torch.cat((extents, extents.new_full((extents.shape[0], added, 2), -1)), dim=1)
        object.__setattr__(carrier, '_concept_extents', extents)
        if torch.is_tensor(positions):
            positions = torch.cat((positions, positions.new_zeros(
                positions.shape[0], positions.shape[1], added, positions.shape[3], 2)), dim=2)
            object.__setattr__(carrier, '_concept_position_evidence', positions)
    addresses = getattr(carrier, '_concept_inventory_rows', None)
    addresses = (torch.arange(count, device=basis.device) if addresses is None
                 else addresses.to(basis.device).clone())
    ids = getattr(carrier, '_concept_ids', None)
    ids = (torch.tensor([space.concept_id_at_row(int(r)) or -1 for r in addresses], device=basis.device)
           if ids is None else ids.to(basis.device).clone())
    query = field.new_zeros(field.shape)
    for reference in seeds:
        try:
            index = reference[1] if reference[0] == 'row' else _existing_row(space, reference)
        except ValueError:
            continue
        if sparse:
            cid = space.concept_id_at_row(index)
            bound_ids = ids[:, row] if ids.ndim == 2 else ids
            bound_rows = addresses[:, row] if addresses.ndim == 2 else addresses
            slots = (bound_ids == cid).nonzero().flatten() if cid is not None else (bound_rows == index).nonzero().flatten()
            if not len(slots):
                order = space._concept_source_order(cid) if cid is not None else 0
                start, end = space.order_slice(order)
                slots = (bound_rows[start:end] < 0).nonzero().flatten() + start
                if len(slots):
                    bound_rows[slots[0]] = index
                    bound_ids[slots[0]] = -1 if cid is None else cid
            index = int(slots[0]) if len(slots) else -1
        if 0 <= index < len(query):
            query[index, row, location, 0] = 1.
    # A new imagined extent has no observed field. Existing extents select
    # only cases supported in the current attentive field.
    observed = field if location < previous_locations else None
    inferred = space.cs_reverse_presence(query, observed=observed,
                                         inventory_rows=addresses) if sparse else query
    if sparse:
        # The paired sparse attribution follows the definition. Include inferred
        # literals even when no separate relation record names the edge.
        roots = (query[:, row].amax(dim=(1, 2)) > 0).nonzero().flatten().tolist()
        supported = (inferred[:, row].amax(dim=(1, 2)) > 0).nonzero().flatten().tolist()
        for index in roots + [i for i in supported if i not in roots]:
            if not work.consume('effect_node'):
                break
            previous, added = field[index, row], inferred[index, row]
            field[index, row] = torch.maximum(previous, added)
    pending, seen = ([] if sparse else list(seeds)), set()
    while pending:
        reference = pending.pop()
        if reference in seen:
            continue
        seen.add(reference)
        if not work.consume('effect_node'):
            break
        if reference[0] == 'row':
            index, cid = reference[1], space.concept_id_at_row(reference[1])
        else:
            cid = reference[1]
            try:
                index = _existing_row(space, reference)
            except ValueError:
                continue
        if 0 <= index < len(field):
            field[index, row, location, 0] = 1.
        if cid is not None:
            pending.extend(part for part in space.concept_parts(cid)
                           if isinstance(part, tuple) and part[0] == 'sym')
    object.__setattr__(carrier, '_concept_activations', field.detach())
    object.__setattr__(carrier, '_concept_codes',
                       space._field_codes(basis, addresses).detach() if sparse else basis[:count].detach())
    object.__setattr__(carrier, '_concept_ids', ids.detach())
    object.__setattr__(carrier, '_concept_inventory_rows', addresses.detach())
    if owner is not None:
        object.__setattr__(carrier, '_concept_code_owner', owner)
