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
    if result is None or result.semantic_id not in ('quantize', 'lookup', 'what'):
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
    basis = _basis(space)
    existing = getattr(carrier, '_concept_activations', None)
    batch = max(row + 1, int(existing.shape[1]) if torch.is_tensor(existing) else 1)
    field = basis.new_zeros(len(basis), batch)
    if torch.is_tensor(existing):
        n = min(len(field), len(existing))
        field[:n, :existing.shape[1]] = existing[:n].detach()
    row_ids = {value: cid for (_order, cid), value in space._csw_rows.items()}
    pending, seen = list(seeds), set()
    while pending:
        reference = pending.pop()
        if reference in seen:
            continue
        seen.add(reference)
        if not work.consume('effect_node'):
            break
        if reference[0] == 'row':
            index, cid = reference[1], row_ids.get(reference[1])
        else:
            cid = reference[1]
            try:
                index = _existing_row(space, reference)
            except ValueError:
                continue
        if 0 <= index < len(field):
            field[index, row] = 1.
        if cid is not None:
            pending.extend(part for part in space.concept_parts(cid)
                           if isinstance(part, tuple) and part[0] == 'sym')
    object.__setattr__(carrier, '_concept_activations', field.detach())
