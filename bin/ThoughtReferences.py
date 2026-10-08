"""Open references travel with ordinary meanings, never a planner stack."""
from dataclasses import replace
import torch


def bindings(meaning):
    value = meaning.bindings
    if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
           for item in value):
        return dict(value)
    return {'_original_bindings': value} if value else {}


def evidence_pair(meaning):
    return tuple(bindings(meaning).get('_evidence_pair', (0., 0.)))


def open_slots(meaning):
    """Surface mode cannot create or preserve an already filled reference."""
    result = []
    pair = evidence_pair(meaning)
    for kind, role in bindings(meaning).get('_open_references', ()):
        if kind not in ('referent', 'relation', 'evidence'):
            raise ValueError('unknown open-reference kind')
        if kind == 'evidence':
            if pair == (0., 0.):
                result.append((kind, -1))
        elif role not in (0, 1, 2):
            raise ValueError('an open role must name a canonical row slot')
        elif meaning.role_refs[role] is None:
            result.append((kind, role))
    return tuple(result)


def with_slots(meaning, slots, *, pair=None):
    data = bindings(meaning)
    data['_open_references'] = tuple(dict.fromkeys(slots))
    if pair is not None:
        values = tuple(float(x) for x in pair)
        if len(values) != 2 or any(not 0 <= x <= 1 for x in values):
            raise ValueError('thought evidence requires two independent poles in [0,1]')
        data['_evidence_pair'] = values
    result = replace(meaning, bindings=data)
    return replace(result, mode='interrogative' if open_slots(result) else 'assertive')


def question(meaning, slots=None):
    """An explicit question opens its omitted operands or its unknown evidence."""
    if slots is None:
        slots = tuple(('relation' if role == 1 else 'referent', role)
                      for role in range(3) if not bool(meaning.role_mask[role]))
        if not slots:
            slots = (('evidence', -1),)
    return with_slots(meaning, slots, pair=(0., 0.))


def validate_local_references(meaning):
    """Check each meaning's local table before a filling can escape.

    Children own their own tables. Sharing a child is valid; a cycle, malformed
    address or out-of-range local address is a construction error.
    """
    seen, active = set(), set()

    def visit(value):
        if id(value) in active or len(active) > 64:
            raise ValueError('thought constituent cycle or depth limit at fill')
        if id(value) in seen:
            return
        active.add(id(value))

        def check(item):
            if isinstance(item, tuple):
                if item and item[0] == 'constituent':
                    if (len(item) != 2 or type(item[1]) is not int
                            or not 0 <= item[1] < len(value.constituents)):
                        raise ValueError('dangling or malformed local constituent reference at fill')
                else:
                    for part in item:
                        check(part)

        for item in value.metadata().values():
            check(item)
        for child in value.constituents:
            visit(child)
        active.remove(id(value))
        seen.add(id(value))

    visit(meaning)


def fill(meaning, result, *, witnesses=(), operation=None):
    """Bind from a checked result, retaining both poles and its provenance."""
    validate_local_references(meaning)
    evidence = result.evidence if hasattr(result, 'evidence') else result
    pair = (float(evidence.get('support_true', 0.)),
            float(evidence.get('support_false', 0.)))
    supplied = evidence.get('meaning')
    if supplied is None:
        frames = evidence.get('frames', ())
        supplied = frames[0].get('meaning') if len(frames) == 1 else None
    if supplied is not None:
        validate_local_references(supplied)
    frames = evidence.get('frames', ())
    roles, mask = meaning.roles.clone(), meaning.role_mask.clone()
    references = list(meaning.role_refs)
    children = list(meaning.constituents)
    rebased = None

    def import_reference(reference):
        nonlocal rebased
        if rebased is None:
            existing = {id(child): index for index, child in enumerate(children)}
            rebased = {}
            for index, child in enumerate(supplied.constituents):
                if id(child) not in existing:
                    existing[id(child)] = len(children)
                    children.append(child)
                rebased[index] = existing[id(child)]
        if isinstance(reference, tuple):
            if reference and reference[0] == 'constituent':
                return ('constituent', rebased[reference[1]])
            return tuple(import_reference(item) for item in reference)
        return reference

    for kind, role in open_slots(meaning):
        if kind == 'evidence':
            continue
        if supplied is not None and supplied.role_refs[role] is not None:
            references[role] = import_reference(supplied.role_refs[role])
            roles[role], mask[role] = supplied.roles[role], supplied.role_mask[role]
        elif (supplied is not None and len(frames) == 1
              and bool(supplied.role_mask[role])):
            # A fused one-slot row names itself. It need not pretend its
            # occurrence address is a conceptual-space allocation.
            if int(supplied.role_mask.sum()) == 1:
                references[role] = import_reference(frames[0]['occurrence'])
            roles[role], mask[role] = supplied.roles[role], supplied.role_mask[role]
        elif (evidence.get('reference') is not None
              and torch.is_tensor(evidence.get('binding_value', evidence.get('value')))):
            references[role] = evidence['reference']
            roles[role], mask[role] = evidence.get('binding_value', evidence.get('value')).to(roles), True
    data = bindings(meaning)
    data['_thought_witnesses'] = tuple(dict.fromkeys((*data.get('_thought_witnesses', ()),
                                                   *witnesses)))
    if operation is not None:
        data['_producing_operation'] = operation
    updated = replace(meaning, roles=roles, role_mask=mask,
                      role_refs=tuple(references), constituents=tuple(children), bindings=data)
    validate_local_references(updated)
    old = evidence_pair(meaning)
    return with_slots(updated, open_slots(meaning),
                      pair=(max(old[0], pair[0]), max(old[1], pair[1])))


def from_closing(meaning, *, evidence=(0., 0.), unresolved=False):
    """The binder's missing reference, not punctuation, defines questionhood."""
    slots = list(open_slots(meaning))
    if unresolved:
        slots.extend(('relation' if role == 1 else 'referent', role)
                     for role in range(3) if bool(meaning.role_mask[role])
                     and meaning.role_refs[role] is None)
    well_defined = bool(meaning.roles[meaning.role_mask].norm() > 0)
    fully_named = all(meaning.role_refs[role] is not None
                      for role in range(3) if bool(meaning.role_mask[role]))
    if (well_defined and tuple(evidence) == (0., 0.)
            and (fully_named or meaning.mode == 'interrogative')):
        slots.append(('evidence', -1))
    return with_slots(meaning, slots, pair=evidence)
