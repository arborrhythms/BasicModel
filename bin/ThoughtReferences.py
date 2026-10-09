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
    """An omitted relational operand is open even without an annotation.

    A fused idea has one role. A relational row has three roles, including
    the ones still to be filled; its occupancy is part of the contract.
    Surface mode cannot preserve an already filled reference.
    """
    result = []
    for kind, role in bindings(meaning).get('_open_references', ()):
        # Old checkpoints may retain the retired evidence annotation. It
        # denotes ignorance, never a free variable or something bind fills.
        if kind == 'evidence':
            continue
        if kind not in ('referent', 'relation'):
            raise ValueError('unknown open-reference kind')
        if role not in (0, 1, 2):
            raise ValueError('an open role must name a canonical row slot')
        elif meaning.role_refs[role] is None:
            result.append((kind, role))
    if (meaning.sentence_kind == 'relation' or
            (meaning.sentence_kind is None and int(meaning.role_mask.sum()) > 1)):
        for role in range(3):
            if (not bool(meaning.role_mask[role]) or
                    (meaning.role_refs[role] is None and
                     (meaning.sentence_kind == 'relation' or any(meaning.role_refs)))):
                result.append(('relation' if role == 1 else 'referent', role))
    return tuple(dict.fromkeys(result))


def needs_episode(meaning):
    """A region can need evidence without owning an extra operand slot."""
    return bool(open_slots(meaning)) or (meaning.mode == 'interrogative'
                                        and evidence_pair(meaning) == (0., 0.))


def conclude_exhausted(meaning):
    """A source-supported forward name becomes a provisional individual.

    This constructs only a trial-local constituent. The kept episode writes
    that individual through the ordinary LTM writer; a rejected trial has
    no durable allocation. A region without source evidence remains open.
    """
    if tuple(bindings(meaning).get('_source_evidence', evidence_pair(meaning))) == (0., 0.):
        return question(meaning, open_slots(meaning))
    from Meaning import ConceptualMeaning
    refs, children = list(meaning.role_refs), list(meaning.constituents)
    mask = meaning.role_mask.clone()
    data = bindings(meaning)
    records = data.get('_formation_records', ())
    filled = []
    for kind, role in open_slots(meaning):
        if kind != 'referent':
            continue
        provenance = tuple(record for record in records if dict(record).get('role') == role)
        child = with_slots(ConceptualMeaning.from_description(meaning.roles[role]), (),
                           pair=evidence_pair(meaning))
        child_data = bindings(child)
        child_data.update(_producing_operation='mint', _formation_records=provenance,
                          _formation_reason='search_exhausted')
        refs[role] = ('constituent', len(children))
        mask[role] = True
        children.append(replace(child, bindings=child_data))
        filled.append((kind, role))
    if filled:
        # Trial scoring precedes the kept write. The provisional individual
        # already fills this variable here, irrespective of its evidence.
        data['_bound_roles'] = tuple(dict.fromkeys((*data.get('_bound_roles', ()), *filled)))
    data['_forward_references'] = tuple(item for item in data.get('_forward_references', ())
                                         if refs[item[0]] is None)
    return with_slots(replace(meaning, role_refs=tuple(refs), role_mask=mask, bindings=data,
                             constituents=tuple(children)), open_slots(meaning))


def with_slots(meaning, slots, *, pair=None):
    data = bindings(meaning)
    data['_open_references'] = tuple(dict.fromkeys(slot for slot in slots if slot[0] != 'evidence'))
    if pair is not None:
        values = tuple(float(x) for x in pair)
        if len(values) != 2 or any(not 0 <= x <= 1 for x in values):
            raise ValueError('thought evidence requires two independent poles in [0,1]')
        data['_evidence_pair'] = values
    return replace(meaning, bindings=data)


def question(meaning, slots=None):
    """An interrogative region has free variables and no source evidence."""
    if slots is None:
        slots = open_slots(meaning)
    data = bindings(meaning)
    data['_source_evidence'] = (0., 0.)
    return with_slots(replace(meaning, mode='interrogative', bindings=data), slots, pair=(0., 0.))


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


def fill(meaning, result, *, witnesses=(), operation=None, slots=None):
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
        if slots is not None and (kind, role) not in slots:
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
    filled_roles = tuple((kind, role) for kind, role in open_slots(meaning)
                         if kind != 'evidence' and references[role] is not None)
    if filled_roles:
        data['_bound_roles'] = tuple(dict.fromkeys((*data.get('_bound_roles', ()), *filled_roles)))
        if '_forward_references' in data:
            data['_forward_references'] = tuple(item for item in data['_forward_references']
                                                 if references[item[0]] is None)
    data['_thought_witnesses'] = tuple(dict.fromkeys((*data.get('_thought_witnesses', ()),
                                                   *witnesses)))
    if operation is not None:
        data['_producing_operation'] = operation
    updated = replace(meaning, roles=roles, role_mask=mask,
                      role_refs=tuple(references), constituents=tuple(children), bindings=data)
    if '_pending' in data:
        data['_pending'] = bool(open_slots(updated))
        updated = replace(updated, bindings=data)
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
    data = bindings(meaning)
    data['_source_evidence'] = tuple(evidence)
    return with_slots(replace(meaning, bindings=data), slots, pair=evidence)
