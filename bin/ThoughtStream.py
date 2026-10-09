"""Bind grammar operands from the ordinary episode's checked serial results."""
from dataclasses import replace
from collections import deque
import torch

from ThoughtReferences import bindings, evidence_pair, fill, needs_episode, open_slots, question, with_slots


def slots(records):
    """A result slot is its content, two poles, witnesses and producing action."""
    return tuple(record.result for record in records
                 if record.kind in ('thought', 'return') and record.result is not None)


def materialize(result):
    """Every checked result supplies an ordinary bindable serial value."""
    from Queries import _freeze_boundary_value
    from Meaning import ConceptualMeaning
    evidence = dict(result.evidence)
    meaning = evidence.get('meaning')
    if meaning is None:
        frames = evidence.get('frames', ())
        if frames:
            meaning = frames[0]['meaning']
        elif torch.is_tensor(evidence.get('value')) and evidence['value'].ndim == 1:
            meaning = ConceptualMeaning.from_description(evidence['value'])
        else:
            meaning = (result.request.constituents[0] if len(result.request.constituents) == 1
                       else result.request)
    if int(meaning.role_mask.sum()) == 2:
        meaning = replace(meaning, sentence_kind='relation')
    data = bindings(meaning)
    data['_producing_operation'] = result.semantic_id
    data['_thought_witnesses'] = tuple(evidence.get('witnesses', ()))
    meaning = with_slots(replace(meaning, bindings=data), open_slots(meaning),
        pair=(evidence.get('support_true', 0.), evidence.get('support_false', 0.)))
    evidence['meaning'] = meaning
    return replace(result, evidence=_freeze_boundary_value(evidence))


def relation(registry, meaning):
    try:
        name = registry.signature_for(meaning, verify_reference=False).operation.semantic_id
    except (TypeError, ValueError):
        return None
    return {'isPart': 'part', 'isEqual': 'equal', 'isImplied': 'implies'}.get(name)


def query_pattern(goal, kind, left, right, values=None, excluded=()):
    refs = (left, goal.role_refs[1], right)
    roles = goal.roles.clone()
    if values is not None:
        roles[0] = values.to(roles)
    mask = goal.role_mask.clone()
    mask[0], mask[2] = left is not None, right is not None
    data = bindings(goal)
    data['_query_relation'] = kind
    data['_query_exclude'] = tuple(excluded)
    return question(replace(goal, roles=roles, role_mask=mask,
                            role_refs=refs, bindings=data),
                    (('referent', 2),) if right is None else ())


def candidates(registry, root, active, current, records, descriptions=()):
    from Queries import ThoughtOperationCandidate
    results = slots(records)
    meanings = tuple(result.evidence.get('meaning') for result in results)
    meanings = tuple(value for value in meanings if value is not None)
    # Existing grammatical alternatives also bind to each result slot.
    requests = list(registry.controller_candidates(root, active, current,
        descriptions=descriptions, result_meanings=meanings))
    if 'query' not in registry.executable_operation_ids:
        return tuple(requests)
    kind = relation(registry, root)
    patterns = [current]
    frames = [frame for result in results for frame in result.evidence.get('frames', ())]
    excluded = tuple(frame['occurrence'] for frame in frames)
    if kind is None:
        data = bindings(current)
        data['_query_exclude'] = tuple(dict.fromkeys((*data.get('_query_exclude', ()), *excluded)))
        patterns = [replace(current, bindings=data)]
        if data.get('_equality'):
            free = [role for tag, role in open_slots(current) if tag == 'referent']
            bound = [role for tag, role in data.get('_bound_roles', ()) if tag == 'referent']
            roles = free or bound
            if len(roles) == 1 and roles[0] in (0, 2):
                role = 2-roles[0] if free else roles[0]
                pattern = query_pattern(current, 'equal', current.role_refs[role], None,
                                        current.roles[role], excluded)
                patterns.append(replace(pattern, bindings=dict(bindings(pattern), _query_components=True)))
    if kind is not None:
        left, right = root.role_refs[0], root.role_refs[2]
        patterns = [query_pattern(root, kind, left, right),
                    query_pattern(root, kind, left, None, excluded=excluded)]
        if kind == 'implies':
            patterns.insert(0, query_pattern(root, 'truth', left, None))
        for frame in frames:
            meaning = frame['meaning']
            if meaning.role_refs[2] is not None:
                patterns.extend((
                    query_pattern(root, kind, meaning.role_refs[2], right, meaning.roles[2]),
                    query_pattern(root, kind, meaning.role_refs[2], None, meaning.roles[2], excluded)))
    seen = set()
    for pattern in patterns:
        key = (pattern.role_refs, pattern.bindings)
        if key in seen:
            continue
        seen.add(key)
        request = registry.form('query', pattern)
        requests.append(ThoughtOperationCandidate(registry.operation_spec('query'), request, ()))
    return tuple(requests)


def integrate(registry, goal, result, records):
    """Compose witnesses already read into serial slots; never search LTM here."""
    if result is None:
        return goal
    witnesses = result.evidence.get('witnesses', ())
    try:
        signature = registry.signature_for(goal, verify_reference=False)
        same = signature.operation.semantic_id == result.semantic_id
    except (TypeError, ValueError):
        same = False
    if same and result.semantic_id == 'ask' and result.evidence.get('meaning') is not None:
        return result.evidence['meaning']
    if (same and result.semantic_id in ('exist', 'gain', 'not', 'query')
            and result.evidence.get('meaning') is not None
            and (result.support_true > 0 or result.support_false > 0)):
        value = result.evidence['meaning']
        return with_slots(value, open_slots(value),
                          pair=(result.support_true, result.support_false))
    if same and result.request.role_refs == goal.role_refs:
        return fill(goal, result, witnesses=witnesses, operation=result.semantic_id)
    kind = relation(registry, goal)
    if kind is None:
        if result.semantic_id in ('query', 'ask'):
            from ThoughtBinding import integrate as bind_region
            aligned = bind_region(goal, result, tuple(item for item in slots(records) if item is not result))
            if aligned is not None:
                return aligned
            return fill(goal, result, witnesses=witnesses, operation=result.semantic_id)
        return goal
    results = (*slots(records), result)
    frames = [frame for item in results for frame in item.evidence.get('frames', ())]
    left, target = goal.role_refs[0], goal.role_refs[2]
    edges = {}
    kinds = {'part': 1, 'equal': 4, 'implies': 2}
    for frame in frames:
        if frame.get('rel_type') != kinds[kind]:
            continue
        meaning = frame['meaning']
        a, b = meaning.role_refs[0], meaning.role_refs[2]
        if a is not None and b is not None:
            edges.setdefault(a, []).append((b, frame))
            if kind == 'equal':
                edges.setdefault(b, []).append((a, frame))
    proof, support = (), 1.
    if kind == 'implies':
        premises = [frame for frame in frames if int(frame['meaning'].role_mask.sum()) == 1
                    and (frame['meaning'].role_refs[0] == left or frame['occurrence'] == left)]
        if not premises:
            return goal
        premise = max(premises, key=lambda frame:frame['evidence'][0])
        proof, support = (premise['occurrence'],), premise['evidence'][0]
    pending = deque([(left, proof, support)])
    visited = set()
    while pending:
        point, path, strength = pending.popleft()
        if point in visited:
            continue
        visited.add(point)
        for next_point, frame in edges.get(point, ()):
            proof = (*path, frame['occurrence'], *frame.get('temporal_witnesses', ()))
            positive = min(strength, frame['evidence'][0])
            if next_point == target and positive > 0:
                return fill(goal, dict(support_true=positive, support_false=0.),
                            witnesses=proof, operation='isPart' if kind == 'part' else
                            'isEqual' if kind == 'equal' else 'isImplied')
            if positive > 0:
                pending.append((next_point, proof, positive))
    return goal


def write(model, meaning, *, row, _active=None, _bound=None, replace_row=None):
    """The thought writer admits only conclusions and unresolved questions."""
    store = getattr(getattr(model, 'symbolSpace', None), 'ltm_store', None)
    if store is None:
        return None
    if not needs_episode(meaning) and evidence_pair(meaning) == (0., 0.):
        return None
    if _active is None:
        visited, checking = set(), set()
        def check(value):
            if id(value) in checking or len(checking) > 64:
                raise ValueError('thought constituent cycle or depth limit')
            if id(value) in visited:
                return
            store._validate_local_references(value)
            checking.add(id(value))
            for child in value.constituents:
                check(child)
            checking.remove(id(value))
            visited.add(id(value))
        check(meaning)
        if len(store) + len(visited) > store.capacity:
            raise OverflowError('thought row capacity exhausted; forgetting is required')
    active = set() if _active is None else _active
    bound = {} if _bound is None else _bound
    if id(meaning) in active or len(active) > 64:
        raise ValueError('thought constituent cycle or depth limit')
    active.add(id(meaning))
    if meaning.constituents:
        references = []
        for child in meaning.constituents:
            if id(child) not in bound:
                value = child if needs_episode(child) or evidence_pair(child) != (0., 0.) else question(child)
                index = write(model, value, row=row, _active=active, _bound=bound)
                bound[id(child)] = store.occurrence_of(index)
            references.append(bound[id(child)])
        def remap(item):
            if isinstance(item, tuple):
                if item and item[0] == 'constituent':
                    return references[item[1]]
                return tuple(remap(value) for value in item)
            return item
        from Meaning import ConceptualMeaning
        original = meaning
        meaning = ConceptualMeaning(meaning.roles, meaning.role_mask,
            **{key: remap(value) for key, value in meaning.metadata().items()})
        active.remove(id(original))
    else:
        active.remove(id(meaning))
    kind = 'question' if needs_episode(meaning) else 'inference'
    if kind == 'inference' and evidence_pair(meaning) == (0., 0.):
        return None
    if replace_row is not None:
        store.upsert_resolved(replace_row, meaning)
        return replace_row
    from Occurrence import source_at
    slot = int(getattr(model, '_open_sentence_slot', 0) or 0)
    document, sentence = source_at(model, row, slot)
    turn = ('thought', document, sentence)
    ordinals = model.__dict__.setdefault('_thought_ordinals', {})
    ordinal = ordinals.get(turn, 0)
    registry = getattr(model, 'grammatical_thoughts', None)
    kind_name = relation(registry, meaning) if registry is not None else None
    if kind_name is None and registry is not None:
        try:
            kind_name = registry.signature_for(meaning, verify_reference=False).operation.semantic_id
        except (ValueError, TypeError):
            pass
    relation_type = {'part': store.REL_PARTOF, 'equal': store.REL_DEF,
                     'implies': store.REL_IMPLIES}.get(kind_name)
    result = store.append_meaning(meaning, kind=kind, evidence=evidence_pair(meaning),
        stream=row, document_key=turn, sentence_index=ordinal, trust=0., rel_type=relation_type)
    if result >= 0:
        ordinals[turn] = ordinal+1
    return result


def commit(model, result, *, row):
    """Publish only the kept walk, after both complete trials were compared."""
    derived = {}
    def resolved_witnesses(meaning):
        data = bindings(meaning)
        data['_thought_witnesses'] = tuple(derived.get(ref, ref)
            for ref in data.get('_thought_witnesses', ()))
        return replace(meaning, bindings=data)
    for record in result.records:
        if record.kind == 'return' and record.meaning is not None and not open_slots(record.meaning):
            # A direct child lookup supplies evidence to its caller; it is
            # not a newly derived proposition. Derived returns are published
            # first so the parent's proof can cite their native occurrences.
            if bindings(record.meaning).get('_direct_binding'):
                continue
            index = write(model, resolved_witnesses(record.meaning), row=row)
            witness = None if record.result is None else record.result.evidence.get('derivation_witness')
            if index is not None and witness is not None:
                derived[witness] = model.symbolSpace.ltm_store.occurrence_of(index)
    result = replace(result, meaning=resolved_witnesses(result.meaning))
    store = getattr(getattr(model, 'symbolSpace', None), 'ltm_store', None)
    occurrence = bindings(result.meaning).get('_query_occurrence')
    prior = None if store is None else store._index_occurrences.get(occurrence)
    if prior is not None:
        # This episode owns the variable in the already committed source
        # row. An exhausted search must not create a second independent
        # variable for later copulas to fill.
        original = store.meaning_of(prior)
        positive, negative = evidence_pair(result.meaning)
        resolved = fill(original, dict(meaning=result.meaning,
            support_true=positive, support_false=negative),
            witnesses=bindings(result.meaning).get('_thought_witnesses', ()),
            operation=bindings(result.meaning).get('_producing_operation'))
        data = bindings(resolved)
        data['_pending'] = bool(open_slots(resolved))
        data['_forward_references'] = tuple(item for item in data.get('_forward_references', ())
                                             if resolved.role_refs[item[0]] is None)
        resolved = replace(resolved, bindings=data)
        if needs_episode(result.meaning):
            return write(model, resolved, row=row, replace_row=prior)
        index = write(model, result.meaning, row=row)
        if index is not None:
            # Rebase any newly minted local children through the concluded
            # row before filling the original region, without replacing its
            # already-bound roles with an unrelated serial result.
            resolved = fill(original, dict(meaning=store.meaning_of(index),
                support_true=positive, support_false=negative),
                witnesses=bindings(result.meaning).get('_thought_witnesses', ()),
                operation=bindings(result.meaning).get('_producing_operation'))
            data = bindings(resolved)
            data['_pending'] = bool(open_slots(resolved))
            data['_forward_references'] = tuple(item for item in data.get('_forward_references', ())
                                                 if resolved.role_refs[item[0]] is None)
            store.upsert_resolved(prior, replace(resolved, bindings=data))
        return index
    index = None
    # A zero-work closing already persisted this exact open row. Reuse its
    # occurrence rather than manufacturing a duplicate exhausted question.
    slot = int(getattr(model, '_open_sentence_slot', 0) or 0)
    fields = getattr(model, '_sentence_fields', {}).get(slot, ())
    if store is not None and result.work.spent == 0 and open_slots(result.meaning) and row < len(fields):
        field = fields[row]
        prior = None if field is None else store.index_of_row(field.row_id)
        if (prior is not None and store.KINDS[int(store.record_kind[prior])] == 'question'
                and torch.equal(store.slots[prior], result.meaning.roles)):
            index = prior
    if index is None:
        index = write(model, result.meaning, row=row)
    if index is None or index < 0:
        return index
    # Pending questions are already LTM rows. Address upserts, not a second
    # row-keyed question cache, fill them when their referenced occurrence arrives.
    return index
