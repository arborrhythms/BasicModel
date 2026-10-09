"""Equality substitution over the rows already read by an episode.

There is no store, vocabulary or operation evaluator here. A proof uses
native identities, the two cached operands of an ended point, or the three
roles of a relative row, and positive equalities in checked serial results.
"""
from dataclasses import replace

from ThoughtReferences import bindings, evidence_pair, fill, open_slots, with_slots


def equality(meaning):
    return bool(bindings(meaning).get('_equality'))


def integrate(goal, result, previous):
    if not equality(goal):
        return None
    frames = result.evidence.get('frames', ())
    if not frames:
        return goal
    terms, edges = {}, {}
    for item in (*previous, result):
        terms.update(item.evidence.get('components', {}))
    for item in previous:
        values = item.evidence.get('frames', ())
        returned = item.evidence.get('returned_meaning')
        if returned is not None and equality(returned):
            values = (*values, dict(meaning=returned, evidence=evidence_pair(returned),
                occurrence=item.evidence.get('derivation_witness'),
                witnesses=bindings(returned).get('_thought_witnesses', ())))
        for frame in values:
            meaning = frame['meaning']
            if not equality(meaning) or frame['evidence'][0] <= 0:
                continue
            a, b = meaning.role_refs[0], meaning.role_refs[2]
            witness = frame.get('occurrence')
            proof = (witness,) if witness is not None else tuple(frame.get('witnesses', ()))
            for left, right in ((a, b), (b, a)):
                if left is not None and right is not None:
                    edges.setdefault(left, []).append((right, proof, frame['evidence'][0]))

    def equivalent(left, right, visiting=frozenset()):
        if left is None or right is None:
            return None
        if left == right:
            return (), 1.
        key = left, right
        if key in visiting or len(visiting) >= 64:
            return None
        visiting = visiting | {key}
        for source, other, reverse in ((left, right, False), (right, left, True)):
            term = terms.get(source)
            if (term is not None and term['kind'] == 0 and len(term['operands']) == 1
                    and term.get('constructor') == ('reference', True)):
                alias = term['operands'][0]
                found = equivalent(other, alias, visiting) if reverse else equivalent(alias, other, visiting)
                if found is not None:
                    return found
        for other, witnesses, support in edges.get(left, ()):
            found = equivalent(other, right, visiting)
            if found is not None:
                return (*witnesses, *found[0]), min(support, found[1])
        a, b = terms.get(left), terms.get(right)
        if a is None or b is None or a['kind'] != b['kind']:
            return None
        if a['kind'] == 0 and (a.get('constructor') is None or a.get('constructor') != b.get('constructor')):
            return None
        operands_a, operands_b = a['operands'], b['operands']
        if not operands_a or len(operands_a) != len(operands_b):
            return None
        proof, strength = (), 1.
        for first, second in zip(operands_a, operands_b):
            found = equivalent(first, second, visiting)
            if found is None:
                return None
            proof, strength = (*proof, *found[0]), min(strength, found[1])
        return proof, strength

    free = [role for kind, role in open_slots(goal) if kind == 'referent']
    bound = [role for kind, role in bindings(goal).get('_bound_roles', ()) if kind == 'referent']
    roles = free or bound
    if len(roles) != 1 or roles[0] not in (0, 2):
        return goal
    target = roles[0]
    # An initial match supplies the free variable opposite the fixed
    # operand. Subsequent selected matches rewrite that variable's region.
    region = goal.role_refs[2-target] if free else goal.role_refs[target]
    for frame in frames:
        meaning = frame['meaning']
        if not equality(meaning) or frame['evidence'][0] <= 0:
            continue
        for source, answer in ((0, 2), (2, 0)):
            found = equivalent(region, meaning.role_refs[source])
            if found is None or meaning.role_refs[answer] is None:
                continue
            if not free and meaning.role_refs[answer] == goal.role_refs[2-target]:
                continue  # The defining premise read backwards makes no progress.
            proof, support = found
            witnesses = tuple(dict.fromkeys((*bindings(goal).get('_thought_witnesses', ()),
                                             *proof, frame['occurrence'])))
            refs = list(goal.role_refs)
            refs[target] = None
            opened = with_slots(replace(goal, role_refs=tuple(refs)), (('referent', target),))
            value = fill(opened, dict(reference=meaning.role_refs[answer],
                binding_value=meaning.roles[answer], support_true=min(support, frame['evidence'][0]),
                support_false=min(support, frame['evidence'][1])), witnesses=witnesses, operation='query')
            data = bindings(value)
            data['_derived_binding'] = not bool(free)
            data['_direct_binding'] = bool(free)
            # A derived chain's support is its weakest link, never the max
            # accumulated by independent direct witnesses of one region.
            if not free:
                support = min(evidence_pair(goal)[0], support)
                value = with_slots(value, (), pair=(min(support, frame['evidence'][0]),
                                                    min(support, frame['evidence'][1])))
            return replace(value, bindings=dict(bindings(value), **{
                key: data[key] for key in ('_derived_binding', '_direct_binding')}))
    return goal
