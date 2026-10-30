"""One bounded derivation over a held premise and a held rule."""
from dataclasses import replace

from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, evidence_pair, open_slots, with_slots


def derive(context, arguments):
    from ThoughtChunks import read
    refs = [arguments[role] for role in ('argin1', 'argin2')]
    if any(not isinstance(ref, tuple) or len(ref) != 2 or ref[0] != 'chunk' for ref in refs):
        raise ValueError('infer requires a premise chunk and a rule chunk addressed by .where')
    premise, rule = [read(context.stm, ref[1], context.row) for ref in refs]
    if not all(bool(rule.role_mask[i]) and rule.role_refs[i] is not None for i in (0, 2)):
        raise ValueError('infer rule requires two referenced operands')
    data = bindings(rule)
    rule_row = data.get('_rule_row')
    if rule_row is None:
        raise ValueError('infer rule requires its recalled row witness')
    source, target = (0, 2) if context.direction == (0, 2) else (2, 0)
    backward = context.direction == (2, 0)
    negated = bool(bindings(premise).get('_operand_negated'))
    if negated and not backward:
        source, target = target, source
    antecedent, consequent = rule.role_refs[source], rule.role_refs[target]
    witnesses = tuple(bindings(premise).get('_thought_witnesses', ()))
    equality = data.get('_rule_kind') == 'equal' or data.get('_equality', False)
    match, subject = None, None
    match_support = 1.
    substitutions = {}

    def equivalent(left, right):
        nonlocal witnesses, match_support
        if left == right:
            return True
        if left is None or right is None:
            return False
        proof = context.ltm.relation_evidence('equal', left, right,
            max_records=context.max_records, work=context.work)
        if proof.get('support_true', 0.) <= 0:
            return False
        witnesses += tuple(proof.get('witnesses', ()))
        match_support = min(match_support, float(proof['support_true']))
        return True

    occupied = [i for i in range(3) if bool(premise.role_mask[i])]
    matching = next((i for i in occupied if premise.role_refs[i] == antecedent), None)
    if equality and matching is None:
        antecedent, consequent = consequent, antecedent
        source, target = target, source
        matching = next((i for i in occupied if premise.role_refs[i] == antecedent), None)
    if (matching is None and antecedent[0] != 'constituent'
            and bindings(premise).get('_rule_row') != antecedent):
        matching = next((i for i in occupied if equivalent(premise.role_refs[i], antecedent)), None)
    if bindings(premise).get('_rule_row') == antecedent:
        match = 'identity'
    elif matching is not None:
        match = 'identity' if len(occupied) == 1 else 'instantiation'
        subject = premise.role_refs[0] if matching == 2 else None
    elif antecedent[0] == 'constituent':
        template = rule.constituents[antecedent[1]]
        if template.role_refs == premise.role_refs:
            match = 'identity'
        elif template.role_refs[1] == premise.role_refs[1]:
            matches = []
            for slot in (0, 2):
                left, right = premise.role_refs[slot], template.role_refs[slot]
                if equivalent(left, right):
                    matches.append(True)
                elif left is not None and right is not None:
                    proof = context.taxonomy.evidence(left, right, max_nodes=context.max_nodes,
                        max_records=context.max_records, max_steps=context.max_steps,
                        max_expansions=context.max_expansions, work=context.work)
                    matches.append(proof.get('support_true', 0.) > 0)
                    match_support = min(match_support, float(proof.get('support_true', 0.)))
                    witnesses += tuple(edge.owner for edge in proof.get('path', ()))
                else:
                    matches.append(False)
            if all(matches):
                match = 'membership'
        if match is not None:
            substitutions = {template.role_refs[slot]: (premise.role_refs[slot], premise.roles[slot])
                             for slot in (0, 2) if template.role_refs[slot] is not None}
    else:
        subject = premise.role_refs[0]
        if subject is not None:
            proof = context.taxonomy.evidence(subject, antecedent,
                max_nodes=context.max_nodes, max_records=context.max_records,
                max_steps=context.max_steps, max_expansions=context.max_expansions, work=context.work)
            if proof.get('support_true', 0.) > 0:
                match = 'membership'
                match_support = min(match_support, float(proof['support_true']))
                witnesses += tuple(edge.owner for edge in proof.get('path', ()))
    if match is None:
        return dict(value=None, support_true=0., support_false=0., witnesses=(),
                    incomplete=('antecedent_mismatch',), reason='unmatched')
    if equality:
        roles, role_refs = premise.roles.clone(), list(premise.role_refs)
        for slot in occupied:
            if slot == matching or role_refs[slot] == antecedent:
                roles[slot], role_refs[slot] = rule.roles[target], consequent
        conclusion = replace(premise, roles=roles, role_refs=tuple(role_refs))
        match = 'equality'
    elif consequent[0] == 'constituent':
        conclusion = rule.constituents[consequent[1]]
        roles, role_refs = conclusion.roles.clone(), list(conclusion.role_refs)
        for slot in (0, 2):
            if role_refs[slot] in substitutions:
                role_refs[slot], roles[slot] = substitutions[role_refs[slot]]
        conclusion = replace(conclusion, roles=roles, role_refs=tuple(role_refs))
    elif subject is not None and (len(occupied) > 1 or match == 'membership'):
        roles = rule.roles.clone()
        roles[0], roles[2] = premise.roles[0], rule.roles[target]
        conclusion = replace(rule, roles=roles, role_refs=(subject, rule.role_refs[1], consequent),
                             bindings=(), constituents=rule.constituents)
    else:
        conclusion = replace(ConceptualMeaning.from_description(rule.roles[target]),
                             role_refs=(consequent, None, None))
    p, r = evidence_pair(premise), evidence_pair(rule)
    result_pair = (min(p[0], match_support, r[0]), min(p[0], match_support, r[1]))
    if negated and not backward:
        result_pair = result_pair[::-1]
    witnesses = tuple(dict.fromkeys((*witnesses, rule_row, *data.get('_thought_witnesses', ()))))
    metadata = bindings(conclusion)
    metadata.update(_producing_operation='infer', _thought_witnesses=witnesses,
                    _evidence_kind='subgoal' if backward else 'inference', _alternative_pairs=())
    if not backward:
        metadata['_bound_roles'] = (('referent', 2 if int(conclusion.role_mask.sum()) > 1 else 0),)
    if backward:
        result_pair = (0., 0.)
        metadata['_subgoal_subject'] = conclusion.role_refs[0]
        metadata['_query_relation'] = data.get('_rule_kind', 'part')
        role_refs = (None, *conclusion.role_refs[1:])
        conclusion = replace(conclusion, role_refs=role_refs)
    conclusion = with_slots(replace(conclusion, bindings=metadata),
        (('referent', 0),) if backward else (), pair=result_pair)
    return dict(value=conclusion.roles[0], meaning=conclusion,
        support_true=result_pair[0], support_false=result_pair[1], witnesses=witnesses,
        sources=(refs[0][1], rule_row), reason=match,
        evidence_kind='subgoal' if backward else 'inference',
        derivation=dict(premise_pair=p, match_support_true=match_support, rule_pair=r,
                        backward=backward, negated=negated),
        incomplete=())


def replay_pair(evidence):
    proof = evidence['derivation']
    if proof['backward']:
        return (0., 0.)
    p, r = proof['premise_pair'], proof['rule_pair']
    match_support = proof['match_support_true']
    pair = (min(p[0], match_support, r[0]), min(p[0], match_support, r[1]))
    pair = pair[::-1] if proof['negated'] else pair
    for _where, alternative in evidence.get('alternatives', ()):
        pair = tuple(max(a, b) for a, b in zip(pair, alternative))
    return pair
