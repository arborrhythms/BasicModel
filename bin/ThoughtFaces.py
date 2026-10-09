"""The conceptual and symbolic thought faces return content and two poles."""
import torch


def pair(value=None, positive=0., negative=0., *, witnesses=(), **metadata):
    return dict(value=value, support_true=positive, support_false=negative,
                witnesses=tuple(witnesses), **metadata)


def part(context, arguments):
    from Queries import _vector
    from Layers import Ops
    left, right = (_vector(context, arguments[role]) for role in ('I1', 'I2'))
    residual = Ops.part(left, right).detach()
    support = float(Ops.part(left, right, scalar=True))
    return pair(residual, support, result_kind='concept', evidence_kind='meronymy')


def is_part(context, arguments):
    if len(arguments) == 1:
        role, reference = next(iter(arguments.items()))
        found = dict(context.taxonomy.neighbors(reference,
            direction='up' if role == 'I1' else 'down', max_nodes=context.max_nodes,
            max_records=context.max_records, max_expansions=context.max_expansions,
            work=context.work))
        values = found.get('value', ())
        # These are direct, held alternatives. The selected symbolic face
        # binds the first strongest filling, carrying its native edge witness.
        best = max(values, key=lambda item: item.get('trust', 0.)) if values else None
        return dict(found, support_true=0. if best is None else best['trust'],
                    support_false=0., reference=None if best is None else best['reference'],
                    binding_value=None if best is None else best['value'],
                    witnesses=() if best is None else (best['source'].owner,),
                    evidence_kind='taxonomy', result_kind='set')
    left, right = arguments['I1'], arguments['I2']
    result = context.ltm.relation_evidence('part', left, right,
        max_records=context.max_records, work=context.work)
    if result['support_true'] or result['support_false']:
        return result
    evidence = dict(context.taxonomy.evidence(left, right,
        max_nodes=context.max_nodes, max_records=context.max_records,
        max_steps=context.max_steps, max_expansions=context.max_expansions,
        work=context.work))
    evidence['witnesses'] = tuple(edge.owner for edge in evidence.get('path', ()))
    evidence['value'] = dict(context.operand_values).get('I1')
    return evidence


def equal(context, arguments):
    from Queries import _vector
    left, right = (_vector(context, arguments[role]) for role in ('I1', 'I2'))
    score = context.conceptual_space.equal(left, right)
    return pair(left.detach().clone(), score, operands=(left, right), result_kind='concept')


def is_equal(context, arguments):
    return context.ltm.relation_evidence('equal', arguments['I1'], arguments['I2'],
        max_records=context.max_records, work=context.work)


def implies(context, arguments):
    # Implication is containment of conceptual regions, with the same two poles.
    return part(context, arguments)


def is_implied(context, arguments):
    link = context.ltm.relation_evidence('implies', arguments['I1'], arguments['I2'],
        max_records=context.max_records, work=context.work)
    premise = context.ltm.premise_evidence(arguments['I1'],
        max_records=context.max_records, work=context.work)
    return pair(link.get('value'), min(link['support_true'], premise['support_true']),
        min(link['support_false'], premise['support_true']),
        witnesses=(*premise.get('witnesses',()), *link.get('witnesses',())),
        frames=(*premise.get('frames',()), *link.get('frames',())),
        evidence_kind='implication')


def is_true(context, arguments):
    references = tuple(ref for _role, ref in context.operand_references
                       if ref and ref[0] == 'ltm')
    if references:
        return context.ltm.end_evidence(references[0], max_records=context.max_records,
                                        work=context.work)
    return context.ltm.truth_evidence(arguments['I1'],
        max_records=context.max_records, work=context.work)


def exist(context, arguments):
    from Queries import _vector
    value = _vector(context, arguments['I1'])
    extent = context.conceptual_space.extent(value)
    return dict(extent, result_kind='concept', evidence_kind='conceptual-presence', witnesses=())


def query(context, arguments):
    references = tuple(ref for _, ref in context.operand_references if ref and ref[0] == 'ltm')
    return context.ltm.best_match(arguments['I1'],
        max_records=context.max_records, work=context.work, references=references)


def ask(context, arguments):
    from ThoughtReferences import needs_episode
    requested = arguments['I1']
    if needs_episode(requested) and callable(context.continuation):
        return returned_subgoal(context.continuation(requested))
    references = tuple(ref for _, ref in context.operand_references if ref and ref[0] == 'ltm')
    found = context.ltm.best_match(requested, max_records=context.max_records,
                                    work=context.work, references=references)
    return found


def returned_subgoal(result):
    """The return half of ask, shared with a suspended episode continuation."""
    if hasattr(result, 'evidence'):
        return dict(result.evidence, value=result, result_kind='subgoal')
    return pair(incomplete=('open_reference',)) if result is None else dict(result)


def negate(context, arguments):
    from Queries import _vector
    from Meaning import ConceptualMeaning
    from ThoughtReferences import bindings, evidence_pair, with_slots, open_slots
    value = arguments['I1']
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        return pair(None, value[1], value[0], evidence_kind='inference')
    if isinstance(value, ConceptualMeaning):
        positive, negative = evidence_pair(value)
        from dataclasses import replace
        roles = (value.roles if context is None else torch.stack([
            context.conceptual_space.negate(role) for role in value.roles]))
        result = with_slots(replace(value, roles=roles), open_slots(value), pair=(negative, positive))
        return pair(result, negative, positive, meaning=result,
                    witnesses=bindings(value).get('_thought_witnesses', ()))
    tensor = _vector(context, value)
    extent = context.conceptual_space.extent(tensor)
    return pair(context.conceptual_space.negate(tensor), extent['support_false'], extent['support_true'],
                result_kind='concept', evidence_kind='inference')


def gain(context, arguments):
    from Queries import _vector
    value = _vector(context, arguments['I1']).detach()
    magnitude = value.norm().clamp(0, 1)
    return pair(value, float(magnitude), gain=float(magnitude),
                evidence_kind='gain', result_kind='concept')
