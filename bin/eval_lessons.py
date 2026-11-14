"""Lesson readings and declared criteria, independent of training control."""
import math


def fraction(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return None if not values else sum(bool(value) for value in values) / len(values)


def minimum(values):
    values = list(values)
    return None if not values or any(v is None for v in values) else min(values)


def criterion_reading(criterion, readings, *, operations=()):
    """An undeclared bar or unavailable reading can never register an artifact."""
    measure = criterion['measure']
    trust = readings['trust']
    answer = fraction(readings.get('answers', ()), 'text_correct')
    requested = [trust['operations'].get(name) for name in operations]
    definite = minimum(None if row is None or not row['measurable'] else
        minimum(row['definiteness']) for row in requested)
    reason = None
    if measure == 'definition_trust_fraction':
        value = trust['stable_word_fraction'] if trust['admitted_words'] else None
    elif measure == 'denotation_fraction':
        value = trust['denoting_word_fraction'] if trust['named_words'] else None
    elif measure == 'heldout_answers':
        value = answer
    elif measure == 'occupant_definiteness':
        value = definite
    elif measure == 'occupant_definiteness_and_answers':
        value = minimum((definite, answer))
    elif measure == 'relation_groups':
        value = None
        reason = 'Numbered relation groups were retired; per-operation gates are reported separately.'
    else:
        raise ValueError('unknown lesson criterion: ' + measure)
    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('lesson measurement must be a finite fraction')
    threshold = criterion['threshold']
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('lesson threshold must be a finite fraction')
    status = ('undeclared' if threshold is None else 'unavailable' if value is None
              else 'passed' if value >= threshold else 'failed')
    return dict(measure=measure, value=value, threshold=threshold, status=status,
                passed=status == 'passed', reason=reason, operations=list(operations),
                answer_accuracy=answer, occupant_definiteness=definite)


def class_readings(model):
    """Read the native argument geometry, radii and memberships without refresh."""
    import torch
    from ConceptHierarchy import direction
    speech = model.languageSpace.parts_of_speech
    owner = speech.owner
    names = speech.arguments
    vectors = [direction(owner, speech.concepts[name]) for name in names]
    cosines = [] if not vectors else (torch.stack(vectors) @ torch.stack(vectors).T).cpu().tolist()
    return dict(arguments=list(names), radii=speech.radii().detach().cpu().tolist(),
        cosines=cosines, members={name: list(speech.members(name)) for name in names},
        operations={name: speech.operation(name) for name in speech.operations})


def lesson_readings(model, answers):
    """Call on the isolated evaluation model; never advance a gate to measure it."""
    owner = model._concept_owner()
    speech = model.languageSpace.parts_of_speech
    gates = owner._concept_allocator.parts.gates
    measurements = gates.measurements(owner, speech)
    return dict(trust=owner.trust_measurements(), answers=answers,
        classes=class_readings(model),
        gates={name: dict(open=gates.open[name], measurement=measurements[name])
               for name in gates.names})


def require_criteria(results):
    if not results or not all(row['passed'] and row['status'] == 'passed' for row in results):
        raise ValueError('all declared lesson criteria must pass before registration')
