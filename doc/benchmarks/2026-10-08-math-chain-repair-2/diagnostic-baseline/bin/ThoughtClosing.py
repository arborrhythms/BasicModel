"""Thinking at committed closings; source text never selects an executor."""
from dataclasses import replace
import torch

from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, evidence_pair, fill, from_closing, open_slots, question, with_slots


def closing_question(meaning, clause, parsed=None, program=None):
    """Preserve a binder's nulls, including a lexical idea with no referent.

    Numerical ideas are already their own conceptual contents. Surface mode
    is evidence only: a known pair and all bound roles close even a wh-row.
    """
    if not open_slots(clause.meaning):
        # A headed outer phrase does not close a reference left in one of
        # its completed constituents. Ask on that existing inner row.
        for child in clause.children:
            nested = closing_question(child.meaning, child)
            if nested is not None:
                return nested
    source = parsed if parsed is not None else meaning
    if (parsed is not None and torch.equal(parsed.roles, meaning.roles)
            and torch.equal(parsed.role_mask, meaning.role_mask)):
        source = replace(source, role_refs=meaning.role_refs, constituents=meaning.constituents)
    surface_open = source.mode == 'interrogative'
    slots = list(dict.fromkeys((*open_slots(source), *open_slots(clause.meaning))))
    clause_open = open_slots(clause.meaning)
    if clause_open:
        refs = list(source.role_refs)
        for kind, role in clause_open:
            if kind != 'evidence':
                refs[role] = None
        data = bindings(source)
        data['_forward_references'] = bindings(clause.meaning).get('_forward_references', ())
        source = replace(source, role_refs=tuple(refs), bindings=data)
    forms = () if program is None else (program.lexical_forms or ())
    lexical = any(isinstance(word, (str, bytes)) and word for word in forms)
    if lexical and clause.relation is None and clause.refs[0] in (-1, 0):
        source = replace(source, role_refs=(None, *source.role_refs[1:]))
        slots.append(('referent', 0))
    source = with_slots(source, slots, pair=clause.evidence)
    if surface_open and tuple(clause.evidence) == (0., 0.):
        source = question(source, slots or (('evidence', -1),))
    result = from_closing(source, evidence=clause.evidence)
    if (parsed is not None and open_slots(parsed) == (('evidence', -1),)
            and evidence_pair(parsed) == (0., 0.) and tuple(clause.evidence) == (0., 0.)):
        result = question(result, (('evidence', -1),))
    return result if open_slots(result) else None


def _later_binding(model, fields, *, sentence, ready):
    """Later matching content can fill a held reference within its document."""
    from Occurrence import source_at
    from ThoughtStream import write
    pending = model.__dict__.setdefault('_open_thought_rows', {})
    for row in ready:
        field = fields[row]
        document = source_at(model, row, sentence)[0]
        remaining = []
        for owner_document, source, occurrence in pending.pop(row, ()):
            if owner_document != document:
                continue  # its durable question is already stored
            store = model.symbolSpace.ltm_store
            held = store._index_occurrences.get(occurrence)
            arrived = None if held is None else store.meaning_of(held)
            if arrived is not None and not open_slots(arrived):
                write(model, arrived, row=row)
                continue
            if field.query is not None:
                remaining.append((owner_document, source, occurrence))
                continue
            # Only the new occurrence is offered, by code identity; there is
            # no scan or re-execution of a thought episode on later text.
            roles = [role for kind, role in open_slots(source) if kind != 'evidence']
            compared = roles or [role for role in range(3) if bool(source.role_mask[role])]
            if any(not bool(field.meaning.role_mask[role]) for role in compared):
                remaining.append((owner_document, source, occurrence))
                continue
            exact = all(torch.allclose(source.roles[role], field.meaning.roles[role]) for role in compared)
            if not exact or field.row_id in (-1, 0) or tuple(field.evidence) == (0., 0.):
                remaining.append((owner_document, source, occurrence))
                continue
            store = model.symbolSpace.ltm_store
            witness = store.occurrence_of(store.index_of_row(field.row_id))
            refs = list(field.meaning.role_refs)
            for role in roles:
                refs[role] = refs[role] or witness
            supplied = replace(field.meaning, role_refs=tuple(refs))
            resolved = fill(source, dict(meaning=supplied, support_true=field.evidence[0],
                support_false=field.evidence[1]), witnesses=(occurrence, witness), operation='bind')
            write(model, resolved, row=row)
        if remaining:
            pending[row] = remaining


def absence(model, row, observed):
    """Cancel the expected direction against observation before negating it."""
    from ThoughtFaces import negate
    from ThoughtStream import write
    disc = getattr(getattr(model, 'symbolSpace', None), 'expectation', None)
    image = getattr(model, '_closing_images', {}).get(row)
    if disc is None or image is None or not image.concept_width:
        return None
    comparison = disc.last_expectation_comparison(row)
    prior = None if comparison is None else comparison.estimate
    if prior is None:
        return None
    start = image.image.shape[-1]-image.concept_width
    expected = -image.image[:, start:]
    norm = expected.square().sum(-1)
    # Negative projection in the shifted origin is the uncancelled image.
    residual = -(image.conceived[:, start:]*expected).sum(-1)
    remaining = torch.where(norm > 0, (residual/norm.clamp_min(1e-12)).clamp(0, 1), 0.)
    confidence = (remaining * prior.presence_logits.sigmoid()
                  * float(getattr(model, 'expectation_gain', 1.))).clamp(0, 1)
    if not bool((confidence > 0).any()):
        return None
    meaning = ConceptualMeaning(prior.roles.detach(), confidence > 0,
        sentence_kind=observed.sentence_kind)
    meaning = with_slots(meaning, (), pair=(float(confidence.max()), 0.))
    result = negate(None, {'I1': meaning})
    value = replace(result['meaning'], polarity=False)
    data = bindings(value)
    data['_producing_operation'] = 'not'
    data['_thought_witnesses'] = tuple(getattr(comparison, 'source_occurrences', ()))
    value = replace(value, bindings=data)
    return write(model, value, row=row)


def close(model, fields, *, sentence, ready, score=None):
    """Exactly one root episode per open closing, then the presented binding."""
    from ThoughtCredit import observe
    observe(model, [None if field is None else field.meaning for field in fields], sentence=sentence)
    _later_binding(model, fields, sentence=sentence, ready=ready)
    results = []
    for row in ready:
        field = fields[row]
        if field.query is None:
            absence(model, row, field.meaning)
            continue
        result = model.run_selected_thought(field.query, row=row,
            work_budget=getattr(model, 'attention_budget', 32),
            score=None if score is None else lambda result,row=row:score(row,result))
        results.append((row, result))
        fields[row] = replace(field, thought_completed=True)
        if not open_slots(result.meaning):
            # The answer is the binding; retain the original source field's
            # address separately from the newly concluded inference.
            if int(result.meaning.role_mask.sum()) in (1, 3):
                fields[row] = replace(fields[row], meaning=result.meaning, query=None,
                                      evidence=evidence_pair(result.meaning))
        model._end_finished_selected_thought_episodes()
    model._sentence_fields[sentence] = tuple(fields)
    model._last_closing_thoughts = tuple(results)
    return tuple(results)
