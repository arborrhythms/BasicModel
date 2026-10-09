"""Thinking at committed closings; source text never selects an executor."""
from dataclasses import replace
import torch

from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, evidence_pair, fill, from_closing, needs_episode, open_slots, question, with_slots


def closing_question(meaning, clause, parsed=None, program=None, *, store=None, written_rows=None):
    """Preserve a binder's nulls, including a lexical idea with no referent.

    Numerical ideas are already their own conceptual contents. Surface mode
    is evidence only: a known pair and all bound roles close even a wh-row.
    """
    # The numerical journal has not installed its canonical role references.
    # Its selected native/child refs, rather than those temporary nulls,
    # determine whether the operation actually left an operand unfilled.
    index = None if written_rows is None else written_rows.get(id(clause))
    committed = None if store is None or index is None else store.meaning_of(index)
    named = committed if committed is not None else replace(clause.meaning, role_refs=tuple(
        ('selected', ref.identity if hasattr(ref, 'identity') else ref)
        if bool(clause.meaning.role_mask[role]) and ref not in (-1, 0) else None
        for role, ref in enumerate(clause.refs)))
    clause_open = open_slots(named)
    if not clause_open:
        # A headed outer phrase does not close a reference left in one of
        # its completed constituents. Ask on that existing inner row.
        for child in clause.children:
            nested = closing_question(child.meaning, child, store=store, written_rows=written_rows)
            if nested is not None:
                return nested
    source = parsed if parsed is not None else meaning
    if committed is not None:
        source = committed
    if (committed is None and parsed is not None and torch.equal(parsed.roles, meaning.roles)
            and torch.equal(parsed.role_mask, meaning.role_mask)):
        source = replace(source, role_refs=meaning.role_refs, constituents=meaning.constituents)
    surface_open = source.mode == 'interrogative'
    slots = list(dict.fromkeys((*open_slots(source), *clause_open)))
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
    if lexical and committed is None and clause.relation is None and clause.refs[0] in (-1, 0):
        source = replace(source, role_refs=(None, *source.role_refs[1:]))
        slots.append(('referent', 0))
    pair = evidence_pair(committed) if committed is not None else clause.evidence
    source = with_slots(source, slots, pair=pair)
    if surface_open and tuple(pair) == (0., 0.):
        source = question(source, slots)
    result = from_closing(source, evidence=pair)
    if committed is not None:
        data = bindings(result)
        data['_query_occurrence'] = store.occurrence_of(index)
        data['_source_evidence'] = tuple(clause.evidence)
        result = replace(result, bindings=data)
    return result if needs_episode(result) else None


def absence(model, row, observed, image=None):
    """Cancel the expected direction against observation before negating it."""
    from ThoughtFaces import negate
    from ThoughtStream import write
    disc = getattr(getattr(model, 'symbolSpace', None), 'expectation', None)
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


def close(model, fields, *, sentence, ready, score=None, images=None):
    """Exactly one root episode per open closing, then the presented binding."""
    results = []
    for row in ready:
        field = fields[row]
        if field.query is None:
            absence(model, row, field.meaning, None if images is None else images.get(row))
            continue
        previous = getattr(model, '_thought_image', None)
        model._thought_image = None if images is None else images.get(row)
        try:
            result = model.run_selected_thought(field.query, row=row,
                work_budget=getattr(model, 'attention_budget', 32),
                score=None if score is None else lambda result,row=row:score(row,result))
        finally:
            if previous is None:
                del model._thought_image
            else:
                model._thought_image = previous
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
