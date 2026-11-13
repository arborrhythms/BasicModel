"""A continuous native modality beside text, sharing basic percept collections."""
from dataclasses import replace
import torch
from NumericPercepts import feeling, number
from Understanding import ConceptualField


def code(owner, value):
    """The felt value in the part and whole towers' own scalar coordinates."""
    basic = owner.basic_percepts
    point = owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
    felt = feeling(torch.tensor(float(value), device=point.device, dtype=torch.float64)).to(point)
    point[basic.identity_width-1] = felt
    point[basic.width-1] = felt
    return point


def bare(owner, points):
    basic = owner.basic_percepts
    a, b = basic.identity_width-1, basic.width-1
    return (points[..., a] > 0) & (points[..., a] == points[..., b]) & (points.ne(0).sum(-1) == 2)


def perceive(owner, value):
    point = code(owner, value)
    if not bool(point.any()):
        return None
    return owner.basic_percepts.perceive_value(point[owner.basic_percepts.identity_width-1])


def concepts(owner):
    from PerceptCollections import definitions
    return {row: edges for row, edges in definitions(owner).items()
            if all(owner.basic_percepts.is_value(percept) for percept, _ in edges)}


def observe(owner, word, value, occurrence, *, stream):
    from PerceptCollections import observe as collect
    percept = perceive(owner, value)
    return None if percept is None else collect(owner, int(word), percept, occurrence, stream=stream)


def commit(model, sid, active):
    """The kept close supplies paired word/percept observations to one writer."""
    present = getattr(model, '_scalar_present', {})
    if not present or sid != 0 or not model.training:
        return
    from PerceptCollections import observe as collect
    owner = model._concept_owner()
    for batch, reading in present.items():
        if not bool(active[batch]):
            continue
        for form in set(model._attention_forms[0][batch]):
            word = owner.interpret.word_for_form(form)
            if word is not None and form:
                collect(owner, word, reading['percept'], reading['occurrence'], stream=batch)


def reading(owner, points):
    """One unplaced reading through the text modality's concept reader."""
    from PerceptField import PerceptField
    basic = owner.basic_percepts
    matrix = owner._concept_allocator.layer(0).features
    rows = sorted({col // 2 for col in matrix._cols})
    B, E = points.shape[:2]
    boundaries = basic.read_points(points, rows)
    native = torch.tensor([min(basic.occurrences[row][0]) < -basic.identity_width for row in rows],
                          dtype=torch.bool, device=points.device)
    boundaries = boundaries * native[None, :, None, None, None]
    values = torch.minimum(boundaries[0], boundaries[1])
    where = torch.zeros(B, E, 2, device=points.device, dtype=torch.long)
    perception = PerceptField(tuple(('bp', row) for row in rows), values,
        values[..., None, :], where, where, where, boundaries=boundaries)
    names = ('_percept_field', '_cs_field_where', '_cs_field_when', '_cs_field_caps',
             '_cs_field_rows', '_cs_field_concept_ids', '_cs_order0_raw')
    previous = {name: owner.__dict__.get(name) for name in names}
    try:
        evidence = owner.cs_read_percept_field(perception)
        return ConceptualField(owner._cs_field_concept_ids, evidence, where, where, None)
    finally:
        for name, value in previous.items():
            if value is None:
                owner.__dict__.pop(name, None)
            else:
                object.__setattr__(owner, name, value)


def reconstruct(owner, *, word=None, value=None):
    from PerceptCollections import objects, code as concept_code
    if value is None:
        known = tuple(cid for cid in objects(owner, word) if owner._csw_row_of(cid) in concepts(owner))
        if len(known) != 1:
            return None
        point = concept_code(owner, owner._csw_row_of(known[0]))
        return float(number(point[owner.basic_percepts.identity_width-1]))
    if value <= 0:
        return ()
    field = reading(owner, code(owner, value)[None, None])
    ids = field.concept_ids
    ids = ids[:, 0] if ids.ndim == 2 else ids
    collections = concepts(owner)
    return tuple(form for cid, evidence in zip(ids.tolist(), field.evidence[:, 0, 0, 0])
                 if owner._csw_row_of(cid) in collections and float(evidence.detach()) >= 1.
                 and (form := owner.interpret.lexical_form(cid)) is not None)


def decode(owner, field, percepts=()):
    collections = concepts(owner)
    result = []
    for batch in range(field.evidence.shape[1]):
        ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, batch]
        evidence = field.evidence[:, batch, :, 0].amax(-1)
        objects = [cid for cid, value in zip(ids.tolist(), evidence)
                   if owner._csw_row_of(cid) in collections and float(value.detach()) >= 1.]
        words = tuple(form for cid in objects if (form := owner.interpret.lexical_form(cid)) is not None)
        percept = percepts[batch] if batch < len(percepts) else None
        if percept is not None:
            felt = owner.basic_percepts.code(percept)[owner.basic_percepts.identity_width-1]
            value = float(number(felt))
        else:
            named = [owner.interpret.reverse(cid) for cid in objects]
            if not named:
                named = [cid for cid, value in zip(ids.tolist(), evidence) if float(value.detach()) > 0
                         and cid in owner.definitions.word_ids]
            candidates = [value for word in named if (value := reconstruct(owner, word=word)) is not None]
            value = candidates[0] if len(set(candidates)) == 1 else None
        result.append(dict(number=value, words=words))
    return tuple(result)


def join(model, sid, field):
    model._scalar_present = {}
    values = getattr(model.inputSpace, '_scalar_values', None)
    if values is None or sid != 0 or all(value is None for value in values):
        return field
    owner = model._concept_owner()
    B = field.evidence.shape[1]
    points = owner.similarity_codebook.W.new_zeros(B, 1, owner.similarity_codebook.W.shape[-1])
    pool = owner.mtm
    for batch, value in enumerate(values):
        percept = None if value is None else perceive(owner, value)
        if percept is None:
            continue
        point = code(owner, value)
        reference = ('bp', percept)
        key = pool.open(reference, kind=None, when=0, value=point, pending=False)
        entry = pool.entries[key]
        pool.entries[key] = replace(entry, refinement=dict(entry.refinement,
            members=((reference, reference),), promoted_percept=percept))
        pool.record_link(key, reference, batch, kind='percept', order=-1)
        model._scalar_present[batch] = dict(candidate=key, occurrence=reference, percept=percept, point=point)
        points[batch, 0] = point
    scalar = reading(owner, points)
    # The modalities share concept identities, but retain separate readings.
    ids = sorted({int(cid) for source in (field, scalar) for cid in source.concept_ids.flatten() if int(cid) >= 0})
    E = field.evidence.shape[2]
    evidence = field.evidence.new_zeros(len(ids), B, E+1, 2)
    for batch in range(B):
        for source, target in ((field, slice(0, E)), (scalar, slice(E, E+1))):
            column = source.concept_ids if source.concept_ids.ndim == 1 else source.concept_ids[:, batch]
            for index, cid in enumerate(column.tolist()):
                if cid >= 0:
                    evidence[ids.index(cid), batch, target] = source.evidence[index, batch]
    return ConceptualField(field.concept_ids.new_tensor(ids), evidence,
        torch.nn.functional.pad(field.where, (0, 0, 0, 1)),
        torch.nn.functional.pad(field.when, (0, 0, 0, 1)), None)
