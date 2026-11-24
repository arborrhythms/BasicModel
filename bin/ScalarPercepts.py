"""A continuous native modality beside text, sharing basic percept collections."""
from dataclasses import replace
import torch
from NumericPercepts import feeling, number
from Understanding import ConceptualField


def _felt(owner, value):
    point = owner.similarity_codebook.W
    dtype = torch.float32 if point.device.type == 'mps' else torch.float64
    return feeling(torch.tensor(float(value), device=point.device, dtype=dtype))


def code(owner, value):
    """A value's code (rule 55): in each half the bins of the felt value at
    or below it. Zero, felt as nothing, has no code."""
    basic = owner.basic_percepts
    point = owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
    felt = _felt(owner, value)
    if float(felt) > 0:
        point[:basic.identity_width] = basic.value_code(felt, basic.identity_width)
        point[basic.identity_width:basic.width] = basic.value_code(felt, basic.property_width)
    return point


def bare(owner, points):
    """Points that are a value's code and nothing else."""
    basic = owner.basic_percepts
    registry = getattr(owner, 'where_registry', None)
    if registry is None or 'scalar' not in registry.slices:
        # No scalar place: no point is a value's code.
        return torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
    return (basic.bin_of(points[..., :basic.width]) >= 0) & ~points[..., basic.width:].ne(0).any(-1)


def perceive(owner, value):
    felt = _felt(owner, value)
    if not float(felt) > 0:
        return None
    # The value beside the words is promoted at first sight.
    return owner.basic_percepts.promote(owner.basic_percepts.perceive_value(felt.to(owner.basic_percepts.value_codes)))


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
    """One scalar-place reading: each percept collection read at the points
    by its boundaries, the reader that ``FieldProjection.stimulate`` uses
    (``PerceptCollections.read``)."""
    from PerceptCollections import read
    B, E = points.shape[:2]
    collections = read(owner, points)
    rows = sorted(collections)
    ids = torch.tensor([owner.concept_id_at_row(row) for row in rows], device=points.device, dtype=torch.long)
    evidence = (torch.stack([collections[row] for row in rows]) if rows
                else points.new_zeros(0, B, E, 2))
    where = owner.where_registry.intervals('scalar', torch.zeros(B, E, device=points.device, dtype=torch.long))
    return ConceptualField(ids, evidence, where, torch.zeros_like(where), None)


def holds(evidence):
    """A collection holds a value where it reads it for more than against:
    the boundary is 0 or 1 (rule 55), so no threshold is needed."""
    return evidence[..., 0] > evidence[..., 1]


def reconstruct(owner, *, word=None, value=None):
    from PerceptCollections import objects, definitions
    if value is None:
        collections = concepts(owner)
        known = tuple(cid for cid in objects(owner, word) if owner._concept_row_of(cid) in collections)
        if len(known) != 1:
            return None
        # The way back: a collection's number is its lower bound, the least
        # of its members' values, read from the table of values.
        members = [percept for percept, _ in definitions(owner)[owner._concept_row_of(known[0])]]
        return float(number(owner.basic_percepts.felt(members).amin()))
    if value <= 0:
        return ()
    field = reading(owner, code(owner, value)[None, None])
    collections = concepts(owner)
    return tuple(form for cid, evidence in zip(field.concept_ids.tolist(), field.evidence[:, 0, 0])
                 if owner._concept_row_of(cid) in collections and bool(holds(evidence.detach()))
                 for form in named_forms(owner, cid))


def named_forms(owner, cid):
    """The forms of the words that name a collection: those tied to it at
    the definition threshold or above (:func:`PerceptCollections.words`)."""
    from PerceptCollections import words
    return tuple(form for word in words(owner, cid)
                 if (form := owner.interpret.lexical_form(word)) is not None)


def _scalar_readings(owner, field, batch):
    """The readings of ``field`` at the scalar place."""
    start, end = owner.where_registry.slices['scalar']
    where = field.where[batch]
    return (where[..., 0] >= start) & (where[..., 1] <= end) & (where[..., 1] > where[..., 0])


def decode(owner, field, percepts=()):
    collections = concepts(owner)
    result = []
    for batch in range(field.evidence.shape[1]):
        ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, batch]
        lanes = field.evidence[:, batch].detach()
        scalar = _scalar_readings(owner, field, batch).to(lanes.device)
        held = (holds(lanes) & scalar).any(-1)
        objects = [cid for cid, value in zip(ids.tolist(), held)
                   if owner._concept_row_of(cid) in collections and bool(value)]
        words = tuple(form for cid in objects for form in named_forms(owner, cid))
        percept = percepts[batch] if batch < len(percepts) else None
        if percept is not None:
            value = float(number(owner.basic_percepts.felt([percept])[0]))
        else:
            named = [owner.interpret.reverse(cid) for cid in objects]
            if not named:
                named = [cid for cid, value in zip(ids.tolist(), lanes[..., 0].amax(-1)) if float(value) > 0
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
        pool._release_id(entry)
        pool.entries[key] = replace(entry, transient=None, refinement=dict(entry.refinement,
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
        torch.cat((field.where, scalar.where), 1),
        torch.cat((field.when, scalar.when), 1), None)
