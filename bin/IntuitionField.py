"""Observe the open field; intuit deposits its own map's intuited field and
closes definitions."""
import torch


def observed(owner, field):
    """Place the field's independent lanes at their fixed inventory addresses."""
    from SentenceField import inventory
    rows = inventory(owner, field)
    evidence = field.evidence.amax(2)
    knowing = evidence.new_zeros(evidence.shape[1], owner.nVectors, 2)
    if rows.ndim == 1:
        valid = rows >= 0
        knowing[:, rows[valid]] = evidence[valid].transpose(0, 1)
    else:
        for batch in range(evidence.shape[1]):
            valid = rows[:, batch] >= 0
            knowing[batch, rows[valid, batch]] = evidence[valid, batch]
    return knowing, rows


def open_field(model, field, *, bracket=None):
    """Capture the observed field once, before expect reads its basic block."""
    owner = model._concept_owner()
    knowing, rows = observed(owner, field)
    model._intuition_bracket = bracket
    with torch.no_grad():
        owner._observed_knowing[:len(knowing)].copy_(knowing.detach())
    return field


def intuited(owner, observed):
    """The intuited field ``[B, W0, 2]`` of the observed concepts of order
    zero ``observed`` ``[B, W0, 2]``: intuit's own map (a free Pi, the Not and
    a Sigma), one signed number for each concept held as its two lanes."""
    from Layers import signed_lanes
    if owner.intuit_map is None:
        return torch.zeros_like(observed)
    return signed_lanes(owner.intuit_map(observed.transpose(-1, -2)))


def read(owner, *, row, work):
    """Deposit the intuited field in the open basic field, marked as
    intuition, and propagate; no operand."""
    from SentenceField import knowing, publish
    from FieldProjection import close, inventory
    from Understanding import ConceptualField
    work.require('field')
    model = getattr(owner, '_model', None)
    if model is None:
        return
    if not getattr(owner, '_concept_prediction_open', False):
        return
    width = owner._order_caps()[0]
    prediction = intuited(owner, owner._observed_knowing[row:row + 1, :width].detach())
    model._intuited = dict(model.__dict__.get('_intuited') or {})
    model._intuited[row] = prediction
    field = knowing(model)
    rows = inventory(owner)
    ids = field.concept_ids.new_tensor([owner.concept_id_at_row(r) for r in rows])
    lookup = {int(cid): index for index, cid in enumerate(ids)}
    evidence = field.evidence.new_zeros(len(rows), *field.evidence.shape[1:])
    # Preserve every existing order and stream; only this stream's basic
    # block receives the new independent evidence.
    for stream in range(evidence.shape[1]):
        current = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:,stream]
        for index, cid in enumerate(current.tolist()):
            if cid in lookup:
                evidence[lookup[cid],stream] = field.evidence[index,stream]
    locations = field.where[row,:,1] > field.where[row,:,0]
    bracket = getattr(model, '_intuition_bracket', None)
    if bracket is not None:
        locations &= (field.where[row] == bracket[row]).all(-1)
    for index, address in enumerate(rows):
        if address < prediction.shape[1]:
            addition = prediction[0,address][None] * locations[:,None]
            evidence[index,row] = torch.maximum(evidence[index,row], addition)
    # Closure writes only the active stream; another stream's field is held.
    from CandidateGrowth import row_scope
    with row_scope(owner, row):
        evidence[:,row:row+1] = close(owner, rows, evidence[:,row:row+1])
    publish(model, ConceptualField(ids, evidence, field.where, field.when, field.percepts))


def complete(model, present, registry):
    """intuit's map is taught by the field that thought leaves: what it
    intuited from the observed concepts against the concepts of order zero
    that thought left, detached. The output's cost reaches it through what
    it deposited."""
    owner = model._concept_owner()
    called = model.__dict__.pop('_intuited', None)
    if owner.intuit_map is None or not called:
        return
    from SentenceField import knowing
    left, _ = observed(owner, knowing(model))
    width = owner._order_caps()[0]
    error = present.new_zeros(len(present), dtype=torch.float)
    for row, prediction in called.items():
        if row < len(present):
            target = left[row:row + 1, :width].detach().to(prediction)
            error = error.index_put((torch.tensor([row], device=error.device),),
                                    (prediction - target).square().mean((1, 2)).to(error))
    rows = torch.tensor(sorted(r for r in called if r < len(present)), dtype=torch.long, device=present.device)
    active = torch.zeros_like(present).index_fill(0, rows, True) & present
    registry.error('intuition.field', error, 1., mask=active, category='output')
