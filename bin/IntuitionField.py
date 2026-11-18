"""Observe the open field; intuit deposits expect and closes definitions."""
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


def read(owner, *, row, work):
    """Deposit the prediction in the open basic field and propagate; no operand."""
    from SentenceField import knowing, publish
    from FieldProjection import close, inventory
    from Understanding import ConceptualField
    work.require('field')
    model = getattr(owner, '_model', None)
    if model is None:
        return
    prediction = owner._expected_knowing[:, :owner._order_caps()[0]]
    if not getattr(owner, '_concept_prediction_open', False):
        return
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
            addition = prediction[row,address].detach()[None] * locations[:,None]
            evidence[index,row] = torch.maximum(evidence[index,row], addition)
    # Closure writes only the active stream; another stream's field is held.
    evidence[:,row:row+1] = close(owner, rows, evidence[:,row:row+1])
    publish(model, ConceptualField(ids, evidence, field.where, field.when, field.percepts))
