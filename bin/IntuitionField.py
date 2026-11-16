"""The live learned map on a complete field, before a thought is taken."""
from dataclasses import replace
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
    """Evaluate once on observation; a numeric result belongs to its bracket."""
    owner = model._concept_owner()
    knowing, rows = observed(owner, field)
    model._intuition_observation = field
    model._intuition_teacher = knowing.detach().clone()
    with torch.no_grad():
        owner._observed_knowing[:len(knowing)].copy_(knowing.detach())
    stack = owner.intuit_stack
    if stack is None:
        model._intuition_prediction = None
        model._intuition_field = replace(field, evidence=torch.zeros_like(field.evidence))
        return field
    prediction = stack(knowing.detach().flatten(1)).reshape_as(knowing)
    model._intuition_prediction = prediction
    if rows.ndim == 1:
        additions = prediction[:, rows.clamp_min(0)].transpose(0, 1)
        additions = additions * (rows >= 0)[:, None, None]
    else:
        additions = prediction.gather(1, rows.T.clamp_min(0)[..., None].expand(-1, -1, 2)).transpose(0, 1)
        additions = additions * (rows >= 0)[..., None]
    locations = field.where[..., 1] > field.where[..., 0]
    if bracket is not None:
        locations = locations & (field.where == bracket[:, None]).all(-1)
    learned = additions[:, :, None] * locations[None, :, :, None]
    model._intuition_field = replace(field, evidence=learned)
    evidence = torch.maximum(field.evidence, learned)
    return replace(field, evidence=evidence)


def read(owner, *, row, work):
    """State the already computed field; neither run the stack nor read LTM."""
    from Understanding import ConceptualField
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots
    from Extension import ConceptSet, Node
    from FieldProjection import key
    work.require('field')
    field = getattr(getattr(owner, '_model', None), '_intuition_field', None)
    if field is None:
        ids = torch.empty(0, dtype=torch.long, device=owner.similarity_codebook.W.device)
        poles = owner.similarity_codebook.W.new_zeros(0, 2)
        value = owner.similarity_codebook.W.new_zeros(owner.outputShape[-1])
    else:
        ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, row]
        valid = ids >= 0
        ids = ids[valid]
        poles = field.evidence[valid, row].amax(1).detach()
        selected = ConceptualField(ids, poles[:, None, None],
            field.where[row:row+1, :1], field.when[row:row+1, :1], None)
        value = key(owner, selected)[0, 0].detach()
    pair = poles.amax(0) if len(poles) else value.new_zeros(2)
    meaning = with_slots(replace(ConceptualMeaning.from_description(value),
        bindings={'_evidence_kind': 'intuition'},
        extension=(Node(ConceptSet(ids, poles)),)), (), pair=pair)
    return dict(value=value, meaning=meaning, support_true=float(pair[0]),
        support_false=float(pair[1]), evidence_kind='intuition',
        learned_field=True, witnesses=(), incomplete=())


def complete(model, present, registry):
    """The kept field at every order teaches the one open-pass prediction."""
    prediction = getattr(model, '_intuition_prediction', None)
    if prediction is None:
        return
    teacher = model._intuition_teacher
    model._intuition_target = teacher.detach()
    error = (prediction - teacher.detach()).square().mean((1, 2))
    registry.error('intuition.field', error, 1., mask=present, category='output')
