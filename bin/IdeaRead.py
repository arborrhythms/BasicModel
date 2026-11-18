"""A kept sentence's idea teaches an independent read of its open field."""
import torch


def open_read(model, observed):
    owner = model._concept_owner()
    if getattr(owner, 'idea_stack', None) is not None:
        # The input is fixed at the sentence's open. Later thoughts and
        # definition closure cannot change the teacher's input.
        model._idea_observation = observed[:, :owner._order_caps()[0]].detach().clone()


def complete(model, record, present, registry):
    owner = model._concept_owner()
    if getattr(owner, 'idea_stack', None) is None:
        return
    prediction = owner.idea_stack(model._idea_observation.flatten(1))
    target = record.meaning_root.detach()
    if prediction.shape != target.shape:
        raise ValueError('field-to-idea read and kept semantic root have different shapes')
    error = (prediction - target).square().mean(-1)
    registry.error('expectation.idea', error, 1., mask=present, category='expectation')


def expected_point(owner, *, streams):
    """Read expect's detached basic field on demand; no production consumer."""
    if owner.idea_stack is None:
        raise ValueError('a model without sentence comprehension has no learned idea read')
    field = owner._expected_knowing[:streams, :owner._order_caps()[0]].detach()
    return owner.idea_stack(field.flatten(1))
