"""Omission-trained expectation; consumers read the detached open prediction."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Prediction:
    value: torch.Tensor
    rows: torch.Tensor
    active: torch.Tensor
    target: torch.Tensor | None = None


def rule_order(owner, row):
    value = owner._expected_knowing[row].detach().amax(-1)
    return tuple(value.argsort(descending=True, stable=True).tolist())


def expected(owner, knowing):
    """Evaluate the existing omission-trained map without publishing a prediction."""
    B, width = knowing.shape[:2]
    if owner.concept_predictor is not None:
        from Expectation import ActivationPredictor
        rows = torch.arange(width, device=knowing.device)
        keys = owner.similarity_codebook.lookup_rows(rows).detach()[None].expand(B, -1, -1).clone()
        context = ActivationPredictor.context_from(keys * knowing.amax(-1)[..., None], knowing.any(-1))
        return owner.concept_predictor(keys, context, knowing).sigmoid()
    if owner.expect_stack is not None:
        # The stack reads each concept's two lanes and gives one signed
        # number for each, held again as its two lanes.
        from Layers import signed_lanes
        return signed_lanes(owner.expect_stack(knowing.transpose(-1, -2)))
    return torch.zeros_like(knowing)


def open_prediction(model, present, *, sid=None):
    owner = model._concept_owner()
    owner._serial_concept_path = bool(getattr(getattr(model, 'inputSpace', None), '_serial_word_major', False)
                                      or getattr(model, 'word_brackets', False) and getattr(model, 'serial_object_meta', False))
    owner._concept_prediction_open = True
    B = len(present)
    if B > owner.nStreams:
        raise RuntimeError('concept prediction exceeds constructed streams')
    rows = torch.arange(owner._order_caps()[0], device=present.device)
    knowing = owner._observed_knowing[:B, rows].detach().clone()
    from IdeaRead import open_read
    open_read(model, knowing)
    prediction = expected(owner, knowing)
    from Omission import concept_teacher
    target, hidden = concept_teacher(model, sid=sid, present=present)
    model._concept_prediction = Prediction(prediction, rows, present & hidden, target)
    with torch.no_grad():
        stored = owner._expected_knowing[:B]
        stored.copy_(torch.where(present[:, None, None], 0., stored))
        stored[:, rows] = torch.where(present[:, None, None], prediction.detach(), stored[:, rows])


def loss(model):
    prediction = getattr(model, '_concept_prediction', None)
    if prediction is None or prediction.target is None or not bool(prediction.active.any()):
        return None
    error = (prediction.value - prediction.target.detach()).square().mean((1, 2))
    return torch.where(prediction.active, error, 0.)


def complete(model, present, registry):
    """Expectation learns the full input, with omission as its only teacher."""
    value = loss(model)
    if value is not None:
        registry.error('expectation.concepts', value, 1., mask=present, category='expectation')
