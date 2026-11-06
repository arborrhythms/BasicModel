"""Bottom-up next-state expectation; all consumers read a frozen prediction."""
from dataclasses import dataclass, replace
import torch


@dataclass(frozen=True)
class Prediction:
    keys: torch.Tensor
    context: torch.Tensor
    knowing: torch.Tensor
    value: torch.Tensor
    rows: torch.Tensor
    active: torch.Tensor
    target: torch.Tensor | None = None
    target_rows: torch.Tensor | None = None


def rule_order(owner, row):
    if getattr(owner, '_serial_concept_path', False):
        return ()
    value = owner._expected_knowing[row].detach().amax(-1)
    return tuple(index for index in value.argsort(descending=True, stable=True).tolist()
                 if bool(value[index] > 0))


def open_prediction(model, present):
    owner = model._concept_owner()
    owner._serial_concept_path = bool(getattr(getattr(model, 'inputSpace', None), '_serial_word_major', False)
                                      or getattr(model, 'word_brackets', False) and getattr(model, 'serial_object_meta', False))
    owner._concept_prediction_open = not owner._serial_concept_path
    if owner._serial_concept_path:
        model._concept_prediction = None
        owner._expected_knowing.zero_()
        return
    B = len(present)
    if B > owner.nStreams:
        raise RuntimeError('concept prediction exceeds constructed streams')
    rows = torch.tensor(sorted(owner._row_to_concept()), device=present.device, dtype=torch.long)
    keys = owner.similarity_codebook.lookup_rows(rows).detach()[None].expand(B, -1, -1).clone()
    knowing = owner._observed_knowing[:B, rows].detach().clone()
    from Expectation import ActivationPredictor
    content = keys * knowing.amax(-1)[..., None]
    context = ActivationPredictor.context_from(content, knowing.any(-1))
    prediction = owner.concept_predictor(keys, context, knowing).sigmoid()
    model._concept_prediction = Prediction(keys, context, knowing,
        prediction.detach().clone(), rows, present.detach().clone())
    with torch.no_grad():
        target = owner._expected_knowing[:B]
        target.copy_(torch.where(present[:, None, None], 0., target))
        target[:, rows] = torch.where(present[:, None, None], prediction.detach(), target[:, rows])


def observe(model, field, addresses, present):
    """The next observed field is a teacher, never a differentiable target."""
    owner = model._concept_owner()
    field = field.detach().amax(2)
    target = torch.zeros_like(owner._observed_knowing[:len(present)])
    valid_rows = torch.zeros_like(target[..., 0], dtype=torch.bool)
    for b in range(len(present)):
        rows = addresses[:, b] if addresses.ndim == 2 else addresses
        valid = rows >= 0
        target[b].scatter_reduce_(0, rows[valid, None].expand(-1, 2),
                                 field[valid, b], reduce='amax', include_self=True)
        valid_rows[b, rows[valid]] = True
    previous = getattr(model, '_concept_prediction', None)
    if previous is not None:
        model._concept_prediction = replace(previous, target=target[:, previous.rows].detach(),
                                            target_rows=valid_rows[:, previous.rows])
    with torch.no_grad():
        owner._observed_knowing[:len(present)].copy_(torch.where(
            present[:, None, None], target, owner._observed_knowing[:len(present)]))


def loss(model):
    prediction = getattr(model, '_concept_prediction', None)
    if prediction is None or prediction.target is None:
        return None
    predictor = model._concept_owner().concept_predictor
    live = predictor(prediction.keys, prediction.context, prediction.knowing).sigmoid()
    frozen = live + (prediction.value - live).detach()
    square = (frozen - prediction.target.detach()).square()
    mask = prediction.target_rows
    error = (square * mask[..., None]).sum((1, 2)) / (2 * mask.sum(-1)).clamp_min(1)
    return torch.where(prediction.active, error, 0.)
