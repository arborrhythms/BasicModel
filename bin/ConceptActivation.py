"""Read the two activations of a signless conceptual code."""
import torch


def poles(value, code, form_width):
    form = code[..., :form_width]
    presence = (value[..., :form_width] * form).sum(-1) / form.square().sum(-1).clamp_min(1e-12)
    positive = torch.stack((presence.clamp_min(0), (-presence).clamp_min(0)), -1)
    if form_width == code.shape[-1]:
        return positive
    width = (code.shape[-1] - form_width) // 2
    if width <= 0 or form_width + 2 * width != code.shape[-1]:
        raise ValueError('concept activation requires two equal meaning blocks')
    meaning = code[..., form_width:form_width + width]
    norm = meaning.square().sum(-1)
    pair = torch.stack(tuple((block * meaning).sum(-1) / norm.clamp_min(1e-12)
                            for block in value[..., form_width:].split(width, -1)), -1).clamp_min(0)
    return torch.where((norm > 0)[..., None], pair, positive)
