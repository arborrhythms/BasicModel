"""Give Claude's stack simulator the whole STM, in chronological order."""
import torch


def full_choice(state, choice, names):
    buffer, depth = state[:2]
    batch, capacity, width = buffer.shape
    slots = torch.arange(capacity, device=buffer.device)[None]
    source = (depth[:, None] - 1 - slots).clamp(0, capacity - 1)
    values = buffer.gather(1, source[..., None].expand(batch, capacity, width))
    values = torch.where((slots < depth[:, None])[..., None], values, 0.)
    position = torch.where(choice.kind == 1, depth - 2, depth - 1 - choice.position)
    return dict(x=values.detach().double().clone(), depth=depth.detach().tolist(),
        kind=choice.kind.detach().tolist(), op=choice.local_op.detach().tolist(),
        pos=position.detach().tolist(), valid=choice.valid.detach().tolist(), names=names)
