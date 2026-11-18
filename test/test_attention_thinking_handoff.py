"""A deterministic attention chooser for handoff certificates."""
from types import SimpleNamespace
import torch



def first_legal(*, fields=False):
    def attend(keys, legal, space, **kwargs):
        flat = legal.flatten(1)
        indices = torch.arange(flat.shape[1], device=keys.device)
        priority = -indices + ((indices % 6 >= 3).long()*flat.shape[1] if fields else 0)
        action = priority[None].expand_as(flat).masked_fill(~flat, -10**6).argmax(-1)
        action = torch.where(flat.any(-1), action, flat.shape[1])
        return action, keys.new_ones(len(keys)), flat.sum(-1)>1, dict(
            probability=keys.new_ones(len(keys)), alternative_count=(flat.sum(-1)-1).clamp_min(0))
    return SimpleNamespace(attention_operations=(0, 1, 2, 4) if fields else (0, 1, 2), attend=attend)
