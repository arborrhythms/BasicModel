"""Fixed joint-residual choice among detached candidate pairs.

The decomposition teacher retired on 2026-11-04. Form and both meaning
lanes remain in the residual; no objective writes the chooser's weights.
"""
import warnings
import torch
from torch import nn


class DecompositionChooser(nn.Module):
    feature_names = ('negative_pair_residual', 'left_activation', 'right_activation',
                     'left_priming', 'right_priming')

    def __init__(self):
        super().__init__()
        # A device/dtype-following constant, excluded from both parameters and
        # checkpoints. A saved learned preference must never replace it.
        self.register_buffer('weight', torch.tensor([1., 0., 0., 0., 0.]), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'weight'
        if key in state_dict:
            state_dict.pop(key)
            warnings.warn(f'Dropped retired decomposition chooser checkpoint key {key}; '
                          'the fixed joint residual argmin is unchanged.', UserWarning)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, features, allowed):
        logits = (features.detach() * self.weight).sum(-1)
        logits = logits.masked_fill(~allowed, -torch.inf).flatten(1)
        ready = allowed.flatten(1).any(-1)
        logits = torch.where(ready[:, None], logits, torch.zeros_like(logits))
        probabilities = logits.softmax(-1)
        return logits.argmax(-1), probabilities, logits

