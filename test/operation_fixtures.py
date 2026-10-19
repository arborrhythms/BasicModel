"""Unit-test action injection at the chooser seam; never a replay API."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


@contextmanager
def selected_action(layer, action):
    def choose(logits, **kwargs):
        valid = torch.isfinite(logits).any(-1)
        chosen = action.to(device=logits.device, dtype=torch.long)
        legal = torch.isfinite(logits).gather(1, chosen[:, None]).squeeze(1)
        probabilities = torch.where(valid[:, None], logits, 0.).softmax(-1)
        result = (chosen, probabilities, legal)
        if kwargs.get('return_proposal', False):
            # This fixture forces one action with proposal probability one;
            # the learned policy probabilities above still own its credit.
            mask = kwargs.get('masked_action')
            correction = (logits.new_zeros(len(logits)) if mask is None else
                          (mask >= 0).to(logits))
            return (*result, correction)
        return result
    with patch.object(layer, 'select_logits', choose):
        yield
