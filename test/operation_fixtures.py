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
        return chosen, probabilities, legal
    with patch.object(layer, 'select_logits', choose):
        yield
