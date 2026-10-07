"""Reconstruction's supervised choice among detached candidate pairs."""
import torch
from torch import nn
from torch.nn import functional as F


class DecompositionChooser(nn.Module):
    feature_names = ('negative_relative_residual', 'left_activation', 'right_activation',
                     'left_priming', 'right_priming')

    def __init__(self):
        super().__init__()
        # No random initialization: preserve the old residual argmin exactly.
        self.weight = nn.Parameter(torch.tensor([1., 0., 0., 0., 0.]))

    def forward(self, features, allowed):
        logits = (features.detach() * self.weight).sum(-1)
        logits = logits.masked_fill(~allowed, -torch.inf).flatten(1)
        ready = allowed.flatten(1).any(-1)
        logits = torch.where(ready[:, None], logits, torch.zeros_like(logits))
        probabilities = logits.softmax(-1)
        return logits.argmax(-1), probabilities, logits

    @staticmethod
    def teacher_loss(details, left_rows, right_rows, targets):
        """Absent identities contribute no CE; a coincident code is not an ID."""
        left = left_rows.gather(1, details['left_indices'])
        right = right_rows.gather(1, details['right_indices'])
        matches = ((left[:, :, None] == targets[:, 0, None, None])
                   & (right[:, None, :] == targets[:, 1, None, None])
                   & (targets >= 0).all(-1)[:, None, None]
                   & details['allowed']).flatten(1)
        present = matches.any(-1)
        target = matches.long().argmax(-1)
        selected_true = matches.gather(1, details['selected'][:, None]).squeeze(1)
        loss = F.cross_entropy(details['logits'], target, reduction='none')
        loss = torch.where(present, loss, torch.zeros_like(loss))
        return loss, present, selected_true
