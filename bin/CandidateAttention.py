"""A shared MLP scores supported candidates; selection is a discrete action."""
import torch
from torch import nn
from Layers import Layer


class CandidateAttention(Layer):
    def __init__(self, context_width, candidate_width, *, hidden_width=64, hidden_layers=1):
        for value in (context_width, candidate_width, hidden_width, hidden_layers):
            if type(value) is not int or value < 1:
                raise ValueError('candidate scorer dimensions must be positive integers')
        super().__init__(context_width + candidate_width, 1)
        self.context_width, self.candidate_width = context_width, candidate_width
        widths = [self.nInput] + [hidden_width] * hidden_layers
        self.hidden = nn.Sequential(*(module for left, right in zip(widths, widths[1:])
                                     for module in (nn.Linear(left, right), nn.Tanh())))
        self.readout = nn.Linear(hidden_width, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def migrate_expectation_checkpoint(self, state_dict, prefix=''):
        spec = getattr(self, '_expectation_layout', None)
        key = prefix+'hidden.0.weight'
        old = state_dict.get(key)
        if spec is None or old is None or old.shape == self.hidden[0].weight.shape:
            return False
        at, count, removed_at, removed_count = spec
        if old.shape != (self.hidden[0].weight.shape[0], self.hidden[0].weight.shape[1]-count+removed_count):
            return False
        state_dict[key] = torch.cat((old[:, :at], old.new_zeros(len(old), count),
            old[:, at:removed_at], old[:, removed_at+removed_count:]), -1)
        return True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.migrate_expectation_checkpoint(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, context, candidates):
        if (context.ndim != 2 or candidates.ndim != 3
                or context.shape != (candidates.shape[0], self.context_width)
                or candidates.shape[-1] != self.candidate_width):
            raise ValueError('candidate scorer requires aligned context and candidate features')
        # A shared field can contain tens of thousands of occupied features.
        # Its size must not drive every tanh unit to the same saturated value
        # before the candidate contributes. Bound this block's length while
        # retaining its direction and leaving candidate differences intact.
        context = context / context.norm(dim=-1, keepdim=True).clamp_min(1.)
        inputs = torch.cat((context[:, None].expand(-1, candidates.shape[1], -1), candidates), -1)
        # Scoring is a reader. Named-part lessons and paired costs train it;
        # neither keys nor needs are extra writers of their source owners.
        return self.readout(self.hidden(inputs.detach())).squeeze(-1)
