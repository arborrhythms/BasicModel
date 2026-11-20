"""A fixed, contiguous row of fused Pi and Sigma pairs."""
import torch
from torch import nn

from Layers import SigmaPiLayer


class SigmaPiStack(nn.Module):
    """``pairs`` fused :class:`SigmaPiLayer` pairs, each a Pi and a Sigma.

    The stack reads concepts as their two lanes, ``[..., lane, i]`` with
    lane 0 for and lane 1 against; a signed input ``x`` is given as
    ``torch.stack((x, -x), -2)``. Every pair gives one signed number in
    ``[-1, 1]`` for each of its outputs, which the next pair reads as its
    lanes. The last Sigma is zero at construction, so an untaught stack
    gives zero.

    ``few``: each row of the first Pi starts with weight on ``few`` inputs
    drawn at random on each lane, and none on the rest; by default a row
    starts with a small weight on every input.
    """
    def __init__(self, inputs, outputs, *, hidden=64, pairs=1, seed=0, few=None):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (inputs, outputs, hidden, pairs)):
            raise ValueError('stack inputs, outputs, hidden width and pair count must be positive integers')
        self.inputs, self.outputs, self.hidden, self.pairs = inputs, outputs, hidden, pairs
        widths = [inputs, *([hidden] * (2*pairs-1)), outputs]
        self.stages = nn.ModuleList()
        # The constructors draw CPU weights. Isolate their stream;
        # do not seed or advance any accelerator's generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for index in range(pairs):
                source, middle, target = widths[2*index:2*index+3]
                self.stages.append(SigmaPiLayer(source, middle, target, zero=index == pairs-1))
            if few is not None:
                raw = self.stages[0].pi.raw
                with torch.no_grad():
                    count = min(int(few), inputs)
                    keep = torch.rand(raw.shape[0], hidden, inputs).argsort(-1)[..., :count]
                    start = torch.zeros_like(raw).transpose(-1, -2)
                    start.scatter_(-1, keep, torch.rand(raw.shape[0], hidden, count) / count)
                    raw.copy_(start.transpose(-1, -2))

    def forward(self, lanes):
        if lanes.shape[-2:] != (2, self.inputs):
            raise ValueError('stack input differs from its constructed width')
        value = self.stages[0](lanes)
        for stage in self.stages[1:]:
            value = stage(torch.stack((value, -value), -2))
        return value
