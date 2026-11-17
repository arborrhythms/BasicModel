"""Fixed, contiguous stacks of the shared PiLayer and SigmaLayer."""
import torch
from torch import nn

from Layers import PiLayer, SigmaLayer


def _field_value(value):
    # Preserve the inward derivative at zero, the last stage's initial value.
    return torch.where(value < 0, 0., torch.where(value > 1, 1., value))


class SigmaPiStack(nn.Module):
    """Alec's layer count counts Pi/Sigma pairs, with separate evidence lanes.

    Pi keeps its native bounded nonlinear map and signed weights. Sigma
    uses its native linear map: its optional atanh entry chart would cancel
    Pi's tanh and remove the hidden nonlinearity. Each additive result is
    clipped into the field's [0,1] range before another pair or a consumer.
    The grammar layers and their parameter laws are unchanged.
    """
    def __init__(self, inputs, outputs, *, hidden=64, pairs=1, seed=0):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (inputs, outputs, hidden, pairs)):
            raise ValueError('stack inputs, outputs, hidden width and pair count must be positive integers')
        self.inputs, self.outputs, self.hidden, self.pairs = inputs, outputs, hidden, pairs
        widths = [inputs, *([hidden] * (2*pairs-1)), outputs]
        self.stages = nn.ModuleList()
        # The native constructors draw CPU weights. Isolate their stream;
        # do not seed or advance any accelerator's generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for index, (source, target) in enumerate(zip(widths, widths[1:])):
                stage = (PiLayer(source, target, nonlinear=True)
                         if index % 2 == 0 else SigmaLayer(source, target, nonlinear=False))
                if index == 2*pairs-1:
                    with torch.no_grad():
                        stage.layer.W.zero_()
                self.stages.append(stage)

    def forward(self, value):
        if value.shape[-1] != self.inputs:
            raise ValueError('stack input differs from its constructed width')
        value = _field_value(value)
        for stage in self.stages:
            # Native layers expose the same arithmetic without their
            # diagnostic activation write, for recurrent compiled bodies.
            value = stage.compute_forward(value)
            if isinstance(stage, SigmaLayer):
                value = _field_value(value)
        return value
