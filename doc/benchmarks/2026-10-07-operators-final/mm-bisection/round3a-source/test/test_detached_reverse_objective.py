"""Gradient-boundary tests for stable construction/reconstruction training."""

import torch

from Language import (
    OperationSelectionLayer,
    ReconstructionStack,
)


class _UnaryScale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return torch.tanh(x * self.scale)


class _BinaryMix(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, left, right):
        return torch.tanh(self.scale * left + (1.0 - self.scale) * right)








def test_fixed_forward_loss_slab_preserves_chooser_gradient():
    stack = ReconstructionStack(batch=1, max_depth=8)
    stack.prepare_choices(
        1, 4, device="cpu", unary_rule_ids=(1, 2),
        binary_rule_ids=(3, 4))
    parameter = torch.nn.Parameter(torch.tensor(0.25, device="cpu"))
    stack.record_choice(
        1, torch.tensor([3]), arity=2, mask=torch.tensor([True]),
        local_structural_loss=(2.0 * parameter).reshape(1))
    loss = stack.forward_loss()
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert torch.allclose(parameter.grad, torch.tensor(2.0))


def test_compose_has_no_auxiliary_local_policy_objective():
    layer = OperationSelectionLayer(d_model=4, ops=[_BinaryMix(.5)],
                                     unary_ops=[_UnaryScale(.5)])
    assert not hasattr(layer, 'local_objective_enabled')
    _, _, routing = layer(torch.ones(1, 2, 4))
    assert 'local_structural_loss' not in routing
