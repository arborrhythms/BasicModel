"""Supplied syntax lessons have no inherited batch question context."""
from types import SimpleNamespace

import torch

from GrammarLessons import compose_loss
from Language import OperationSelectionLayer


def test_teacher_state_uses_neutral_context_without_mutating_the_reading():
    class Sum(torch.nn.Module):
        def forward(self, left, right):
            return left + right
    class Difference(torch.nn.Module):
        def forward(self, left, right):
            return left - right
    layer = OperationSelectionLayer(d_model=4, ops=(Sum(), Difference()), chooser='mlp')
    context = torch.ones(3, layer.chooser.WHAT_CONTEXT_DIM)
    layer._what_context = context
    language = SimpleNamespace(_tree_layer=lambda _: layer,
        _compose_binary_rules=(SimpleNamespace(method_name='sum'), SimpleNamespace(method_name='difference')),
        _compose_unary_rules=())
    program = SimpleNamespace(leaves=torch.eye(4)[:2])
    lesson = {'tree': ['sum', 0, 1]}
    first = compose_loss(language, [program], [lesson])
    assert first is not None and first.requires_grad and torch.isfinite(first)
    assert layer._what_context is context
    layer._what_context = context * -5
    second = compose_loss(language, [program], [lesson])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.backward()
    assert any(parameter.grad is not None and bool(parameter.grad.any())
               for parameter in layer.chooser.parameters())
