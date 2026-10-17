"""The bind marker stays numerical; reference is deferred to the close."""
import torch
from Language import ContextualBindLayer


def test_bind_marker_preserves_its_input_until_the_close():
    layer = ContextualBindLayer()
    marker = torch.randn(1, 1, 4, requires_grad=True)
    other = torch.randn(1, 1, 4, requires_grad=True)
    value = layer.compose(marker, other)
    torch.testing.assert_close(value, marker)
    value.sum().backward()
    torch.testing.assert_close(marker.grad, torch.ones_like(marker))
    assert other.grad is None
    assert (layer.rule_name, layer.arity, layer.space_role) == ('bind', 2, 'CS')
