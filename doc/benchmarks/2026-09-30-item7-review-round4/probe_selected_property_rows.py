"""A selected property read gathers definitions before expanding over positions."""
import torch
from Spaces import Codebook
from PerceptProperties import PrimitiveProperties


def test_selected_property_rows_are_gathered_before_byte_read(monkeypatch):
    basis = Codebook()
    basis.primitive_properties = PrimitiveProperties(513, dtype=torch.float64)
    with torch.no_grad():
        basis.primitive_properties.members.copy_((torch.arange(513 * 256).reshape(513, 256) % 5) / 4.)
    selected = torch.tensor([2, 7, 2])
    byte_ids = torch.tensor([[4, 7, 8], [0, 1, 255]])
    observed_widths = []
    read = basis.primitive_properties.forward
    def observed(*args, **kwargs):
        result = read(*args, **kwargs)
        observed_widths.append(result.shape[-1])
        return result
    monkeypatch.setattr(basis.primitive_properties, 'forward', observed)
    actual = basis.materialize_property(selected, byte_ids.shape[-1], input_bytes=byte_ids)
    members = basis.primitive_properties.members.detach().clone().requires_grad_()
    expected = members[selected].t()[byte_ids].amax(-1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_grad, = torch.autograd.grad(actual.sum(), basis.primitive_properties.members)
    expected_grad, = torch.autograd.grad(expected.sum(), members)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert observed_widths == [len(selected)], observed_widths
