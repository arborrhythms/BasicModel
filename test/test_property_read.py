"""Stage-zero diagnostics do not retain the previous sentence's property graph."""
import torch
from test_wholespace_property_migration import _small_property_model


def test_property_diagnostic_releases_its_graph_at_tick_end(tmp_path):
    model = _small_property_model(tmp_path)
    owner = model.wholeSpace
    owner._stage0_carrier(torch.tensor([[97, 98]]), None)
    assert owner._stage0_property_membership.grad_fn is not None
    model.post_tick_compact()
    assert owner._stage0_property_membership.grad_fn is None


"""Property reads retain the basis and support, never their Cartesian product."""
import pytest
import torch
from PerceptProperties import PrimitiveProperties


@pytest.mark.parametrize('conjunctive', [True, False])
@pytest.mark.parametrize('complement', [True, False])
def test_property_read_preserves_values_and_tie_gradients(conjunctive, complement):
    primitive = PrimitiveProperties(513, dtype=torch.float64)
    with torch.no_grad():
        primitive.members.copy_((torch.arange(513 * 256).reshape(513, 256) % 5) / 4.)
        primitive.members[0].fill_(1.)
        primitive.members[1].zero_()
    counts = torch.zeros(2, 3, 256, dtype=torch.float64)
    counts[0, 0, [7, 9, 31]] = 1.
    counts[0, 1, [0, 4]] = 2.
    counts[1, 0] = 1.
    counts[1, 2, [1, 128, 255]] = 3.
    members = primitive.members.detach().clone().requires_grad_()
    weights = 1 - members if complement else members
    values = torch.where(counts.unsqueeze(-2) > 0, weights, 1. if conjunctive else 0.)
    expected = values.amin(-1) if conjunctive else values.amax(-1)
    expected = expected * (counts.sum(-1, keepdim=True) > 0)
    actual = primitive.on_counts(counts, conjunctive=conjunctive, complement=complement)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    cotangent = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual) / actual.numel()
    expected_grad, = torch.autograd.grad(expected, members, cotangent)
    actual_grad, = torch.autograd.grad(actual, primitive.members, cotangent)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


def test_property_read_does_not_save_event_property_primitive_product():
    primitive = PrimitiveProperties(513)
    counts = torch.ones(2, 3, 256)
    saved = []
    def pack(value):
        saved.append(tuple(value.shape))
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        primitive.on_counts(counts).sum().backward()
    assert (2, 3, 513, 256) not in saved, saved


def test_property_read_and_gradient_compile_as_one_graph():
    primitive = PrimitiveProperties(513)
    counts = torch.ones(2, 3, 256)
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(primitive.on_counts, backend=backend, fullgraph=True)
    expected = primitive.on_counts(counts)
    actual = compiled(counts)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert len(graphs) == 1
    assert torch.isfinite(primitive.members.grad).all()


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
