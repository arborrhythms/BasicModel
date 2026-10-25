"""AOT's stale effect interface must not shift values, strides or gradients."""
import pytest
import torch

from CompilerEffects import unlift_dead_tokens


def graph_with_token(*, effect=True, used=False):
    from torch._functorch._aot_autograd.descriptors import ForwardTokenAOTInput
    owner = torch.nn.Module()
    owner.register_buffer('empty_token', torch.empty(0))
    graph = torch.fx.Graph()
    token = graph.placeholder('token')
    if effect:
        token.meta['desc'] = ForwardTokenAOTInput(0)
    value = graph.placeholder('value')
    value.meta['val'] = torch.zeros(2, 3)
    empty = graph.get_attr('empty_token')
    empty.meta['val'] = owner.empty_token
    doubled = graph.call_function(torch.mul, (value, 2))
    if used:
        graph.call_function(torch.numel, (token,))
    output = graph.output((empty, doubled))
    output.meta['original_output_strides'] = [(1,), (3, 1)]
    output.meta['user_visible_output_idxs'] = [1]
    return torch.fx.GraphModule(owner, graph)


def test_unused_unlifted_token_preserves_output_and_gradient():
    graph = graph_with_token()
    value = torch.arange(6.).reshape(2, 3).requires_grad_()
    unlift_dead_tokens(graph, (value,))
    result, = graph(value)
    torch.testing.assert_close(result, value*2, rtol=0, atol=0)
    gradient, = torch.autograd.grad(result.sum(), (value,))
    torch.testing.assert_close(gradient, torch.full_like(value, 2), rtol=0, atol=0)
    output = next(iter(graph.graph.find_nodes(op='output')))
    assert output.meta['original_output_strides'] == [(3, 1)]
    assert output.meta['user_visible_output_idxs'] == [0]


@pytest.mark.parametrize('effect,used,complete', [(False, False, False),
                                               (True, True, False),
                                               (True, False, True)])
def test_live_or_ordinary_inputs_and_complete_interfaces_are_untouched(effect, used, complete):
    graph = graph_with_token(effect=effect, used=used)
    before = graph.code
    inputs = (torch.empty(0), torch.zeros(2, 3)) if complete else (torch.zeros(2, 3),)
    unlift_dead_tokens(graph, inputs)
    assert graph.code == before
