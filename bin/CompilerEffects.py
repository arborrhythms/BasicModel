"""Normalize an already-unlifted, unused AOT effect interface for Inductor.

Some AOT builds remove an effect token from the call metadata but leave an
unused placeholder and empty output when no top-level effects survive.
Only that stale interface is removed; live effects and their checks stay.
"""
import torch


def unlift_dead_tokens(graph, inputs):
    from torch._functorch._aot_autograd.descriptors import ForwardTokenAOTInput

    placeholders = list(graph.graph.find_nodes(op='placeholder'))
    extra = len(placeholders) - len(inputs)
    if extra <= 0:
        return
    tokens = placeholders[:extra]
    if not all(isinstance(node.meta.get('desc'), ForwardTokenAOTInput)
               and not node.users for node in tokens):
        return
    output = next(iter(graph.graph.find_nodes(op='output')))
    values = output.args[0]
    if not isinstance(values, (tuple, list)) or len(values) < extra:
        return
    if not all(isinstance(node, torch.fx.Node) and node.op == 'get_attr'
               and isinstance(node.meta.get('val'), torch.Tensor)
               and tuple(node.meta['val'].shape) == (0,)
               and set(node.users) == {output} for node in values[:extra]):
        return
    output.args = (values[extra:],)
    if 'original_output_strides' in output.meta:
        output.meta['original_output_strides'] = output.meta['original_output_strides'][extra:]
    if 'user_visible_output_idxs' in output.meta:
        output.meta['user_visible_output_idxs'] = [
            index-extra for index in output.meta['user_visible_output_idxs'] if index >= extra]
    for token in tokens:
        graph.graph.erase_node(token)
    graph.graph.lint()
    graph.recompile()


def compile_inductor(graph, inputs):
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner

    def inner(module, arguments, *args, **kwargs):
        unlift_dead_tokens(module, arguments)
        return compile_fx_inner(module, arguments, *args, **kwargs)

    return compile_fx(graph, inputs, inner_compile=inner)
