"""Fixed-input certificates and timings for the leaves retained by item 1.8.

Only batch axes vary. Width, dtype, topology, mode flags and device are a
compile signature. No RNG is read, seeded or advanced by these fixtures.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
import math
import statistics
import time

import torch
from torch._dynamo.decorators import mark_unbacked


@dataclass
class Case:
    name: str
    function: object
    inputs: object
    batch_axis: object = 0
    parameters: tuple = ()


def cases(device='cpu', dtype=torch.float32):
    from MeaningKernels import householder, householder_action, LaneKernel
    from Layers import cube_chart, cube_unchart, SparseLayer
    from IdentityRows import meet_ratios
    from Spaces import WhereEncoding

    def fixed(shape, cube=False):
        x = torch.arange(math.prod(shape), device=device, dtype=dtype).reshape(shape)
        return x.remainder(17) / 16 if cube else (x.remainder(19) - 9) / 8

    vectors = lambda b: (fixed((b, 4, 16)),)
    action = lambda b: (fixed((b, 2, 16)), fixed((b, 4, 16)))
    yield Case('householder', householder, vectors)
    yield Case('householder_action', householder_action, action)
    yield Case('householder_action_transpose', partial(householder_action, transpose=True), action)
    for neutral in (True, False):
        lane = LaneKernel(32).to(device=device, dtype=dtype)
        if not neutral:
            with torch.no_grad():
                lane.log_scale.copy_(fixed((16,)) / 8)
                lane.bias.copy_(fixed((16,)) / 16)
                lane.context.copy_(fixed((16, 16)) / 64)
        for inverse in (False, True):
            yield Case(f'lane_{"neutral" if neutral else "learned"}_{"inverse" if inverse else "forward"}',
                       partial(lane, inverse=inverse),
                       lambda b: (fixed((b, 32), True), fixed((b, 32), True)),
                       parameters=tuple(lane.parameters()))
    for bounded in (True, False):
        yield Case(f'cube_chart_{bounded}', partial(cube_chart, bounded=bounded),
                   lambda b: (fixed((b, 32), True),))
    yield Case('cube_unchart', cube_unchart, lambda b: (fixed((b, 32)),))
    yield Case('meet_ratios', meet_ratios,
               lambda b: (fixed((b, 32), True), fixed((b, 24), True)))
    for conjunction in (False, True):
        for dual in (False, True):
            layer = SparseLayer(16, 8, device=torch.device(device))
            for row, col, weight in ((0, 0, 1.), (1, 0, .5), (1, 2, 1.),
                                     (2, 3, 0.), (3, 5, 2.), (4, 15, 1.), (7, 7, .75)):
                layer.add_edge(row, col, weight)
            layer.to(device=device, dtype=dtype)
            layer._missing_sources = torch.tensor([False, False, True, False, False, False, False], device=device)
            for ranged in (False, True):
                start, end = (1, 6) if ranged else (0, 8)
                yield Case(f'fold_{conjunction}_{dual}_{ranged}',
                    partial(layer.fold_presence, conjunctive=conjunction, dual=dual, start=start, end=end),
                    lambda b: (fixed((16, b), True),), batch_axis=1, parameters=(layer.values,))
            def own_fold(u, own, mask, known, edge, edge_known, *, layer=layer,
                         conjunction=conjunction, dual=dual):
                return layer.fold_presence(u, start=1, end=6, conjunctive=conjunction, dual=dual,
                    own=own, own_mask=mask, own_known=known,
                    edge_memberships=edge, edge_known=edge_known)
            yield Case(f'fold_own_{conjunction}_{dual}', own_fold,
                lambda b: (fixed((16, b), True), fixed((5, b), True),
                           torch.tensor([True, False, True, True, False], device=device),
                           fixed((5, b), True) > .3, fixed((7, b), True),
                           fixed((7, b), True) > .2),
                batch_axis=(1, 1, None, 1, 1, 1), parameters=(layer.values,))
    empty = SparseLayer(16, 8, device=torch.device(device))
    yield Case('fold_empty', empty.fold_presence, lambda b: (fixed((16, b), True),), batch_axis=1)
    unsorted = SparseLayer(16, 8, device=torch.device(device))
    for row, col, weight in ((7, 7, .75), (1, 0, .5), (0, 0, 1.), (4, 15, 1.),
                             (3, 5, 2.), (1, 2, 1.), (2, 3, 0.)):
        unsorted.add_edge(row, col, weight)
    unsorted.to(device=device, dtype=dtype)
    yield Case('fold_unsorted', partial(unsorted.fold_presence, start=1, end=6),
               lambda b: (fixed((16, b), True),), batch_axis=1, parameters=(unsorted.values,))
    for width in (2, 4):
        encoding = WhereEncoding(nWhere=width).set_capacity(2**26)
        def positions(b):
            return torch.arange(b, device=device, dtype=torch.int64) * 2718391
        yield Case(f'where_encode_{width}', encoding.encode, lambda b: (positions(b),))
        for name in ('decode', 'decode_index'):
            yield Case(f'where_{name}_{width}', getattr(encoding, name),
                       lambda b, encoding=encoding: (encoding.encode(positions(b)),))


def tensors(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


CASE_NAMES = (
    'householder', 'householder_action', 'householder_action_transpose',
    *[f'lane_{state}_{direction}' for state in ('neutral', 'learned') for direction in ('forward', 'inverse')],
    'cube_chart_True', 'cube_chart_False', 'cube_unchart', 'meet_ratios',
    *[name for conjunction in (False, True) for dual in (False, True)
      for name in (*[f'fold_{conjunction}_{dual}_{ranged}' for ranged in (False, True)],
                   f'fold_own_{conjunction}_{dual}')],
    'fold_empty', 'fold_unsorted',
    *[f'where_{method}_{width}' for width in (2, 4) for method in ('encode', 'decode', 'decode_index')],
)


def mark_batch(case, inputs):
    axes = ((case.batch_axis,) * len(inputs) if isinstance(case.batch_axis, int) else case.batch_axis)
    for value, axis in zip(inputs, axes, strict=True):
        if axis is not None:
            mark_unbacked(value, axis, shape_id='batch')
            mark_unbacked(value, axis, strict=True)


def evaluate(function, inputs, parameters, *, gradients):
    value = tensors(function(*inputs))
    leaves = tuple(x for x in inputs if x.requires_grad) + parameters
    outputs = [x for x in value if x.requires_grad]
    grads = (torch.autograd.grad(sum(x.sum() for x in outputs), leaves, allow_unused=True)
             if gradients and leaves and outputs else ())
    return tuple(x.detach().clone() for x in value), tuple(None if x is None else x.detach().clone() for x in grads)


def certify(case, *, backend='inductor', gradients=True, timing=False, require_clean=True):
    """One graph across positive batch sizes, including singleton and tails.

    Unbacked batch annotations forbid batch specialization; all non-batch
    dimensions stay static. Scalar configuration floats are compile constants,
    rather than CPU scalar tensor inputs requiring item() at execution.
    The options are scoped to this certificate, never the sentence owner.
    """
    from torch.fx.experimental import _config as shapes
    from CompilerEffects import compile_inductor
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward if backend == 'eager' else compile_inductor(graph, inputs)

    torch._dynamo.reset()
    expected, actual = {}, {}
    with ExitStack() as stack:
        stack.enter_context(torch._dynamo.config.patch(specialize_float=True))
        stack.enter_context(shapes.patch(backed_size_oblivious=True, use_duck_shape=False))
        function = torch.compile(case.function, backend=capture, fullgraph=True, dynamic=False)
        started = time.perf_counter()
        for batch in (3, 1, 2, 4, 8, 17):
            inputs = case.inputs(batch)
            if gradients:
                for value in inputs:
                    if value.is_floating_point():
                        value.requires_grad_(True)
            # Also stages any host-owned sparse topology/range before capture.
            expected[batch] = evaluate(case.function, inputs, case.parameters, gradients=gradients)
            mark_batch(case, inputs)
            actual[batch] = evaluate(function, inputs, case.parameters, gradients=gradients)
            torch.testing.assert_close(actual[batch], expected[batch], rtol=2e-5, atol=2e-6)
            for found, wanted in zip(actual[batch][0], expected[batch][0], strict=True):
                if not found.is_floating_point():
                    assert torch.equal(found, wanted), case.name
        compile_seconds = time.perf_counter() - started
        if require_clean:
            assert len(graphs) == 1, (case.name, len(graphs))
        forbidden = ('nonzero', '_local_scalar_dense', '.item')
        if require_clean:
            for graph in graphs:
                for node in graph.graph.nodes:
                    assert not any(name in str(node.target) for name in forbidden), (case.name, node)
                    assert not (node.op == 'call_method' and node.target == 'item'), (case.name, node)
        # Inspect execution as well as Dynamo's graph: a boolean gather can
        # be captured in fullgraph yet lower to a synchronizing nonzero call.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            evaluate(function, inputs, case.parameters, gradients=gradients)
        sync = {'aten::nonzero', 'aten::item', 'aten::_local_scalar_dense'}
        observed = {event.key for event in profile.key_averages()} & sync
        if require_clean:
            assert not observed, (case.name, observed)
        result = dict(graphs=len(graphs), batches=[3, 1, 2, 4, 8, 17],
                      graph_breaks=0, recompiles=max(0, len(graphs)-1),
                      host_sync_ops=sorted(observed), compile_seconds=compile_seconds)
        if timing:
            # Match the grad/inference mode that was certified above.
            inputs = case.inputs(8)
            for value in inputs:
                if gradients and value.is_floating_point():
                    value.requires_grad_(True)
            mark_batch(case, inputs)
            iterations = 1000 if inputs[0].device.type == 'cpu' else 100
            def elapsed(fn):
                samples = []
                for _ in range(5):
                    if inputs[0].device.type == 'mps': torch.mps.synchronize()
                    start = time.perf_counter()
                    for _ in range(iterations): fn(*inputs)
                    if inputs[0].device.type == 'mps': torch.mps.synchronize()
                    samples.append((time.perf_counter() - start) / iterations)
                return samples
            eager, compiled = elapsed(case.function), elapsed(function)
            result.update(eager_seconds=statistics.median(eager), compiled_seconds=statistics.median(compiled),
                          eager_samples_seconds=eager, compiled_samples_seconds=compiled,
                          timing_iterations=iterations)
            assert len(graphs) == result['graphs']
    return result, expected
