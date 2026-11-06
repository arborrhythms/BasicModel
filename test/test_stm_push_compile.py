"""§52 item 114: tensor STM pushes cannot specialize on host depth."""
from types import SimpleNamespace

import pytest
import torch

from CompileGuard import guarded_compile
from Layers import ShortTermMemory
from Spaces import ConceptualSpace


@pytest.mark.parametrize('capacity', (1, 4))
@pytest.mark.parametrize('backend', ('eager', 'inductor'))
def test_repeated_compiled_push_ignores_host_depth_and_keeps_tensor_state(capacity, backend):
    eager = SimpleNamespace(stm=ShortTermMemory(batch=2, capacity=capacity, concept_dim=3))
    owner = SimpleNamespace(stm=ShortTermMemory(batch=2, capacity=capacity, concept_dim=3))
    names = ('_buffer', '_depth', '_orders', '_grammar_orders',
             '_concept_rows', '_concept_activations')
    def push(idea, order, grammar, concept, activation):
        ConceptualSpace._stm_shift_and_push(owner, idea, order, grammar, concept, activation)
        return tuple(getattr(owner.stm, name).clone() for name in names)
    torch._dynamo.reset()
    before = torch._dynamo.utils.counters['stats']['unique_graphs']
    try:
        call = guarded_compile(push, backend=backend, fullgraph=True, dynamic=False)
        for step in range(6):
            idea = torch.arange(6, dtype=torch.float32).reshape(2, 3) + step
            order = torch.tensor([step, step + 1])
            grammar = torch.tensor([step + 2, step + 3])
            concept = torch.tensor([step + 10, -1])
            activation = torch.tensor([.75, .5])
            expected_args = (idea, order, grammar, concept, activation)
            ConceptualSpace._stm_shift_and_push(eager, *expected_args)
            result = call(*expected_args)
            for name, value in zip(names, result, strict=True):
                torch.testing.assert_close(value, getattr(eager.stm, name), rtol=0, atol=0)
            # Host bookkeeping changes must not become a graph input. The
            # ordinary whole-sentence boundary synchronizes it after the body.
            if step == 3:
                owner.stm._max_depth_host = 77
        assert owner.stm._max_depth_host == 77
        assert eager.stm._max_depth_host == capacity
        assert torch._dynamo.utils.counters['stats']['unique_graphs'] - before == 1
    finally:
        torch._dynamo.reset()
