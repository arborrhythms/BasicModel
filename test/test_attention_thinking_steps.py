"""Exact reference certificates for the attention walk's public steps."""
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import torch


def original_attention():
    """The specified base is the oracle, never another path in production."""
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ['git', 'show', '3926005f12:bin/Attention.py'], cwd=root, text=True)
    module = ModuleType('attention_at_3926005')
    exec(compile(source, '3926005f12:bin/Attention.py', 'exec'), module.__dict__)
    return module


def exact(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            exact(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('departure', [False, True])
@pytest.mark.parametrize('field', [False, True])
def test_public_steps_equal_every_base_field_and_replay_stream(departure, field):
    from Language import OperationSelectionLayer
    import Attention
    base = original_attention()
    chooser = OperationSelectionLayer(d_model=4)
    chooser.attention_operations = tuple(range(6)) if field else (0, 1, 2)
    keys = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).remainder(9).div(8)
    spans = torch.tensor([[[0, 3], [4, 7], [8, 11]], [[0, 3], [4, 7], [0, 0]]])
    known = torch.tensor([[True, False, True], [False, True, False]])
    kwargs = dict(budget=32, identities=torch.tensor([[0, 1, 0], [2, 3, -1]]))
    if departure:
        kwargs.update(exploit=base.narrow_words(chooser, keys, spans, known, **kwargs),
                      departure=torch.tensor([0, 1]))
    initial = torch.get_rng_state()
    wanted = base.narrow_words(chooser, keys, spans, known, **kwargs)
    final = torch.get_rng_state()
    torch.set_rng_state(initial)
    walk = Attention.open_narrowing(chooser, keys, spans, known, **kwargs)
    for _ in range(32):
        Attention.narrow_step(walk)
    found = Attention.close_narrowing(walk)
    exact(found, wanted)
    exact(torch.get_rng_state(), final)
    torch.set_rng_state(initial)
    exact(Attention.narrow_words(chooser, keys, spans, known, **kwargs), wanted)
    exact(torch.get_rng_state(), final)


def test_empty_walk_preserves_the_base_optional_fields():
    from Language import OperationSelectionLayer
    import Attention
    chooser = OperationSelectionLayer(d_model=4)
    args = (chooser, torch.empty(2, 0, 4), torch.empty(2, 0, 2, dtype=torch.long),
            torch.empty(2, 0, dtype=torch.bool))
    wanted = original_attention().narrow_words(*args, budget=32)
    walk = Attention.open_narrowing(*args, budget=32)
    exact(Attention.close_narrowing(walk), wanted)


@pytest.mark.parametrize('departure', [False, True])
def test_exhausted_walk_skips_chooser_work_but_keeps_the_full_base_carrier(departure):
    from Language import OperationSelectionLayer
    import Attention
    chooser = OperationSelectionLayer(d_model=4)
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).div(12)
    spans = torch.tensor([[[0, 3], [4, 7], [8, 11]]])
    known = torch.ones(1, 3, dtype=torch.bool)
    kwargs = dict(budget=32)
    base = original_attention()
    if departure:
        kwargs.update(exploit=base.narrow_words(chooser, keys, spans, known, **kwargs),
                      departure=torch.tensor([1]))
    initial = torch.get_rng_state()
    wanted = base.narrow_words(chooser, keys, spans, known, **kwargs)
    final = torch.get_rng_state()
    calls = []
    attend = chooser.attend
    def count(*args, **options):
        calls.append(None)
        return attend(*args, **options)
    chooser.attend = count
    torch.set_rng_state(initial)
    found = Attention.narrow_words(chooser, keys, spans, known, **kwargs)
    exact(found, wanted)
    exact(torch.get_rng_state(), final)
    assert len(calls) < 32, 'exhausted walk still evaluates its chooser to the budget'


def test_captured_stop_keeps_discrete_bits_and_policy_gradients():
    from Language import OperationSelectionLayer
    import Attention
    chooser = OperationSelectionLayer(d_model=4)
    keys = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).div(8)
    spans = torch.tensor([[[2**33, 2**33+3], [2**33+4, 2**33+7]]])
    known = torch.ones(1, 2, dtype=torch.bool)
    def run(keys):
        return Attention.narrow_words(chooser, keys, spans, known, budget=8)
    wanted = original_attention().narrow_words(chooser, keys, spans, known, budget=8)
    found = torch.compile(run, backend='aot_eager', fullgraph=True)(keys)
    exact(found, wanted)
    weights = torch.arange(8, dtype=keys.dtype)[None]
    parameters = tuple(chooser.parameters())
    before = torch.autograd.grad((wanted.probabilities*weights).sum(), parameters, allow_unused=True)
    after = torch.autograd.grad((found.probabilities*weights).sum(), parameters, allow_unused=True)
    exact(after, before)
