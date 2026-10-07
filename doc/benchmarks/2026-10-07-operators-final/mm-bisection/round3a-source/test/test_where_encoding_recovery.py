"""Native concept IDs replace the retired WholeSpace position namespace."""
from types import SimpleNamespace

import torch

from test_structural_checkpoint import _allocator, _model_with


def _owner():
    allocator = _allocator()
    space = SimpleNamespace(_concept_allocator=allocator)
    return allocator, space, _model_with(space, SimpleNamespace())


def test_allocate_starts_at_one():
    allocator, _space, _model = _owner()
    assert allocator.new_concept() == 1


def test_allocate_monotonic():
    allocator, _space, _model = _owner()
    positions = [allocator.new_concept() for _ in range(8)]
    assert positions == list(range(1, 9))


def test_allocate_position_persists_via_structural_extras(tmp_path):
    allocator, _space, model = _owner()
    for _ in range(5):
        allocator.new_concept()
    assert allocator.new_concept() == 6
    path = tmp_path / 'concept-counter.ckpt'
    model.save_weights(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    blob = saved['structural_extras']['conceptual_spaces'][0]['allocator']
    assert 'next_id' in blob
    assert int(blob['next_id']) == 7
    restored, _space2, model2 = _owner()
    assert model2.load_weights(path)
    assert restored.new_concept() == 7


def test_restore_without_counter_defaults_to_one():
    allocator, space, model = _owner()
    model._restore_allocator_extras(space, {})
    assert allocator.new_concept() == 1
