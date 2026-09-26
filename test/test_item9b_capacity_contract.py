"""Admission grows occupancy inside a fixed perceptual allocation (9b section 4e)."""
import pytest
import torch

from Layers import RadixLayer
from Spaces import Codebook, Embedding


def test_admission_at_capacity_is_atomic_and_names_the_physical_knob():
    store = RadixLayer(4, initial_cap=2)
    store.insert(b'a')
    store.insert(b'b')
    parameter = store._basis.W
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError, match='nVectors'):
        store.insert(b'c')
    assert len(store) == 2 and store.get_id(b'c') is None
    assert store._basis.W is parameter
    torch.testing.assert_close(parameter, before)


def test_atomic_byte_admission_preflights_the_complete_batch():
    store = RadixLayer(4, initial_cap=2)
    store.insert(b'a')
    parameter = store._basis.W
    with pytest.raises(RuntimeError, match='nVectors'):
        store.ensure_atomic_bytes([b'b', b'c'])
    assert len(store) == 1
    assert store.get_id(b'b') is None and store.get_id(b'c') is None
    assert store._basis.W is parameter


def test_admission_preserves_optimizer_parameter_and_moments():
    # Standalone radix fixtures default to a nonlearned tensor; production
    # PartSpace owns a Parameter. Match that ownership before checking Adam.
    basis = Codebook()
    basis.create(1, 4, 4, customVQ=False)
    basis.setW(torch.nn.Parameter(basis.W.detach().clone()))
    store = RadixLayer(4, initial_cap=4, basis=basis)
    store.insert(b'a')
    parameter = store._basis.W
    optimizer = torch.optim.Adam([parameter], lr=.01)
    store.active_prototypes().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    moment = optimizer.state[parameter]['exp_avg'].clone()
    store.ensure_atomic_bytes([b'b', b'c'], optimizer=optimizer)
    assert store._basis.W is parameter
    assert optimizer.param_groups[0]['params'] == [parameter]
    torch.testing.assert_close(optimizer.state[parameter]['exp_avg'], moment)
    assert tuple(parameter.shape) == (4, 4)


def test_checkpoint_cannot_resize_an_existing_percept_codebook():
    source = RadixLayer(4, initial_cap=4)
    source.insert(b'a')
    target = RadixLayer(4, initial_cap=2)
    parameter = target._basis.W
    with pytest.raises(ValueError, match='nVectors'):
        target.load_vocab_extras(source.vocab_extras())
    assert target._basis.W is parameter and len(target) == 0


def test_lexicon_reserves_its_physical_capacity_before_admission():
    lexicon = Embedding()
    lexicon.create(nInput=4, nVectors=256, nDim=4)
    parameter = lexicon.wv._vectors
    assert tuple(parameter.shape) == (256, 4)
    lexicon.insert('new-word')
    lexicon.insert('another-word')
    assert lexicon.wv._vectors is parameter
    assert tuple(parameter.shape) == (256, 4)
    assert lexicon.pretrain.optimizer.param_groups[0]['params'] == [parameter]


def test_where_ranges_are_fixed_and_owned_by_the_model(tmp_path):
    from test_grounded_xor import grounded_model
    first, _ = grounded_model(tmp_path)
    layout = first.where_registry
    inputs, parts, wholes, symbols = [layout.slices[key] for key in
                                      ('input', 'parts', 'wholes', 'symbols')]
    assert inputs[0] == 0 and inputs[1] == parts[0]
    assert parts[1] == wholes[0] and wholes[1] == symbols[0]
    assert first.perceptualSpace.subspace.what.where_offset == parts[0]
    assert first.wholeSpaces[0].subspace.what.where_offset == wholes[0]
    assert first.symbolSpace.subspace.what.where_offset == symbols[0]
    first.perceptualSpace.percept_store.ensure_atomic_bytes([b'x', b'y'])
    assert first.where_registry is layout
    second, _ = grounded_model(tmp_path)
    assert second.where_registry.slices == layout.slices
    assert second.where_registry is not layout


def test_input_occurrences_and_thought_symbols_use_their_own_ranges():
    from WhereRegistry import WhereRegistry
    registry = WhereRegistry([('input', 256), ('parts', 64), ('wholes', 8),
                              ('symbols', 128)])
    rows = torch.tensor([[12, 12, -1]])
    positions = torch.tensor([[0, 1, 2]])
    where = registry.intervals('input', torch.where(rows >= 0, positions, -1))
    assert where[0, 0, 1] <= where[0, 1, 0]
    assert where.dtype == torch.int64
    assert where[0, 0, 0] == registry.slices['input'][0]
    assert where[0, 2].tolist() == [-1., -1.]
    thought = registry.intervals('symbols', rows)
    assert thought[0, 0, 0] == registry.slices['symbols'][0] + 12
    torch.testing.assert_close(thought[0, 0], thought[0, 1])
    assert registry.slices['symbols'][1] - registry.slices['symbols'][0] == 128


def test_large_where_ranges_keep_symbol_row_addresses_exact():
    from WhereRegistry import WhereRegistry
    registry = WhereRegistry([('parts', 1 << 25), ('symbols', 1 << 20)])
    rows = torch.tensor([1 << 19, (1 << 19) + 1, (1 << 19) + 2])
    where = registry.intervals('symbols', rows)
    assert (where[:, 1] - where[:, 0]).tolist() == [1, 1, 1]
    assert where[0, 1] == where[1, 0] and where[1, 1] == where[2, 0]
