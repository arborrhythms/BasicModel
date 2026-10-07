import pytest

import torch

from Layers import RadixLayer

from Spaces import Codebook, Embedding


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



from functools import wraps

import pytest

import torch


def test_model_owns_one_ladder_across_all_spaces(tmp_path):
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    spaces = (model.inputSpace, model.perceptualSpace,
              *model.wholeSpaces, model.symbolSpace)
    assert len({id(space.subspace.whereEncoding) for space in spaces}) == 1
    assert len({id(space.subspace.whenEncoding) for space in spaces}) == 1
    assert model.where_encoding.maxVal > model.where_registry.capacity
    assert model.where_encoding.period_hf <= 256


def test_large_registry_addresses_roundtrip_through_both_phases():
    from WhereRegistry import WhereRegistry
    registry = WhereRegistry([('input', 8192), ('parts', 400_000),
                              ('wholes', 400_000), ('symbols', 512_000_000)])
    encoding = registry.encoding
    addresses = torch.tensor([0, 8191, 8192, 2**24 + 1, 2**28 - 1,
                              registry.capacity - 2, registry.capacity - 1])
    bands = encoding.encode(addresses)
    assert bands.shape == (7, 4)
    assert bands.dtype == torch.float32
    torch.testing.assert_close(encoding.decode_index(bands), addresses)
    assert not torch.equal(bands[-1], bands[-2])


def test_copy_preserves_the_single_registry_ladder():
    import copy
    from WhereRegistry import WhereRegistry
    registry = WhereRegistry([('input', 32), ('parts', 512), ('symbols', 1024)])
    ladder, copied = copy.deepcopy((registry.encoding, registry))
    assert copied.encoding is ladder
    assert ladder is not registry.encoding


def test_content_only_encode_decode_preserves_all_coordinates(tmp_path):
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    for space in (model.conceptualSpace, model.outputSpace):
        sub = space.subspace
        assert sub.nWhere == sub.nWhen == 0
        values = torch.arange(1., 2 * 3 * sub.nWhat + 1).reshape(2, 3, sub.nWhat)
        torch.testing.assert_close(sub.encode(values.clone()), values)
        decoded, where, when = sub.decode(values.clone())
        torch.testing.assert_close(decoded, values)
        assert where.count_nonzero() == when.count_nonzero() == 0

