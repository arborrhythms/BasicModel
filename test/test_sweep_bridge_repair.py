"""Only initialized bridge prefixes conduct energy, including after shrinkage."""
from types import SimpleNamespace
import torch
from test_review13_subspaces import fixture


def test_bridge_projection_reads_only_its_counted_prefix(monkeypatch):
    cs, _, _, alloc = fixture(meaning_pairs=8)
    rows = {101: 1, 102: 2}
    words = [31, 32]
    alloc.word_descriptions.update({
        31: dict(parts=((4, 7, 4, -1, 10**9),), wholes=(3,)),
        32: dict(parts=((7,),), wholes=(5,)),
    })
    monkeypatch.setattr(cs, '_word_definition_ids', lambda: tuple(words))
    monkeypatch.setattr(cs, '_csw_row_of', rows.get)
    monkeypatch.setattr(cs.definitions, 'objects', lambda word: (101 if word == 31 else 102,))
    monkeypatch.setattr(cs.interpret, 'no_object', lambda word: False)
    cs._priming_boosts.fill_(1.)
    cs._priming_boosts[0, 1:3] = torch.tensor([2., 3.])
    monkeypatch.setattr(cs, 'priming_weights', lambda **kwargs: cs._priming_boosts[:1])
    # An invalid source and valid destination make any tail read fail, rather
    # than letting the destination filter hide an uninitialized source access.
    cs._priming_bridge_src.fill_(cs.nVectors + 1 - cs._priming_bridge_src_bias)
    cs._priming_bridge_dst.fill_(-cs._priming_bridge_dst_bias)
    cs._priming_bridge_bits.fill_(-1)
    pointers = (cs._priming_bridge_src.data_ptr(), cs._priming_bridge_dst.data_ptr())
    targets = [torch.ones(1, 10), torch.ones(1, 10)]
    towers = [SimpleNamespace(_priming_dim=lambda: 10, _priming_decay=1.,
        _priming_surface=lambda *args, value=value, **kwargs: value) for value in targets]
    cs.project_priming_to_towers(*towers, batch=1)
    expected = torch.ones(1, 10); expected[0, 4] = 2.; expected[0, 7] = 4.
    torch.testing.assert_close(targets[0], expected)
    expected = torch.ones(1, 10); expected[0, 3] = 2.; expected[0, 5] = 3.
    torch.testing.assert_close(targets[1], expected)
    count = cs._priming_bridge_rebuilds
    for value in targets: value.fill_(1.)
    cs.project_priming_to_towers(*towers, batch=1)
    assert cs._priming_bridge_rebuilds == count
    words.pop()
    alloc.word_descriptions[31] = dict(parts=((4,),), wholes=())
    cs.interpret._description_revision += 1
    for value in targets: value.fill_(1.)
    cs.project_priming_to_towers(*towers, batch=1)
    expected = torch.ones(1, 10); expected[0, 4] = 2.
    torch.testing.assert_close(targets[0], expected)
    torch.testing.assert_close(targets[1], torch.ones(1, 10))
    assert cs._priming_bridge_counts == (1, 0)
    assert pointers == (cs._priming_bridge_src.data_ptr(), cs._priming_bridge_dst.data_ptr())


def test_bridge_endpoint_storage_is_lossless_at_inventory_boundaries():
    from Spaces import _bridge_index_layout
    for capacity, expected in ((1, torch.int16), (32768, torch.int16),
                               (65536, torch.int16), (65537, torch.int32),
                               (2**32, torch.int32), (2**32 + 1, torch.int64)):
        dtype, bias = _bridge_index_layout(capacity)
        assert dtype == expected
        values = torch.tensor([0, capacity // 2, capacity - 1], dtype=torch.long)
        encoded = (values - bias).to(dtype)
        torch.testing.assert_close(encoded.long() + bias, values, rtol=0, atol=0)
    # Native inventories: 65,536 sources, 32,768 parts and nine properties.
    # Same logical relation, represented by bounded 32-destination masks.
    slots = 65536 * ((32768 + 31)//32 + (9 + 31)//32)
    src, _ = _bridge_index_layout(65536)
    dst, _ = _bridge_index_layout((32768 + 31)//32)
    assert slots * (torch.empty((), dtype=src).element_size() +
                    torch.empty((), dtype=dst).element_size() + 4) == 537395200


def test_bridge_masks_cover_bit_31_and_adjacent_blocks(monkeypatch):
    from test_cs_sparse_weights import _cs
    from Spaces import _concept_alloc_of
    cs = _cs(nS=64, order=0)
    alloc = _concept_alloc_of(cs)
    alloc.word_descriptions[31] = dict(parts=(tuple(range(64)),), wholes=())
    monkeypatch.setattr(cs, '_word_definition_ids', lambda: (31,))
    monkeypatch.setattr(cs, '_csw_row_of', lambda identity: 63)
    monkeypatch.setattr(cs.definitions, 'objects', lambda word: (101,))
    monkeypatch.setattr(cs.interpret, 'no_object', lambda word: False)
    cs._priming_boosts.fill_(1.)
    cs._priming_boosts[0, 63] = 3.
    monkeypatch.setattr(cs, 'priming_weights', lambda **kwargs: cs._priming_boosts[:1])
    target = torch.ones(1, 64)
    tower = SimpleNamespace(_priming_dim=lambda: 64, _priming_decay=1.,
        _priming_surface=lambda *args, **kwargs: target)
    cs.project_priming_to_towers(tower, None, batch=1)
    torch.testing.assert_close(target, torch.full_like(target, 3.), rtol=0, atol=0)
    assert cs._priming_bridge_counts == (2, 0)
    assert cs._priming_bridge_bits[:2].tolist() == [-1, -1]
