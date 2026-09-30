"""Alec's four September 26 corrections; no initialization-seed selection."""
from functools import wraps

import pytest
import torch


def test_default_interpret_reuses_existing_kind_without_minting():
    from test_item9b_interpret import _operator
    from Spaces import _concept_alloc_of
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [], form='cat')
    kind = interpret.forward(word, order=2)
    before = dict(_concept_alloc_of(cs).placement)
    assert interpret.forward(word) == kind
    assert interpret.forward(word, order=1) == kind
    assert dict(_concept_alloc_of(cs).placement) == before


def test_default_interpret_reuses_a_witnessed_association():
    from test_item9b_interpret import _operator
    from Spaces import _concept_alloc_of
    cs, interpret = _operator()
    word = interpret.lookup_word([7], [], form='cat')
    kind = cs.new_concept()
    _concept_alloc_of(cs).reference_orders[kind] = 2
    cs.bind_word_concept('cat', kind)
    assert interpret.forward(word) == kind
    assert interpret.reverse(kind) == word


def test_context_pass_cannot_backpropagate_or_step(tmp_path, monkeypatch):
    from ModeSchedule import ModeSchedule
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    model.mode_schedule = ModeSchedule('interleave:2')
    model._last_primary_costs = {'input_reconstruction_reverse': torch.tensor(0.)}
    calls = []
    @wraps(model._run_batch_once)
    def observe(*args, **kwargs):
        calls.append((model.serial, kwargs['train'], kwargs['optimizer'],
                      torch.is_grad_enabled()))
        return None, 0
    monkeypatch.setattr(model, '_run_batch_once', observe)
    optimizer = object()
    raw = model.inputSpace.prepInput(['the cat sat'])
    model.runBatch(train=True, optimizer=optimizer, split='runtime',
        batch_override=(raw, torch.empty(1, 0)),
        schedule_context=('the cat sat', 'the cat flew'))
    assert calls == [(False, False, None, False), (True, True, optimizer, True)]


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


def test_reset_clears_only_the_selected_fields_time_band(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    cs = model._concept_owner()
    cs.add_concept_feature(0, 'ps', 7, 1.)
    spans = torch.tensor([[[0, 1]], [[0, 1]]])
    cs.cs_read_memberships((torch.tensor([[7], [7]]), spans, None,
                           torch.tensor([[65], [65]]), spans), spans)
    before = cs._percept_field.when_band.clone()
    cs._clear_percept_field(batch=0)
    assert cs._percept_field.when_band[0].count_nonzero() == 0
    torch.testing.assert_close(cs._percept_field.when_band[1], before[1])


def test_compiled_byte_reconstruction_handles_lookahead_and_nul():
    from types import SimpleNamespace
    from Models import BasicModel
    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
    idea = torch.arange(1., 129.).reshape(2, 64).requires_grad_()
    bank = torch.nn.functional.normalize(torch.arange(1., 2049.).reshape(2, 16, 64), dim=-1)
    tokens = torch.arange(2 * 16 * 9).reshape(2, 16, 9) % 256
    tokens[:, :, 7] = 0
    valid = torch.ones_like(tokens, dtype=torch.bool)
    target = tokens[:, :8, :8].clone()
    target_mask = torch.ones_like(target, dtype=torch.bool)
    def score(value):
        return BasicModel._byte_word_cost(owner, value, torch.tensor(0), bank,
            tokens, valid, target, target_mask, True)
    expected = score(idea)
    compiled = torch.compile(score, backend='inductor', fullgraph=True)
    actual = compiled(idea)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), idea)[0],
                               torch.autograd.grad(expected.sum(), idea)[0])


def test_native_program_captures_the_symbol_band_and_one_field_time(tmp_path):
    from test_packed_reconstruction_parity import build_model
    from What import What
    model = build_model(tmp_path, word_capacity=8)
    raw = model.inputSpace.prepInput(['the cat sat'])
    from reading_fixtures import capture_readings
    with capture_readings(model) as readings, torch.no_grad():
        model.runBatch(train=False, split='validation', batchSize=1,
            batch_override=(raw, torch.empty(1, 0)),
            questions=(What.present(0, split='validation'),))
    program, = readings[0]
    assert program.symbol_where.shape == (*program.rows.shape, 4)
    addresses = model.where_encoding.decode_index(program.symbol_where)
    starts = model.inputSpace._ar_word_part_offsets[0,
        model.inputSpace._word_active_mask[0], 0]
    expected = model.where_registry.intervals('input', starts)[..., 0]
    torch.testing.assert_close(addresses, expected)
    assert program.symbol_when.shape == program.symbol_where.shape
    torch.testing.assert_close(program.symbol_when,
        model.when_encoding.encode(model.when_time).expand_as(program.symbol_when))


def test_native_percepts_encode_input_starts_and_share_time(tmp_path):
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    parts = model.perceptualSpace
    ids = torch.tensor([[1, 2, 3]])
    offsets = torch.tensor([[0, 8, 16]])
    model._advance_when_time()
    event = parts._radix_part_events(ids, offsets)
    bands = event[..., parts._radix_where_indices]
    torch.testing.assert_close(model.where_encoding.decode_index(bands),
        offsets + model.where_registry.slices['input'][0])
    torch.testing.assert_close(event[..., -4:],
        model.when_encoding.encode(model.when_time).expand_as(event[..., -4:]))
    first_time = event[..., -4:].clone()
    model._advance_when_time()
    later = parts._radix_part_events(ids, offsets)
    assert not torch.equal(later[..., -4:], first_time)
    torch.testing.assert_close(model.when_encoding.decode_index(later[..., -4:]),
        torch.full_like(ids, 2))


def test_percept_group_keeps_each_members_band_and_one_field_time(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    cs = model._concept_owner()
    cs.add_concept_feature(0, 'ps', 7, 1.)
    spans = torch.tensor([[[0, 1]]])
    cs.cs_read_memberships((torch.tensor([[7]]), spans, None, torch.tensor([[65]]), spans), spans)
    field = cs._percept_field
    assert field.percept_where.shape[-1] == field.when_band.shape[-1] == 4
    for key, bands in zip(field.keys, field.percept_where):
        role, group = key
        group = group if isinstance(group, tuple) else (group,)
        expected = torch.tensor(group) + model.where_registry.slices['parts' if role == 'ps' else 'wholes'][0]
        torch.testing.assert_close(model.where_encoding.decode_index(bands[:len(group)]), expected)
    torch.testing.assert_close(field.when_band,
        model.when_encoding.encode(model.when_time).expand_as(field.when_band))
