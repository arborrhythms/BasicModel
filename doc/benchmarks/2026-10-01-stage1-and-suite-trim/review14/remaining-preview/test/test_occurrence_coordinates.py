from functools import wraps

import pytest

import torch


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



"""Occurrence starts and field time have different reconstruction roles."""
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def model(tmp_path):
    from test_packed_reconstruction_parity import build_model
    owner = build_model(tmp_path, word_capacity=8)
    try:
        yield owner
    finally:
        owner.End()
        owner.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_symbol_range_has_one_address_per_pole_without_word_slots(model):
    start, end = model.where_registry.slices['symbols']
    symbols = model.symbolSpace.subspace.what
    rows = max(int(getattr(symbols, 'lexicon_capacity', symbols.nVectors)),
               int(model.conceptualSpaces[0].nVectors))
    assert end - start == 2 * rows


def test_repeated_percept_occurrences_use_input_starts_and_one_time(model):
    parts = model.perceptualSpace
    ids = torch.tensor([[1, 1, 2]])
    starts = torch.tensor([[0, 8, 19]])
    model._advance_when_time()
    event = parts._radix_part_events(ids, starts)
    decoded = model.where_encoding.decode_index(event[..., parts._radix_where_indices])
    torch.testing.assert_close(decoded, starts + model.where_registry.slices['input'][0])
    torch.testing.assert_close(event[..., -4:],
        model.when_encoding.encode(model.when_time).expand_as(event[..., -4:]))
    before = event[..., -4:].clone()
    model._advance_when_time()
    later = parts._radix_part_events(ids, starts)
    assert not torch.equal(later[..., -4:], before)


def test_input_symbols_keep_their_word_starts_even_when_the_word_repeats(model):
    from What import What
    text = 'cat cat sat'
    from reading_fixtures import capture_readings
    with capture_readings(model) as readings, torch.no_grad():
        model.runBatch(train=False, split='validation', batchSize=1,
            batch_override=(model.inputSpace.prepInput([text]), torch.empty(1, 0)),
            questions=(What.present(0, split='validation'),))
    program, = readings[0]
    # This grammar retains the two spaces as units too.
    assert len(program.rows) == 5
    torch.testing.assert_close(model.where_encoding.decode_index(program.symbol_where),
                               torch.tensor([0, 3, 4, 7, 8]))
    torch.testing.assert_close(program.symbol_when,
        model.when_encoding.encode(model.when_time).expand_as(program.symbol_when))


def test_input_event_time_uses_the_subjective_clock(model):
    model._advance_when_time()
    before = model.when_time.clone()
    with torch.no_grad():
        model.runBatch(train=False, split='validation', batchSize=1,
            batch_override=(model.inputSpace.prepInput(['cat cat sat']), torch.empty(1, 0)))
    torch.testing.assert_close(model.when_time, before + 1)
    events = model.inputSpace._ar_embedded[:, :5]
    torch.testing.assert_close(events[..., -4:],
        model.when_encoding.encode(model.when_time).expand_as(events[..., -4:]))


def test_d3_scores_content_and_position_without_repeated_time_credit():
    from Models import BasicModel
    from Layers import ModelLoss
    target = torch.zeros(1, 3, 16)
    pred = torch.zeros_like(target)
    pred[..., 0] = .2
    pred[..., 8] = .3
    pred[..., -4:] = 20.
    pred.requires_grad_()
    owner = SimpleNamespace(
        inputSpace=SimpleNamespace(subspace=SimpleNamespace(nWhere=4, nWhen=4),
                                   _ar_embedded=target),
        perceptualSpace=None, loss=ModelLoss(),
        _stm_single_S=torch.zeros(1, 16), _reverse_from_S=lambda _: pred)
    owner._reverse_event_loss = BasicModel._reverse_event_loss.__get__(owner)
    loss, metric = BasicModel._d3_reconstruction_loss(owner)
    expected = .7 * pred[..., :8].square().mean() + .2 * pred[..., 8:12].square().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert pred.grad[..., :12].count_nonzero() > 0
    assert pred.grad[..., -4:].count_nonzero() == 0
    target[..., -4:] = -100.
    again, metric_again = BasicModel._d3_reconstruction_loss(owner)
    torch.testing.assert_close(again, loss)
    torch.testing.assert_close(metric_again, metric)
    # A general event comparison may still score temporal differences.
    assert owner._reverse_event_loss(pred, target) > loss


def test_leaf_distillation_excludes_field_time():
    from Models import BasicModel
    from Layers import LeafDecoderHead
    root = torch.ones(1, 8, requires_grad=True) * 2
    leaves = torch.ones(1, 2, 12)
    head = LeafDecoderHead(8, 2, 12)
    owner = SimpleNamespace(_stm_single_S=root,
        _reconstruction_stack=lambda: SimpleNamespace(leaves=lambda: leaves),
        _leaf_distill_head_module=head,
        inputSpace=SimpleNamespace(subspace=SimpleNamespace(nWhen=4)))
    loss = BasicModel._leaf_distill_loss(owner)
    loss.backward()
    assert head.out.weight.grad[-4:].count_nonzero() == 0
    assert head.out.weight.grad[:-4].count_nonzero() > 0
    leaves[..., -4:] = -100.
    torch.testing.assert_close(BasicModel._leaf_distill_loss(owner), loss)


