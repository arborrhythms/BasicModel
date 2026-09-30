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


@pytest.mark.parametrize('packed', [False, True])
def test_detached_word_teacher_also_excludes_field_time(packed):
    from Language import ReconstructionStack, ReverseConstructionChooser
    chooser = ReverseConstructionChooser(idea_dim=8, n_rules=2, max_words=2,
        max_steps=6, leaf_dim=12, hidden=16)
    idea = torch.ones(1, 8, requires_grad=True)
    stack = ReconstructionStack(batch=1, max_depth=8)
    leaves = torch.ones(1, 2, 12)
    stack.store_word_parts(torch.ones(1, 2, 1, dtype=torch.long),
                           torch.ones(1, 2, 1, dtype=torch.bool))
    def score():
        stack.store_leaves(leaves)
        if packed:
            return chooser.packed_loss(idea[:, None], stack,
                word_positions=torch.tensor([[0, 1]]),
                sentence_ids=torch.zeros(1, 2, dtype=torch.long),
                sentence_end_mask=torch.tensor([[False, True]]), nWhen=4)[0]
        return chooser.loss(idea, stack, nWhen=4)[0]
    loss = score()
    loss.backward()
    assert idea.grad is None
    assert chooser.leaf_decoder.out.weight.grad[-4:].count_nonzero() == 0
    assert chooser.leaf_decoder.out.weight.grad[:-4].count_nonzero() > 0
    leaves[..., -4:] = -100.
    torch.testing.assert_close(score(), loss)
