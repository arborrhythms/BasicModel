"""Decision 7: working-memory load and deadlines in the one-softmax chooser."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import Language
from test_compose_operations import Add


def unary_first():
    step = Language.OperationSelectionLayer(d_model=1, ops=[Add()],
                                            unary_ops=[nn.Identity()])
    with torch.no_grad():
        step.reduce_anchor.zero_()
        step.apply_anchor.fill_(1000.)
        step.stop_anchor.zero_()
    return step


def test_online_deadline_uses_full_depth_behind_the_two_slot_window():
    from Models import BasicModel
    from Spaces import ConceptualSpace
    step = unary_first()
    owner = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=step),
                            _structural_context=lambda **kwargs: None)
    # Before the word arrives there are K-1 terms. After its deposit, d=K.
    state = (torch.ones(1, 8, 1), torch.tensor([8]),
             torch.zeros(1, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
             torch.full((1, 8), -1, dtype=torch.long), torch.ones(1, 8))
    choice = Language.LanguageSpace.choose_operation(owner, state, torch.tensor([True]),
        slots=1, allowance=7, rounds_left=1)
    assert choice.kind.tolist() == [1]
    following = ConceptualSpace.apply_language_choice(state, choice)
    assert following[1].tolist() == [7]
    model = SimpleNamespace(conceptualSpace=SimpleNamespace(
        stm=SimpleNamespace(capacity=8, _depth=following[1])))
    assert BasicModel._compose_admission(model, torch.tensor([[True]])).all()


def test_full_stack_arrival_is_an_assertion_instead_of_dropping_the_word():
    from Models import BasicModel
    model = SimpleNamespace(conceptualSpace=SimpleNamespace(
        stm=SimpleNamespace(capacity=8, _depth=torch.tensor([8]))))
    with pytest.raises(RuntimeError, match='full stack'):
        BasicModel._compose_admission(model, torch.tensor([[True]]))


@pytest.mark.parametrize('pressure', [0., 1.])
def test_unary_preference_still_completes_absolute_and_relative_ends(pressure):
    step = unary_first()
    step.reduce_pressure = pressure
    result = step.derive(torch.ones(2, 8, 1), slots=torch.tensor([1, 3]),
                         rounds=16, greedy=True)
    assert result['complete'].all()
    assert result['depth'].tolist() == [1, 3]
    kinds = torch.stack([t['kind'] for t in result['traces']], 1)
    assert (kinds == 1).sum(1).tolist() == [7, 5]
    assert (kinds == 2).any(), 'unary candidates are legal while slack remains'


def test_row_allowance_is_one_or_three_never_two():
    slots = Language.sentence_row_slots(torch.tensor([False, True, False]))
    assert slots.tolist() == [1, 3, 1]
    assert not (slots == 2).any()


def test_pressure_is_zero_when_empty_and_increases_with_load_and_urgency():
    step = unary_first()
    occupancy = step.reduction_pressure(torch.arange(9.), allowance=7, rounds_left=3)
    assert occupancy[0] == 0
    assert (occupancy[1:] > occupancy[:-1]).all()
    urgency = step.reduction_pressure(torch.tensor([4., 4., 4.]),
        allowance=2, rounds_left=torch.tensor([4., 2., 1.]))
    assert (urgency[1:] > urgency[:-1]).all()
    torch.testing.assert_close(urgency, torch.tensor([2.5, 3., 4.]))


def test_pressure_enters_credit_but_temperature_does_not():
    step = unary_first()
    with torch.no_grad():
        step.apply_anchor.zero_()
    x = torch.ones(1, 2, 1)
    step.reduce_pressure = 0.
    _, _, unloaded = step(x, slots=2, allowance=2, rounds_left=3)
    step.reduce_pressure = 1.
    for temperature in (0., .25, 2.):
        step.temperature = temperature
        _, _, loaded = step(x, slots=2, allowance=2, rounds_left=3)
        torch.testing.assert_close(loaded['probabilities'],
                                   torch.tensor([[1., 0., 0., 0.]]).softmax(-1))
        assert loaded['probabilities'][0, 0] > unloaded['probabilities'][0, 0]


def test_deadline_masks_unary_and_stop_in_the_credit_distribution():
    step = unary_first()
    # The relative row fits three slots, but the online allowance is two.
    _, _, route = step(torch.ones(1, 3, 1), slots=3, allowance=2, rounds_left=1)
    assert route['kind'].tolist() == [1]
    assert route['unary_probabilities'].count_nonzero() == 0
    assert route['probabilities'][0, -1] == 0


def test_early_stop_cannot_leave_a_small_relative_stack_full():
    step = unary_first()
    with torch.no_grad():
        step.stop_anchor.fill_(2000.)
    _, _, route = step(torch.ones(1, 2, 1), slots=3, allowance=1, rounds_left=3)
    assert route['kind'].tolist() == [2]
    assert route['probabilities'][0, -1] == 0


def test_pressure_and_deadlines_keep_fullgraph_forward_and_backward():
    step = unary_first()
    x = torch.ones(2, 4, 1, requires_grad=True)
    def run(value, count):
        _, path, route = step(value, depth=count, slots=3, allowance=3, rounds_left=1)
        return path, route['probabilities'], route['kind']
    compiled = torch.compile(run, backend='aot_eager', fullgraph=True)
    for depth in (torch.tensor([4, 2]), torch.tensor([3, 1])):
        eager, captured = run(x, depth), compiled(x, depth)
        for expected, actual in zip(eager, captured):
            torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(eager[0].square().sum(), x)[0]
        b = torch.autograd.grad(captured[0].square().sum(), x)[0]
        torch.testing.assert_close(a, b)


def test_native_packed_unary_preference_still_records_every_sentence(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'unary_pressure_memory', [
        ('<architecture>', '<architecture><ltmConsolidation>true</ltmConsolidation>'),
        ('<serialWordCapacity>8</serialWordCapacity>', '<serialWordCapacity>16</serialWordCapacity>'),
        ('<serialWordBuckets>8</serialWordBuckets>', '<serialWordBuckets>16</serialWordBuckets>'),
    ])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    # This mechanism measures native admission/recording, without a reverse
    # objective. The three existing observation fixtures retain their losses.
    model.reconstruct_in_loop = False
    model.loss.reconstruction_scale = 0.
    chooser = model.languageSpace._tree_layer(2).chooser
    original = chooser.score_unary
    def prefer_unary(*args, **kwargs):
        stop, unary = original(*args, **kwargs)
        return stop, unary + 1e6
    monkeypatch.setattr(chooser, 'score_unary', prefer_unary)
    try:
        store = model.symbolSpace.ltm_store
        before = int((store.rel_type[:len(store)] != store.REL_DEF).sum())
        inputs = model.inputSpace.prepPackedInput([['1 plus 2', '3 plus 4']])
        from types import SimpleNamespace
        from reading_fixtures import capture_operation_traces
        with capture_operation_traces(model) as traces:
            model.runBatch(train=False, batchSize=1, split='runtime',
                           batch_override=(inputs, torch.zeros(1, 1, 0)))
        assert int((store.rel_type[:len(store)] != store.REL_DEF).sum()) - before == 2
        assert (model._tensor_sentence_roots_depth[0, :2] > 0).all()
        # The caller observes the open readings; the completed rows discard
        # their operation records. Retain the same unary-execution assertion.
        trace = SimpleNamespace(**{name: torch.cat([getattr(item, name) for item in traces], 1)
            for name in ('_choice_arities', '_choice_mask')})
        assert (trace._choice_arities[trace._choice_mask] == 1).any()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.parametrize('value', [-1., float('inf'), float('nan')])
def test_pressure_weight_must_be_finite_and_nonnegative(value):
    with pytest.raises(ValueError, match='pressure'):
        Language.OperationSelectionLayer(d_model=1, ops=[Add()], reduce_pressure=value)


def test_pressure_default_and_xml_override_share_the_compose_layer():
    from test_compose_pair_driver import _build
    assert unary_first().reduce_pressure == 1.
    model = _build('<reducePressure>0.75</reducePressure>')
    try:
        step = model.symbolSpace.languageLayer.operation_layer
        assert step.reduce_pressure == .75
        assert step is model.languageSpace._tree_layer(2)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
