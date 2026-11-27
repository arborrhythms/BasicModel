"""Working-memory load and deadlines in the one-softmax chooser: the load
left is a path cost, and the chooser reads the rounds left as a code."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import Language
from test_compose_operations import Add
from chooser_steering import stand_in


def unary_first():
    step = Language.OperationSelectionLayer(d_model=1, ops=[Add()],
                                            unary_ops=[nn.Identity()])
    stand_in(step)
    with torch.no_grad():
        step.stand_in.reduce.zero_()
        step.stand_in.unary.fill_(1000.)
        step.stand_in.stop.zero_()
    return step


def test_online_deadline_uses_full_depth_behind_the_two_slot_window():
    from Models import BasicModel
    from Spaces import ConceptualSpace
    from ChunkJournal import ChunkJournal
    step = unary_first()
    owner = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=step),
                            _compose_binary_rules=(), _compose_unary_rules=(),
                            _structural_context=lambda **kwargs: None,
                            # The chosen rule's head role is read from the chunk
                            # journal's rule tables (573f1a5ca1).
                            chunk_journal=ChunkJournal((), ()))
    owner._symbol_space = SimpleNamespace(conceptualSpace=SimpleNamespace(
        _model=SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=SimpleNamespace(when_start=None)))))
    # Before the word arrives there are K-1 terms. After its deposit, d=K.
    state = (torch.ones(1, 8, 1), torch.tensor([8]),
             torch.zeros(1, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
             torch.full((1, 8), -1, dtype=torch.long), torch.ones(1, 8))
    from chunk_fixtures import from_stack
    state = from_stack(*state)
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


def test_unary_preference_still_completes_absolute_and_relative_ends():
    step = unary_first()
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


def test_the_load_left_is_a_path_cost_linear_in_what_is_left():
    """Rule 75: no learned number multiplies a count. A program's cost is
    the depth held at each step over the allowance, times reductionCost:
    three words read and folded by two binaries hold 1, 2, 1, 2, 1."""
    from WholePaths import load_cost
    language = SimpleNamespace(reduction_cost=1.,
        _compose_binary_rules=(SimpleNamespace(method_name='sum'), SimpleNamespace(method_name='bind')))
    model = SimpleNamespace(languageSpace=language,
                            conceptualSpace=SimpleNamespace(stm=SimpleNamespace(capacity=8)))
    program = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 2], [1, 0, 0], [-1, 0, 0]])
    bound = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 1, 0], [-1, 0, 0]])
    cost = load_cost(model, dict(entries=(program, bound, None)), torch.zeros(3))
    torch.testing.assert_close(cost, torch.tensor([7 / 7, (1 + 2 + 2) / 7, 0.]))
    language.reduction_cost = 2.
    cost = load_cost(model, dict(entries=(program,)), torch.zeros(1))
    torch.testing.assert_close(cost, torch.tensor([2.]))
    assert not any('pressure' in name for name, _ in unary_first().named_parameters())


def test_the_chooser_reads_the_rounds_left_as_a_sinusoid_and_can_learn_the_cost():
    """Rule 75: the chooser is shown how much is left of the budget in hand,
    in the measure of .where; its reader starts at zero and the gradient of
    a score reaches it."""
    from Language import budget_code, budget_encoding, OperationSelectionLayer
    code = budget_code(budget_encoding(), torch.tensor([3, 2, 1, 0]), 4, device='cpu', dtype=torch.float32)
    assert code.shape == (4, 4) and bool((code >= 0).all() and (code <= 1).all())
    assert len({tuple(row) for row in code.tolist()}) == 4
    step = OperationSelectionLayer(d_model=2, ops=[Add()])
    x = torch.ones(1, 3, 2)
    _, _, early = step(x, slots=3, allowance=3, rounds_left=3)
    _, _, late = step(x, slots=3, allowance=3, rounds_left=1)
    torch.testing.assert_close(early['logits'], late['logits'])
    with torch.no_grad():
        step.chooser.budget_projection.weight.normal_()
    _, _, early = step(x, slots=3, allowance=3, rounds_left=3)
    _, _, late = step(x, slots=3, allowance=3, rounds_left=1)
    assert not torch.equal(early['logits'], late['logits'])
    early['probability'].sum().backward()
    assert step.chooser.budget_projection.weight.grad.abs().sum() > 0


def test_temperature_does_not_enter_credit():
    step = unary_first()
    with torch.no_grad():
        step.stand_in.unary.zero_()
    x = torch.ones(1, 2, 1)
    _, _, cold = step(x, slots=2, allowance=2, rounds_left=3)
    for temperature in (.25, 2.):
        step.temperature = temperature
        _, _, warm = step(x, slots=2, allowance=2, rounds_left=3)
        torch.testing.assert_close(warm['probabilities'], cold['probabilities'])
    torch.testing.assert_close(cold['probabilities'], torch.tensor(
        [[-step.operator_cost, -step.operator_cost, -step.operator_cost, 0.]]).softmax(-1))


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
        step.stand_in.stop.fill_(2000.)
    _, _, route = step(torch.ones(1, 2, 1), slots=3, allowance=1, rounds_left=3)
    assert route['kind'].tolist() == [2]
    assert route['probabilities'][0, -1] == 0


def test_deadlines_keep_fullgraph_forward_and_backward():
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
    model.inputSpace.data.has_supervised_outputs = False  # Input/storage probe has no answer target.
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
        before = len(store) + len(store.parts)
        completed = []
        write = store.write_clause
        def observed_write(*args, **kwargs):
            row = write(*args, **kwargs)
            completed.append(row)
            return row
        monkeypatch.setattr(store, 'write_clause', observed_write)
        # MTM nominates the words on their concept_mint_count-th reading
        # (sigma-pi §21, rule 20). Read once, the words are not nominated,
        # and in about half the walks each sentence's closing thought is
        # written as a second root beside its clause.
        from nomination_fixtures import nominated_forms
        nominated_forms(model, ['1 plus 2', '3 plus 4'])
        inputs = model.inputSpace.prepPackedInput([['1 plus 2', '3 plus 4']])
        from types import SimpleNamespace
        from reading_fixtures import capture_operation_traces
        with capture_operation_traces(model) as traces:
            model.runBatch(train=False, batchSize=1, split='runtime',
                           batch_override=(inputs, torch.zeros(1, 1, 0)))
        # A completed sentence can own embedded clause rows. Count its
        # two root writes, preserving the nested rows the grammar chose.
        assert len(completed) == 2 and len(set(completed)) == 2
        assert all(store.row(row)['meaning'] is not None for row in completed)
        assert len(store) + len(store.parts) - before >= 2
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
def test_reduction_cost_must_be_finite_and_nonnegative(value, monkeypatch):
    read = Language.TheXMLConfig.get
    monkeypatch.setattr(Language.TheXMLConfig, 'get', lambda key, *args, **kwargs:
        value if key == 'architecture.reductionCost' else read(key, *args, **kwargs))
    with pytest.raises(ValueError, match='reductionCost'):
        Language.configured_reduction_cost()


def test_reduction_cost_default_and_xml_override_and_the_retired_pressure():
    from test_compose_pair_driver import _build
    model = _build('<reductionCost>0.75</reductionCost>')
    try:
        assert model.languageSpace.reduction_cost == .75
        assert not hasattr(model.symbolSpace.languageLayer.operation_layer, 'reduce_pressure')
    finally:
        model.End()
        model.symbolSpace.soft_reset()
    from pathlib import Path
    from util import XMLConfig
    defaults = XMLConfig(defaults_path=str(Path(__file__).resolve().parents[1] / 'data/model.xml'))
    assert defaults.get('architecture.reductionCost') == 1
    with pytest.raises(ValueError, match='reducePressure'):
        _build('<reducePressure>0.75</reducePressure>')


def test_last_relative_closing_round_cannot_reduce_three_slots_to_two():
    step = unary_first()
    x = torch.ones(1, 2, 1)
    # The serial chooser's window contains only two operands. The deadline
    # concerns the three occupied chunks behind it.
    for closing, rounds, reducing in ((True, 1, False), (True, 2, True), (False, 1, True)):
        _, _, route = step(x, depth=torch.tensor([2]), slots=3, allowance=3, load_depth=torch.tensor([3]),
                           rounds_left=rounds, closing=closing)
        assert bool(route['logits'][0, 0].isfinite()) == reducing
        assert route['logits'][0, -1].isfinite()
    _, _, route = step(x, slots=1, load_depth=torch.tensor([2]), rounds_left=1, closing=True)
    assert route['kind'].tolist() == [1]
