"""Item 3.0 step 4: chooser opening, structural references and supplied silence."""
from dataclasses import fields, replace
from types import SimpleNamespace
import ast
import inspect
import json
import pytest
import torch

from Meaning import ConceptualMeaning
from Models import BasicModel
from ThoughtReferences import bindings, evidence_pair, has_open_slots, open_slots, with_slots
from What import What, WhatAnswer


def test_native_every_completed_field_reaches_the_chooser(tmp_path, monkeypatch, eager_reading):
    import ThoughtClosing
    from test_parts_of_speech import test_kept_readings_alone_supply_occupation_counts
    pending, offered, menus, work_rows = [], [], [], []
    close, run, choose = ThoughtClosing.close, BasicModel.run_selected_thought, BasicModel._choose_selected_thought_action
    def boundary(model, values, **kwargs):
        pending.extend((id(model), b) for b in kwargs['ready'])
        return close(model, values, **kwargs)
    def offer(model, meaning, **kwargs):
        offered.append((id(model), kwargs.get('row', 0)))
        return run(model, meaning, **kwargs)
    def menu(model, root, active, actions, **kwargs):
        memory = model._what_memory()
        state = memory.thought_state(b=kwargs['row'])
        first = state is None or state.finished or state.work_spent == 0
        if first:
            opened = has_open_slots(active)
            advances = any(a is not None and a.semantic_id != 'not' for a in actions)
            if not opened:
                assert None in actions
            elif kwargs['work'].remaining and advances:
                assert None not in actions
            menus.append((opened, None in actions))
        return choose(model, root, active, actions, **kwargs)
    path_cost = BasicModel._sentence_path_cost
    def cost(model, state, sid, active):
        # A trial sees its provisional brackets as already paid. Scoring
        # silence must not read with the pre-bracket remaining allowance.
        reading = getattr(model, '_attention_words', None)
        if reading is not None:
            for b, meter in enumerate(model._attention_meters):
                if bool(active[b]):
                    assert meter.spent >= int(reading.table.spent[b])
                    work_rows.append(dict(row=b, spent=meter.spent, remaining=meter.remaining))
        return path_cost(model, state, sid, active)
    monkeypatch.setattr(BasicModel, '_sentence_path_cost', cost)
    monkeypatch.setattr(ThoughtClosing, 'close', boundary)
    monkeypatch.setattr(BasicModel, 'run_selected_thought', offer)
    monkeypatch.setattr(BasicModel, '_choose_selected_thought_action', menu)
    test_kept_readings_alone_supply_occupation_counts(tmp_path, monkeypatch, eager_reading)
    assert pending and offered == pending and menus
    print('RETIRE_MODE_OFFERS', json.dumps(dict(completed=len(pending), offered=len(offered), menus=menus, work=work_rows)))


@pytest.mark.parametrize('opening', [False, True])
def test_closed_first_decision_records_only_a_chosen_episode(monkeypatch, opening):
    from test_normal_thought_controller import _catalog_world
    from meaning_fixtures import replay_first_choice
    model, registry, memory, part, whole = _catalog_world()
    value = registry.form('isPart', part, whole)
    assert not has_open_slots(value)
    replay_first_choice(model, monkeypatch, 0 if opening else -1)
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(value, work_budget=16)
    if opening:
        assert result.records[0].kind == 'begin' and result.records[0].opened_by_chooser
        assert result.work.spent == memory.thought_state().work_spent > 0
    else:
        assert result.records == () and memory.thought_history() == []
        assert result.work.spent == 0 and result.result is None
        assert result.evidence['evidence_kind'] == 'bound'
        torch.testing.assert_close(result.meaning.roles, value.roles, rtol=0, atol=0)


def test_open_first_menu_keeps_the_search_rule(monkeypatch):
    from test_normal_thought_controller import _catalog_world
    from meaning_fixtures import replay_thought_plan
    model, registry, memory, part, _ = _catalog_world()
    value = registry.form('isPart', part, open_roles=('argin2',))
    def plan(root, active, actions, **kwargs):
        if memory.thought_state().work_spent == 0:
            assert None not in actions
            return next(a for a in actions if a.semantic_id == 'query')
        return None if None in actions else next(a for a in actions if a.semantic_id == 'query')
    replay_thought_plan(model, monkeypatch, plan)
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(value, work_budget=16)
    assert result.records and not result.records[0].opened_by_chooser


def test_mode_api_and_old_semantic_layout_are_rejected():
    import ThoughtReferences
    from Language import AskLayer, Grammar
    from Layers import LTM0
    assert 'mode' not in {f.name for f in fields(ConceptualMeaning)}
    assert 'mode' not in ConceptualMeaning.from_description(torch.ones(4)).metadata()
    assert not hasattr(AskLayer, 'meaning_mode') and not hasattr(ThoughtReferences, 'question')
    assert 'meaning_mode' not in Grammar.RuleDef._fields
    store = LTM0(4, capacity=4)
    old = dict(store.semantic_extras(), version=9)
    with pytest.raises(ValueError, match='version'):
        store.load_semantic_extras(old)


@pytest.mark.parametrize('body', ['atom', ('part', 'atom', 'whole')])
def test_ask_kept_head_zeroes_positive_evidence_in_every_recovery_path(monkeypatch, body):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import finish_reading
    f = SentenceFixture(monkeypatch)
    def recover(tree):
        reading = f.program(tree)
        reading = replace(reading, leaf_evidence=torch.tensor([[.8, .2]]).expand(len(reading.leaves), 2))
        return finish_reading(f.language, reading, registry=f.registry)
    asserted, asked = recover(body), recover(('ask', body))
    assert asserted.evidence == pytest.approx((.8, .2))
    assert asked.evidence == (0., 0.) and evidence_pair(asked.meaning) == (0., 0.)
    assert bindings(asked.meaning)['_source_evidence'] == (0., 0.)


def test_closed_zero_is_unverified_and_only_open_references_make_question(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from Models import _append_observed_meaning
    f = SentenceFixture(monkeypatch)
    closed = f.clause(('ask', ('part', 'atom', 'whole')))
    index = _append_observed_meaning(f.store, closed)
    assert f.store.row(index)['kind'] == 'unverified'
    assert f.store.row(index)['evidence'] == (0., 0.)
    from ReferenceClose import ReferenceClose
    assert ReferenceClose._commit_events(SimpleNamespace(), {'prepared_root': closed}, {}, when=1) == set()
    from ClauseRow import Clause
    child = Clause(with_slots(ConceptualMeaning.from_description(closed.meaning.roles[0]),
        (('referent', 0),), pair=(0., 0.)), point=closed.meaning.roles[0])
    opened = replace(closed, composition=None, children=(child,), refs=(('clause', 0), *closed.refs[1:]),
        meaning=with_slots(replace(closed.meaning, role_refs=(None, None, None)),
                           (('referent', 0),), pair=(0., 0.)))
    row = _append_observed_meaning(f.store, opened)
    assert f.store.row(row)['kind'] == 'question'
    with f.store.clause_assertions(trust=1., origin=f.store.ORIGIN_USER, text='source'):
        with pytest.raises(ValueError, match='open references'):
            f.store.write_clause(opened)
        fact = f.store.write_clause(f.clause('atom'))
    assert f.store.row(fact)['kind'] == 'fact'


def test_silence_is_available_only_on_a_supplied_split():
    from data import Data
    data = Data()
    data.train_input = ['atom', 'whole', 'other']
    data.train_output = [None, '', torch.tensor([1., 2.])]
    data.has_supervised_outputs = True
    for row in (0, 1):
        answer = data.what(What.supervised(row))
        assert answer.available and answer.what == '' and answer.provenance == 'silence'
    assert data.what(What.supervised(2)).provenance == 'data'
    assert not data.what(What.inference(0, split='train')).available
    data.has_supervised_outputs = False
    assert not data.what(What.supervised(0)).available


def test_tensor_silence_is_zero_and_unavailable_rows_are_masked():
    questions = tuple(What.supervised(i) for i in range(3))
    desired = (WhatAnswer.silence(questions[0]), WhatAnswer(questions[1], torch.tensor([2., 3.])),
               WhatAnswer.unavailable(questions[2], 'no target'))
    model = BasicModel()
    # This isolated target fixture has three rows and no space construction.
    model._answer_silence_mask = torch.zeros(3, dtype=torch.bool)
    object.__setattr__(model, 'inputSpace', SimpleNamespace(data=SimpleNamespace(has_supervised_outputs=True)))
    model._last_what_desired = desired
    target, mask = model._what_answer_target(questions, torch.ones(3, 1, 2), supervised_only=True)
    assert mask.tolist() == [True, True, False]
    assert model._answer_silence_mask.tolist() == [True, False, False]
    torch.testing.assert_close(target, torch.tensor([[[0., 0.]], [[2., 3.]], [[0., 0.]]]))


def test_surface_silence_uses_the_real_empty_embedding(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=2, word_buckets='16')
    try:
        monkeypatch.setattr(model.inputSpace.data, 'has_supervised_outputs', True)
        questions = (What.supervised(0), What.supervised(1))
        empty = model._embed_answer_texts(['', ''])
        spoken = model._embed_answer_texts(['atom', ''])
        assert torch.is_tensor(empty) and empty.shape[1] > 0
        model._last_answer_construction = SimpleNamespace(surface=empty)
        model._last_what_desired = (WhatAnswer.silence(questions[0]), WhatAnswer(questions[1], 'atom'))
        target, mask = model._what_answer_target(questions, torch.ones(2, 1, 1), supervised_only=True)
        assert mask.tolist() == [True, True]
        assert model._answer_silence_mask.tolist() == [True, False]
        assert model._reverse_event_loss(empty[:1], target[:1]) == 0
        assert model._reverse_event_loss(spoken[:1], target[:1]) > 0
        records = [dict(expected='', actual='atom', provenance='silence'),
                   dict(expected='atom', actual='', provenance='data')]
        print('RETIRE_MODE_SILENCE', json.dumps(dict(records=records,
            spoken_when_silent=1, silent_when_asked=1, empty_shape=list(empty.shape))))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('opened', [(), (0,), (1,), (2,), (0, 2), (0, 1, 2)])
def test_expectation_spares_exactly_the_structural_open_roles(opened):
    from Models import _open_reference_mask
    calls = []
    def slot(role):
        calls.append(role)
        return {'argin1': 0, 'argin2': 2}[role]
    refs = tuple(None if i in opened else ('sym', i+1) for i in range(3))
    meaning = ConceptualMeaning(torch.eye(3), torch.ones(3, dtype=torch.bool),
        role_refs=refs, sentence_kind='relation')
    mask = _open_reference_mask(meaning, SimpleNamespace(_slot_for_operand=slot))
    assert mask.tolist() == [float(i in opened) for i in range(3)]
    assert calls == [name for i, name in ((0,'argin1'),(2,'argin2')) if i in opened]


def test_public_open_field_uses_the_same_controller(monkeypatch):
    from test_normal_thought_controller import _catalog_world
    from meaning_fixtures import replay_thought_plan
    model, registry, memory, part, _ = _catalog_world()
    value = registry.form('isPart', part, open_roles=('argin2',))
    def plan(root, active, actions, **kwargs):
        return None if None in actions else next(a for a in actions if a.semantic_id == 'query')
    replay_thought_plan(model, monkeypatch, plan)
    result = model._run_public_thought(value, work_budget=16)
    assert result is not None and result.records[0].kind == 'begin'
    assert memory.thought_state().finished and not memory.in_episode(0)


def test_opening_fork_can_keep_zero_work_without_history(monkeypatch):
    import ThoughtStream, WalkTrials
    from test_normal_thought_controller import _catalog_world
    from meaning_fixtures import replay_thought_plan
    model, registry, memory, part, whole = _catalog_world()
    value = registry.form('isPart', part, whole)
    candidates = ThoughtStream.candidates
    def one_operation(*args, **kwargs):
        return tuple(a for a in candidates(*args, **kwargs) if a.semantic_id == 'isPart')[:1]
    monkeypatch.setattr(ThoughtStream, 'candidates', one_operation)
    capture = WalkTrials.capture_thought_fork
    def opening_only(model, trace, meter, **kwargs):
        if len(trace['actions']) == 1:
            capture(model, trace, meter, **kwargs)
    monkeypatch.setattr(WalkTrials, 'capture_thought_fork', opening_only)
    replay_thought_plan(model, monkeypatch, lambda root, active, actions, **kwargs:
        actions[0] if memory.thought_state() is None else None)
    model.train()
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(value, work_budget=16,
            score=lambda result: float(bool(result.records)))
    comparison = model._last_thought_comparison
    assert comparison['departure'] == 0 and comparison['explore_kept']
    assert comparison['work_spent'][0] > 0 and comparison['work_spent'][1] == 0
    assert result.records == () and result.work.spent == 0
    assert memory.thought_history() == [] and not memory.in_episode(0)


@pytest.mark.parametrize('operation', ['query', 'exist'])
@pytest.mark.parametrize('budget', [0, 8])
def test_chooser_opened_read_reuses_unchanged_chain_and_stamp(monkeypatch, budget, operation):
    from ClauseRow import attach_clause_index
    from Layers import LTM0
    from meaning_fixtures import replay_thought_plan
    from test_normal_thought_controller import _catalog_world
    model, registry, memory, part, whole = _catalog_world()
    store = LTM0(8, capacity=8)
    attach_clause_index(model, store, model.conceptualSpace)
    model.symbolSpace.ltm_store = store
    row = store.append_relation(part[1], whole[1], truth_plus=1.)
    reference = store.occurrence_of(row)
    value = store.meaning_of(row)
    value = replace(value, bindings={**bindings(value), '_query_occurrence': reference})
    before = {name:tensor.clone() for name,tensor in store.state_dict().items()}
    replay_thought_plan(model, monkeypatch, [operation, None])
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(value, work_budget=budget)
    assert result.work.spent <= budget and result.records[0].opened_by_chooser
    assert result.records[-1].kind == 'finish'
    if budget == 0:
        assert [record.kind for record in result.records] == ['begin', 'cutoff', 'finish']
    else:
        assert any(record.operation == operation for record in result.records)
    for name, tensor in store.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert store.location(reference) == row


@pytest.mark.parametrize('exposures', [(1., 1.), (.5, .5), (1., .5, .5)])
@pytest.mark.parametrize('target', [0., 2.])
def test_thought_reader_exposure_keeps_priority_and_baselines(exposures, target):
    from Layers import Error
    from ThoughtAnswer import train_readers
    count = len(exposures)
    predicted = torch.arange(1, count+1, dtype=torch.float32, requires_grad=True)
    expected = torch.full_like(predicted, target)
    trials = []
    for row, exposure in enumerate(exposures):
        active = torch.arange(count) == row
        comparison = Error(row_mask=active)
        comparison.squared('answer.what', predicted, expected, weight=.7)
        trials.append((Error(row_mask=active), comparison,
                       dict(reader_weight=0., comparison_weight=exposure)))
    model = SimpleNamespace(errors=Error())
    train_readers(model, trials)
    term = model.errors._terms['thought_comparison.answer.what']
    assert term['weight'] == .7
    denominator = count * (target**2 if target else 1.)
    weighted = (predicted-expected).square() * predicted.new_tensor(exposures)
    wanted = .7 * weighted.sum() / denominator
    actual = model.errors.total()
    torch.testing.assert_close(actual, wanted)
    gradient, = torch.autograd.grad(actual, predicted, retain_graph=True)
    wanted_gradient, = torch.autograd.grad(wanted, predicted)
    torch.testing.assert_close(gradient, wanted_gradient)
    assert all(comparison._terms['answer.what']['weight'] == .7
               for _, comparison, _ in trials)
    if len(set(exposures)) == 1:
        previous = Error()
        for _, comparison, _ in trials:
            previous.merge(comparison, prefix='thought_comparison.', weight=exposures[0])
        torch.testing.assert_close(actual, previous.total())


def test_silent_premise_keeps_full_trial_and_still_trains_output():
    from Layers import Error
    from SentenceCredit import components, comparison, reader_costs, reader_weights
    from test_supplied_answer_training import _sentence
    model, full, parameters = _sentence(True)
    questions = tuple(What.supervised(row) for row in range(2))
    desired = (WhatAnswer.silence(questions[0]), WhatAnswer(questions[1], torch.tensor([0.])))
    model._sentence_answer_questions = questions
    model.inputSpace.data.what = lambda question: desired[question.where]
    model._last_what_desired = desired
    # The fixed two-stream fixture reserves one provenance flag per row.
    model._answer_silence_mask = torch.zeros(2, dtype=torch.bool)
    target, available = BasicModel._what_answer_target(model, questions, torch.zeros(2, 1),
                                                       supervised_only=True)
    assert available.tolist() == [True, True]
    assert model._answer_silence_mask.tolist() == [True, False]
    model._sentence_supplied_answers = target
    held = model._answer_silence_mask.data_ptr()
    quiet = list(full[1])
    for index in (6, 10):
        quiet[index] = quiet[index].clone()
        quiet[index][0] = quiet[index][0] * .1
    lossy = (full[0], tuple(quiet), full[2])
    active = torch.ones(2, dtype=torch.bool)
    registries, parts, raw, answers = [], [], [], []
    for state in (full, lossy):
        registry = Error(row_mask=active)
        answer = BasicModel._sentence_answer_error(model, state, 0, active,
                                                   {'record': None}, registry=registry)
        raw.append(registry.total(objective='output'))
        answers.append(answer)
        # Both paths reconstruct equally; only their returned content differs.
        registry.error('reconstruction', torch.ones(2), 1., objective='reconstruction')
        parts.append(components(registry, None, torch.zeros(2), answer=answer))
        registries.append(registry)
        assert answer[0] == 0 and answer[1] == raw[-1][1]
        assert model._answer_silence_mask.data_ptr() == held
    assert raw[0][0] > raw[1][0] > 0  # The full reading speaks; the lossy one is quieter.
    values = torch.stack(parts, 1)
    kept = comparison(values, active)
    assert kept['keep_decision'][0] == 'tie:greedy' and not kept['wins'][0]
    assert values[0, :, 2].tolist() == [0., 0.]
    assert not kept['answer_against_keep'][0]
    weights = reader_weights(values, active, None)
    assert weights[0].tolist() == [1., 0.]
    assert all(term['trained'] for term in registries[0]._terms.values()
               if term['objective'] == 'output')
    # The ordinary output-owner path still receives the kept silence error.
    training = reader_costs(registries, weights)
    assert sum(value.sum() for value in training.values()) > 0
    ordinary = BasicModel._sentence_reader_error(model, full, 0, active, {'record': None})
    assert ordinary[0] == raw[0][0]  # The thought scorer's shared reader is unchanged.
    asked_gradient, = torch.autograd.grad(answers[0][1], parameters[-1], retain_graph=True)
    assert bool(asked_gradient.abs().any())
    gradient, = torch.autograd.grad(raw[0][0], parameters[-1])
    assert bool(gradient.abs().any())
    print('SILENT_PREMISE_COMPARISON', json.dumps(dict(
        raw_answer=[float(value[0].detach()) for value in raw],
        comparison_answer=values[0, :, 2].tolist(), keep=kept['keep_decision'][0],
        training_silence=float(raw[0][0].detach()), training_gradient=True)))
