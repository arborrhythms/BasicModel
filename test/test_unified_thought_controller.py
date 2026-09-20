"""Item 1: all public reasoning uses the ordinary grammatical controller."""

import pytest
import torch

from Models import SelectedThoughtResult
from test_normal_thought_controller import _catalog_world


@pytest.mark.parametrize("entry", ["reason_about", "think_about"])
def test_public_reasoning_uses_one_completed_boundary_and_episode(entry, monkeypatch):
    model, registry, memory, part, whole = _catalog_world()
    model.reasoning_iterations = model.thinking_budget = 64
    question = registry.form("part", part, whole)
    calls = []
    run = model.run_selected_thought

    def checked(meaning, **kwargs):
        model._assert_query_boundary(kwargs.get("row", 0))
        calls.append(meaning)
        return run(meaning, **kwargs)

    monkeypatch.setattr(model, "run_selected_thought", checked)
    result = getattr(model, entry)(question)

    assert isinstance(result, SelectedThoughtResult)
    assert calls == [question]
    assert result.evidence["support_true"] == 1.0
    assert result.work.spent == memory.thought_state().work_spent
    assert memory.thought_state().finished
    assert not memory.in_episode(0)
    assert model._query_ready_rows is None
    assert [record.kind for record in memory.thought_history()].count("begin") == 1


def test_public_answer_accepts_the_completed_meaning_without_surface_dispatch(monkeypatch):
    model, registry, memory, part, whole = _catalog_world()
    model.reasoning_iterations = 64
    question = registry.form("part", part, whole)
    monkeypatch.setattr(model, "_detect_query", lambda *_args: pytest.fail(
        "a surface dispatcher selected the thought"), raising=False)

    payload = model.answer_query(question)

    assert payload["support_true"] == 1.0
    assert isinstance(model._last_selected_thought, SelectedThoughtResult)
    assert memory.thought_state().finished and not memory.in_episode(0)
    assert "kernel" not in payload


def test_candidate_payloads_are_detached_from_the_hard_choice():
    model, registry, _memory, part, whole = _catalog_world()
    root = registry.form("part", part, whole)
    from dataclasses import replace
    values = root.roles.detach().clone().requires_grad_()
    candidate = replace(root, roles=values)
    context = model._selected_thought_context(
        root, root, candidate, level=0, pressure=0.0)
    chooser = model._selected_thought_chooser(root)
    with torch.no_grad():
        for parameter in chooser.parameters():
            parameter.fill_(0.01)
    logits = chooser.logits(torch.stack((context, context)), (False, True))
    logits.sum().backward()

    assert values.grad is None or not bool(values.grad.any())
    assert any(parameter.grad is not None and bool(parameter.grad.any())
               for parameter in chooser.parameters())


def test_child_context_can_refine_before_returning_its_causal_result(monkeypatch):
    from QueryWork import QueryWorkBudget
    model, registry, memory, part, whole = _catalog_world()
    child = registry.form("part", part, whole)
    record = memory.begin_thought_episode(child, work_budget=2)
    reference = memory.thought_reference(record)
    memory.finish_thought(child)
    memory.end_what_episode()
    child_choices = []

    def choose(_root, _active, actions, *, level, evidence=None, **_kwargs):
        if level == 0:
            if evidence is not None:
                return None
            return next(action for action in actions
                        if action is not None and action.semantic_id == "what")
        child_choices.append(evidence)
        if len(child_choices) > 2:
            return None
        target = "part" if len(child_choices) == 1 else "equal"
        return next(action for action in actions
                    if action is not None and action.semantic_id == target)

    monkeypatch.setattr(model, "_choose_selected_thought_action", choose)
    with model._query_boundary_scope((0,)):
        outer = registry.form("what", reference, context=model._thought_grammar_context(
            child, row=0, work=QueryWorkBudget(8), continuation=None))
        selected = model.run_selected_thought(outer, work_budget=128)
    assert len(child_choices) == 3
    assert child_choices[1]["support_true"] == 1.0
    assert selected.result.value.semantic_id == "equal"
    child_ops = [record.operation for record in selected.records
                 if record.level == 1 and record.kind == "thought"]
    assert child_ops == ["part", "equal", "conclude"]
    returned = next(record for record in selected.records if record.kind == "return")
    conclusion = next(record for record in selected.records
                      if record.level == 0 and record.operation == "conclude")
    assert memory.thought_reference(returned) in conclusion.sources
    assert selected.work.spent == memory.thought_state().work_spent <= 128
    assert memory.thought_state().level == 0 and memory.thought_state().finished
    memory.end_what_episode()


def test_retired_controllers_cannot_be_constructed_or_restored():
    import reasoning
    from Models import BasicModel
    assert not hasattr(reasoning, 'NeuralToolUser')
    assert not hasattr(BasicModel, '_what_step_chooser')
    assert not hasattr(BasicModel, '_thinking_policy_loss')
    assert not hasattr(BasicModel, 'reason_predict_next')
    assert not hasattr(reasoning, 'NextIdeaScorer')
    model, *_ = _catalog_world()
    state = {prefix + 'mlp.0.weight': torch.ones(2, 2) for prefix in (
        'what_step_chooser.', '_next_op_policy.', '_predict_next_scorer.')}
    model._materialize_answer_path_from_checkpoint(state)
    assert state == {}


def test_policy_charges_shared_episode_work_not_the_number_of_choices():
    model, registry, memory, part, whole = _catalog_world()
    model.selected_thought_policy_weight = 1
    model.eval()
    model._selected_thought_policy_baseline = 0.0
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(registry.form('part', part, whole), work_budget=64)
    loss = model._selected_thought_policy_loss(torch.tensor(0.0))
    report = model._what_report_state()
    assert report['policy_selected_thought_return_sum'] == pytest.approx(
        -model.WHAT_STEP_COST * result.work.spent)
    assert loss is not None and loss.requires_grad
    memory.end_what_episode()


def test_runbatch_credits_each_controller_row_from_its_own_answer(monkeypatch):
    from test_output_walk import _model, _capture_program_probe
    from What import What
    model = _model()
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    try:
        with torch.no_grad():
            understood = _capture_program_probe(model, ['12 plus 1', '3 plus 4'])
        model._staged_in_sub = None
        registry = model.grammatical_thoughts
        ref = ('sym', int(understood.answer_program[0].concept_ids[0]))
        query = registry.form('equal', ref, ref)
        monkeypatch.setattr(model.languageSpace, 'program_meaning', lambda *_a: query)
        model.selected_thought_policy_weight = 1
        model.selected_thought_budget = 64
        observed = []
        policy_loss = model._selected_thought_policy_loss
        def credit(errors, mask=None):
            pred = model._align_output_pred(model._last_answer_construction.actual,
                                            model._last_answer_target)
            expected = torch.stack([model.loss.compute(pred[b:b + 1],
                model._last_answer_target[b:b + 1]) for b in range(2)])
            assert errors.shape == (2,) and mask.tolist() == [True, True]
            torch.testing.assert_close(errors, expected.detach())
            assert not errors.requires_grad
            result = policy_loss(errors, mask)
            assert result is not None and result.requires_grad
            observed.append(True)
            return result
        monkeypatch.setattr(model, '_selected_thought_policy_loss', credit)
        optimizer = model.getOptimizer(lr=.001)
        batch = (model.inputSpace.prepInput(['12 plus 1', '3 plus 4']), torch.zeros(2, 1, 1))
        model.runBatch(train=True, batchSize=2, optimizer=optimizer, batch_override=batch,
            questions=(What.supervised(0), What.supervised(1)))
        assert observed == [True]
        chooser = next(iter(model.selected_thought_choosers.values()))
        assert bool(chooser.mlp[-1].weight.detach().abs().any())
        owned = [id(p) for group in optimizer.param_groups for p in group['params']]
        assert all(owned.count(id(p)) == 1 for p in chooser.parameters())
        assert not model._selected_thought_policy_records
        assert all(not model._what_memory().in_episode(row) for row in range(2))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('budget', [0, 1, 6, 12, 24])
def test_nested_cutoff_drains_the_actual_depth_without_fresh_work(budget):
    from Layers import TernaryTruthStore
    model, registry, memory, part, whole = _catalog_world()
    model.symbolSpace.ltm_store = TernaryTruthStore(8, capacity=32)
    question = registry.form('part', part, whole)
    for _ in range(4):
        question = registry.form('what', question)
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(question, work_budget=budget)
    state = memory.thought_state()
    assert state.finished and state.level == 0
    assert state.work_spent == result.work.spent <= budget
    cutoff = next((record for record in result.records if record.kind == 'cutoff'), None)
    if cutoff is not None:
        tail = result.records[result.records.index(cutoff) + 1:]
        assert len(tail) == cutoff.level + 1
        assert all(record.cost == 0 and record.kind in ('return', 'finish') for record in tail)
    memory.end_what_episode()


def test_policy_masks_other_rows_and_drops_failed_episode_graphs(monkeypatch):
    model, registry, memory, part, whole = _catalog_world()
    model.selected_thought_policy_weight = 1
    model.eval()
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(registry.form('part', part, whole), work_budget=64)
    records = list(model._selected_thought_policy_records)
    assert model._selected_thought_policy_loss(torch.tensor(1.), mask=torch.tensor([False])) is None
    assert model._selected_thought_policy_records == records
    model._end_what_episodes()
    assert not model._selected_thought_policy_records
    monkeypatch.setattr(registry, 'execute', lambda *_a: (_ for _ in ()).throw(ValueError('reader error')))
    with pytest.raises(ValueError, match='reader error'), model._query_boundary_scope((0,)):
        model.run_selected_thought(registry.form('part', part, whole), work_budget=64)
    assert memory.thought_state().finished and not model._selected_thought_policy_records
    memory.end_what_episode()
