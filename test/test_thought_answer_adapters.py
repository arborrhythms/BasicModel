"""Checked non-truth results become owned, full-width answer meanings."""

from dataclasses import replace

import pytest
import torch

from Queries import ThoughtResult
from QueryWork import QueryWorkBudget
from test_normal_thought_controller import _catalog_world


def test_code_answer_uses_the_checked_atom_and_keeps_its_reference():
    from Output import thought_answer_meanings
    model, registry, memory, part, _whole = _catalog_world()
    query = registry.form("quantize", registry._payload(part))
    with model._query_boundary_scope((0,)):
        selected = model.run_selected_thought(query, work_budget=128)
    answers = thought_answer_meanings(selected)
    assert len(answers) == 1
    torch.testing.assert_close(answers[0].roles[0], selected.result.value)
    assert answers[0].role_mask.tolist() == [True, False, False]
    assert answers[0].role_refs[0] == selected.result.evidence["reference"]
    assert not answers[0].roles.requires_grad
    memory.end_what_episode()


def test_open_relation_answer_preserves_every_member_without_another_reader():
    from Output import thought_answer_meanings
    model, registry, memory, part, whole = _catalog_world()
    second = ("sym", model.conceptualSpace.new_concept())
    model.conceptualSpace._csw_concept_row(0, second[1])
    model.conceptualSpace.add_whole(second[1], whole)
    query = registry.form("part", whole, open_roles=("I1",))
    with model._query_boundary_scope((0,)):
        selected = model.run_selected_thought(query, work_budget=128)
    cost = selected.work.spent
    answers = thought_answer_meanings(selected)
    assert {answer.role_refs[0] for answer in answers} == {part, second}
    assert all(answer.role_refs[2] == whole for answer in answers)
    assert all(answer.role_mask.tolist() == [True, True, True] for answer in answers)
    for answer in answers:
        torch.testing.assert_close(answer.roles[0], registry._payload(answer.role_refs[0]))
    assert selected.work.spent == cost == memory.thought_state().work_spent
    # Restoring the checked result cannot need a current codebook read.
    checked = ThoughtResult.from_checkpoint(selected.result.checkpoint())
    restored = thought_answer_meanings(replace(selected, result=checked))
    for first, second_answer in zip(answers, restored):
        torch.testing.assert_close(first.roles, second_answer.roles)
    memory.end_what_episode()


def test_subgoal_carries_the_typed_child_result_into_the_answer():
    from Output import thought_answer_meanings
    model, registry, memory, part, whole = _catalog_world()
    inner = registry.form("part", part, whole)
    entry = memory.begin_thought_episode(inner, work_budget=2)
    reference = memory.thought_reference(entry)
    memory.finish_thought(inner)
    memory.end_what_episode()
    with model._query_boundary_scope((0,)):
        outer = registry.form(
            "what", reference,
            context=model._thought_grammar_context(
                inner, row=0, work=QueryWorkBudget(8), continuation=None))
        selected = model.run_selected_thought(outer, work_budget=128)
    assert selected.result.result_kind == "subgoal"
    assert isinstance(selected.result.value, ThoughtResult)
    assert selected.result.value.semantic_id == "part"
    answers = thought_answer_meanings(selected)
    assert len(answers) == 1
    assert answers[0].role_refs == inner.role_refs
    torch.testing.assert_close(answers[0].roles, inner.roles)
    stored = ThoughtResult.from_checkpoint(selected.result.checkpoint())
    assert isinstance(stored.value, ThoughtResult)
    assert stored.value.request.role_refs == inner.role_refs
    memory.end_what_episode()


def test_missing_code_is_incomplete_rather_than_a_zero_or_request_answer():
    from Output import thought_answer_meanings
    from types import MappingProxyType, SimpleNamespace
    model, registry, _memory, part, _whole = _catalog_world()
    query = registry.form("quantize", registry._payload(part))
    checked = ThoughtResult(
        "quantize", "conceptual-codebook", "code", "concept-codebook",
        query, MappingProxyType({"value": None, "incomplete": ("work_budget",)}))
    assert thought_answer_meanings(SimpleNamespace(meaning=query, result=checked)) == ()




@pytest.mark.parametrize('kind', ['set', 'code', 'subgoal'])
def test_typed_results_survive_resolve_and_reverse_without_execution(monkeypatch, kind):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from Understanding import AnswerProgram, Understanding
    from What import What
    model, registry, memory, part, whole = _catalog_world()
    query = registry.form('part', part, whole)
    if kind == 'set':
        extra = ('sym', model.conceptualSpace.new_concept())
        model.conceptualSpace._csw_concept_row(0, extra[1])
        model.conceptualSpace.add_whole(extra[1], whole)
        query = registry.form('part', whole, open_roles=('I1',))
    elif kind == 'code':
        query = registry.form('quantize', registry._payload(part))
    else:
        from Layers import TernaryTruthStore
        model.symbolSpace.ltm_store = TernaryTruthStore(8, capacity=16)
        query = registry.form('what', query)
    entry = AnswerProgram(rows=torch.tensor([1]), word_rows=torch.tensor([1]),
        concept_ids=torch.tensor([part[1]]), activations=torch.ones(1),
        leaves=registry._payload(part)[None], actions=torch.tensor([[0, -1, 0]]),
        targets=torch.tensor([-1]), end_state=torch.zeros(3, 8))
    object.__setattr__(model, 'languageSpace', SimpleNamespace(
        program_meaning=lambda item, _registry: query))
    object.__setattr__(model.conceptualSpace, 'stm', SimpleNamespace(concept_dim=8))
    model._materialize_entries = lambda _entries, base, _budget: (base, None)
    model._what_grammar_context = lambda *_a, **_k: (torch.zeros(1, 8), None)
    model._select_perceptual_bindings = lambda *_a: ()
    model._condition_answer_on_question = lambda idea, _context: idea
    model._synthesis_guard = nullcontext
    model.conceptualSpace.synthesize_idea = lambda idea, **_k: idea
    object.__setattr__(model, 'perceptualSpace', SimpleNamespace(synthesize=lambda idea, **_k: idea))
    object.__setattr__(model, 'outputSpace', SimpleNamespace(from_percepts=lambda idea: idea))
    model.selected_thought_budget = 128
    model.reconstruct_in_loop = False
    understanding = Understanding(answer_program=(entry,))
    model.eval()
    derivation = model.resolveAnswer(understanding, What.supervised(0))
    assert derivation.source == 'thought-' + kind
    assert len(derivation.answer_meanings[0]) == (2 if kind == 'set' else 1)
    assert derivation.resolved
    monkeypatch.setattr(registry, 'execute', lambda *_a, **_k: pytest.fail('output executed a query'))
    result = model.reverseOutput(understanding, derivation)
    expected = torch.cat(tuple(item.roles for item in derivation.answer_meanings[0]))
    torch.testing.assert_close(result.actual[0], expected)
    assert memory.thought_state().finished
