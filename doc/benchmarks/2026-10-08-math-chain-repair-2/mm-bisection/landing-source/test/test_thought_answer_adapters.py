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
    query = registry.form("exist", registry._payload(part))
    with model._query_boundary_scope((0,)):
        selected = model.run_selected_thought(query, work_budget=128)
    answers = thought_answer_meanings(selected)
    assert len(answers) == 1
    torch.testing.assert_close(answers[0].roles[0], selected.result.value)
    assert answers[0].role_mask.tolist() == [True, False, False]
    assert answers[0].role_refs[0] is None
    assert not answers[0].roles.requires_grad
    memory.end_what_episode()


def _stored_query(model, registry, part, whole):
    from Layers import TernaryTruthStore
    store=TernaryTruthStore(8,capacity=16)
    store.configure_leaf_index(unfold=lambda idea,limit,**kwargs:((7,),1,True))
    model.symbolSpace.ltm_store=store
    source=registry.form('part',part,whole,mode='assertive')
    store.append_meaning(source,kind='fact',evidence=(.7,.1))
    with model._query_boundary_scope((0,)):
        context=model._thought_grammar_context(source,row=0,work=QueryWorkBudget(32),continuation=None)
        return registry.form('query',store.occurrence_of(0),context=context)


def test_open_relation_answer_preserves_every_member_without_another_reader():
    """Query returns one best match and its answer needs no further read."""
    from Output import thought_answer_meanings
    model,registry,memory,part,whole=_catalog_world()
    request=_stored_query(model,registry,part,whole)
    with model._query_boundary_scope((0,)):
        selected=model.run_selected_thought(request,work_budget=128)
    cost=selected.work.spent
    answers=thought_answer_meanings(selected)
    assert len(answers)==1
    assert answers[0].role_refs[0]==part and answers[0].role_refs[2]==whole
    assert selected.work.spent==cost==memory.thought_state().work_spent
    checked=ThoughtResult.from_checkpoint(selected.result.checkpoint())
    restored=thought_answer_meanings(replace(selected,result=checked))
    torch.testing.assert_close(answers[0].roles,restored[0].roles)
    assert not answers[0].roles.requires_grad
    memory.end_what_episode()


def test_subgoal_carries_the_typed_child_result_into_the_answer():
    from Output import thought_answer_meanings
    model, registry, memory, part, whole = _catalog_world()
    inner = registry.form("isPart", part, whole)
    entry = memory.begin_thought_episode(inner, work_budget=2)
    reference = memory.thought_reference(entry)
    memory.finish_thought(inner)
    memory.end_what_episode()
    with model._query_boundary_scope((0,)):
        outer = registry.form(
            "ask", reference,
            context=model._thought_grammar_context(
                inner, row=0, work=QueryWorkBudget(8), continuation=None))
        selected = model.run_selected_thought(outer, work_budget=128)
    assert selected.result.result_kind == "subgoal"
    assert isinstance(selected.result.value, ThoughtResult)
    assert selected.result.value.semantic_id == "isPart"
    answers = thought_answer_meanings(selected)
    assert len(answers) == 1
    assert answers[0].role_mask.tolist() == [True, True, True]
    torch.testing.assert_close(answers[0].roles, selected.meaning.roles)
    assert answers[0].scope == inner.scope
    stored = ThoughtResult.from_checkpoint(selected.result.checkpoint())
    assert isinstance(stored.value, ThoughtResult)
    assert stored.value.request.role_refs == inner.role_refs
    memory.end_what_episode()


def test_missing_code_is_incomplete_rather_than_a_zero_or_request_answer():
    from Output import thought_answer_meanings
    from types import MappingProxyType, SimpleNamespace
    model, registry, _memory, part, _whole = _catalog_world()
    query = registry.form("exist", registry._payload(part))
    checked = ThoughtResult(
        "exist", "conceptual-presence", "concept", "conceptual-presence",
        query, MappingProxyType({"value": None, "incomplete": ("work_budget",)}))
    assert thought_answer_meanings(SimpleNamespace(meaning=query, result=checked)) == ()




@pytest.mark.parametrize('kind', ['set', 'concept', 'subgoal'])
def test_typed_results_survive_resolve_and_reverse_without_execution(monkeypatch, kind):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from Understanding import AnswerProgram, Understanding
    from What import What
    model, registry, memory, part, whole = _catalog_world()
    query = registry.form('isPart', part, whole)
    if kind == 'set':
        query = _stored_query(model,registry,part,whole)
    elif kind == 'concept':
        query = registry.form('exist', registry._payload(part))
    else:
        from Layers import TernaryTruthStore
        model.symbolSpace.ltm_store = TernaryTruthStore(8, capacity=16)
        query = registry.form('ask', query)
    entry = AnswerProgram(rows=torch.tensor([1]), word_rows=torch.tensor([1]),
        concept_ids=torch.tensor([part[1]]), activations=torch.ones(1),
        leaves=registry._payload(part)[None], actions=torch.tensor([[0, -1, 0]]),
        targets=torch.tensor([-1]), end_state=torch.zeros(3, 8))
    object.__setattr__(model, 'languageSpace', SimpleNamespace(
        program_meaning=lambda item, _registry: query,
        # This adapter fixture stubs every numerical synthesis operation.
        generation_scope=nullcontext))
    object.__setattr__(model.conceptualSpace, 'stm', SimpleNamespace(concept_dim=8))
    model._what_grammar_context = lambda *_a, **_k: (torch.zeros(1, 8), None)
    model._select_perceptual_bindings = lambda *_a: ()
    model._condition_answer_on_question = lambda idea, _context, **_kwargs: idea
    model._synthesis_guard = nullcontext
    model.conceptualSpace.synthesize_idea = lambda idea, **_k: idea
    object.__setattr__(model, 'perceptualSpace', SimpleNamespace(synthesize=lambda idea, **_k: idea))
    object.__setattr__(model, 'outputSpace', SimpleNamespace(from_percepts=lambda idea: idea))
    model.attention_budget = 128
    model.reconstruct_in_loop = False
    # Numerical generation is an explicit identity stub in this adapter test.
    model._walk_operand=lambda value, **kwargs: value
    def walk(idea, *args, **kwargs):
        model._adapter_words=idea
        return (idea,torch.full((len(idea),),idea.shape[1]),
                torch.zeros(len(idea),dtype=torch.bool),None,
                torch.zeros(len(idea),1,dtype=torch.long))
    model._compiled_output_walk=lambda: walk
    model.conceptualSpace.commit_event=lambda value: None
    model._reverse_body=lambda sub: model._adapter_words
    model._reverse_perceptual=lambda value: value
    from Understanding import SentenceEndState
    from Meaning import ConceptualMeaning
    field = SentenceEndState(ConceptualMeaning.from_description(entry.leaves[0]), query=query)
    understanding = Understanding(sentence_states=(field,))
    model.eval()
    derivation = model.resolveAnswer(understanding, What.supervised(0))
    assert derivation.source == 'thought-' + kind
    assert len(derivation.answer_meanings[0]) == 1
    assert derivation.resolved
    monkeypatch.setattr(registry, 'execute', lambda *_a, **_k: pytest.fail('output executed a query'))
    result = model.reverseOutput(understanding, derivation)
    expected = torch.cat(tuple(item.roles for item in derivation.answer_meanings[0]))
    torch.testing.assert_close(result.actual[0], expected)
    assert memory.thought_state().finished
