"""Checked non-truth results become owned, full-width answer meanings."""
from Memory import STM0

from dataclasses import replace

import pytest
import torch

from Queries import ThoughtResult
from QueryWork import QueryWorkBudget
from test_normal_thought_controller import _catalog_world


def test_code_answer_uses_the_checked_atom_and_keeps_its_reference():
    from Output import thought_answer_meanings
    model, registry, memory, part, _whole = _catalog_world()
    # A code-answer adapter consumes a placed atom, not an unplaced class.
    with torch.no_grad():
        model.conceptualSpace.similarity_codebook.W[model.conceptualSpace._concept_row_of(part[1])].fill_(.5)
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
    from Memory import LTM0
    store=STM0(8, percept_width=0,capacity=16, relation_capacity=8192)
    model.symbolSpace.ltm_store=store
    source=registry.form('part',part,whole)
    from MemoryIndex import configure_model_index
    configure_model_index(model,store)
    row=store.append_relation(part[1],whole[1],truth_plus=.7,truth_minus=.1)
    with model._query_boundary_scope((0,)):
        context=model._thought_grammar_context(source,row=0,work=QueryWorkBudget(32),continuation=None)
        opened = replace(source, role_refs=(None, source.role_refs[1], whole),
            role_mask=torch.tensor([False, True, True]))
        from ThoughtReferences import with_slots
        opened = with_slots(replace(opened, bindings={'_query_relation':'part'}),
                            (('referent',0),), pair=(0.,0.))
        return registry.form('query',opened,context=context)


def test_open_relation_recall_keeps_members_in_stm_and_returns_no_operand():
    from ThoughtChunks import held
    from ThoughtStream import materialize
    model,registry,memory,part,whole=_catalog_world()
    request=_stored_query(model,registry,part,whole)
    with model._query_boundary_scope((0,)):
        work = QueryWorkBudget(128)
        context = model._thought_grammar_context(request,row=0,work=work,continuation=None)
        selected = materialize(registry.execute(request,context))
    assert selected.value is None and selected.evidence.get('meaning') is None
    stored = [value for _,value in held(model.conceptualSpace.stm,0)]
    assert len(stored) == 1
    assert (stored[0].role_refs[0], stored[0].role_refs[2]) == (part,whole)
    checked=ThoughtResult.from_checkpoint(selected.checkpoint())
    assert checked.value is None and checked.evidence.get('meaning') is None
    torch.testing.assert_close(checked.evidence['frames'][0]['meaning'].roles, stored[0].roles)
    assert not stored[0].roles.requires_grad


def _open_goal(model, registry, part, whole):
    from Memory import LTM0
    from ClauseRow import attach_clause_index
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots
    store = LTM0(8, capacity=16)
    attach_clause_index(model, store, model.conceptualSpace)
    model.symbolSpace.ltm_store = store
    store.append_relation(part[1], whole[1], truth_plus=1.)
    model.conceptualSpace._concept_allocator.singletons.add(whole[1])
    return registry.form('isPart', whole, open_roles=('argin1',))


def test_subgoal_carries_the_typed_child_result_into_the_answer(monkeypatch):
    from Output import thought_answer_meanings
    model, registry, memory, part, whole = _catalog_world()
    inner = _open_goal(model, registry, part, whole)
    from meaning_fixtures import replay_thought_plan
    from itertools import chain, repeat
    replay_thought_plan(model, monkeypatch, chain(['ask'], repeat('query')))
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
    assert selected.result.value.semantic_id == "query"
    answers = thought_answer_meanings(selected)
    assert answers == ()  # §48: recalled subgoals still need a subsequent binding.
    assert selected.meaning.scope == inner.scope
    stored = ThoughtResult.from_checkpoint(selected.result.checkpoint())
    assert isinstance(stored.value, ThoughtResult)
    assert stored.value.request.role_refs == selected.result.value.request.role_refs
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
    from Understanding import Reading, Understanding
    from What import What
    model, registry, memory, part, whole = _catalog_world()
    query = registry.form('isPart', part, whole)
    if kind == 'set':
        query = _stored_query(model,registry,part,whole)
        from itertools import repeat
        from meaning_fixtures import replay_thought_plan
        replay_thought_plan(model, monkeypatch, repeat('query'))
    elif kind == 'concept':
        query = registry.form('exist', registry._payload(part))
    else:
        query = registry.form('ask', _open_goal(model, registry, part, whole))
        from meaning_fixtures import replay_thought_plan
        from itertools import chain, repeat
        replay_thought_plan(model, monkeypatch, chain(['ask'], repeat('query')))
    entry = Reading(rows=torch.tensor([1]), word_rows=torch.tensor([1]),
        concept_ids=torch.tensor([part[1]]), activations=torch.ones(1),
        leaves=registry._payload(part)[None], actions=torch.tensor([[0, -1, 0]]),
        targets=torch.tensor([-1]), end_state=torch.zeros(3, 8))
    object.__setattr__(model, 'languageSpace', SimpleNamespace(
        program_meaning=lambda item, _registry: query,
        # This adapter fixture stubs every numerical synthesis operation.
        generation_scope=nullcontext))
    from Memory import ShortTermMemory
    object.__setattr__(model.conceptualSpace, 'stm', ShortTermMemory(concept_dim=8))
    from clock_fixtures import for_stm
    for_stm(model.conceptualSpace.stm, model=model)
    model._what_grammar_context = lambda *_a, **_k: (torch.zeros(1, 8), None)
    model._select_perceptual_bindings = lambda *_a: ()
    model._condition_answer_on_question = lambda idea, _context, **_kwargs: idea
    model._synthesis_guard = nullcontext
    object.__setattr__(model, 'perceptualSpace', SimpleNamespace())
    object.__setattr__(model, 'outputSpace', SimpleNamespace(from_percepts=lambda idea: idea))
    model.attention_budget = 128
    model.reconstruct_in_loop = False
    # Numerical realization is explicit at the new generate/reverse seams.
    model.answer_synthesis = True
    def generate(idea, **kwargs):
        model._adapter_words = idea
        return SimpleNamespace(words=idea, truncated=torch.zeros(len(idea), dtype=torch.bool),
            field=idea, realization_field=None, reconstruction_field=None,
            texts=lambda surface: ())
    model.generate = generate
    model.reverse = lambda carrier, *, field: SimpleNamespace(
        materialize=lambda: field, _reconstruction_percepts=field)
    from Understanding import SentenceEndState
    from Meaning import ConceptualMeaning
    field = SentenceEndState(ConceptualMeaning.from_description(entry.leaves[0]), query=query)
    understanding = Understanding(sentence_states=(field,))
    model.eval()
    derivation = model.resolveAnswer(understanding, What.supervised(0))
    assert derivation.source == 'thought-' + kind
    # Presence remains an executable operation, but its pair is not a proof
    # accepted by the answer boundary. Recall and a recalled subgoal are.
    assert len(derivation.answer_meanings[0]) == 0
    assert not derivation.resolved
    monkeypatch.setattr(registry, 'execute', lambda *_a, **_k: pytest.fail('output executed a query'))
    result = model.reverseOutput(understanding, derivation)
    expected = (torch.cat(tuple(item.roles for item in derivation.answer_meanings[0]))
                if derivation.answer_meanings[0] else derivation.conceptual_answer[0])
    torch.testing.assert_close(result.actual[0], expected)
    assert memory.thought_state().finished
