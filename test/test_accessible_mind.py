"""Accessible-mind boundaries, exact cue indices, and retrieval effects."""
from types import SimpleNamespace

import pytest
import torch

from Layers import LTM0
from Meaning import ConceptualMeaning
from index_fixtures import append_content


def _meaning(a=0, b=1, *, width=8, scope=()):
    roles = torch.zeros(3, width)
    roles[0, a] = 1
    roles[0, b] = 1
    return ConceptualMeaning(roles, torch.tensor([True, False, False]), scope=scope)






def test_thought_effect_uses_cached_row_identity(monkeypatch):
    from AccessibleMind import apply_thought_effect
    from Queries import ThoughtResult
    from QueryWork import QueryWorkBudget
    from types import MappingProxyType
    from test_cs_symbol_table import _cs
    cs = _cs()
    member = cs.new_concept()
    row = cs._csw_concept_row(0, member)
    def forbidden(self):
        raise AssertionError("thought must not rebuild the dictionary reverse map")
    monkeypatch.setattr(type(cs), "_csw_rows", property(forbidden))
    result = ThoughtResult('ask', 'conceptual-subgoal', 'set', 'retrieval', _meaning(),
        MappingProxyType({'frames': ({'reference': ('sym', member)},)}))
    for _ in range(2):
        apply_thought_effect(SimpleNamespace(conceptualSpace=cs), result, row=0, work=QueryWorkBudget(32))
    assert cs.subspace._concept_activations[row, 0, 0, 0] == 1






def test_cue_fan_is_charged_and_scope_filters_before_ranking():
    from QueryWork import QueryWorkBudget
    store = LTM0(8, percept_width=0, capacity=32)
    for i in range(12):
        store.append_meaning(_meaning(scope={"where": (i, i + 1)}), stream=0, sentence_index=len(store))
    work = QueryWorkBudget(5)
    result = store.cued_rows(_meaning(scope={"where": (2, 4)}),
        max_candidates=20, work=work, stream=0)
    assert [row['index'] for row in result['value']] == [2, 3]
    assert result['records_scanned'] == work.spent == 5
    assert 'work_budget' in result['incomplete']




def test_subsystem_permissions_and_structural_capability_refusal():
    from dataclasses import replace
    from AccessibleMind import Subsystem as S, check_access
    from Queries import THOUGHT_EXECUTORS, StructuralGrammarContext, ConceptualSpaceCapability
    from Language import _structural_face_phase
    for descriptor in THOUGHT_EXECUTORS.values():
        check_access('thought', descriptor.read_scope, descriptor.write_scope)
        assert descriptor.write_target in descriptor.write_scope
    with pytest.raises(ValueError, match='permissions'):
        replace(THOUGHT_EXECUTORS['equal'], read_scope=(S.PERCEPT,))
    with pytest.raises(ValueError, match='Subsystem'):
        replace(THOUGHT_EXECUTORS['equal'], read_scope=('ltm.facts',))
    context = StructuralGrammarContext((), ConceptualSpaceCapability(8), None, 'compose')
    object.__setattr__(context, 'ltm', object())
    with pytest.raises(ValueError, match='forbidden'):
        _structural_face_phase(SimpleNamespace(arity=2), (torch.ones(8),) * 2, context=context)


def test_part_thought_writes_detached_vector_residual_without_memory_read(monkeypatch):
    from Layers import Ops
    from test_cs_symbol_table import _cs
    from test_query_vp_boundaries import _context, _signature
    from Queries import ThoughtTaxonomyCapability
    monkeypatch.setattr(ThoughtTaxonomyCapability, 'evidence', lambda *a, **k: pytest.fail('taxonomy read'))
    left, right = torch.rand(8, requires_grad=True), torch.rand(8, requires_grad=True)
    context = _context(_cs())
    result = _signature('part', 'argin1', 'argin2').invoke(context, left, right)
    torch.testing.assert_close(result['value'], Ops.part(left, right))
    assert result['support_true'] == pytest.approx(float(Ops.part(left, right, scalar=True)))
    assert result['result_kind'] == 'concept' and not result['value'].requires_grad
    assert context.work.spent == 1


def test_query_retrieves_every_old_cued_frame_without_an_implicit_write():
    from dataclasses import replace
    from test_cs_symbol_table import _cs
    from test_query_vp_boundaries import _context, _signature
    store = LTM0(8, percept_width=0, capacity=32)
    fact = _meaning()
    oldest = store.append_meaning(fact, trust=.2, sentence_index=len(store))
    for _ in range(20):
        store.append_meaning(_meaning(4, 5), trust=.9, sentence_index=len(store))
    newest = store.append_meaning(fact, trust=.3, sentence_index=len(store))
    context = _context(_cs(), store=store)
    cue = replace(fact, role_mask=torch.tensor([True, False, False]),
                  role_refs=(fact.role_refs[0], None, None))
    found = _signature('query', 'argin1').invoke(context, cue)
    assert len(found['frames']) == 1
    assert {frame['occurrence'] for frame in found['frames']} == {
        store.occurrence_of(oldest)}
    assert sorted(frame['trust'] for frame in found['frames']) == pytest.approx([.2])
    assert context.ltm.held_frames() == ()
    assert found['result_kind'] == 'set'
    with pytest.raises((TypeError, ValueError)):
        __import__('Language').Grammar().configure({'thought': {'rule': 'quantize_argout1 = quantize.thought(quantize_argin1)'}})


def test_higher_order_missing_edge_keeps_content_with_zero_evidence():
    from test_cs_symbol_table import _cs
    from test_query_vp_boundaries import _context, _signature
    cs = _cs()
    a, b = cs.new_concept(), cs.new_concept()
    higher = cs.synthesize_higher_order([('sym', a)])
    for cid in (a, b, higher):
        cs._csw_concept_row(0, cid)
    result = _signature('isPart', 'argin1', 'argin2').invoke(_context(cs), ('sym', higher), ('sym', b))
    assert result['support_true'] == 0.
    assert result['result_kind'] == 'concept' and result['evidence_kind'] == 'taxonomy'
    assert result['value'] is None and result['witnesses']==()



def test_thought_retrieval_keeps_history_and_restores_episode_knowing(monkeypatch):
    from dataclasses import replace
    from QueryWork import QueryWorkBudget
    from test_normal_thought_controller import _catalog_world
    from Queries import ThoughtResult
    model, registry, memory, part, whole = _catalog_world()
    store = model.symbolSpace.ltm_store = LTM0(8, percept_width=0, capacity=32)
    fact = replace(ConceptualMeaning.from_description(torch.eye(8)[0]),
                   role_refs=(part, None, None))
    from MemoryIndex import configure_model_index
    configure_model_index(model, store)
    oldest = append_content(store, fact, trust=.7)
    for _ in range(20):
        append_content(store, ConceptualMeaning.from_description(torch.eye(8)[1]), trust=.1)
    cue = replace(fact)
    cue_row = store.append_meaning(cue, kind='question', sentence_index=len(store))
    with model._query_boundary_scope((0,)):
        meter = QueryWorkBudget(256)
        context = model._thought_grammar_context(cue, row=0, work=meter, continuation=None)
        question = registry.form('query', store.occurrence_of(cue_row), context=context)
    def choose(root, active, actions, **kw):
        if kw.get('evidence') is not None:
            return None
        return next(action for action in actions if action and action.semantic_id == 'query')
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    prior_knowing = getattr(model.conceptualSpace.subspace, '_concept_activations', None)
    with model._query_boundary_scope((0,)):
        before = model._selected_thought_memory(question, row=0, work=QueryWorkBudget(128))
        result = model.run_selected_thought(question, work_budget=256)
        after = model._selected_thought_memory(question, row=0, work=QueryWorkBudget(128))
    assert not bool(before[1].any()) and bool(after[1].any())
    assert result.result.semantic_id == 'query' and result.result.result_kind == 'set'
    frame, = memory.retrieved_frames()
    assert frame['occurrence'] == store.occurrence_of(oldest)
    assert frame['occurrence'] in memory.retained_ltm_occurrences(frame['occurrence'][1])
    assert not frame['meaning'].roles.requires_grad
    saved = ThoughtResult.from_checkpoint(result.result.checkpoint())
    assert saved.evidence['frames'][0]['occurrence'] == frame['occurrence']
    # Mutating an LTM row never changes the retained, owned frame.
    store.slots[oldest].fill_(100.)
    torch.testing.assert_close(frame['meaning'].roles, fact.roles)
    # The checked frames remain in the credit/history trail; the episode's
    # transient conceptual activation is not another lasting effect (§4.4).
    assert getattr(model.conceptualSpace.subspace, '_concept_activations', None) is prior_knowing


def test_higher_order_retrieval_seeds_discontinuous_members_only():
    from types import MappingProxyType
    from AccessibleMind import apply_thought_effect
    from Queries import ThoughtResult
    from QueryWork import QueryWorkBudget
    from test_cs_symbol_table import _cs
    cs = _cs()
    members = [cs.new_concept() for _ in range(3)]
    rows = [cs._csw_concept_row(0, cid) for cid in members]
    higher = cs.synthesize_higher_order([('sym', members[0]), ('sym', members[2])])
    cs._csw_concept_row(1, higher)
    meaning = _meaning()
    held = ConceptualMeaning.from_description(torch.ones(8))
    from dataclasses import replace
    held = replace(held, role_refs=(('sym', higher), None, None))
    result = ThoughtResult('ask', 'conceptual-subgoal', 'set', 'retrieval', meaning,
        MappingProxyType({'frames': ({'meaning': held},)}))
    apply_thought_effect(SimpleNamespace(conceptualSpace=cs), result, row=0, work=QueryWorkBudget(32))
    field = cs.subspace._concept_activations[:, 0, 0, 0]
    assert field[rows[0]] == field[rows[2]] == 1 and field[rows[1]] == 0
    assert not field.requires_grad


def test_budget_changes_context_and_forces_closure():
    from test_normal_thought_controller import _catalog_world
    model, registry, memory, a, b = _catalog_world()
    question = registry.form('part', a, b)
    make = model._selected_thought_context
    assert not torch.equal(make(question, level=0, pressure=0.), make(question, level=0, pressure=1.))
    with model._query_boundary_scope((0,)):
        result = model.run_selected_thought(question, work_budget=2)
    assert memory.thought_state().finished and result.work.spent <= 2






def test_structural_context_keeps_live_values_but_rejects_owner_escape():
    from Queries import StructuralGrammarContext, ConceptualSpaceCapability
    from test_cs_symbol_table import _cs
    value = torch.ones(8, requires_grad=True)
    with pytest.raises(ValueError, match='geometry-only'):
        StructuralGrammarContext(value, _cs(), None, 'compose')
    held = StructuralGrammarContext(value, ConceptualSpaceCapability(8), value, 'compose')
    assert held.word_stream.data_ptr() != value.data_ptr()
    held.word_stream.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))




def test_narrowed_descriptor_cannot_keep_the_original_taxonomy_grant():
    from dataclasses import replace
    from AccessibleMind import Subsystem as S
    from Queries import THOUGHT_EXECUTORS, _descriptor_context
    from test_query_vp_boundaries import _context
    from test_cs_symbol_table import _cs
    descriptor = replace(THOUGHT_EXECUTORS['part'], read_scope=(S.SERIAL, S.BUDGET))
    context = _descriptor_context(_context(_cs()), descriptor)
    assert not hasattr(context.taxonomy, 'evidence')
    assert not hasattr(context.conceptual_space, 'payload')
    assert context.primed_symbols is None






def test_nested_writes_are_shared_across_batch_positions():
    from dataclasses import replace
    child = _meaning()
    parent = replace(child, role_mask=torch.tensor([True, False, False]),
                     role_refs=(('constituent', 0), None, None), constituents=(child,))
    store = LTM0(8, percept_width=0, capacity=4)
    store.append_meaning(parent, kind='observation', stream=2, sentence_index=len(store))
    assert not hasattr(store, 'index_stream')
    first = store.cued_rows(child, stream=0)['value']
    moved = store.cued_rows(child, stream=2)['value']
    assert first and [row['occurrence'] for row in first] == [row['occurrence'] for row in moved]


def test_executor_cannot_emit_outside_its_declared_write_scope():
    from dataclasses import replace
    from AccessibleMind import Subsystem as S
    from test_query_vp_boundaries import _signature, _context
    from test_cs_symbol_table import _cs
    signature = _signature('equal', 'argin1', 'argin2')
    signature = replace(signature, descriptor=replace(signature.descriptor,
        executor=lambda *args: {'write_target': S.PERCEPT.value}))
    with pytest.raises(ValueError, match='write scope'):
        signature.invoke(_context(_cs()), torch.ones(8), torch.ones(8))


def test_unnamed_vectors_cannot_offer_open_taxonomy_candidates():
    from test_normal_thought_controller import _catalog_world
    model, registry, _, _, _ = _catalog_world()
    query = registry.form('part', torch.ones(8), torch.ones(8))
    candidates = registry.controller_candidates(query, query, query)
    assert any(candidate.semantic_id == 'part' for candidate in candidates)
    assert not any(candidate.semantic_id == 'part' and candidate.open_roles
                   for candidate in candidates)
    with pytest.raises((TypeError, ValueError), match='reference'):
        registry.form('isPart', torch.ones(8), open_roles=('argin2',))
