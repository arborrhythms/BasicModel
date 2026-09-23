"""Accessible-mind boundaries, exact cue indices, and retrieval effects."""
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning


def _meaning(a=0, b=1, *, width=8, scope=()):
    roles = torch.zeros(3, width)
    roles[0, a] = 1
    roles[2, b] = 1
    return ConceptualMeaning(roles, torch.tensor([True, False, True]),
        role_refs=(("sym", a + 1), None, ("sym", b + 1)), scope=scope)


def test_leaf_index_retrieves_an_old_row_by_any_derivation_leaf():
    store = TernaryTruthStore(8, capacity=32)
    first = store.append_meaning(_meaning(), leaf_codes=((3, 17, 9), (), (6,)))
    for i in range(20):
        store.append_meaning(_meaning(2, 3), leaf_codes=((40 + i,), (), (70 + i,)))
    assert store.rows_for_code(17, role=0) == (first,)
    assert store.rows_for_code(17, role=2) == ()
    assert store.rows_for_code(1000) == ()
    assert store.leaf_terms(first, 0) == (3, 17, 9)


def test_leaf_column_growth_is_amortized_and_checkpoint_append_is_safe():
    import io
    store = TernaryTruthStore(8, capacity=256)
    pointers = set()
    for _ in range(128):
        store.append_meaning(_meaning(), leaf_codes=((3, 17), (), (6,)))
        pointers.add(store.leaf_codes.data_ptr())
    assert len(pointers) <= 5, "append must not copy the entire leaf history"
    checkpoint = io.BytesIO()
    torch.save(store.state_dict(), checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    column = state["leaf_codes"]
    assert column.numel() == 384
    assert column.untyped_storage().nbytes() == column.numel() * column.element_size()
    restored = TernaryTruthStore(8, capacity=256)
    restored.load_state_dict(state)
    restored.load_semantic_extras(store.semantic_extras())
    row = restored.append_meaning(_meaning(), leaf_codes=((9,), (), (10,)))
    assert restored.leaf_terms(127, 0) == (3, 17)
    assert restored.leaf_terms(row, 0) == (9,)
    restored.reset()
    row = restored.append_meaning(_meaning(), leaf_codes=((11,), (), (12,)))
    assert restored.leaf_terms(row, 0) == (11,)


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
    result = ThoughtResult('what', 'conceptual-subgoal', 'set', 'retrieval', _meaning(),
        MappingProxyType({'frames': ({'leaf_codes': ((row,), (), ())},)}))
    for _ in range(2):
        apply_thought_effect(SimpleNamespace(conceptualSpace=cs), result, row=0, work=QueryWorkBudget(32))
    assert cs.subspace._concept_activations[row, 0, 0, 0] == 1


def test_index_checkpoint_compaction_and_codebook_remap():
    import copy
    store = TernaryTruthStore(8, capacity=8)
    old = store.append_meaning(_meaning(), leaf_codes=((3, 17), (), (6,)))
    kept = store.append_meaning(_meaning(2, 3), leaf_codes=((17, 40), (), (70,)))
    store.set_origin(old, store.ORIGIN_USER)
    reference = store.occurrence_of(kept)
    restored = TernaryTruthStore(8, capacity=8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(store.semantic_extras())
    assert restored.rows_for_code(17) == (0, 1)
    assert restored.clear_origin(store.ORIGIN_USER) == 1
    assert restored.occurrence_of(0) == reference
    assert restored.rows_for_code(3) == ()
    assert restored.rows_for_code(17) == (0,)
    restored.remap_leaf_codes({17: 2, 40: 4, 70: 7})
    assert restored.leaf_terms(0, 0) == (2, 4)
    assert restored.rows_for_code(17) == ()
    assert restored.rows_for_code(2) == (0,)
    restored.reset()
    assert not restored.rows_for_code(2)


def test_no_derivation_uses_bounded_unfold_instead_of_snapping_the_root():
    store = TernaryTruthStore(8, capacity=4)
    calls = []
    def unfold(idea, limit):
        calls.append((idea.clone(), limit))
        return (11, 23), 3, True
    store.configure_leaf_index(unfold=unfold)
    idea = torch.full((1, 8), .23, requires_grad=True)
    meaning = ConceptualMeaning.from_description(idea)
    row = store.append_meaning(meaning)
    assert store.leaf_terms(row, 0) == (11, 23)
    assert len(calls) == 1 and not calls[0][0].requires_grad
    assert bool(store.leaf_complete[row].all())


def test_cue_fan_is_charged_and_scope_filters_before_ranking():
    from QueryWork import QueryWorkBudget
    store = TernaryTruthStore(8, capacity=32)
    store.configure_leaf_index(code_row=lambda ref: ref[1] - 1)
    for i in range(12):
        store.append_meaning(_meaning(scope={"where": (i, i + 1)}), stream=0)
    work = QueryWorkBudget(5)
    result = store.cued_rows(_meaning(scope={"where": (2, 4)}),
        max_candidates=20, work=work, stream=0)
    assert [row['index'] for row in result['value']] == [2, 3]
    assert result['records_scanned'] == work.spent == 5
    assert 'work_budget' in result['incomplete']


def test_priming_adds_code_disjoint_rows_and_contiguity_breaks_equal_matches():
    store = TernaryTruthStore(8, capacity=8)
    store.configure_leaf_index(code_row=lambda ref: ref[1] - 1)
    first = store.append_meaning(_meaning(), leaf_codes=((90,), (), (91,)), stream=0)
    second = store.append_meaning(_meaning(), leaf_codes=((92,), (), (93,)), stream=1)
    third = store.append_meaning(_meaning(), leaf_codes=((94,), (), (95,)), stream=1)
    cue = _meaning()
    assert not store.cued_rows(cue)['value']
    found = store.cued_rows(cue, primed=(90, 94))['value']
    assert [row['index'] for row in found] == [first, third]
    found = store.cued_rows(cue, primed=(90, 94), retrieved=(store.occurrence_of(second),))['value']
    assert [row['index'] for row in found] == [third, first]


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
    result = _signature('part', 'I1', 'I2').invoke(context, left, right)
    torch.testing.assert_close(result['value'], Ops.part(left, right))
    assert result['support_true'] == pytest.approx(float(Ops.part(left, right, scalar=True)))
    assert result['result_kind'] == 'concept' and not result['value'].requires_grad
    assert context.work.spent == 1


def test_what_retrieves_old_cued_frame_and_exist_keeps_it_out_of_stm():
    from dataclasses import replace
    from test_cs_symbol_table import _cs
    from test_query_vp_boundaries import _context, _signature
    store = TernaryTruthStore(8, capacity=32)
    store.configure_leaf_index(code_row=lambda ref: ref[1] - 1)
    fact = _meaning()
    oldest = store.append_meaning(fact, trust=.2)
    for _ in range(20):
        store.append_meaning(_meaning(4, 5), trust=.9)
    context = _context(_cs(), store=store)
    first = _signature('exist', 'I1').invoke(context, fact)
    store.append_meaning(fact, trust=.3)
    second = _signature('exist', 'I1').invoke(context, fact)
    assert second['support_true'] > first['support_true']
    assert context.ltm.held_frames() == ()
    cue = replace(fact, role_mask=torch.tensor([True, False, False]),
                  role_refs=(fact.role_refs[0], None, None), mode='interrogative')
    found = _signature('what', 'I1').invoke(context, cue)
    assert len(found['frames']) == 1
    assert found['frames'][0]['occurrence'] == store.occurrence_of(oldest)
    assert found['result_kind'] == 'set'
    # Other readers cannot bring the same unseen row into serial context.
    assert not _signature('lookup', 'I1', 'I2').invoke(context, fact.roles[0], fact.roles[2])['value']
    with pytest.raises((TypeError, ValueError)):
        _signature('quantize', 'I1').invoke(context, store.occurrence_of(oldest))


def test_higher_order_missing_edge_is_symbolic_zero_not_a_residual():
    from test_cs_symbol_table import _cs
    from test_query_vp_boundaries import _context, _signature
    cs = _cs()
    a, b = cs.new_concept(), cs.new_concept()
    higher = cs.synthesize_higher_order([('sym', a)])
    for cid in (a, b, higher):
        cs._csw_concept_row(0, cid)
    result = _signature('part', 'I1', 'I2').invoke(_context(cs), ('sym', higher), ('sym', b))
    assert result['symbolic_value'] == 0.
    assert result['result_kind'] == 'truth' and result['evidence_kind'] == 'taxonomy'
    assert not torch.is_tensor(result.get('value'))


def test_normal_what_effect_enters_recency_and_detached_knowing(monkeypatch):
    from dataclasses import replace
    from QueryWork import QueryWorkBudget
    from test_normal_thought_controller import _catalog_world
    from Queries import ThoughtResult
    model, registry, memory, part, whole = _catalog_world()
    store = model.symbolSpace.ltm_store = TernaryTruthStore(8, capacity=32)
    fact = registry.form('part', part, whole, mode='assertive')
    oldest = store.append_meaning(fact, trust=.7)
    for _ in range(20):
        store.append_meaning(replace(fact, roles=fact.roles + 20,
                                    role_refs=(('sym', 500), None, ('sym', 501))), trust=.1)
    cue = replace(fact, mode='interrogative', role_mask=torch.tensor([True, True, False]),
                  role_refs=(part, fact.role_refs[1], None))
    cue_row = store.append_meaning(cue, kind='question')
    with model._query_boundary_scope((0,)):
        meter = QueryWorkBudget(256)
        context = model._thought_grammar_context(cue, row=0, work=meter, continuation=None)
        question = registry.form('what', store.occurrence_of(cue_row), context=context)
    def choose(root, active, actions, **kw):
        if kw.get('evidence') is not None:
            return None
        return next(action for action in actions if action and action.semantic_id == 'what')
    monkeypatch.setattr(model, '_choose_selected_thought_action', choose)
    with model._query_boundary_scope((0,)):
        before = model._selected_thought_memory(question, row=0, work=QueryWorkBudget(128))
        result = model.run_selected_thought(question, work_budget=256)
        after = model._selected_thought_memory(question, row=0, work=QueryWorkBudget(128))
    assert not bool(before[1].any()) and bool(after[1].any())
    assert result.result.semantic_id == 'what' and result.result.result_kind == 'set'
    frame, = memory.retrieved_frames()
    assert frame['occurrence'] == store.occurrence_of(oldest)
    assert frame['occurrence'] in memory.retained_ltm_occurrences(frame['occurrence'][1])
    assert not frame['meaning'].roles.requires_grad
    saved = ThoughtResult.from_checkpoint(result.result.checkpoint())
    assert saved.evidence['frames'][0]['occurrence'] == frame['occurrence']
    # Mutating an LTM row never changes the retained, owned frame.
    store.slots[oldest].fill_(100.)
    torch.testing.assert_close(frame['meaning'].roles, fact.roles)
    field = model.conceptualSpace.subspace._concept_activations
    assert field is not None and not field.requires_grad


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
    result = ThoughtResult('what', 'conceptual-subgoal', 'set', 'retrieval', meaning,
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


def test_codebook_compaction_notifies_the_live_ltm_index(monkeypatch):
    from MemoryIndex import configure_model_index
    from test_selected_relation_meaning import _program_owner
    cs, _, registry, language, _, _, _, _ = _program_owner(monkeypatch)
    store = TernaryTruthStore(8, capacity=4)
    host = SimpleNamespace(conceptualSpace=cs, languageSpace=language,
                           grammatical_thoughts=registry)
    configure_model_index(host, store)
    store.append_meaning(_meaning(), leaf_codes=((3, 6), (), (9,)))
    cs.similarity_codebook.remove([1, 4])
    assert store.leaf_terms(0, 0) == (2, 4)
    assert store.leaf_terms(0, 2) == (7,)


def test_recorded_forest_indexes_each_role_without_unfolding():
    from MemoryIndex import recorded_leaf_terms
    program = SimpleNamespace(rows=torch.tensor([6, 8, 10, 12]),
        actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, 0, -1],
                              [0, -1, 2], [0, -1, 3]]))
    assert recorded_leaf_terms(program, None, 3) == ((10,), (6, 8), (12,))


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


def test_normal_priming_uses_boosted_rows_not_the_identity_mask():
    from dataclasses import replace
    from QueryWork import QueryWorkBudget
    from test_normal_thought_controller import _catalog_world
    from test_query_vp_boundaries import _signature
    model, registry, memory, part, whole = _catalog_world()
    store = model.symbolSpace.ltm_store = TernaryTruthStore(8, capacity=4)
    fact = registry.form('part', part, whole, mode='assertive')
    store.append_meaning(fact, leaf_codes=((50,), (51,), (52,)))
    cue = replace(fact, role_mask=torch.tensor([True, False, False]),
                  role_refs=(part, None, None), mode='interrogative')
    priming = torch.ones(64)
    model.symbolSpace.taxonomy = SimpleNamespace(priming_enabled=True, priming_mask=lambda **kw: priming)
    def retrieve():
        with model._query_boundary_scope((0,)):
            context = model._thought_grammar_context(cue, row=0, work=QueryWorkBudget(64), continuation=None)
            return _signature('what', 'I1').invoke(context, cue)['frames']
    assert not retrieve()
    priming[50] = 1.5
    assert len(retrieve()) == 1
    priming.fill_(1.)
    assert not retrieve()


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


def test_legacy_index_rebuild_uses_real_rows_and_isolates_unknown_streams(monkeypatch):
    import copy
    from dataclasses import replace
    from MemoryIndex import configure_model_index
    from test_selected_relation_meaning import _program_owner
    cs, _, registry, language, _, _, a, b = _program_owner(monkeypatch)
    meaning = registry.form('part', a, b, mode='assertive')
    original = TernaryTruthStore(8, capacity=4)
    original.append_meaning(meaning, kind='fact')
    original.append_meaning(meaning, kind='observation')
    state = copy.deepcopy(original.state_dict())
    for key in ('leaf_codes', 'leaf_offsets', 'leaf_complete', 'index_stream'):
        del state[key]
    store = TernaryTruthStore(8, capacity=4)
    store.load_state_dict(state)
    store.load_semantic_extras(original.semantic_extras())
    configure_model_index(SimpleNamespace(conceptualSpace=cs, languageSpace=language,
                                         grammatical_thoughts=registry), store)
    from Queries import _existing_row
    assert store.leaf_terms(0, 0) == (_existing_row(cs, a),)
    assert store.index_stream[:2].tolist() == [-1, -2]
    found = store.cued_rows(replace(meaning, mode='interrogative'), stream=0)
    assert [row['index'] for row in found['value']] == [0]


def test_rows_written_before_owner_binding_never_index_allocator_ids(monkeypatch):
    from MemoryIndex import configure_model_index
    from Queries import _existing_row
    from test_selected_relation_meaning import _program_owner
    cs, _, registry, language, _, _, a, b = _program_owner(monkeypatch)
    meaning = registry.form('part', a, b, mode='assertive')
    store = TernaryTruthStore(8, capacity=4)
    store.append_meaning(meaning)
    store.append_meaning(meaning, leaf_codes=((3, 7), (), (11,)))
    assert store.leaf_terms(0, 0) == ()
    assert not bool(store.leaf_complete[0, 0])
    configure_model_index(SimpleNamespace(conceptualSpace=cs, languageSpace=language,
                                         grammatical_thoughts=registry), store)
    assert store.leaf_terms(0, 0) == (_existing_row(cs, a),)
    assert store.leaf_terms(0, 2) == (_existing_row(cs, b),)
    assert store.leaf_terms(1, 0) == (3, 7)
    assert store.leaf_terms(1, 2) == (11,)


def test_nested_writes_inherit_stream_isolation():
    from dataclasses import replace
    child = _meaning()
    parent = replace(child, role_mask=torch.tensor([True, False, False]),
                     role_refs=(('constituent', 0), None, None), constituents=(child,))
    store = TernaryTruthStore(8, capacity=4)
    store.configure_leaf_index(code_row=lambda ref: ref[1] - 1)
    store.append_meaning(parent, kind='observation', stream=2)
    assert store.index_stream[:2].tolist() == [2, 2]
    assert not store.cued_rows(child, stream=0)['value']
    assert store.cued_rows(child, stream=2)['value']


def test_executor_cannot_emit_outside_its_declared_write_scope():
    from dataclasses import replace
    from AccessibleMind import Subsystem as S
    from test_query_vp_boundaries import _signature, _context
    from test_cs_symbol_table import _cs
    signature = _signature('equal', 'I1', 'I2')
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
        registry.form('part', torch.ones(8), open_roles=('I2',))
