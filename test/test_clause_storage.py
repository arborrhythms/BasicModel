"""Two-truths storage and provenance contracts (item 7, §§1–4).

The September 28 amendment keeps fixture identification evidence explicit
beside source trust; equal numbers below are independently supplied.
"""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from Layers import TernaryTruthStore, TruthLayer
from Spaces import ConceptualSpace
from ClauseRow import Clause
from Meaning import ConceptualMeaning


def test_end_row_has_shared_references_and_field_coordinates():
    store = TernaryTruthStore(4, capacity=8)
    assert store.refs.shape == (8, 3)
    assert store.refs.dtype == torch.int64
    assert bool((store.refs == -1).all())
    assert store.where.shape == store.when.shape == (8, 4)


def test_both_evidence_poles_survive_checkpoint_separately_from_neither():
    store = TernaryTruthStore(4, capacity=8)
    both = store.append_idea(torch.ones(4))
    neither = store.append_idea(torch.ones(4))
    store.set_evidence(both, .8, .8)
    restored = TernaryTruthStore(4, capacity=8)
    restored.load_state_dict(store.state_dict())
    assert restored.row(both)['evidence'] == pytest.approx((.8, .8))
    assert restored.row(neither)['evidence'] == (0., 0.)
    assert restored.row(both)['corners'] == pytest.approx((.2, .2, .8, .2))


def test_relation_kinds_have_no_catch_all():
    assert TernaryTruthStore.REL_OPERATOR == 3
    assert not hasattr(TernaryTruthStore, 'REL_OTHER')


def test_relation_rows_never_enter_luminosity():
    store = TernaryTruthStore(4, capacity=8)
    idea = store.append_idea(torch.tensor([1., 0., 0., 0.]), trust=.6, evidence=(.6, 0.))
    relation = store.append_relation(torch.ones(4), torch.ones(4),
        torch.ones(4), rel_type=store.REL_PARTOF, trust=1.)
    store.set_origin(idea, store.ORIGIN_PROVISIONED)
    store.set_origin(relation, store.ORIGIN_PROVISIONED)
    light = TruthLayer(4, max_truths=8)
    light.attach_ltm(store)
    assert light.sync_from_ltm() == 1
    torch.testing.assert_close(light.truths[0], torch.tensor([.6, 0., 0., 0.]))


def test_sentence_content_cannot_choose_its_own_trust():
    owner = SimpleNamespace(
        stm=SimpleNamespace(_depth=torch.tensor([3])),
        _incoming_trust_multiplier=lambda: .65,
        _tetralemma_trust=lambda _predicate: (0., 1., 0., 0.),
        _scale_tetralemma_trust=lambda value: value,
        _collapse_trust=lambda _value: pytest.fail('identification cannot supply source trust'))
    actual = ConceptualSpace.stm_end_state_trust(
        owner, torch.ones(1, 3, 4), torch.tensor([True]))
    assert actual == [.65]


def clause_store():
    """A real native allocator with a small, explicitly assigned concept bank."""
    from Layers import ConceptAllocator
    allocator = ConceptAllocator()
    points = {allocator.new_concept(): vector for vector in torch.eye(4)}
    store = TernaryTruthStore(4, capacity=32)
    def allocate(point):
        return allocator.new_concept()
    store.configure_clause_index(allocate=allocate, concept_point=points.get,
        predicate_kind=lambda ref: 'part' if ref == 3 else 'implies' if ref == 4 else 'operator')
    return store, points


def part_clause(left=1, right=2, *, polarity=True):
    return Clause(ConceptualMeaning(torch.eye(4)[[left - 1, 2, right - 1]], torch.ones(3, dtype=torch.bool),
        polarity=polarity), relation='part', refs=(left, 3, right))


def idea_clause(point=None, *, children=(), refs=(1, 3, -1)):
    return Clause(ConceptualMeaning(torch.eye(4)[:3], torch.ones(3, dtype=torch.bool)),
        point=torch.tensor([.2, .3, .4, .5]) if point is None else point,
        children=children, refs=refs, where=torch.tensor([0., 1., 2., 3.]),
        when=torch.tensor([4., 5., 6., 7.]))


def test_compound_idea_stores_only_fused_point_without_factored_target():
    store, _ = clause_store()
    clause = idea_clause()
    row = store.write_clause(clause, trust=.7)
    assert len(store) == 1 and int(store.rel_type[row]) == store.REL_NONE
    torch.testing.assert_close(store.slots[row, 0], clause.point)
    assert not bool(store.slots[row, 1:].any())
    torch.testing.assert_close(store.meaning_of(row).roles[0], clause.point)
    assert store.meaning_of(row).role_mask.tolist() == [True, False, False]
    assert store.refs[row].tolist() == [1, 3, -1]
    torch.testing.assert_close(store.where[row], clause.where)
    torch.testing.assert_close(store.when[row], clause.when)


def test_recursive_absolute_clauses_each_write_one_idea():
    store, _ = clause_store()
    inner = idea_clause()
    middle = idea_clause(children=(inner,))
    outer = idea_clause(children=(middle,))
    assert store.write_clause(outer, trust=.7) == 2
    assert store.ideas().tolist() == [0, 1, 2]
    assert store.trust[:3].tolist() == pytest.approx([0., 0., .7])
    assert bool((store.refs[:3, 2] == -1).all())


def test_attribution_keeps_inner_claim_unasserted_then_direct_assertion_updates_it():
    store, _ = clause_store()
    child = part_clause()
    roles = child.meaning.roles.clone()
    roles[2].zero_()
    outer = Clause(replace(child.meaning, roles=roles), relation='operator',
                   refs=(4, 1, ('clause', 0)), children=(child,))
    assert store.write_clause(outer, trust=.65) == 1
    assert store.row(0)['evidence'] == (0., 0.)
    assert store.refs[1, 2] == store.row_ids[0]
    assert not bool(store.slots[1, 2].any())
    assert store.write_clause(child, trust=.8, evidence=(.8, 0.)) == 0
    assert len(store) == 2
    assert store.row(0)['evidence'] == pytest.approx((.8, 0.))
    store.set_trust(1, 0.)
    assert store.row(0)['evidence'] == pytest.approx((.8, 0.))


def test_second_order_implication_is_row_indexed_and_has_null_vector_operands():
    store, _ = clause_store()
    left, right = part_clause(), part_clause(1, 4)
    roles = left.meaning.roles.clone()
    roles[[0, 2]] = 0
    outer = Clause(replace(left.meaning, roles=roles), relation='implies',
        refs=(('clause', 0), 4, ('clause', 1)), children=(left, right))
    row = store.write_clause(outer, trust=.9)
    assert row == 2 and len(store) == 3
    assert not bool(store.slots[row, (0, 2)].any())
    first, second = store.relation_operands(row)
    assert first['rel_type'] == second['rel_type'] == store.REL_PARTOF
    assert first['evidence'] == second['evidence'] == (0., 0.)
    matches = store.consequents_by_row(int(store.row_ids[0]), rel_type=store.REL_IMPLIES)
    assert matches[0][:2] == (2, int(store.row_ids[1]))


def test_equality_is_two_independent_part_rows_and_never_merges_operands():
    store, points = clause_store()
    source = part_clause()
    converse = replace(source, refs=(2, 3, 1),
                       meaning=replace(source.meaning, roles=source.meaning.roles[[2, 1, 0]]))
    value = replace(source, companions=(converse,))
    store.write_clause(value, trust=.75, evidence=(.75, 0.))
    assert store.relations(store.REL_PARTOF).tolist() == [0, 1]
    assert store.refs[:2].tolist() == [[1, 3, 2], [2, 3, 1]]
    store.set_evidence(0, .2, .7)
    assert store.row(1)['evidence'] == pytest.approx((.75, 0.))
    assert not torch.equal(points[1], points[2])
    forward = store.meaning_of(0)
    reverse = store.meaning_of(1)
    assert reverse.role_refs == forward.role_refs[::-1]
    torch.testing.assert_close(reverse.roles, forward.roles[[2, 1, 0]])


def test_vector_and_row_readers_keep_both_poles_and_weight_substitution():
    store, points = clause_store()
    row = store.write_clause(part_clause(), evidence=(.7, .4))
    torch.testing.assert_close(store.evaluate(points[1], points[3], points[2]),
                               torch.tensor([.7, .4]))
    torch.testing.assert_close(store.evaluate_rows(1, 3, 2), torch.tensor([.7, .4]))
    match, = store.consequents(points[1])
    assert match[0] == row and match[1] == pytest.approx(.7)
    torch.testing.assert_close(match[3], points[2])


def test_luminosity_preserves_both_instead_of_erasing_it_to_unknown():
    store, _ = clause_store()
    row = store.write_clause(idea_clause(torch.tensor([1., 0., 0., 0.])),
        kind='fact', origin=store.ORIGIN_USER, evidence=(.8, .6))
    light = TruthLayer(4, max_truths=8)
    light.attach_ltm(store)
    assert light.sync_from_ltm() == 2
    torch.testing.assert_close(light.truths[:2, 0], torch.tensor([.8, -.6]))
    assert store.row(row)['evidence'] == pytest.approx((.8, .6))




def test_negation_contributes_negative_evidence_without_erasing_positive():
    store, _ = clause_store()
    store.write_clause(part_clause(), trust=.8, evidence=(.8, 0.))
    store.write_clause(part_clause(polarity=False), trust=.7, evidence=(0., .7))
    assert len(store) == 1
    assert store.row(0)['evidence'] == pytest.approx((.8, .7))


def test_clause_end_state_and_evidence_checkpoint_roundtrip():
    import copy
    store, _ = clause_store()
    clause = idea_clause()
    row = store.write_clause(clause, trust=.6, evidence=(.6, 0.))
    restored, _ = clause_store()
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    torch.testing.assert_close(restored.meaning_of(row).roles[0], clause.point)
    assert restored.row(row)['evidence'] == pytest.approx((.6, 0.))
    torch.testing.assert_close(restored.refs, store.refs)


def test_migration_drops_catch_all_rows_and_initializes_missing_refs():
    store = TernaryTruthStore(4, capacity=8)
    store.append_idea(torch.ones(4), trust=.5, evidence=(.5, 0.))
    store.append_relation(torch.ones(4), torch.ones(4), torch.ones(4))
    state = store.state_dict()
    for key in ('truth_schema', 'refs', 'row_ids', 'where', 'when'):
        state.pop(key)
    restored = TernaryTruthStore(4, capacity=8)
    with pytest.warns(UserWarning, match='legacy REL_OTHER'):
        restored.load_state_dict(state)
    assert len(restored) == 1
    assert bool((restored.refs == -1).all())
    assert restored.row(0)['evidence'] == (.5, 0.)


def test_migration_discards_all_old_provisioned_rows_for_reprovisioning():
    store = TernaryTruthStore(4, capacity=8)
    store.append_idea(torch.ones(4), trust=.5, evidence=(.5, 0.))
    old = store.append_idea(torch.ones(4), trust=.8)
    store.set_origin(old, store.ORIGIN_PROVISIONED)
    state = store.state_dict()
    state.pop('truth_schema')
    restored = TernaryTruthStore(4, capacity=8)
    with pytest.warns(UserWarning, match='re-provisioned'):
        restored.load_state_dict(state)
    assert len(restored) == 1
    assert restored.origin[0] != store.ORIGIN_PROVISIONED


def test_verification_keeps_conflicting_observations_as_two_poles():
    from test_truth_ideas_routing import _cs
    cs = _cs()
    store, points = clause_store()
    row = store.write_clause(part_clause(), evidence=(.8, .7))
    cs.verify_relation(row, [(points[1], points[2]), (points[1], points[4])],
                       store=store, support_weight=.9)
    assert store.row(row)['evidence'] == pytest.approx((.9, .9))


def test_invalid_parent_never_partially_writes_its_child():
    store, _ = clause_store()
    child = part_clause()
    outer = idea_clause(children=(child,), refs=(1, ('clause', 0), -1))
    with pytest.raises(ValueError, match='cannot fuse'):
        store.write_clause(outer)
    assert len(store) == 0


def test_migration_discards_dependents_of_retired_rows():
    import copy
    store = TernaryTruthStore(4, capacity=8)
    old = store.append_idea(torch.ones(4))
    store.set_origin(old, store.ORIGIN_PROVISIONED)
    child = ConceptualMeaning(torch.ones(3, 4), torch.ones(3, dtype=torch.bool),
        role_refs=(store.occurrence_of(old), None, None))
    store.append_meaning(child, kind='observation', rel_type=store.REL_PARTOF)
    store.append_idea(torch.zeros(4))
    state, extras = copy.deepcopy(store.state_dict()), copy.deepcopy(store.semantic_extras())
    state.pop('truth_schema')
    restored = TernaryTruthStore(4, capacity=8)
    with pytest.warns(UserWarning):
        restored.load_state_dict(state)
        restored.load_semantic_extras(extras)
    assert len(restored) == 1 and int(restored.occurrence_id[0]) == 2


def test_observation_writer_accepts_a_completed_field_without_a_program():
    from Models import _append_observed_meaning
    store, _ = clause_store()
    field = part_clause()
    row = _append_observed_meaning(store, field)
    torch.testing.assert_close(store.slots[row], field.slots, rtol=0, atol=0)
    assert store.refs[row].tolist() == list(field.refs)


def test_full_store_can_update_an_existing_relation():
    store, points = clause_store()
    store.write_clause(part_clause(), trust=.2)
    for _ in range(store.capacity - 1):
        store.append_idea(points[1])
    assert store.write_clause(part_clause(), trust=.8, evidence=(.8, 0.)) == 0
    assert len(store) == store.capacity
    assert store.row(0)['evidence'] == pytest.approx((.8, 0.))


def test_query_returns_end_clause_content_and_preserves_both_poles():
    from test_query_vp_boundaries import _world, _context
    cs, _grammar, registry, _a, _b, _ = _world()
    store = TernaryTruthStore(8, capacity=8)
    store.configure_clause_index(allocate=lambda _point: cs.new_concept(), concept_point=lambda _cid: None)
    point = torch.arange(8.).float() / 8
    meaning = ConceptualMeaning.from_description(point)
    row = store.write_clause(Clause(meaning, point=point), kind='fact', evidence=(.8, .7))
    context = _context(cs, store=store)
    question = registry.form('what', store.occurrence_of(row), context=context)
    result = registry.execute(question, context)
    assert result.evidence['semantic_id'] == 'what'
    torch.testing.assert_close(result.value[0]['meaning'].roles, meaning.roles)
    assert result.support_true == pytest.approx(.8)
    assert result.support_false == pytest.approx(.7)
