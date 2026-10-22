"""Two-truths storage and provenance contracts (item 7, §§1–4).

The September 28 amendment keeps fixture identification evidence explicit
beside source trust; equal numbers below are independently supplied.
"""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from Layers import LTM0, TruthLayer
from Spaces import ConceptualSpace
from ClauseRow import Clause
from Meaning import ConceptualMeaning


def test_end_row_has_shared_references_and_field_coordinates():
    store = LTM0(4, capacity=8)
    assert store.refs.shape == (8, 3)
    assert store.refs.dtype == torch.int64
    assert bool((store.refs == -1).all())
    assert store.when.shape == (8, 4)


def test_both_evidence_poles_survive_checkpoint_separately_from_neither():
    store = LTM0(4, capacity=8)
    both = store.append_idea(torch.ones(4))
    neither = store.append_idea(torch.ones(4), sentence_index=1)
    store.set_evidence(both, .8, .8)
    restored = LTM0(4, capacity=8)
    restored.load_state_dict(store.state_dict())
    assert restored.row(both)['evidence'] == pytest.approx((.8, .8))
    assert restored.row(neither)['evidence'] == (0., 0.)
    assert restored.row(both)['corners'] == pytest.approx((.2, .2, .8, .2))


def test_relation_has_no_kind_column_or_catch_all():
    from LTM1 import LTM1
    assert LTM1.columns == ('refs','truth_plus','truth_minus','access_count','surprise','when')
    assert not any(name.startswith('REL_') for name in vars(LTM0))


def test_relation_rows_never_enter_luminosity():
    store, _ = clause_store()
    idea = store.append_idea(torch.tensor([1.,0.,0.,0.]), trust=.6, evidence=(.6,0.))
    store.write_clause(part_clause(), evidence=(1.,0.))
    store.set_origin(idea, store.ORIGIN_PROVISIONED)
    light = TruthLayer(4, max_truths=8); light.attach_ltm(store)
    assert light.sync_from_ltm() == 1
    torch.testing.assert_close(light.truths[0], torch.tensor([.6,0.,0.,0.]))


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
    from part_fixtures import native_store
    store, symbols = native_store(points=torch.eye(4))
    points = dict(zip(symbols, torch.eye(4)))
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
    assert len(store) == 1 and not len(store.parts)
    torch.testing.assert_close(store.slots[row, 0], clause.point)
    assert not bool(store.slots[row, 1:].any())
    torch.testing.assert_close(store.meaning_of(row).roles[0], clause.point)
    assert store.meaning_of(row).role_mask.tolist() == [True, False, False]
    assert store.refs[row].tolist() == [-1, -1, -1]
    from Spaces import WhenEncoding
    from Occurrence import document_digest
    assert int(store.sentence_index[row]) == 0
    assert bytes(store.document_keys[row].tolist()) == document_digest(('direct', 'observation'))
    encoding = WhenEncoding(n_when=4).set_capacity(store.capacity)
    assert int(encoding.decode_index(store.when[row])) == 0


def test_recursive_absolute_clauses_each_write_one_idea():
    store, _ = clause_store()
    inner = idea_clause()
    middle = idea_clause(children=(inner,))
    outer = idea_clause(children=(middle,))
    from Occurrence import sentence_key
    with store.at_address('nested-document', 0, sentence_key((b'nested',))):
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
    row = store.write_clause(outer, trust=.65)
    child_row = ('sym', store.parts.find(1,2))
    assert store.row(child_row)['evidence'] == (0.,0.)
    assert store.row(row)['refs'].tolist() == [4,child_row[1]]
    assert store.write_clause(child, trust=.8, evidence=(.8,0.)) == child_row
    assert len(store) == 0 and len(store.parts) == 2
    assert store.row(child_row)['evidence'] == pytest.approx((.8,0.))
    store.parts.truth_plus[row[1]] = 0
    assert store.row(child_row)['evidence'] == pytest.approx((.8,0.))


def test_second_order_implication_names_two_part_symbols():
    store, _ = clause_store()
    left, right = part_clause(), part_clause(1,4)
    outer = Clause(left.meaning, relation='implies',
        refs=(('clause',0),4,('clause',1)), children=(left,right))
    row = store.write_clause(outer, evidence=(.9,0.))
    assert len(store) == 0 and len(store.parts) == 3
    first, second = store.relation_operands(row)
    assert first['refs'].tolist() == [1,2] and second['refs'].tolist() == [1,4]
    assert first['evidence'] == second['evidence'] == (0.,0.)
    assert store.row(row)['readings'] == ('conditional',)
    matches = store.consequents_by_row(first['row_id'], reading='implies')
    assert matches[0][:2] == (row[1],second['row_id'])


def test_equality_is_one_singleton_membership_and_never_merges_native_content():
    store, points = clause_store()
    before = store._concept_owner().similarity_codebook.W.detach().clone()
    row = store.write_clause(replace(part_clause(), relation='equal'), evidence=(.75,0.))
    assert len(store.parts) == 1 and store.row(row)['refs'].tolist() == [1,2]
    assert 'equality' in store.row(row)['readings']
    assert store.parts.find(2,1) is None
    assert not torch.equal(points[1],points[2])
    torch.testing.assert_close(store._concept_owner().similarity_codebook.W,before,rtol=0,atol=0)


def test_vector_and_row_readers_keep_both_poles_and_weight_substitution():
    store, points = clause_store()
    row = store.write_clause(part_clause(), evidence=(.7,.4))
    torch.testing.assert_close(store.evaluate(points[1],points[2]),torch.tensor([.7,.4]))
    torch.testing.assert_close(store.evaluate_rows(1,2),torch.tensor([.7,.4]))
    match, = store.consequents(points[1])
    assert match[0] == row[1] and match[1] == pytest.approx(.7)


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
    assert len(store.parts) == 1
    assert store.row(store.location(store.parts.find(1,2)))['evidence'] == pytest.approx((.8, .7))


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


def test_retired_relation_columns_are_rejected_without_dropping_rows():
    store = LTM0(4,capacity=8)
    store.append_idea(torch.ones(4),evidence=(.5,0.))
    state = store.state_dict(); state['rel_type'] = torch.ones(8,dtype=torch.long)
    restored = LTM0(4,capacity=8)
    with pytest.raises(ValueError, match='retired'):
        restored.load_state_dict(state)
    assert len(restored) == 0


def test_native_provisioned_content_survives_a_schema_stamp_upgrade():
    store = LTM0(4,capacity=8)
    row = store.append_idea(torch.ones(4),evidence=(.8,0.))
    store.set_origin(row,store.ORIGIN_PROVISIONED)
    state = store.state_dict();state.pop('truth_schema')
    restored = LTM0(4,capacity=8);restored.load_state_dict(state)
    assert len(restored) == 1 and restored.origin[0] == store.ORIGIN_PROVISIONED
    assert restored.row(0)['evidence'] == pytest.approx((.8,0.))


def test_verification_keeps_conflicting_observations_as_two_poles():
    from test_truth_ideas_routing import _cs
    cs = _cs()
    store, points = clause_store()
    row = store.write_clause(part_clause(), evidence=(.8, .7))
    cs.verify_relation(row[1], [(points[1], points[2]), (points[1], points[4])],
                       store=store, support_weight=.9)
    assert store.row(row)['evidence'] == pytest.approx((.9, .9))


def test_invalid_parent_never_partially_writes_its_child():
    store, _ = clause_store()
    child = part_clause()
    outer = idea_clause(torch.full((4,),-.1),children=(child,),refs=(1,('clause',0),-1))
    before = store._concept_owner()._concept_allocator.next_id
    with pytest.raises(RuntimeError,match='identity map'):
        store.write_clause(outer)
    assert len(store) == len(store.parts) == 0
    assert store._concept_owner()._concept_allocator.next_id == before


def test_retired_relation_checkpoint_is_rejected_with_dependents_intact():
    store = LTM0(4,capacity=8)
    old = store.append_idea(torch.ones(4))
    child = ConceptualMeaning(torch.ones(3,4),torch.ones(3,dtype=torch.bool),
        role_refs=(store.occurrence_of(old),None,None))
    store.append_meaning(child,kind='question')
    state=store.state_dict();state['rel_type']=torch.tensor([0,1,0,0,0,0,0,0])
    restored=LTM0(4,capacity=8)
    with pytest.raises(ValueError,match='retired'):
        restored.load_state_dict(state)
    assert len(restored) == 0 and len(store) == 2


def test_observation_writer_accepts_a_completed_field_without_a_program():
    from Models import _append_observed_meaning
    store,_=clause_store();field=part_clause()
    row=_append_observed_meaning(store,field)
    assert len(store)==0 and store.row(row)['refs'].tolist()==[1,2]


def test_full_store_can_update_an_existing_relation():
    store,points=clause_store()
    row=store.write_clause(part_clause(),evidence=(.2,0.))
    for position in range(store.capacity):
        store.append_idea(points[1],sentence_index=position)
    assert store.write_clause(part_clause(),evidence=(.8,0.))==row
    assert len(store)==store.capacity and len(store.parts)==1
    assert store.row(row)['evidence']==pytest.approx((.8,0.))


def test_query_returns_end_clause_content_and_preserves_both_poles():
    from test_query_vp_boundaries import _world, _context
    cs, _grammar, registry, _a, _b, _ = _world()
    store = LTM0(8, capacity=8)
    from ClauseRow import attach_clause_index
    attach_clause_index(SimpleNamespace(),store,cs)
    point = torch.arange(8.).float() / 8
    meaning = ConceptualMeaning.from_description(point)
    row = store.write_clause(Clause(meaning, point=point), kind='fact', evidence=(.8, .7))
    context = _context(cs, store=store)
    question = registry.form('ask', store.occurrence_of(row), context=context)
    result = registry.execute(question, context)
    assert result.evidence['semantic_id'] == 'ask'
    torch.testing.assert_close(result.value[0]['meaning'].roles, meaning.roles)
    assert result.support_true == pytest.approx(.8)
    assert result.support_false == pytest.approx(.7)
