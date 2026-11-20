from placement_fixtures import place_types
"""Formal argument declarations allocate concepts, never learned evidence."""
import pytest
import torch

from test_review13_subspaces import fixture
from nomination_fixtures import recurrent_word


def operation_measurement(occupations, name='change'):
    """The reported definiteness remains a lesson measurement, without a gate."""
    value = occupations.operation(name)
    return min(value['definiteness']) if value['measurable'] and all(
        v is not None for v in value['definiteness']) else None


def test_formal_arguments_are_prebuilt_zero_radius_concepts_above_words():
    cs, bank, _, alloc = fixture(meaning_pairs=4)
    names = ('change_argin1', 'change_argin2', 'change_argout1')
    pointers = (bank.W.data_ptr(), bank.trust.lengths.data_ptr())
    concepts = cs.prebuild_formal_arguments(names)
    assert set(concepts) == set(names)
    assert len(set(concepts.values())) == len(names)
    for cid in concepts.values():
        assert alloc.order_of(cid) == 1  # The native word order is zero.
        row = cs._concept_row_of(cid)
        assert row is not None
        assert 'formal-argument' in alloc.symbol_types[cid]
        assert not bank.lookup_rows(row).any()
        assert not bank.trust.lengths[row].any()
        assert int(bank.trust.views[row]) == 0
        assert not cs.is_frozen(cid)
    next_id = alloc.next_id
    assert cs.prebuild_formal_arguments(names) == concepts
    assert alloc.next_id == next_id
    assert pointers == (bank.W.data_ptr(), bank.trust.lengths.data_ptr())


def test_insufficient_argument_capacity_refuses_the_whole_declaration():
    cs, bank, _, alloc = fixture(meaning_pairs=4)
    before = alloc.next_id, dict(alloc.layer()._tensor_row_keys), bank.W.clone()
    with pytest.raises(RuntimeError, match='formal argument'):
        cs.prebuild_formal_arguments(tuple(f'op_argin{i}' for i in range(33)))
    assert alloc.next_id == before[0]
    assert alloc.layer()._tensor_row_keys == before[1]
    torch.testing.assert_close(bank.W, before[2])


def test_group_of_argument_concepts_is_one_order_above_arguments():
    cs, _, _, alloc = fixture(meaning_pairs=4)
    arguments = cs.prebuild_formal_arguments(('change_argin1', 'verb_argin2'))
    group = cs.new_concept()
    for argument in arguments.values():
        cs.add_part(group, ('sym', argument))
    assert alloc.order_of(group) == 2
    assert cs._csw_concept_row(2, group) is not None


def test_argument_declarations_survive_native_structural_checkpoint():
    import copy
    from types import SimpleNamespace
    from Models import BaseModel
    cs, bank, _, alloc = fixture(meaning_pairs=4)
    names = ('change_argin1', 'change_argin2')
    concepts = cs.prebuild_formal_arguments(names)
    host = SimpleNamespace(conceptualSpaces=[cs], wholeSpace=None, named_modules=lambda: ())
    saved = BaseModel._collect_structural_extras(host)['conceptual_spaces'][0]['allocator']
    vectors = copy.deepcopy(bank.state_dict())
    restored, restored_bank, _, restored_alloc = fixture(meaning_pairs=4)
    BaseModel._restore_allocator_extras(host, restored, saved)
    restored_bank.load_state_dict(vectors)
    next_id = restored_alloc.next_id
    assert restored.prebuild_formal_arguments(names) == concepts
    assert restored_alloc.next_id == next_id == alloc.next_id
    for cid in concepts.values():
        assert restored_alloc.order_of(cid) == alloc.order_of(cid)
        assert restored._concept_row_of(cid) == cs._concept_row_of(cid)


def argument_fixture(*names):
    from PartsOfSpeech import PartsOfSpeech
    from definition_fixtures import agreeing_views
    cs, bank, _, alloc = fixture(meaning_pairs=4)
    speech = PartsOfSpeech(names or ('change_argin1', 'change_argin2'), cs)
    words = tuple(recurrent_word(cs.interpret, [part], (), form=form)
                  for part, form in ((4, 'dax'), (7, 'wug')))
    for i, word in enumerate(words):
        row = cs._concept_row_of(word)
        bank.mereology.placement[row].copy_(torch.eye(4)[i])
        bank.mereology.placement_witnesses[row] = 1
    agreeing_views(cs, *words)
    return cs, bank, alloc, speech, words


def test_argument_views_learn_radius_and_shared_words_do_not_divide_membership():
    import math
    cs, bank, alloc, speech, (word, _) = argument_fixture()
    for _ in range(3):
        speech.observe_use('change', ((word,), (word,)))
    for identity in speech.concepts.values():
        row = cs._concept_row_of(identity)
        bank.mereology.placement[row] = torch.eye(4)[0]
        bank.mereology.placement_witnesses[row] = 1
    radius = -math.expm1(-3 / bank.trust.trust_views)
    from ConceptTrust import positive_length
    assert 'trust' not in speech.operation('change')
    assert positive_length(cs, speech.seats['change'].concept) == 0.
    assert speech.operation('change')['definiteness'] == pytest.approx((radius, radius))
    assert speech.members('change_argin1') == speech.members('change_argin2') == (word,)
    assert not hasattr(speech, 'counts') and not hasattr(speech, 'distribution')
    for argument in speech.arguments:
        cid = speech.concepts[argument]
        assert alloc.order_of(cid) == alloc.order_of(word) + 1
        row = cs._concept_row_of(cid)
        assert int(bank.trust.views[row]) == 3
        torch.testing.assert_close(bank.lookup_rows(row)[3:7], torch.eye(4)[0]*radius)
    # Scores are geometry with confidence, not probabilities that must sum to one.
    torch.testing.assert_close(speech.similarities(word), torch.ones(2))




def test_higher_order_context_cannot_move_word_meanings_forms_or_radii():
    cs, bank, _, speech, words = argument_fixture('group_argin1')
    for word in words:
        speech.add(word, 'group_argin1')
    place_types(cs)
    rows = [cs._concept_row_of(word) for word in words]
    forms = bank.lookup_rows(torch.tensor(rows))[:, :3].clone()
    radii = bank.trust.lengths.clone()
    before = bank.mereology.placement[rows].clone()
    parent = speech._direction(speech.concepts['group_argin1']).clone()
    context = torch.zeros(cs.nVectors)
    context[cs._concept_row_of(speech.concepts['group_argin1'])] = 1.
    bank.mereology.negative_count = 0
    bank.mereology.place_rows(rows, context, b'other-order')
    after = bank.mereology.placement[rows]
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(bank.trust.lengths, radii, rtol=0, atol=0)
    torch.testing.assert_close(bank.lookup_rows(torch.tensor(rows))[:, :3], forms, rtol=0, atol=0)


def test_empty_argument_views_supply_no_operation_authority_or_definiteness(monkeypatch):
    cs, _, _, speech, _ = argument_fixture()
    for _ in range(3):
        speech.observe_use('change', ((), ()))
    assert speech.operation('change')['uses'] == 3
    from ConceptTrust import positive_length
    assert positive_length(cs, speech.seats['change'].concept) == 0.
    assert operation_measurement(speech) == 0.


@pytest.mark.parametrize('formal', [False, True])
def test_form_only_charts_record_members_without_learning_from_form_geometry(formal):
    import math
    from PartsOfSpeech import PartsOfSpeech
    from ConceptTrust import positive_length
    cs, bank, _, _ = fixture(meaning_pairs=0)
    from ConceptHierarchy import observe_member
    word = recurrent_word(cs.interpret, [4], (), form='dax')
    if formal:
        speech = PartsOfSpeech(('change_argin1',), cs)
        parent = speech.concepts['change_argin1']
    else:
        parent = cs.new_concept()
        cs.add_part(parent, ('sym', word))
        cs._csw_concept_row(1, parent)
    for _ in range(3):
        observe_member(cs, parent, word)
    assert positive_length(cs, word) == 1.
    from ConceptHierarchy import members, direction
    row = cs._concept_row_of(parent)
    assert int(bank.trust.views[row]) == 0
    assert float(bank.trust.lengths[row, 0]) == 0.
    assert members(cs, parent) == (word,)
    assert direction(cs, parent).numel() == 0
    before = bank.W.clone()
    place_types(cs)
    torch.testing.assert_close(bank.W, before, rtol=0, atol=0)


def test_ordinary_nested_classes_place_within_each_order_without_a_grammar_catalogue():
    from ConceptHierarchy import observe_member, direction
    from definition_fixtures import agreeing_views
    cs, bank, _, alloc = fixture(meaning_pairs=4)
    words = [recurrent_word(cs.interpret, [part], (), form=form)
             for part, form in ((4, 'dax'), (7, 'wug'), (8, 'zup'))]
    rows = [cs._concept_row_of(word) for word in words]
    for i, row in enumerate(rows):
        bank.mereology.placement[row] = torch.eye(4)[i]
        bank.mereology.placement_witnesses[row] = 1
    agreeing_views(cs, *words)

    def group(members):
        cid = cs.new_concept()
        for member in members:
            cs.add_part(cid, ('sym', member))
        cs._csw_concept_row(alloc.order_of(cid), cid)
        for _ in range(3):
            for member in members:
                observe_member(cs, cid, member)
        place_types(cs)
        return cid

    first, second = group(words[:2]), group(words[2:])
    upper = group((first, second))
    assert (alloc.order_of(first), alloc.order_of(upper)) == (1, 2)
    assert not alloc.formal_arguments
    before = bank.mereology.placement[rows].clone()
    target = direction(cs, upper).clone()
    forms = bank.lookup_rows(torch.tensor(rows))[:, :3].clone()
    radii = bank.trust.lengths.clone()
    bank.mereology.negative_count = 0  # No same-order negative update in this check.
    first_before = direction(cs, first).clone()
    context = torch.zeros(cs.nVectors)
    context[cs._concept_row_of(upper)] = 1.
    bank.mereology.place_rows(rows + [cs._concept_row_of(first)], context, b'higher-order-context')
    torch.testing.assert_close(bank.mereology.placement[rows], before, rtol=0, atol=0)
    torch.testing.assert_close(bank.trust.lengths, radii, rtol=0, atol=0)
    torch.testing.assert_close(bank.lookup_rows(torch.tensor(rows))[:, :3], forms, rtol=0, atol=0)
    torch.testing.assert_close(direction(cs, first), first_before, rtol=0, atol=0)


def test_learned_members_placements_and_confidence_restore_into_a_fresh_owner():
    import copy
    from types import SimpleNamespace
    from Models import BaseModel
    cs, bank, _, speech, words = argument_fixture()
    for _ in range(3):
        speech.observe_use('change', ((words[0],), (words[1],)))
    place_types(cs)
    host = SimpleNamespace(conceptualSpaces=[cs], wholeSpace=None, named_modules=lambda: ())
    structure = BaseModel._collect_structural_extras(host)['conceptual_spaces'][0]['allocator']
    vectors, grammar = copy.deepcopy(bank.state_dict()), copy.deepcopy(speech.state_dict())
    fresh, fresh_bank, _, restored, fresh_words = argument_fixture()
    assert fresh_words == words
    pointers = fresh_bank.W.data_ptr(), fresh_bank.trust.lengths.data_ptr()
    BaseModel._restore_allocator_extras(host, fresh, structure)
    fresh_bank.load_state_dict(vectors)
    restored.load_state_dict(grammar)
    assert restored.operation('change') == speech.operation('change')
    for name in speech.arguments:
        assert restored.members(name) == speech.members(name)
        torch.testing.assert_close(restored._direction(restored.concepts[name]), speech._direction(speech.concepts[name]))
    assert pointers == (fresh_bank.W.data_ptr(), fresh_bank.trust.lengths.data_ptr())


@pytest.mark.parametrize('formal', [False, True])
def test_explicit_class_views_on_a_whole_conceptual_chart_use_the_same_radius_rule(formal):
    import math
    from test_cs_sparse_weights import _cs
    from ConceptHierarchy import observe_member
    from ConceptTrust import positive_length
    from PartsOfSpeech import PartsOfSpeech
    cs = _cs(nS=32, order=3)
    bank = cs.similarity_codebook
    assert getattr(bank, 'mereology', None) is None
    word = recurrent_word(cs.interpret, [4], (), form='dax')
    if formal:
        speech = PartsOfSpeech(('change_argin1',), cs)
        parent = speech.concepts['change_argin1']
    else:
        parent = cs.new_concept()
        cs.add_part(parent, ('sym', word))
        cs._csw_concept_row(1, parent)
    for _ in range(3):
        observe_member(cs, parent, word)
    assert positive_length(cs, word) == 1.
    assert positive_length(cs, parent) == pytest.approx(-math.expm1(-1.))
    bank.trust.project()
    assert not bank.W[cs._concept_row_of(parent)].any()  # Extension is not the class's position.


def test_revised_separability_evidence_and_contiguous_classes():
    import math
    from placement_fixtures import place_types
    cs, bank, _, speech, (a,b) = argument_fixture('change_argin1','change_argin2','verb_argin1')
    for view in range(1,6):
        speech.observe_use('change', ((a,), (b,)))
        for name, point in zip(('change_argin1','change_argin2'), torch.eye(4)[:2]):
            row = cs._concept_row_of(speech.concepts[name])
            bank.mereology.placement[row] = point
            bank.mereology.placement_witnesses[row] = 1
        result=speech.operation('change')
        expected=-math.expm1(-view/bank.trust.trust_views)
        assert [v['d_j'] for v in result['argument_measurements'].values()] == [1.,1.]
        assert result['definiteness'] == pytest.approx((expected,expected))
        assert operation_measurement(speech) == pytest.approx(expected)
        assert (expected >= .75) == (view == 5)
    speech.add(a,'verb_argin1')
    row = cs._concept_row_of(speech.concepts['verb_argin1'])
    bank.mereology.placement[row] = torch.eye(4)[0]
    bank.mereology.placement_witnesses[row] = 1
    assert speech.operation('change')['argument_measurements']['change_argin1']['d_j']==1.
    # Overlap competes only outside the contiguity angle: a shared member
    # at the competing centroid is no longer won strictly by the mixed class.
    row=cs._concept_row_of(speech.concepts['change_argin1'])
    bank.mereology.placement[row]=torch.tensor([1.,1.,0.,0.])/(2**.5)
    speech.add(b,'change_argin1')
    result=speech.operation('change')
    assert result['argument_measurements']['change_argin1']['d_j'] < 1.
    assert operation_measurement(speech) < .75


def test_revised_fit_is_coverage_and_radius_is_commitment():
    from placement_fixtures import place_types
    from IdentityRows import meet_ratios
    from ConceptHierarchy import extension
    cs,bank,_,speech,(a,b)=argument_fixture('group_argin1')
    speech.add(a,'group_argin1');speech.add(b,'group_argin1');place_types(cs)
    row=cs._concept_row_of(speech.concepts['group_argin1'])
    bank.mereology.placement[row] = torch.tensor([1., 1., 0., 0.]) / 2**.5
    outsider=recurrent_word(cs.interpret, [9],(),form='outsider')
    outside_row=cs._concept_row_of(outsider)
    bank.mereology.placement[outside_row]=torch.nn.functional.normalize(torch.tensor([.2,.98,0.,0.]),dim=0)
    bank.mereology.placement_witnesses[outside_row]=1
    assert outsider not in speech.members('group_argin1')
    scores=[]
    for radius in (.1,.3,.6,1.):
        bank.trust.lengths[row,0]=radius
        score=float(speech.similarities(outsider)[0]);scores.append(score)
        assert score == pytest.approx(float(meet_ratios(speech._direction(outsider),extension(cs, speech.concepts['group_argin1'])*radius)[0]))
    assert all(x>=y for x,y in zip(scores,scores[1:])) and scores[0]>scores[-1]
    bank.trust.lengths[row,0]=0.;assert float(speech.similarities(a)[0])==0.


def test_revised_boundary_scope_preserves_absent_members_and_objects(monkeypatch):
    from placement_fixtures import place_types
    from definition_fixtures import agreeing_views
    from IdentityRows import mint
    cs,bank,_,speech,(a,b)=argument_fixture('group_argin1')
    obj=mint(cs,torch.eye(cs.nDim)[3]);agreeing_views(cs,obj,count=3)
    cs.interpret.define(a,obj)
    binding=cs.definitions.objects(a)
    speech.add(a,'group_argin1');speech.add(b,'group_argin1')
    place_types(cs)
    before=bank.mereology.placement.clone()
    context = torch.zeros(cs.nVectors)
    context[cs._concept_row_of(b)] = 1.
    bank.mereology.negative_count = 0
    bank.mereology.placement[cs._concept_row_of(a)] = torch.tensor([1.,0.,0.,0.])
    bank.mereology.placement[cs._concept_row_of(b)] = torch.tensor([0.,1.,0.,0.])
    before=bank.mereology.placement.clone()
    bank.mereology.place_rows([cs._concept_row_of(a)], context, b'scoped-row')
    assert not torch.equal(before[cs._concept_row_of(a)],bank.mereology.placement[cs._concept_row_of(a)])
    for identity in (b,obj):torch.testing.assert_close(before[cs._concept_row_of(identity)],bank.mereology.placement[cs._concept_row_of(identity)],rtol=0,atol=0)
    assert cs.definitions.objects(a)==binding




def test_revised_form_only_operation_reports_unmeasurable_definiteness():
    from PartsOfSpeech import PartsOfSpeech
    cs,bank,_,alloc=fixture(meaning_pairs=0)
    speech=PartsOfSpeech(('change_argin1','change_argin2'),cs)
    a,b=[recurrent_word(cs.interpret, [i],(),form=f'term{i}') for i in (4,7)]
    speech.observe_use('change',((a,),(b,)))
    from ConceptTrust import positive_length
    assert positive_length(cs, speech.seats['change'].concept) == 0.
    assert operation_measurement(speech) is None
    row=alloc.parts.add(a,b,truth_plus=.8,truth_minus=.3)
    torch.testing.assert_close(torch.stack(alloc.parts.evidence(row)), torch.tensor([.8, .3]))
    torch.testing.assert_close(torch.stack((alloc.parts.truth_plus[row], alloc.parts.truth_minus[row])), torch.tensor([.8, .3]))


def test_revised_relation_membership_is_enumerated_once():
    from test_priming_diffusion import class_fixture
    from Priming import part_edges
    cs,derived,(a,b,c),_=class_fixture()
    src,dst,w=part_edges(cs,authority=False)
    pairs=list(zip(src.tolist(),dst.tolist()))
    assert len(pairs)==len(set(pairs))
    selected=w[dst==c];torch.testing.assert_close(selected,torch.tensor([1.,.5]))


def test_revised_membership_proof_names_formal_class():
    from placement_fixtures import place_types
    from Queries import ThoughtTaxonomyCapability
    from QueryWork import QueryWorkBudget
    cs,bank,_,speech,(a,b)=argument_fixture('group_argin1')
    for _ in range(3):speech.add(a,'group_argin1')
    place_types(cs);parent=speech.concepts['group_argin1']
    # An observed occupant alone does not assert the taxonomy relation.
    reader = ThoughtTaxonomyCapability(cs)
    bounds = dict(max_nodes=64,max_records=64,max_steps=8,max_expansions=64)
    assert not reader.evidence(('sym',a),('sym',parent),work=QueryWorkBudget(128),**bounds)['path']
    cs._concept_allocator.parts.add(a,parent,truth_plus=1.)
    proof=reader.evidence(('sym',a),('sym',parent),max_nodes=64,max_records=64,max_steps=8,max_expansions=64,work=QueryWorkBudget(128))
    assert proof['path']
    assert proof['path'][0].part==('sym',a) and proof['path'][0].whole==('sym',parent)
    assert proof['formal_arguments']==(('group_argin1',parent),)
