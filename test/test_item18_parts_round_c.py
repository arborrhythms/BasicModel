"""Round C's mechanism certificates use declared argument participation."""
from types import SimpleNamespace as NS

import pytest
import torch

from argument_fixtures import argument_inventory
from placement_fixtures import place_at
from test_parts_of_speech import rule
from test_parts_of_speech import chooser
from PartsOfSpeech import formal_arguments


def positive_rotation():
    # A random pair of axis swaps is orthogonal with determinant +1 and
    # stays inside the native positive cone; an arbitrary QR need not.
    axes = torch.randperm(4)
    mapping = torch.arange(4)
    mapping[axes] = axes.roll(2)
    rotation = torch.eye(4)[mapping]
    assert torch.linalg.det(rotation) == 1
    return rotation


def test_fit_reads_the_extension_and_not_the_class_position():
    owner, speech, words = argument_inventory(('modifier_argin1', 'modifier_argin2'))
    argument = 'modifier_argin1'
    speech.add(words[0], argument)
    place_at(owner, speech.concepts[argument], [1., 0., 0., 0.])
    before = speech.similarities(words[0]).clone()
    rotation = positive_rotation()
    row = owner._csw_row_of(speech.concepts[argument])
    owner.similarity_codebook.mereology.placement[row] = torch.tensor([1.,0.,0.,0.]) @ rotation
    after = speech.similarities(words[0])
    torch.testing.assert_close(before, after, atol=1e-6, rtol=0)
    assert after[speech.index[argument]] > 0


def test_unplaced_chart_does_not_read_trust_content_as_a_position():
    from ConceptTrust import ConceptTrust
    bank = NS(W=torch.zeros(2, 4))
    owner = NS(similarity_codebook=bank)
    trust = ConceptTrust(owner, bank)
    trust.allocated[1] = True
    trust.lengths[1] = 1.
    trust.content[1] = torch.tensor([1., 0., 0., 0.])
    trust.project()
    torch.testing.assert_close(bank.W[1], torch.zeros(4))
    torch.testing.assert_close(trust.content[1], torch.tensor([1., 0., 0., 0.]))


def test_kept_arguments_are_located_at_the_words_that_filled_them():
    from Understanding import ConceptualField
    binary = (rule('modifier'), rule('verb'))
    owner, speech, words = argument_inventory(formal_arguments(binary))
    # The declared tree is verb(modifier(modifier(cat, fast), the), ran).
    program = NS(word_ids=torch.tensor(words),
        actions=torch.tensor([[0,-1,2],[0,-1,1],[1,0,0],[0,-1,0],[1,0,0],[0,-1,3],[1,1,0]]),
        symbol_where=torch.tensor([[0,3],[4,8],[9,12],[13,16]]),
        symbol_when=torch.zeros(4,2,dtype=torch.long),
        leaf_evidence=torch.tensor([[1.,0.],[.7,.2],[.9,0.],[1.,0.]]))
    program.word_extents = program.symbol_where
    field = ConceptualField(torch.tensor(words), program.leaf_evidence[:,None,None].expand(-1,1,4,-1),
                           program.symbol_where[None], program.symbol_when[None], None)
    result = speech.observe_kept(program, binary, (), field=field, stream=0)
    lookup = {int(cid): i for i,cid in enumerate(result.concept_ids)}
    for word, names in ((2, ('modifier_argin1','modifier_argout1','verb_argin1','verb_argout1')),
                        (1, ('modifier_argin2',)), (0, ('modifier_argin2',)),
                        (3, ('verb_argin2',))):
        location = (result.where[0] == program.symbol_where[word]).all(-1)
        for name, cid in speech.concepts.items():
            expected = program.leaf_evidence[word] if name in names else torch.zeros(2)
            torch.testing.assert_close(result.evidence[lookup[cid],0,location].amax(0), expected)


def test_generate_fit_is_invariant_to_argument_rotation():
    from IdentityRows import point
    owner, speech, words = argument_inventory(('modifier_argin1', 'modifier_argin2'))
    speech.add(words[0], 'modifier_argin1')
    place_at(owner, speech.concepts['modifier_argin1'], [1.,0.,0.,0.])
    speech.prepare_values()
    value = point(owner, words[0])[None]
    before = speech.value_scores(value)
    rotation = positive_rotation()
    row = owner._csw_row_of(speech.concepts['modifier_argin1'])
    owner.similarity_codebook.mereology.placement[row] = torch.tensor([1.,0.,0.,0.]) @ rotation
    speech.prepare_values()
    torch.testing.assert_close(speech.value_scores(value), before, atol=1e-6, rtol=0)


def test_grouping_moves_each_order_only_by_its_own_members_and_keeps_frozen_rows():
    from copy import deepcopy
    binary = (rule('modifier'),)
    owner, speech, words = argument_inventory(formal_arguments(binary))
    program = NS(word_ids=torch.tensor(words[:2]), actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,0]]))
    for i, word in enumerate(words[:2]):
        speech.add(word, f'modifier_argin{i+1}')
    speech.add(words[0], 'modifier_argout1')
    speech.add(words[2], 'modifier_argin1')
    for i,name in enumerate(speech.arguments):
        place_at(owner, speech.concepts[name], torch.eye(4)[i].tolist())
    bank = owner.similarity_codebook
    before = deepcopy(bank.state_dict())
    word_rows = [owner._csw_row_of(word) for word in words]
    argument_rows = [owner._csw_row_of(speech.concepts[name]) for name in speech.arguments]
    speech.observe_kept(program, binary, ())
    moved = bank.mereology.placement.clone()
    assert not torch.equal(moved[word_rows[:2]], before['mereology.placement'][word_rows[:2]])
    assert not torch.equal(moved[argument_rows], before['mereology.placement'][argument_rows])
    torch.testing.assert_close(moved[word_rows[2:]], before['mereology.placement'][word_rows[2:]])
    # Move the other order by the first legal rotation, without a seed search.
    rotation = positive_rotation()
    for changed, held in ((argument_rows, word_rows), (word_rows, argument_rows)):
        bank.load_state_dict(before)
        bank.mereology.placement[changed] = bank.mereology.placement[changed] @ rotation
        speech.observe_kept(program, binary, ())
        torch.testing.assert_close(bank.mereology.placement[held], moved[held], atol=1e-6, rtol=0)
    bank.load_state_dict(before)
    owner._frozen_concepts = {words[0], speech.concepts['modifier_argin1']}
    frozen = [owner._csw_row_of(cid) for cid in owner._frozen_concepts]
    values = bank.mereology.placement[frozen].clone()
    speech.observe_kept(program, binary, ())
    torch.testing.assert_close(bank.mereology.placement[frozen], values, atol=0, rtol=0)


def test_centroid_refreshes_when_its_members_move_without_moving_the_class():
    from ConceptHierarchy import extension
    owner, speech, words = argument_inventory(('modifier_argin1', 'modifier_argin2'))
    cid = speech.concepts['modifier_argin1']
    speech.add(words[0], 'modifier_argin1')
    speech.add(words[1], 'modifier_argin1')
    place_at(owner, cid, [0.,0.,0.,1.])
    own = owner.similarity_codebook.mereology.placement[owner._csw_row_of(cid)].clone()
    torch.testing.assert_close(extension(owner,cid), torch.tensor([.5,.5,0.,0.]))
    place_at(owner, words[1], [0.,0.,1.,0.])
    torch.testing.assert_close(extension(owner,cid), torch.tensor([.5,0.,.5,0.]))
    torch.testing.assert_close(owner.similarity_codebook.mereology.placement[owner._csw_row_of(cid)], own)


def test_omitted_word_has_no_position_or_placement_seed():
    binary = (rule('modifier'),)
    owner,speech,words = argument_inventory(formal_arguments(binary))
    program = NS(word_ids=torch.tensor([words[0],-1]),
        actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,0]]))
    speech.observe_kept(program,binary,())
    assert speech.members('modifier_argin1') == (words[0],)
    assert speech.members('modifier_argin2') == ()


def test_expected_argument_presence_is_added_to_both_occupation_reads(chooser):
    from Expectation import SymbolRead
    from GenerateOccupation import features
    owner, speech, words = argument_inventory(formal_arguments((rule('modifier'),)))
    speech.add(words[0], 'modifier_argin1')
    speech.prepare_values()
    rows = torch.tensor([owner._csw_row_of(speech.concepts[name]) for name in speech.arguments])
    read = SymbolRead(torch.zeros(1,len(rows),4), torch.zeros(1,len(rows),2), 1.,1e-6,'clip',rows)
    read.expected[0,speech.index['modifier_argin1'],0] = .7
    like = owner.similarity_codebook.W
    priming = speech.expected_arguments(read, like[:1])
    assert priming[0,speech.index['modifier_argin1']] == .7
    base = speech.argument_scores(torch.tensor([words[0]]), like)
    torch.testing.assert_close((base+priming)-base, priming)
    language = NS(parts_of_speech=speech, _generate_binary_rules=(rule('modifier'),), _generate_unary_rules=())
    value = owner.similarity_codebook.lookup_rows(owner._csw_row_of(words[0]))[None]
    left = right = value[:,None].expand(-1,2,-1)
    without = features(language,value,left,right)
    with_prime = features(language,value,left,right,expected=priming)
    torch.testing.assert_close(with_prime-without, torch.tensor([[[0.,.7,0.],[0.,0.,0.]]]))
    layer, native = chooser
    addresses = torch.tensor([native.owner._csw_row_of(native.concepts[name]) for name in native.arguments])
    prediction = SymbolRead(torch.zeros(1,len(addresses),4), torch.zeros(1,len(addresses),2),
                            1.,1e-6,'clip',addresses)
    prediction.expected[0,native.index['lower_argin1'],0] = .7
    values = torch.ones(1,2,4)
    occupation = values.new_zeros(1,2,len(native.arguments))
    base = layer(values,slots=2,occupation=occupation)[2]['logits']
    occupation[:,0] += native.expected_arguments(prediction, values)
    actual = layer(values,slots=2,occupation=occupation)[2]['logits']
    torch.testing.assert_close(actual-base, torch.tensor([[.7,0.,0.,0.,0.]]))


def test_argument_field_entry_and_placements_round_trip_without_a_word_argument_table(tmp_path):
    from Models import BaseModel
    from types import MethodType
    from Memory import LTM0
    from Understanding import ConceptualField
    from EpisodicMemory import select, write
    binary = (rule('modifier'),)
    owner, speech, words = argument_inventory(formal_arguments(binary))
    program = NS(word_ids=torch.tensor(words[:2]), actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,0]]),
        word_extents=torch.tensor([[0,1],[2,3]]),leaf_evidence=torch.tensor([[1.,0.],[.7,.2]]))
    field = ConceptualField(torch.tensor(words[:2]),torch.eye(2)[:,None,:,None]*torch.tensor([1.,0.]),
        program.word_extents[None],torch.zeros(1,2,2,dtype=torch.long),None)
    result = speech.observe_kept(program,binary,(),field=field,stream=0)
    for name,value in (('activations',result.evidence),('ids',result.concept_ids),
                       ('where',result.where),('when',result.when)):
        object.__setattr__(owner.subspace,'_concept_'+name,value)
    store = LTM0(owner.nWhat+4,capacity=4,episodic_concepts=16)
    row = store.append_idea(torch.nn.functional.pad(owner.similarity_codebook.lookup_rows(owner._csw_row_of(words[0])),(0,4)))
    entry = select(result,store.episodic_capacity,torch.tensor([True]))
    write(store,row,tuple(value[0] for value in entry))
    checkpoint = tmp_path/'parts.pt'
    structural = BaseModel._collect_structural_extras(NS(conceptualSpaces=[owner],conceptualSpace=owner))
    torch.save(dict(owner=owner.state_dict(),parts=speech.state_dict(),store=store.state_dict(),structural=structural),checkpoint)
    fresh, fresh_parts, _ = argument_inventory(formal_arguments(binary))
    restored = LTM0(owner.nWhat+4,capacity=4,episodic_concepts=16)
    state = torch.load(checkpoint,weights_only=False)
    wrapper = NS(conceptualSpaces=[fresh],conceptualSpace=fresh,named_modules=lambda: ())
    wrapper._restore_allocator_extras = MethodType(BaseModel._restore_allocator_extras,wrapper)
    BaseModel._restore_structural_extras(wrapper,state['structural'])
    fresh.load_state_dict(state['owner']); fresh_parts.load_state_dict(state['parts']); restored.load_state_dict(state['store'])
    torch.testing.assert_close(restored.episodic_concepts,store.episodic_concepts)
    torch.testing.assert_close(restored.episodic_values,store.episodic_values)
    torch.testing.assert_close(fresh.similarity_codebook.mereology.placement,owner.similarity_codebook.mereology.placement)
    torch.testing.assert_close(fresh.subspace._concept_activations,result.evidence)
    torch.testing.assert_close(fresh.subspace._concept_where,result.where)
    assert {name:fresh_parts.members(name) for name in speech.arguments} == {name:speech.members(name) for name in speech.arguments}
    assert not any(value.ndim == 2 and tuple(value.shape) == (len(words),len(speech.arguments))
                   for value in speech.state_dict().values() if torch.is_tensor(value))
