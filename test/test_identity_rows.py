"""Step 1b: regions have one native owner; indexes cannot retain stale values."""
from dataclasses import replace
import torch
import pytest

from IdentityRows import meet_ratios, point, witness_count, refold
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from test_reference_close import fixture, close, form_exposure


def test_meet_ties_kind_requests_and_excludes_missing_defining_content():
    resolver, _, atoms = fixture()
    q = atoms[0]
    candidates = {1:dict(content=q+atoms[1],role=0,position=(0,0)),
                  2:dict(content=q+atoms[2]+atoms[3],role=0,position=(0,0))}
    _, phi, logits = resolver.score(q, q*0, q*0, 0, candidates)
    assert phi[:,0].tolist() == pytest.approx([.5, 1/3])
    assert phi[:,1].tolist() == [1.,1.]
    assert logits.tolist() == [1.,1.]
    _, _, specific = resolver.score(q+atoms[1], q*0, q*0, 0, candidates)
    assert specific[0] > specific[1]
    with pytest.raises(ValueError, match='native concept'):
        resolver.coordinates(-q)


def test_native_row_is_the_only_content_and_witnesses_are_live_queries():
    resolver, store, a = fixture()
    frame, address, _ = close(resolver,a[0],a[6],'mint',(),document='one',sentence=0)
    cid = int(store.refs[store.index_of_row(address),0])
    assert witness_count(resolver.components.owner,cid) == 1
    owner = resolver.components.owner
    assert not hasattr(resolver.components,'nouns')
    assert all(not hasattr(resolver,k) for k in ('seen','cooccurrence','associations','predicates','cue_policy'))
    # Identity reads do not invoke an event chart, including after a row edit.
    resolver.components._ready = lambda: (_ for _ in ()).throw(AssertionError('identity chart'))
    with torch.no_grad(): owner.similarity_codebook.W[owner._csw_row_of(cid)].copy_(a[1])
    candidates = resolver.individuals(resolver.accessible((frame,)))
    torch.testing.assert_close(candidates[cid]['content'],a[1],rtol=0,atol=0)
    store.refs[store.index_of_row(address),0] = -1
    assert witness_count(owner,cid) == 0


def test_forgetting_does_not_subtract_but_reassignment_refolds_retained_rows():
    resolver, store, a = fixture()
    first, address, _ = close(resolver,a[0]+a[1],a[6],'mint',(),document='fold',sentence=0)
    second, address2, _ = close(resolver,a[0],a[6],'bind',(first,),document='fold',sentence=1)
    cid = int(store.refs[store.index_of_row(address2),0])
    # Remove the first occurrence from the binding query, as forgetting does.
    store.refs[store.index_of_row(address),0] = -1
    resolver.revise(resolver.accessible((second,)))
    torch.testing.assert_close(point(resolver.components.owner,cid),a[0]+a[1])
    refold(resolver.components.owner,cid)
    torch.testing.assert_close(point(resolver.components.owner,cid),a[0])


def test_form_region_is_a_concept_and_known_exposure_changes_containment():
    resolver, _, a = fixture()
    form_exposure(resolver,a[4],a[0])
    candidates={1:dict(content=a[0],role=0,position=(0,0)),
                2:dict(content=a[1],role=0,position=(0,0))}
    _, features, _ = resolver.score(a[0]*0,a[0]*0,a[4],0,candidates)
    assert features[:,4].tolist() == [1.,0.]
    assert len(resolver.components.owner._concept_allocator.form_regions) == 1


def test_stm_retains_addresses_and_reads_reinterpretation_without_a_copy():
    from test_sentence_expectation import layer, observe
    from ReferenceContext import SituationAddress
    discourse = layer()
    store = TernaryTruthStore(4,capacity=8)
    discourse._ltm_store = store
    observed=torch.eye(4)[:3]
    observe(discourse,observed)
    row=store.append_meaning(ConceptualMeaning.from_description(observed[0]),order=1)
    discourse.bind_observation_occurrence(0,store.occurrence_of(row))
    held=discourse._inter_context[0][-1]
    assert isinstance(held,SituationAddress) and not any(torch.is_tensor(v) for v in vars(held).values())
    assert discourse._inter_context_ids[0][-1] is None and discourse._inter_context_lanes[0][-1] is None
    store.slots[row,0].copy_(observed[1])
    torch.testing.assert_close(discourse.situation_references()[0].roles[0],observed[1])
    store._index_occurrences.clear()
    assert discourse.situation_references() == ()
    assert discourse.latest() is None


def test_model_time_is_independent_of_source_position_and_covers_the_run_horizon():
    from RowTime import encoder,HORIZON
    store=TernaryTruthStore(4,capacity=8)
    tick=[731]
    object.__setattr__(store,'_model_present',lambda:tick[0])
    row=store.append_idea(torch.ones(4),document_key='first',sentence_index=9000)
    assert int(encoder().decode_index(store.when[row])) == 731
    tick[0]+=1
    other=store.append_idea(torch.ones(4),document_key='second',sentence_index=0)
    assert int(encoder().decode_index(store.when[other])) == 732
    torch.testing.assert_close(store.slots[row],store.slots[other],rtol=0,atol=0)
    trials=torch.tensor([0,731,1000000,HORIZON-1,HORIZON])
    torch.testing.assert_close(encoder().decode_index(encoder().encode(trials)),trials,rtol=0,atol=0)
    tick[0]=HORIZON+1
    with pytest.raises(OverflowError,match='horizon'):
        store.append_idea(torch.ones(4),document_key='third')


def test_old_checkpoint_migrates_identity_parameters_once_and_marks_unknown_row_time():
    from IdentityRows import migrate_checkpoint
    from IndependentComponents import _pack_state,_unpack_state
    from torch import nn
    resolver,store,a=fixture()
    host=nn.Module();host.add_module('components',resolver.components)
    state=host.state_dict();old=torch.arange(5,dtype=torch.float32)
    state['components.reference_close.policy']=old
    state['components.reference_close.cue_policy']=torch.ones(12,5)
    state['components.reference_close._extra_state']=_pack_state(dict(version=2,seen={1:20}))
    state['components.nouns.directions.27']=torch.randn(1024)
    drift,transforms={},{}
    migrate_checkpoint(host,state,host.state_dict(),drift,transforms)
    assert not any('nouns' in k or 'cue_policy' in k for k in state)
    assert _unpack_state(state['components.reference_close._extra_state']) == {'version':3}
    assert state['components.reference_close.policy'].tolist() == [0,0,1,1,0,2,3,4]
    assert resolver.components._migrating_individuals == (27,)
    stable={};migrate_checkpoint(host,state,host.state_dict(),stable,{})
    assert not stable
    store.append_idea(a[0])
    saved=store.state_dict();saved.pop('model_clock')
    target=TernaryTruthStore(12,capacity=512);target.load_state_dict(saved)
    assert not target.when.any()
    torch.testing.assert_close(target.slots,store.slots,rtol=0,atol=0)


def test_reference_closing_reads_surface_meaning_and_never_mints_a_relation_address(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from ClauseJournal import finish_clause
    from ReferenceClose import constituents
    f=SentenceFixture(monkeypatch,native_cube=True)
    entry=f.program(('lower_unary',('surface','marker','object')))
    entry=record_reading(f.language,replace(entry,close_references=True,
        leaf_orders=torch.zeros(2,dtype=torch.long)))
    clause=finish_clause(f.language,entry,registry=f.registry)
    member,=constituents(clause)
    torch.testing.assert_close(member.point,entry.leaves[1],rtol=0,atol=0)
    relation=f.program(('lower_unary',('part','object','kind')))
    relation=record_reading(f.language,replace(relation,close_references=True,
        leaf_orders=torch.zeros(2,dtype=torch.long)))
    closed=finish_clause(f.language,relation,registry=f.registry)
    assert not constituents(closed)
    assert closed.relation == 'part'
