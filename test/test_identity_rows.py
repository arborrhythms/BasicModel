"""Step 1b: regions have one native owner; indexes cannot retain stale values."""
from dataclasses import replace
import torch
import pytest

from IdentityRows import meet_ratios, point, witness_count
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from test_reference_close import fixture, close, form_exposure


def native_close(resolver, content, predicate, *, document, sentence):
    """Supply the explicit native individual these bank certificates examine."""
    from types import SimpleNamespace
    from IdentityRows import mint
    identity = mint(resolver.components.owner, content)
    assert identity is not None
    frame = SimpleNamespace(symbol_ids=(identity,), row_id=-1, activations=())
    return close(resolver, content, predicate, 'bind', (frame,),
                 document=document, sentence=sentence)


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


def test_content_meet_cannot_read_form_while_form_can_identify_its_region():
    from types import SimpleNamespace
    resolver, _, atoms = fixture()
    owner = resolver.components.owner
    # The same paired layout as the production owner: form, band, meaning.
    object.__setattr__(owner.similarity_codebook, 'mereology',
                       SimpleNamespace(percept_width=4, percept_event_width=6))
    q = atoms[0] + atoms[6]
    candidates = {1: dict(content=atoms[0]+atoms[7], role=0, position=(0,0)),
                  2: dict(content=atoms[1]+atoms[6], role=0, position=(0,0))}
    _, phi, logits = resolver.score(q, q*0, q*0, 0, candidates)
    assert phi[:, :2].tolist() == [[0., 0.], [1., 1.]]
    assert logits[1] > logits[0]
    changed_form = atoms[1] + atoms[6]
    _, changed, _ = resolver.score(changed_form, q*0, q*0, 0, candidates)
    torch.testing.assert_close(changed[:, :2], phi[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(resolver.cue(q), q, rtol=0, atol=0)
    torch.testing.assert_close(resolver.coordinates(q), atoms[6], rtol=0, atol=0)
    # A narrowed value keeps its address and never borrows form as meaning.
    assert not resolver.coordinates(q[:4]).any()


def test_individual_axes_in_the_change_residual_exclude_form_and_its_gradient():
    from types import SimpleNamespace
    resolver, _, atoms = fixture()
    native_close(resolver, atoms[0]+atoms[6], atoms[7],
          document='separate-metrics', sentence=0)
    owner = resolver.components.owner
    object.__setattr__(owner.similarity_codebook, 'mereology',
                       SimpleNamespace(percept_width=4, percept_event_width=6))
    observation = (.25*atoms[6]).expand(3, -1).clone().requires_grad_()
    prediction = torch.zeros_like(observation)
    residual = resolver.components.innovation(observation, prediction)
    changed_form = resolver.components.innovation(observation+.5*atoms[1], prediction)
    torch.testing.assert_close(changed_form, residual, rtol=0, atol=0)
    assert not torch.equal(
        resolver.components.innovation(observation+.5*atoms[6], prediction), residual)
    residual.sum().backward()
    assert not observation.grad[:, :6].any()
    assert observation.grad[:, 6].abs().sum() > 0


def test_native_row_is_the_only_content_and_witnesses_are_live_queries():
    resolver, store, a = fixture()
    frame, address, _ = native_close(resolver,a[0],a[6],document='one',sentence=0)
    cid = resolver.last_audit[0]['reference']
    assert witness_count(resolver.components.owner,cid) == 1
    owner = resolver.components.owner
    assert not hasattr(resolver.components,'nouns')
    assert all(not hasattr(resolver,k) for k in ('seen','cooccurrence','associations','predicates','cue_policy'))
    # Identity reads do not invoke an event chart, including after a row edit.
    resolver.components._ready = lambda: (_ for _ in ()).throw(AssertionError('identity chart'))
    with torch.no_grad(): owner.similarity_codebook.W[owner._csw_row_of(cid)].copy_(a[1])
    candidates = resolver.individuals(resolver.accessible((frame,)))
    torch.testing.assert_close(point(owner,cid),a[1],rtol=0,atol=0)
    torch.testing.assert_close(candidates[cid]['content'],a[0]+a[1],rtol=0,atol=0)
    store.parts.remove_symbol(cid)
    assert witness_count(owner,cid) == 0


def test_forgetting_and_reinterpretation_never_refold_a_native_identity():
    resolver, store, a = fixture()
    first, address, _ = native_close(resolver,a[0]+a[1],a[6],document='fold',sentence=0)
    cid = resolver.last_audit[0]['reference']
    second, address2, _ = close(resolver,a[0],a[6],'bind',(first,),document='fold',sentence=1)
    # Read only the surviving occurrence, as after forgetting. Its working
    # interpretation must leave the original native code intact.
    original = store.slots.clone(), store.refs.clone()
    resolver.individuals(resolver.accessible((second,)))
    torch.testing.assert_close(point(resolver.components.owner,cid),a[0]+a[1])
    torch.testing.assert_close(store.slots,original[0],rtol=0,atol=0)
    torch.testing.assert_close(store.refs,original[1],rtol=0,atol=0)


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
    drift,transforms={},{}
    migrate_checkpoint(host,state,host.state_dict(),drift,transforms)
    assert not any('nouns' in k or 'cue_policy' in k for k in state)
    assert _unpack_state(state['components.reference_close._extra_state']) == {'version':3}
    assert state['components.reference_close.policy'].tolist() == [0,0,1,1,0,2,3,4]
    stable={};migrate_checkpoint(host,state,host.state_dict(),stable,{})
    assert not stable
    store.append_idea(a[0])
    saved=store.state_dict();saved.pop('model_clock')
    target=TernaryTruthStore(12,capacity=512);target.load_state_dict(saved)
    assert not target.when.any()
    torch.testing.assert_close(target.slots,store.slots,rtol=0,atol=0)


@pytest.mark.parametrize('native_bank', (False, True))
def test_loader_permanently_rejects_signed_identity_without_modifying_the_row(native_bank):
    resolver, store, atoms = fixture()
    frame, address, _ = native_close(resolver, atoms[0], atoms[6],
                          document='strict-load', sentence=0)
    row = store.index_of_row(address)
    identity = resolver.last_audit[0]['reference']
    owner = resolver.components.owner
    if native_bank:
        with torch.no_grad():
            owner.similarity_codebook.W[owner._csw_row_of(identity),0] = -.125
    else:
        store.slots[row, 0, 0] = -.125
    held = store.slots.clone()
    native = point(resolver.components.owner, identity).clone()
    with pytest.raises(ValueError, match='native concept coordinates'):
        resolver.components.migrate_identity_rows()
    torch.testing.assert_close(store.slots, held, rtol=0, atol=0)
    torch.testing.assert_close(point(resolver.components.owner, identity), native, rtol=0, atol=0)


@pytest.mark.parametrize('value', (-.125, .5))
def test_retired_chart_checkpoint_is_rejected_before_keys_are_removed(value):
    from IdentityRows import migrate_checkpoint
    from torch import nn
    resolver, _, _ = fixture()
    host=nn.Module();host.add_module('components',resolver.components)
    state=host.state_dict()
    state['components.nouns.directions.27']=torch.full((12,),value)
    keys, drift, transforms = set(state), {}, {}
    with pytest.raises(ValueError, match='retired signed-chart'):
        migrate_checkpoint(host,state,host.state_dict(),drift,transforms)
    assert set(state) == keys and not drift and not transforms


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
