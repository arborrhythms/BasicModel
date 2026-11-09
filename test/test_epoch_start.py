"""Thinking §40. Preserve given knowledge and start a new history at a training boundary."""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from part_fixtures import stored_pair
import torch

from FamiliarTier import promote_identity
from Memory import OCCURRENCE, RELATION, SYMBOL
from Meaning import ConceptualMeaning
from test_presentation_memory import begin, fixture, model_for, write


def test_validation_before_training_does_not_become_epoch_start(tmp_path, monkeypatch, eager_reading):
    import Forgetting
    from data import TheData
    from Models import BaseModel
    from reading_fixtures import force_absolute_reading
    from test_corpus import manifest

    path = tmp_path/'validation.xml'
    config = ET.parse(Path(__file__).resolve().parents[1]/'data/MM_corpus_orders.xml')
    config.find('./architecture/training/packSentences').text = 'true'
    config.find('./architecture/training/numWorkers').text = '0'
    config.write(path)
    TheData.load('curriculum', dat={'corpusManifest':str(manifest(tmp_path,[
        dict(text='alpha.', split='train',stage=0,document='train',sentence=0),
        dict(text='beta.', split='validation',stage=0,document='validation',sentence=0)]))})
    model = BaseModel.from_config(str(path), data=TheData)[0]
    try:
        force_absolute_reading(model)
        store = model.symbolSpace.ltm_store
        model.runEpoch(batchSize=1, split='validation')
        assert len(store) and len(store.parts)
        observed = set(store.address_keys[:len(store)].tolist())
        starts = []
        boundary = Forgetting.model_boundary
        def inspect(target, **options):
            result = boundary(target, **options)
            if options.get('epoch_boundary'):
                starts.append((len(store),len(store.parts),len(model._concept_owner().mtm.entries)))
            return result
        monkeypatch.setattr(Forgetting,'model_boundary',inspect)
        optimizer = model.getOptimizer(lr=.001)
        for _ in range(2):
            model.runEpoch(optimizer=optimizer,batchSize=1,split='train')
            assert observed.isdisjoint(store.address_keys[:len(store)].tolist())
        assert starts == [(0,0,0)]*2
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def _supply_relation(store, point, left, right, *, trust=.9):
    from ClauseRow import Clause, Composition, predicate_identity
    term = Composition(point, (Composition(point, reference=left),
                               Composition(point, reference=right)), operation='implies')
    clause = Clause(ConceptualMeaning.from_description(point), relation='implies',
                    refs=(left, predicate_identity('implies'), right), composition=term)
    with store.clause_assertions(trust=trust, origin=store.ORIGIN_USER, text='supply'):
        return store.write_clause(clause)[1]


def test_new_given_knowledge_retains_its_relation_and_transitive_dependencies():
    resolver,store,points=fixture()
    owner=resolver.components.owner
    bank,model=store.parts,model_for(owner,store)
    symbol=owner.new_concept()
    dependency=write(store,points[0],'historic')
    index=store.location(dependency)
    store.set_evidence(index,0.,0.)
    store.access_count[index]=3
    store.surprise[index]=.4
    begin(model)
    store.set_evidence(index,.6,.2)
    store.access_count[index]+=9
    store.surprise[index]=.8
    child=bank.add((OCCURRENCE,dependency[-1]),(SYMBOL,symbol),truth_plus=.35,truth_minus=.2)
    saved_child={k:v.clone() for k,v in bank.row(child).items()}
    bank.access_count[child]+=8
    bank.surprise[child]=.9
    given=write(store,points[1],'new-given',store.ORIGIN_USER)
    claim=_supply_relation(store,points[1],given[-1],child)
    unrelated=bank.add(symbol,owner.new_concept(),truth_plus=.8)
    assert tuple(map(float,bank.evidence(claim))) == pytest.approx((.9*.35,0.))
    saved_claim={k:v.clone() for k,v in bank.row(claim).items()}
    for _ in range(2):
        begin(model)
        assert bank.contains(claim) and bank.contains(child) and not bank.contains(unrelated)
        assert store.location(given) is not None and store.location(dependency) is not None
        assert bank.find((OCCURRENCE,given[-1]),(RELATION,child)) == claim
        for key in ('refs','ref_kinds','percept'):
            assert torch.equal(bank.row(child)[key],saved_child[key])
        assert float(bank.truth_plus[child]) == float(bank.truth_minus[child]) == 0.
        assert int(bank.access_count[child]) == 1 and float(bank.surprise[child]) == -1.
        # The supply's stored pair survives; its closure-only dependency
        # loses evidence at the boundary, so the effective pair becomes zero.
        assert tuple(map(float,bank.evidence(claim))) == (0.,0.)
        for key,value in saved_claim.items():
            expected = torch.zeros_like(value) if key in ('effective_plus','effective_minus') else value
            assert torch.equal(expected,bank.row(claim)[key]), key
        index=store.location(dependency)
        assert store.row(index)['evidence'] == (0.,0.)
        assert int(store.access_count[index]) == 3 and float(store.surprise[index]) == pytest.approx(.4)
        bank.add((OCCURRENCE,dependency[-1]),(SYMBOL,symbol),truth_plus=.99,truth_minus=.98,surprise=.7)
        bank.add((OCCURRENCE,given[-1]),(RELATION,child),truth_plus=.98,surprise=.8)
        store.access_count[index]+=6
        store.surprise[index]=.9


def test_supplied_outer_keeps_history_relation_as_unverified_dependency():
    from test_clause_storage import clause_store, part_clause
    from ClauseRow import Clause
    store,_=clause_store()
    model=model_for(store._concept_owner(),store)
    begin(model)
    child=store.write_clause(part_clause(),evidence=(.6,.2))[1]
    saved={key:value.clone() for key,value in store.parts.row(child).items()}
    outer=Clause(part_clause().meaning,relation='operator',refs=(4,3,('clause',0)),
                 children=(part_clause(),))
    with store.clause_assertions(trust=.7,origin=store.ORIGIN_USER,text='outer'):
        claim=store.write_clause(outer)
    # The held history still owns its live evidence; only the retained copy
    # enters the next epoch unverified.
    assert store.row(('sym',child))['evidence'] == pytest.approx((.6,.2))
    assert float(store.parts.epoch_start['truth_plus'][child]) == 0.
    assert float(store.parts.epoch_start['truth_minus'][child]) == 0.
    for _ in range(2):
        begin(model)
        assert store.row(('sym',child))['evidence'] == (0.,0.)
        assert stored_pair(store,claim) == pytest.approx((.7,0.))
        assert store.row(claim)['evidence'] == (0.,0.)
        for key in ('refs','ref_kinds','percept'):
            assert torch.equal(store.parts.row(child)[key],saved[key])


def test_retained_new_occurrence_restores_its_creation_observation():
    resolver,store,points=fixture()
    model=model_for(resolver.components.owner,store)
    begin(model)
    dependency=write(store,points[0],'history',surprise=.2)
    index=store.location(dependency)
    store.access_count[index]+=8
    store.surprise[index]=.9
    meaning=replace(ConceptualMeaning.from_description(points[1]),role_refs=(dependency,None,None))
    given=store.append_meaning(meaning,kind='observation',evidence=(.7,.1),document_key='supplied')
    store.set_origin(given,store.ORIGIN_USER, asserted=(float(store.truth_plus[given]), float(store.truth_minus[given])))
    supplied=store.occurrence_of(given)
    for _ in range(2):
        begin(model)
        index=store.location(dependency)
        assert store.row(index)['evidence'] == (0.,0.)
        assert int(store.access_count[index]) == 1
        assert float(store.surprise[index]) == pytest.approx(.2)
        given=store.location(supplied)
        assert store.row(given)['evidence'] == pytest.approx((.7,.1))
        assert int(store.access_count[given]) == 1 and float(store.surprise[given]) == -1.
        store.access_count[index]+=6
        store.surprise[index]=.8
        store.access_count[given]+=5
        store.surprise[given]=.6


def test_history_relation_to_supplied_occurrence_is_not_supplied_authority():
    resolver,store,points=fixture()
    owner=resolver.components.owner
    model=model_for(owner,store)
    begin(model)
    given=write(store,points[0],'supply',store.ORIGIN_USER)
    relation=store.parts.add((OCCURRENCE,given[-1]),(SYMBOL,owner.new_concept()),truth_plus=.8)
    begin(model)
    assert store.location(given) is not None
    assert not store.parts.contains(relation)


@pytest.mark.parametrize('occurrence_root', [False, True])
def test_withdrawn_supply_does_not_return_in_either_memory(occurrence_root):
    from test_clause_storage import clause_store, part_clause
    store,points=clause_store()
    model=model_for(store._concept_owner(),store)
    begin(model)
    supplied = write(store,points[1],'withdrawn',store.ORIGIN_USER) if occurrence_root else None
    relation = (_supply_relation(store,points[1],supplied[-1],2) if supplied is not None
                else _supply_relation(store,points[1],1,2))
    store.clear_origin(store.ORIGIN_USER)
    assert not int(store.parts.epoch_start['ref_kinds'][relation,0])
    for _ in range(2):
        begin(model)
        assert not store.parts.contains(relation)
        assert supplied is None or store.location(supplied) is None


@pytest.mark.parametrize('node_kind', ['companion','identity','request'])
def test_supply_outer_alone_keeps_authority_for_all_closing_nodes(node_kind):
    from test_clause_storage import clause_store, part_clause
    from ClauseRow import Clause
    store,_=clause_store()
    model=model_for(store._concept_owner(),store)
    begin(model)
    child=replace(part_clause(),evidence=(.5,.1),
                  **({'identity':1} if node_kind=='identity' else
                     {'reference_request':{'mode':'mint'}} if node_kind=='request' else {}))
    outer=Clause(child.meaning,relation='operator',refs=(4,3,('clause',0)),
                 children=(child,),companions=(child,) if node_kind=='companion' else ())
    written={}
    with store.clause_assertions(trust=.7,origin=store.ORIGIN_USER,text='outer only'):
        root=store.write_clause(outer,written_rows=written)
    retained=store.occurrence_of(written[id(child)])
    for _ in range(2):
        begin(model)
        assert stored_pair(store,root)==pytest.approx((.7,0.))
        assert store.row(root)['evidence'] == pytest.approx((.7 if node_kind == 'request' else 0., 0.))
        assert store.row(store.location(retained))['evidence']==(0.,0.)


@pytest.mark.parametrize('known_at_start',[False,True])
def test_mid_epoch_withdrawal_does_not_restore_observations(known_at_start):
    resolver,store,points=fixture()
    model=model_for(resolver.components.owner,store)
    if not known_at_start:
        begin(model)
    row=store.append_meaning(ConceptualMeaning.from_description(points[0]),evidence=(.6,.2),
                             document_key='history',surprise=.2)
    if known_at_start:
        begin(model)
    else:
        store._epoch_rows.pop('initial_observations',None)  # pre-item67 checkpoint shape
    store.access_count[row]=8
    store.surprise[row]=.9
    store.withdraw_origin(store.ORIGIN_CONVERSATION)
    assert int(store.access_count[row])==8
    assert float(store.surprise[row])==pytest.approx(.9)
    assert store.row(row)['evidence']==(0.,0.)


def test_shared_document_does_not_turn_history_into_a_supply():
    resolver,store,points=fixture()
    model=model_for(resolver.components.owner,store)
    begin(model)
    history=store.append_meaning(ConceptualMeaning.from_description(points[0]),evidence=(.4,.4),
        document_key='same-document',sentence_index=0,surprise=.2)
    history_ref=store.occurrence_of(history)
    store.access_count[history]=6
    store.surprise[history]=.9
    meaning=replace(ConceptualMeaning.from_description(points[1]),role_refs=(history_ref,None,None))
    given=store.append_meaning(meaning,evidence=(.7,.1),document_key='same-document',sentence_index=1)
    store.set_origin(given,store.ORIGIN_USER,asserted=(.7,.1))
    begin(model)
    row=store.location(history_ref)
    assert store.row(row)['evidence']==(0.,0.)
    assert int(store.access_count[row])==1
    assert float(store.surprise[row])==pytest.approx(.2)


def test_new_mid_epoch_occurrence_is_retained_through_a_relation_endpoint():
    resolver,store,points=fixture()
    owner=resolver.components.owner
    model=model_for(owner,store)
    begin(model)
    history=write(store,points[0],'mid-epoch',surprise=.2)
    row=store.location(history)
    store.access_count[row]=7
    store.surprise[row]=.9
    child=store.parts.add((OCCURRENCE,history[-1]),(SYMBOL,owner.new_concept()),truth_plus=.5)
    given=write(store,points[1],'supply',store.ORIGIN_USER)
    claim=_supply_relation(store,points[1],given[-1],child)
    for _ in range(2):
        begin(model)
        assert store.parts.contains(claim) and store.parts.contains(child)
        assert float(store.parts.truth_plus[child])==0.
        row=store.location(history)
        assert store.row(row)['evidence']==(0.,0.)
        assert int(store.access_count[row])==1
        assert float(store.surprise[row])==pytest.approx(.2)


def test_supply_requires_explicit_asserted_pair():
    from PresentationMemory import supplied_occurrence
    resolver,store,points=fixture()
    row=store.append_meaning(ConceptualMeaning.from_description(points[0]),evidence=(.7,.1))
    with pytest.raises(TypeError,match='asserted'):
        supplied_occurrence(store,row)


def test_forgetting_a_supply_releases_its_two_memory_snapshots():
    from Forgetting import ForgettingConfig, forget
    from ClauseRow import Clause
    from test_clause_storage import clause_store,part_clause
    store,points=clause_store()
    model=model_for(store._concept_owner(),store)
    begin(model)
    claim=store.write_clause(part_clause(),evidence=(.5,.1))[1]
    clause=Clause(ConceptualMeaning.from_description(points[1]),point=points[1],refs=(claim,-1,-1))
    with store.clause_assertions(trust=.7,origin=store.ORIGIN_USER,text='forgotten'):
        given=store.occurrence_of(store.write_clause(clause))
    # An origin outside forgetProtectOrigins is eligible. The first pass
    # releases source text; the second releases its unreferenced occurrence.
    config=ForgettingConfig(high_water=1/store.capacity,low_water=0,min_interval=0,
                            protect_origins=())
    for _ in range(2):
        forget(store,config,document_boundary=True)
    assert store.location(given) is None
    for _ in range(2):
        begin(model)
        assert store.location(given) is None and not store.parts.contains(claim)


@pytest.mark.parametrize('legacy',[False,True])
def test_mid_epoch_checkpoint_load_reuses_instance_without_stale_observations(tmp_path,legacy):
    import copy
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model=_make_model(_SERIAL_CONFIG,stateless=True)
    try:
        store=model.symbolSpace.ltm_store
        point=model._concept_owner().similarity_codebook.W[0].detach().clone()
        begin(model)
        dependency=write(store,point,'checkpoint-history',surprise=.2)
        dep_row=store.location(dependency)
        store.access_count[dep_row]=7
        store.surprise[dep_row]=.9
        meaning=replace(ConceptualMeaning.from_description(point),role_refs=(dependency,None,None))
        given_row=store.append_meaning(meaning,evidence=(1.,0.),document_key='checkpoint-given')
        store.set_origin(given_row,store.ORIGIN_PROVISIONED,asserted=(1.,0.))
        given=store.occurrence_of(given_row)
        user=write(store,point,'checkpoint-user',store.ORIGIN_USER)
        path=tmp_path/'mid-epoch.pt'
        model.save_weights(str(path))
        if legacy:
            saved=torch.load(path,weights_only=False)
            extras=saved['structural_extras']
            # §50 item 102: v3 requires creation observations. This fixture
            # explicitly represents the older v2 envelope and its imports.
            extras['version']=2
            for semantics in extras['truth_semantics'].values():
                semantics.pop('supplies',None)
            for start in extras['epoch_rows'].values():
                if start is None:
                    continue
                if '_base' in start:
                    states={int(address):dict(columns={name:values[i:i+1].clone()
                        for name,values in start['columns'].items()},records=[copy.deepcopy(start['records'][i])],
                        document_sources=dict(start['document_sources']))
                        for i,address in enumerate(start['columns']['address_keys'].tolist())}
                    base=copy.deepcopy(start.pop('_base'))
                    base['supplied_rows']={k:v for k,v in states.items()
                        if k not in base['columns']['address_keys'].tolist()
                        and bytes(v['columns']['document_keys'][0].tolist()).hex() in v['document_sources']}
                    start.clear(); start.update(base)
                start.pop('initial_observations',None)
            for entry in extras['conceptual_spaces'].values():
                state=(entry.get('allocator') or {}).get('epoch_relations',(None,))[0]
                if state is not None:
                    state.pop('_base',None)
                    state.pop('_initial_observations',None)
            torch.save(saved,path)
        # Loading into this used object used to consult its previous epoch's
        # creation map in the tensor post-hook, before restoring the new map.
        store._epoch_rows.pop('initial_observations',None)
        assert model.load_weights(str(path),require_match=True)
        assert store.location(user) is None
        store.access_count[store.location(dependency)]=20
        store.surprise[store.location(dependency)]=.6
        for _ in range(2):
            begin(model)
            assert store.location(user) is None
            assert store.row(store.location(given))['evidence']==(1.,0.)
            row=store.location(dependency)
            assert store.row(row)['evidence']==(0.,0.)
            assert int(store.access_count[row])==(7 if legacy else 1)
            assert float(store.surprise[row])==pytest.approx(.9 if legacy else .2)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('asserted_first',[False,True])
def test_embedded_supply_keeps_only_its_own_assertion_authority(asserted_first):
    from test_clause_storage import clause_store, part_clause
    from ClauseRow import Clause
    store,_=clause_store()
    model=model_for(store._concept_owner(),store)
    begin(model)
    child=part_clause()
    if asserted_first:
        with store.clause_assertions(trust=.6,origin=store.ORIGIN_USER,text='first supply'):
            store.write_clause(child)
    outer=Clause(child.meaning,relation='operator',refs=(4,3,('clause',0)),children=(child,))
    with store.clause_assertions(trust=.7,origin=store.ORIGIN_USER,text='outer supply'):
        claim=store.write_clause(outer)
    for _ in range(2):
        begin(model)
        assert store.row(('sym',store.parts.find(1,2)))['evidence'] == pytest.approx((.6 if asserted_first else 0.,0.))
        assert stored_pair(store,claim) == pytest.approx((.7,0.))
        assert store.row(claim)['evidence'] == pytest.approx((.42 if asserted_first else 0.,0.))


@pytest.mark.parametrize('already_given',[False,True])
def test_resupplying_a_snapshot_occurrence_keeps_its_new_ground(already_given):
    resolver,store,points=fixture()
    owner=resolver.components.owner
    bank,model=store.parts,model_for(owner,store)
    from part_fixtures import symbol as native_symbol
    symbol=native_symbol(store,points[1])
    occurrence=write(store,points[0],'baseline',store.ORIGIN_USER if already_given else None)
    begin(model)
    i=store.location(occurrence)
    store.set_evidence(i,.6,.3)
    store.set_origin(i,store.ORIGIN_USER,text='supplied again', asserted=(float(store.truth_plus[i]), float(store.truth_minus[i])))
    claim=_supply_relation(store,points[0],occurrence[-1],symbol,trust=.8)
    store.set_evidence(i,.99,.98)
    for _ in range(2):
        begin(model)
        i=store.location(occurrence)
        assert store.source_document(i)==store.ORIGIN_USER
        assert store.text_of(i)=='supplied again'
        assert (float(store.truth_plus[i]),float(store.truth_minus[i]))==pytest.approx((1.,.3))
        assert bank.find((OCCURRENCE,occurrence[-1]),(SYMBOL,symbol))==claim, bank.references(claim)
        store.set_evidence(i,.85,.02)
        bank.add((OCCURRENCE,occurrence[-1]),(SYMBOL,symbol),truth_minus=.99)
    assert float(bank.epoch_start['truth_minus'][claim])==0.


def test_provisioned_symbol_relation_keeps_its_own_evidence_across_epochs():
    from test_clause_storage import clause_store, part_clause
    store, _ = clause_store()
    model = model_for(store._concept_owner(),store)
    begin(model)
    with store.clause_assertions(trust=.7,origin=store.ORIGIN_PROVISIONED,text='source'):
        row = store.write_clause(part_clause())
    assert row[0] == 'sym'
    store.parts.add(1,2,truth_minus=.8)
    begin(model)
    assert store.row(row)['evidence'] == pytest.approx((.7,0.))
    with store.clause_assertions(trust=-.2,origin=store.ORIGIN_USER,text='counterexample'):
        assert store.write_clause(part_clause()) == row
    store.parts.add(1,2,truth_plus=.95)
    begin(model)
    assert store.row(row)['evidence'] == pytest.approx((.7,.2))
    store.clear_origin(store.ORIGIN_USER)
    for _ in range(2):
        begin(model)
        assert store.row(row)['evidence'] == pytest.approx((.7,0.))


def test_regroup_restored_snapshot_and_orthogonal_new_row():
    from MemoryConsolidation import recurrence_groups
    resolver,store,points=fixture()
    model=model_for(resolver.components.owner,store)
    a=torch.zeros_like(points[0]);b=a.clone()
    a.reshape(-1)[0]=1.;b.reshape(-1)[1]=1.
    first=write(store,a,'snapshot',store.ORIGIN_USER)
    recurrence_groups(store)
    i=store.location(first)
    assert int(store.recurrence_readings[i])>=0
    begin(model)
    begin(model)
    assert int(store.recurrence_readings[i])==-1 and not store.recurrence_used.any()
    second=write(store,b,'new-document')
    recurrence_groups(store)
    assert store.recurrence_readings[i]!=store.recurrence_readings[store.location(second)]


@pytest.mark.parametrize('reference',['given_role','seat','relate_key','relate_key_and_value'])
def test_withdrawn_relation_handle_stays_reserved_while_named(reference):
    resolver,store,points=fixture()
    owner=resolver.components.owner
    alloc=owner._concept_allocator
    bank,model=store.parts,model_for(owner,store)
    a,b,c=(owner.new_concept() for _ in range(3))
    begin(model)
    handle=bank.add(a,b,truth_plus=.6)
    if reference=='given_role':
        meaning=replace(ConceptualMeaning.from_description(points[0]),role_refs=(('sym',handle),None,None))
        given=store.append_meaning(meaning,kind='observation',evidence=(1.,0.),document_key='supplied')
        store.set_origin(given,store.ORIGIN_USER, asserted=(float(store.truth_plus[given]), float(store.truth_minus[given])))
    elif reference=='seat':
        # An order-zero physical seat can be held without a stored order stamp.
        assert alloc.layer().assign_row(('snap',handle),capacity=owner._order_caps()[0]) is not None
    else:
        alloc.relate_idx['singleton',handle]=handle if reference=='relate_key_and_value' else a
    begin(model)
    assert bank.contains(handle) is (reference == 'given_role')
    if reference == 'given_role':
        assert bank.references(handle) == ((SYMBOL,a),(SYMBOL,b))
        assert float(bank.truth_plus[handle]) == float(bank.truth_minus[handle]) == 0.
    new=bank.add(a,c,truth_plus=1.)
    assert new!=handle
    if reference=='given_role':
        assert store.raw_meaning_of(given).role_refs[0]==('sym',handle)


def test_other_order_active_definition_does_not_hide_remembered_identity():
    from order_fixtures import force_order
    resolver,store,points=fixture()
    owner=resolver.components.owner
    model=model_for(owner,store)
    begin(model)
    word=owner.interpret.lookup_word([23],[],form='label')
    first=write(store,points[0],'first',order=0)
    token=owner.mtm.open(first,kind=None,value=points[0],when=0,reading=word)
    known=promote_identity(owner,token,word=word)
    begin(model)
    other=owner.new_concept()
    force_order(owner,other,2)
    owner._csw_concept_row(2,other)
    assert owner.interpret.define(word,other) is not None
    now=write(store,points[1],'again',order=0)
    token=owner.mtm.open(now,kind=None,value=points[1],when=1,reading=word)
    assert promote_identity(owner,token,word=word)==known
    assert owner.interpret.denotations(word)==tuple(sorted((known,other)))


@pytest.mark.parametrize('frozen',[True,False])
def test_read_of_ambiguous_remembered_name_fails_before_opening_mtm(frozen):
    from order_fixtures import force_order
    resolver,store,points=fixture()
    owner=resolver.components.owner
    model=model_for(owner,store)
    begin(model)
    word=owner.interpret.lookup_word([23],[],form='label')
    names=[owner.new_concept(),owner.new_concept()]
    for cid in names:
        force_order(owner,cid,1)
        owner._csw_concept_row(1,cid)
        assert owner.interpret.define(word,cid) is not None
    begin(model)
    owner._online_learning_frozen=frozen
    assert not owner.mtm.entries
    with pytest.raises(ValueError) as exc:
        owner.interpret.reading(word)
    assert 'label' in str(exc.value) and str(names) in str(exc.value)
    assert not owner.mtm.entries


@pytest.mark.parametrize('missing',['epoch_rows','store_start','epoch_relations','epoch_mtm'])
def test_checkpoint_that_loses_epoch_start_rejected(missing,tmp_path):
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model=_make_model(_SERIAL_CONFIG)
    try:
        begin(model)
        extras=model._collect_structural_extras()
        if missing=='epoch_rows':
            del extras[missing]
        elif missing=='store_start':
            del extras['epoch_rows'][next(iter(extras['epoch_rows']))]
        else:
            entry=next(e for e in extras['conceptual_spaces'].values() if 'epoch_mtm' in e)
            del (entry['allocator'] if missing=='epoch_relations' else entry)[missing]
        with pytest.raises(ValueError,match='epoch'):
            model._restore_structural_extras(extras)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_unstarted_checkpoint_preserves_setup_until_its_first_epoch_boundary(tmp_path):
    import copy
    from test_epoch_history import same_state
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG

    model = _make_model(_SERIAL_CONFIG)
    restored = None
    try:
        owner, store = model._concept_owner(), model.symbolSpace.ltm_store
        point = owner.similarity_codebook.W[0].detach().clone()
        symbol = owner.new_concept()
        setup = write(store, point, 'intentional-setup')
        relation = store.parts.add((OCCURRENCE, setup[-1]), (SYMBOL, symbol),
                                   truth_plus=.7, truth_minus=.2)
        assert tuple(map(float,store.parts.evidence(relation))) == pytest.approx((.7,.2))
        owner.mtm.open(setup, kind=None, value=point, when=0)
        extras = model._collect_structural_extras()
        assert extras['epoch_rows'] == {}
        allocators = [entry for entry in extras['conceptual_spaces'].values()
                      if entry.get('allocator')]
        assert allocators
        assert all(entry['allocator']['epoch_relations'][0] is None
                   and 'epoch_mtm' not in entry for entry in allocators)
        history = copy.deepcopy((store.state_dict(), store.semantic_extras(),
                                 store.parts.state_dict(), owner.mtm.state()))

        checkpoint = tmp_path / 'unstarted.ckpt'
        model.save_weights(str(checkpoint))
        restored = _make_model(_SERIAL_CONFIG)
        assert restored.load_weights(str(checkpoint), require_match=True)
        owner, store = restored._concept_owner(), restored.symbolSpace.ltm_store
        assert getattr(store, '_epoch_rows', None) is None
        assert store.parts.epoch_start is None and not hasattr(owner, '_epoch_mtm')
        same_state(history, (store.state_dict(), store.semantic_extras(),
                             store.parts.state_dict(), owner.mtm.state()))

        begin(restored)
        assert store._epoch_rows is not None and store.parts.epoch_start is not None
        assert hasattr(owner, '_epoch_mtm')
        same_state(history, (store.state_dict(), store.semantic_extras(),
                             store.parts.state_dict(), owner.mtm.state()))
        experienced = write(store, point, 'first-epoch')
        during_epoch = store.parts.add((OCCURRENCE, experienced[-1]), (SYMBOL, symbol),
                                      truth_plus=1.)
        owner.mtm.open(experienced, kind=None, value=point, when=1)
        begin(restored)
        assert store.location(experienced) is None and not store.parts.contains(during_epoch)
        assert store.location(setup) is not None and store.parts.contains(relation)
        same_state(history, (store.state_dict(), store.semantic_extras(),
                             store.parts.state_dict(), owner.mtm.state()))
    finally:
        for current in (model, restored):
            if current is not None:
                current.End()
                current.symbolSpace.soft_reset()
