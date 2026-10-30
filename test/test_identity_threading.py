"""Fixed-operation identity certificates; no claim about learned parsing."""
from chunk_fixtures import context as chunk_context
from types import SimpleNamespace

import pytest
import torch

from MediumTermMemory import MediumTermMemory
import STMChunks as chunks
from chunk_fixtures import Walk, choice, value as working_value
from Spaces import LanguageOperationChoice


def address(number):
    return 'ltm', 'address-v1', number


def operations():
    return Walk((SimpleNamespace(method_name='bind',head_role=1),
                 SimpleNamespace(method_name='lift',clause_form='S',head_role=2)), ())



def select_bind(walk,state,mention,identity,column):
    points,ids,indices,valid = chunks.bind_menu(state[0],mention[:,None])
    selected = (ids[0,0] == identity) & valid[0,0]
    at = int(selected.long().argmax())
    assert selected.any()
    operand = points[:,0,at]
    result = torch.where((ids[:,0,at] == 0)[:,None],mention,operand)
    chosen = choice(result,local=0,identity=identity,chunk=int(indices[0,0,at]),bind=True)
    return walk.apply(state,chosen,torch.stack((mention,operand),1),column)



def push(walk,state,value,*,word,index,order=1,resolved=True):
    return walk.shift(state,value,word=word,index=index,order=order,resolved=resolved)



def start(value):
    walk = operations()
    state = walk.empty(value,16)
    state = push(walk,state,value,word=101,index=0)
    return walk,select_bind(walk,state,value,0,0)



def event(walk,state,value,*,index,column):
    state = push(walk,state,torch.ones_like(value)*.2,word=202,index=index,order=0)
    state_stm,live = chunks.refresh(state[0],chunks.stack_indices(state[0])[:,:2])
    state = (state_stm,*state[1:])
    return walk.apply(state,choice(value,local=1,head=2),torch.stack((live[:,1],live[:,0]),1),column)



@pytest.mark.parametrize('compiled', (False, True))
def test_second_verb_receives_first_result_through_bind_with_derivative(compiled):
    first = torch.tensor([[.2,.7,.3]],requires_grad=True)
    scale = torch.tensor(.6,requires_grad=True)
    walk,state = start(first)
    if compiled:
        walk.apply = torch.compile(walk.apply,backend='eager',fullgraph=True)
    changed = first*scale
    state = event(walk,state,changed,index=1,column=1)
    mention = torch.zeros_like(first)
    state = push(walk,state,mention,word=303,index=2)
    points,ids,indices,available = chunks.bind_menu(state[0],mention[:,None])
    selected = available & (ids == 512)
    torch.testing.assert_close(points[selected],changed)
    state = select_bind(walk,state,mention,512,2)
    state_stm,live = chunks.refresh(state[0],chunks.at(state[0])[:,None])
    state = state_stm,*state[1:]
    second_scale = torch.tensor(.4,requires_grad=True)
    state = event(walk,state,live[:,0]*second_scale,index=3,column=3)
    held = working_value(state,512)
    torch.testing.assert_close(held,first*scale*second_scale)
    held.sum().backward()
    torch.testing.assert_close(first.grad,torch.full_like(first,.24))
    torch.testing.assert_close(scale.grad,(first.detach()*second_scale.detach()).sum())
    assert second_scale.grad > 0



def test_latest_verb_replaces_state_and_two_aliases_share_description_join():
    walk,state = start(torch.tensor([[.1,.1,.1]]))
    state = event(walk,state,torch.tensor([[.8,.1,.1]]),index=1,column=1)
    state = push(walk,state,torch.zeros(1,3),word=303,index=2)
    state = select_bind(walk,state,torch.zeros(1,3),512,2)
    state = event(walk,state,torch.tensor([[.1,.8,.1]]),index=3,column=3)
    torch.testing.assert_close(working_value(state,512),torch.tensor([[.1,.8,.1]]))
    description = torch.tensor([[.1,.1,.9]])
    state = push(walk,state,description,word=404,index=4)
    snapshot = working_value(state,512).clone()
    state = select_bind(walk,state,description,512,4)
    torch.testing.assert_close(snapshot,torch.tensor([[.1,.8,.1]]))
    _,live = chunks.refresh(state[0],chunks.stack_indices(state[0])[:,:2])
    torch.testing.assert_close(live,torch.tensor([[[.1,.8,.9],[.1,.8,.9]]]))



@pytest.mark.parametrize('keep_second', (False, True))
def test_trial_values_are_isolated_and_only_selected_values_commit(keep_second):
    from SentenceCompose import select_rows
    walk,original = start(torch.tensor([[.2,.3]]))
    first = event(walk,original,torch.tensor([[.8,.1]]),index=1,column=1)
    second = event(walk,original,torch.tensor([[.1,.9]]),index=1,column=1)
    kept = select_rows(first,second,torch.tensor([keep_second]))
    torch.testing.assert_close(working_value(original,512),torch.tensor([[.2,.3]]))
    pool = MediumTermMemory()
    pool.open(address(1),kind=None,when=1,value=working_value(original,512)[0])
    pool.update(address(1),working_value(kept,512)[0],when=2)
    torch.testing.assert_close(pool.value_of(address(1)),working_value(second if keep_second else first,512)[0])
    assert not pool.value_of(address(1)).requires_grad



def test_names_events_checkpoint_and_reclamation_have_separate_ownership():
    pool = MediumTermMemory(2)
    x, alias, event_row = address(1), address(2), address(3)
    pool.open(x, kind=None, when=1, reading=101, value=torch.tensor([.5, .1]))
    pool.bind(x, alias, kind=None, when=2, reading=102, value=torch.tensor([.1, .7]))
    pool.event(x, event_row, verb=201, role=0, when=3)
    pool.event(x, event_row, verb=201, role=0, when=4)
    pool.update(x, torch.tensor([.2, .9]), when=4)
    assert pool.entries[x].count == 2 and pool.entries[x].occurrences == (x, alias)
    assert len(pool.entries[x].events) == 1
    assert pool.reading_of(x) == 101 and pool.reading_of(alias) == 102
    restored = MediumTermMemory(2)
    restored.restore(pool.state())
    assert restored.names == pool.names and restored.entries[x].events == pool.entries[x].events
    torch.testing.assert_close(restored.value_of(alias), torch.tensor([.2, .9]))
    restored.forget(event_row)
    assert restored.token_of(alias) == x and not restored.entries[x].events
    restored.forget(x)
    assert not restored.entries and not restored.names


def test_frame_access_does_not_leak_a_native_individuals_other_stream_value():
    from dataclasses import replace
    from IdentityRows import mint
    from ReferenceContext import SituationFrame
    from test_medium_term_memory import keep,mention
    from test_reference_close import fixture
    resolver,store,atoms = fixture()
    pool = resolver.components.owner.mtm
    native = mint(resolver.components.owner,atoms[0])
    # Independently opened streams own distinct working tokens even when
    # both recall the same promoted native identity.
    from Meaning import ConceptualMeaning
    rows = [store.append_meaning(ConceptualMeaning.from_description(atoms[0]),
        stream=index,sentence_index=index,document_key='streams',order=1) for index in (0,1)]
    for index,row in enumerate(rows):
        pool.open(store.occurrence_of(row),kind=native,native=native,when=index,value=atoms[0])
    first,second = map(store.occurrence_of,rows)
    assert first != second and pool.token_of(first) != pool.token_of(second)
    pool.update(first,atoms[1],when=20)
    for row,expected in zip(rows,(atoms[1],atoms[0])):
        frame = replace(SituationFrame.from_row(store.row(row)),symbol_ids=(native,),
                        activations=((store.occurrence_of(row),native,0),))
        state = chunk_context(resolver,(frame,))
        records = resolver.accessible(state)
        points = torch.stack([record['value'] for _,record in records.values()])
        torch.testing.assert_close(points,expected.expand_as(points),rtol=0,atol=0)
    assert not resolver.individuals(resolver.accessible(chunk_context(resolver)))



@pytest.mark.parametrize('compiled', (False, True))
def test_compose_bind_locations_share_the_operation_chooser_and_zero_dummy(compiled):
    from Language import BindLayer,OperationSelectionLayer
    from ReferenceContext import prepare_operands
    rule=SimpleNamespace(method_name='bind',head_role=1)
    layer=OperationSelectionLayer(d_model=2,ops=[BindLayer()],unary_ops=[])
    with torch.no_grad():
        layer.reduce_anchor.fill_(1)
        layer.stop_anchor.fill_(-100)
    mention=torch.tensor([[[.1,.2],[0.,0.]]],requires_grad=True)
    antecedent=torch.tensor([[[.8,.7],[.3,.4]]],requires_grad=True)
    bank=SimpleNamespace()
    def run(mention,antecedent):
        state=chunks.empty(1,8,2)
        for index in range(2):
            state,_=chunks.deposit(state,antecedent[:,index],torch.tensor([True]),identity=100+index)
        ids=torch.full((1,2),-1,dtype=torch.long)
        active=torch.tensor([[True,False]])
        data=prepare_operands(mention,ids,ids*0,ids*0,orders=ids+2,
            rules=(rule,),unary_rules=(),bank=bank,live=None,active=active,chunk_state=state)
        _,path,route=layer(mention,reference_data=data,expectation_prime=torch.ones(1,2),
                           depth=torch.ones(1,dtype=torch.long))
        return path,route['identity'],route['logits'],route['probabilities'],route['operands'],data['binary_valid']
    if compiled:
        run=torch.compile(run,backend='aot_eager',fullgraph=True)
    path,identity,logits,probabilities,operands,valid=run(mention,antecedent)
    assert identity.tolist()==[100] and logits[0,0]==0 and int(valid.sum())==3
    torch.testing.assert_close(path[:,:1],antecedent[:,:1])
    torch.testing.assert_close(operands[:,0],mention[:,0])
    torch.testing.assert_close(operands[:,1],antecedent[:,0])
    torch.testing.assert_close(probabilities,logits.softmax(-1))
    path.sum().backward(retain_graph=True)
    torch.testing.assert_close(antecedent.grad,torch.tensor([[[1.,1.],[0.,0.]]]))
    assert layer.reduce_anchor.grad is None or torch.count_nonzero(layer.reduce_anchor.grad)==0
    (-probabilities[0,1].log()).backward()
    assert layer.reduce_anchor.grad is not None and layer.reduce_anchor.grad.abs().sum()>0
    with torch.no_grad():
        layer.reduce_anchor.fill_(-10)
    _,identity,logits,_,_,_=run(mention,antecedent)
    assert identity.tolist()==[0] and logits[0,0]==0 and (logits[0,1:]<0).all()



def test_compose_menu_uses_the_resolvers_observed_exclusions():
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    from test_reference_close import fixture
    resolver,store,atoms = fixture()
    for index in (0,1):
        row=store.write_clause(Clause(ConceptualMeaning.from_description(atoms[index]),
            point=atoms[index],order=1),document_key='supports',sentence_index=index)
        store.witness_count[row]=8
    exclusions=torch.tensor(resolver.exclusion_pairs()).reshape(-1,2)
    state=chunks.empty(1,8,12)
    state,_=chunks.deposit(state,atoms[:1],torch.tensor([True]),identity=101)
    for own in (atoms[1],atoms[0],atoms[0]+atoms[1],atoms[1]*0):
        *_,available=chunks.bind_menu(state,own.reshape(1,1,-1),exclusions=exclusions)
        assert available[0,0,0]
        assert bool(available[0,0,1]) == (not resolver.exclusion_query()(own,atoms[0]))



def recorded_identity_reading(f, tree, *, dummy_binds=()):
    """Fixed grammar operations through the production shared chunks/journal."""
    from chunk_fixtures import record
    from dataclasses import replace
    entry, inputs, handle = record(f,tree,dummy_binds=dummy_binds)
    return replace(entry, leaf_evidence=torch.tensor([[1., 0.]]).expand(len(entry.leaves), 2)), inputs, handle



def test_kept_journal_commits_two_verbs_one_identity_and_immutable_occurrences(monkeypatch):
    from ClauseJournal import finish_clause
    from test_clause_acceptance import SentenceFixture
    from test_medium_term_memory import snapshot, unchanged
    f = SentenceFixture(monkeypatch)
    tree = ('sum',
        ('lift', ('lower_unary', 'subject'), ('verb', 'move', ('lower', 'd', 'place'))),
        ('lift', ('bind', 'alias'), ('verb', 'affect', ('lower', 'd', 'target'))))
    entry, inputs, handle = recorded_identity_reading(f, tree)
    lifts = [record for record in inputs if record[0] == 'lift']
    # The alias is a description mention and joins before the next VP reads it.
    alias = entry.leaves[entry.word_ids.tolist().index(f.words['alias'][0])]
    torch.testing.assert_close(lifts[1][1][:, 0], torch.maximum(lifts[0][2], alias))
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    prepared = resolver.prepare(clause, plan)
    assert not f.cs.mtm.entries and not plan['decisions']
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='identity', sentence_index=0)
    retained = snapshot(resolver)
    resolver.commit(plan, row, written, document='identity', position=0, training=False)
    unchanged(resolver, retained)
    selected = [item for item in plan['items'] if item['written_node'].identity['handle'] == handle]
    names = [written[('occurrence', id(item['written_node']))] for item in selected]
    assert len(names) == 2
    token = f.cs.mtm.token_of(names[0])
    assert f.cs.mtm.token_of(names[1]) == token
    assert f.cs.mtm.entries[token].count == 2
    assert len(f.cs.mtm.entries[token].events) == 2
    torch.testing.assert_close(f.cs.mtm.value_of(token), lifts[1][2][0])
    f.cs.mtm.update(token, torch.zeros(8), when=20)
    unchanged(resolver, retained)


@pytest.mark.parametrize('corruption', ('membership', 'value', 'event', 'reading'))
def test_invalid_identity_checkpoint_does_not_replace_the_pool(corruption):
    pool = MediumTermMemory()
    pool.open(address(1), kind=None, when=1, value=torch.ones(2))
    before = pool.state()
    bad = pool.state()
    if corruption == 'membership':
        bad['entries'][0]['occurrences'] = (address(2),)
    elif corruption == 'value':
        bad['entries'][0]['timelines'][0][1][0] = float('nan')
    elif corruption == 'event':
        bad['entries'][0]['timelines'] = ((((address(3), 5, 1, 3, None, None, False, None, (), False),),
                                           bad['entries'][0]['timelines'][0][1]),)
    else:
        bad['names'] = ((address(1), address(1), 'bad'),)
    with pytest.raises(ValueError):
        pool.restore(bad)
    assert pool.state()['names'] == before['names']
    torch.testing.assert_close(pool.value_of(address(1)), before['entries'][0]['timelines'][0][1])


def test_binding_an_unresolved_object_does_not_complete_its_verb():
    from ChunkJournal import IDENTITY,RESOLVED
    walk,state=start(torch.tensor([[.6,.2]]))
    state=push(walk,state,torch.tensor([[.1,.1]]),word=303,index=2,resolved=False)
    unknown=int(state[1][0,0,IDENTITY])
    state=select_bind(walk,state,torch.tensor([[.1,.1]]),0,2)
    state=push(walk,state,torch.zeros(1,2),word=404,index=3,resolved=False)
    state=select_bind(walk,state,torch.zeros(1,2),unknown,3)
    assert state[1][0,0,RESOLVED] == 0



@pytest.mark.parametrize('readmit', (False, True))
@pytest.mark.parametrize('wrapper', ('sum', 'lower'))
def test_later_bind_keeps_its_earlier_determination_under_a_wrapper(monkeypatch, readmit, wrapper):
    from ClauseJournal import finish_clause
    from test_clause_acceptance import SentenceFixture
    from test_medium_term_memory import snapshot, unchanged
    f = SentenceFixture(monkeypatch)
    entry, _, handle = recorded_identity_reading(f,
        ('bind', (wrapper, ('lower_unary', 'subject'), 'tail')))
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    assert len(plan['items']) == 2
    introduction, use = plan['items']
    assert introduction['mode'] == 'mint'
    assert introduction['node'].identity['handle'] == use['node'].identity['handle'] == handle
    assert use['selected'] == introduction['selected'] == ('new', 0)
    if readmit:
        clause = finish_clause(f.language, entry, registry=f.registry)
        sites = resolver.sites(clause, pending_names=plan['pending_names'])
        plan = dict(plan, items=[dict(item, **site) for item, site in zip(plan['items'], sites)])
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='wrapped', sentence_index=0)
    retained = snapshot(resolver)
    resolver.commit(plan, row, written, document='wrapped', position=0, training=False)
    first, second = [written[('occurrence', id(item['written_node']))] for item in plan['items']]
    assert f.cs.mtm.token_of(second) == first
    assert f.cs.mtm.entries[first].occurrences == (first, second)
    unchanged(resolver, retained)


def test_three_slot_close_keeps_the_introduction_used_by_its_embedded_bind(monkeypatch):
    """Captured final curriculum: a three-slot root orphaned a determined subject."""
    from dataclasses import replace
    from ClauseJournal import finish_clause
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    entry, _, handle = recorded_identity_reading(f, ('sum', ('lower_unary', 'subject'),
        ('sum', 'predicate', ('implies', ('bind', 'alias'), 'tail'))))
    # Stop the fixed operation journal before its final two sums. The three
    # surviving roots are subject, predicate, and an embedded relative clause.
    action_fields = ('actions', 'operation_values', 'operation_refs', 'operation_relations',
        'operation_probabilities', 'identity_actions', 'timeline_actions', 'case_values',
        'case_selected', 'operand_counts', 'form_wholes', 'form_whole_counts')
    entry = replace(entry, **{name: getattr(entry, name)[:-2] for name in action_fields
                              if getattr(entry, name) is not None})
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    introduction = next(item for item in plan['items']
        if item['node'].identity['handle'] == handle and item['node'].identity['introduction'])
    use = next(item for item in plan['items']
        if item['node'].reference_request.get('compose_bind') == handle)
    assert use['selected'] == introduction['selected']
    assert clause.refs[0][0] == 'clause'
    assert clause.children[clause.refs[0][1]] is introduction['node']


def test_dummy_bind_preserves_a_carried_object_with_different_working_content(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from PartClosing import PartClosing
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    reference = f.noun('carrier')
    value = torch.ones(8) * .25
    leaf = Composition(value, reference=reference, order=0)
    bound = Composition(value, children=(leaf,), operation='bind', head_role=1, order=0)
    closing = PartClosing(f.cs, f.store)
    assert not torch.equal(closing._point(reference), value)
    before = int(closing.alloc.next_id), len(closing.parts)
    record = closing._plan_term(bound, True)
    assert record['reference'] == (closing.parts.SYMBOL, reference)
    assert record['resolved']
    assert (int(closing.alloc.next_id), len(closing.parts)) == before


def test_dummy_binds_preserve_the_relative_operand_in_a_three_slot_close(monkeypatch):
    """Captured whole + repeated dummy binds must not become three plain points."""
    from dataclasses import replace
    from types import SimpleNamespace
    from ClauseJournal import finish_clause
    from ClauseScope import ClauseScope
    from ReferenceContext import prepare_operands
    from test_clause_acceptance import SentenceFixture
    import STMChunks
    f = SentenceFixture(monkeypatch)
    tree = ('sum', 'prefix', ('sum', 'predicate',
        ('bind', ('bind', ('whole', 'subject', 'tail')))))
    entry, _, _ = recorded_identity_reading(f, tree, dummy_binds=(0, 1))
    action_fields = ('actions', 'operation_values', 'operation_refs', 'operation_relations',
        'operation_probabilities', 'identity_actions', 'timeline_actions', 'case_values',
        'case_selected', 'operand_counts', 'form_wholes', 'form_whole_counts')
    entry = replace(entry, **{name:getattr(entry, name)[:-2] for name in action_fields
                             if getattr(entry, name) is not None})
    relations = torch.zeros(len(entry.actions), 2, dtype=torch.bool)
    rule = next(rule for rule in f.binary if rule.method_name == 'bind')
    field = STMChunks.empty(1, 8, 8, device=entry.leaves.device, dtype=entry.leaves.dtype)
    for index, (kind, operation, _) in enumerate(entry.actions.tolist()):
        if kind != 1 or f.binary[operation].method_name != 'bind':
            continue
        left = entry.operation_values[index, 0]
        menu = prepare_operands(torch.stack((left*0, left))[None],
            torch.tensor([[0, -2]]), torch.tensor([[0, ClauseScope.RELATIVE | ClauseScope.LOCAL]]),
            torch.tensor([[0, 1]]), orders=torch.zeros(1, 2, dtype=torch.long),
            rules=(rule,), unary_rules=(), bank=SimpleNamespace(interpreter=None,cases=None),
            live=(), active=torch.ones(1, 2, dtype=torch.bool), chunk_state=field)
        relations[index] = menu['binary_relations'][0, 0, 0]
    clause = finish_clause(f.language, replace(entry, operation_relations=relations), registry=f.registry)
    assert clause.relation == 'operator'
    assert clause.refs[2][0] == 'clause'
    child = clause.children[clause.refs[2][1]]
    assert child.relation == 'part' and child.point is None
    assert child.reference_request is None
    # The close still rejects three point roots with no relative grammar.
    from reading_fixtures import record_reading
    plain = record_reading(f.language, f.program(('sum', 'prefix', ('sum', 'predicate', 'tail'))))
    plain = replace(plain, actions=plain.actions[:-2], operation_values=plain.operation_values[:-2],
                    operation_refs=plain.operation_refs[:-2])
    with pytest.raises(ValueError, match='three slots have no selected relative clause'):
        finish_clause(f.language, plain, registry=f.registry)


@pytest.mark.parametrize('open_sibling', (False, True))
def test_source_authority_closes_a_nested_region_without_hiding_a_free_role(open_sibling):
    from dataclasses import replace
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    from ThoughtClosing import closing_question
    from ThoughtReferences import has_open_slots, open_slots, with_slots
    value = replace(ConceptualMeaning.from_description(torch.ones(4)),
                    role_refs=(('sym', 1), None, None))
    child = Clause(replace(value), point=value.roles[0],
                   refs=(1, -1, -1), evidence=(0., 0.))
    children = (child,)
    if open_sibling:
        pending = with_slots(replace(value, role_refs=(None, None, None)),
                             (('referent', 0),), pair=(0., 0.))
        children += (Clause(pending, point=pending.roles[0], evidence=(0., 0.)),)
    parent = Clause(value, point=value.roles[0], refs=(1, -1, -1),
                    children=children, evidence=(1., 0.))
    query = closing_question(value, parent)
    if open_sibling:
        assert query is not None and has_open_slots(query)
        assert open_slots(query) == (('referent', 0),)
    else:
        assert query is not None and not open_slots(query)


@pytest.mark.parametrize('role', (0, 2))
def test_compose_bind_pending_child_keeps_the_direct_question_role(monkeypatch, role):
    from dataclasses import replace
    from ClauseRow import Clause, ClausePredicate, predicate_point
    from Meaning import ConceptualMeaning
    from ThoughtClosing import closing_question
    from ThoughtReferences import bindings, open_slots
    from test_reference_arrivals import world, named, force_named
    resolver, store, atoms, words, _ = world()
    force_named(resolver, monkeypatch, None)
    child = named(atoms[0], words['x'])
    child = replace(child, reference_request=dict(child.reference_request, compose_bind=0))
    pred = predicate_point('equal', atoms[0])
    value = ConceptualMeaning(torch.stack((atoms[0], pred, atoms[0])),
        torch.ones(3, dtype=torch.bool),  bindings={'_equality': True})
    refs = [words['three'], ClausePredicate(pred, 'equal'), words['three']]
    refs[role] = ('clause', 0)
    parent = Clause(value, relation='equal', refs=tuple(refs), children=(child,), evidence=(0., 0.))
    plan = resolver.preview(parent, chunk_context(resolver, ()))
    prepared = resolver.prepare(parent, plan)
    written = {}
    row = store.write_clause(prepared, document_key='direct-question', written_rows=written)
    resolver.commit(plan, row, written, document='direct-question', position=0, training=False)
    raw = store.raw_meaning_of(row)
    query = closing_question(store.meaning_of(row), prepared, store=store, written_rows=written)
    assert raw.role_refs[role][0] == 'ltm'
    assert bindings(raw)['_forward_references'] == ((role, words['x']),)
    assert query is not None and open_slots(query) == (('referent', role),)
    assert bindings(query)['_query_occurrence'] == store.occurrence_of(row)
    assert query.role_refs[2-role] == ('sym', words['three'])
    torch.testing.assert_close(query.roles[1], pred)


def test_copied_store_rebinds_its_current_owner_before_closing(monkeypatch):
    import copy
    from MemoryIndex import configure_model_index
    from test_clause_acceptance import SentenceFixture
    from Spaces import _concept_alloc_of
    f = SentenceFixture(monkeypatch)
    f.noun('held')
    configure_model_index(f.model, f.store)
    owner, store, model = copy.deepcopy((f.cs, f.store, f.model))
    configure_model_index(model, store)
    assert store._concept_owner() is owner
    assert store.parts is _concept_alloc_of(owner).parts and store.parts is not f.store.parts
    assert owner._clause_store_ref() is store
    symbol = f.words['held'][1]
    row = owner._csw_row_of(symbol)
    original = f.store._concept_point(symbol).clone()
    with torch.no_grad():
        owner.similarity_codebook.W[row].fill_(.25)
    torch.testing.assert_close(store._concept_point(symbol), torch.full((8,), .25))
    torch.testing.assert_close(f.store._concept_point(symbol), original)
    configure_model_index(model, store)
    assert store._concept_owner() is owner
    assert not hasattr(store, 'leaf_terms')


def test_transparent_wrapper_retains_one_completed_event(monkeypatch):
    from ClauseJournal import finish_clause
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    entry, _, _ = recorded_identity_reading(f, ('not',
        ('lift', ('lower_unary', 'subject'), ('verb', 'move', ('lower', 'd', 'target')))))
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='wrapped-event', sentence_index=0)
    resolver.commit(plan, row, written, document='wrapped-event', position=0, training=False)
    events = {event.occurrence for token in f.cs.mtm.entries.values() for event in token.events}
    assert events == {written[('occurrence', id(prepared))]}
    assert len(f.store) == 3


@pytest.mark.parametrize('role', (0, 2))
def test_resolved_compose_bind_keeps_the_parent_relation_closed(monkeypatch, role):
    from dataclasses import replace
    from ClauseRow import Clause, ClausePredicate, predicate_point
    from Meaning import ConceptualMeaning
    from ReferenceContext import SituationFrame
    from ThoughtClosing import closing_question
    from ThoughtReferences import open_slots
    from test_reference_arrivals import world, named, force_named
    from test_medium_term_memory import keep, mention, snapshot, unchanged
    resolver, store, atoms, words, _ = world()
    force_named(resolver, monkeypatch, None)
    first, _ = keep(resolver, mention(atoms[0], reference=words['three']))
    token = store.occurrence_of(first)
    child = named(atoms[0], words['x'])
    child = replace(child, reference_request=dict(child.reference_request, compose_bind=1),
        identity=dict(handle=1, origins=torch.tensor([[2, token[-1], 1]])))
    pred = predicate_point('equal', atoms[0])
    value = ConceptualMeaning(torch.stack((atoms[0], pred, atoms[1])),
        torch.ones(3, dtype=torch.bool), bindings={'_equality': True})
    refs = [words['wrong'], ClausePredicate(pred, 'equal'), words['wrong']]
    refs[role] = ('clause', 0)
    parent = Clause(value, relation='equal', refs=tuple(refs), children=(child,), evidence=(1., 0.))
    plan = resolver.preview(parent, chunk_context(resolver, (SituationFrame.from_row(store.row(first)),)))
    prepared = resolver.prepare(parent, plan)
    written = {}
    row = store.write_clause(prepared, document_key='resolved-question', written_rows=written)
    retained = snapshot(resolver)
    resolver.commit(plan, row, written, document='resolved-question', position=1, training=False)
    assert plan['items'][0]['reference'] == token
    assert row[0] == 'sym' and store.parts.contains(row[1])
    assert store.meaning_of(row).role_refs[role] == ('sym', words['three'])
    assert not open_slots(store.meaning_of(row))
    assert not open_slots(closing_question(store.meaning_of(row), prepared, store=store, written_rows=written))
    unchanged(resolver, retained)


@pytest.mark.parametrize('pending', (False, True))
def test_unminted_value_and_anonymous_name_have_distinct_pending_state(pending):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from ReferenceContext import SituationFrame
    from ThoughtReferences import open_slots,with_slots
    from test_reference_close import fixture
    from test_medium_term_memory import snapshot,unchanged
    from chunk_fixtures import prepend
    resolver,store,atoms=fixture()
    row=store.append_meaning(ConceptualMeaning.from_description(atoms[0]),order=1,
        document_key='unminted',sentence_index=0,evidence=(1.,0.))
    token=store.occurrence_of(row);pool=resolver.components.owner.mtm
    pool.open(token,kind=None,pending=pending,when=1,value=atoms[0])
    raw=with_slots(replace(store.raw_meaning_of(row),bindings={'_forward_references':((0,101),)}),(('referent',0),))
    retained=snapshot(resolver)
    read=store.read_reference_bindings(row,raw)
    f=SimpleNamespace(cs=resolver.components.owner,store=store)
    staged=prepend(f,(SituationFrame.from_row(store.row(row)),),1)
    identity=pool.entries[token].transient
    assert (staged[chunks.IDENTITY]==identity).any()
    assert pool.entries[token].pending == pending
    assert bool(open_slots(read))==pending
    if not pending:
        assert read.role_refs[0]==token
        torch.testing.assert_close(read.roles[0],atoms[0])
    restored=MediumTermMemory(pool.capacity,n_vectors=int(pool.transient_ids[0]),field_width=pool.field_width)
    restored.restore(pool.state())
    assert restored.entries[token].pending==pending and restored.entries[token].kind is None
    torch.testing.assert_close(restored.value_of(token),atoms[0])
    unchanged(resolver,retained)



def test_pending_checkpoint_migration_and_validation_are_atomic():
    import copy
    pool = MediumTermMemory()
    first, second = address(1), address(2)
    pool.open(first, kind=None, pending=False, when=1, value=torch.ones(3))
    pool.open(second, kind=101, when=2, value=torch.zeros(3))
    state = pool.state()
    legacy = copy.deepcopy(state)
    legacy['version'] = 1
    for entry in legacy['entries']:
        events, value = entry.pop('timelines')[0]
        entry.update(events=tuple(event[:4] for event in events), value=value)
        del entry['pending']
    restored = MediumTermMemory()
    with pytest.raises(ValueError,match="version"):
        restored.restore(legacy)
    restored.restore(state)
    assert not restored.entries[first].pending and not restored.entries[second].pending
    previous = restored.entries
    malformed = copy.deepcopy(state)
    malformed['entries'][0]['pending'] = 'false'
    with pytest.raises(ValueError, match='pending state'):
        restored.restore(malformed)
    assert restored.entries is previous
    pool.bind(first, address(3), kind=102, when=3)
    assert not pool.entries[first].pending
    pool.open(address(4), kind=None, when=4, value=torch.zeros(3))
    pool.merge(first, address(4), when=5)
    assert not pool.entries[address(4)].pending and pool.entries[address(4)].kind == 102


@pytest.mark.parametrize('pending,wrapped', ((True, True), (False, True), (True, False)))
def test_nested_unresolved_operand_keeps_its_resolution_annotation(pending, wrapped):
    import copy
    from dataclasses import replace
    from ClauseRow import Clause, ClausePredicate, Composition, predicate_point
    from Meaning import ConceptualMeaning
    from ThoughtReferences import bindings, with_slots
    from test_reference_arrivals import world
    from test_medium_term_memory import snapshot, unchanged
    resolver, store, atoms, words, _ = world()
    value = ConceptualMeaning.from_description(atoms[0])
    reference = -1 if pending else words['three']
    if pending:
        value = with_slots(replace(value, bindings={'_forward_references': ((0, words['x']),)}),
                           (('referent', 0),), pair=(0., 1.))
    child = Clause(value, point=atoms[0], refs=(reference, -1, -1),
                   composition=Composition(atoms[0], reference=reference), evidence=(1., 0.))
    operand = child
    if wrapped:
        point = torch.maximum(atoms[0], atoms[1])
        term = Composition(point, children=(child.composition,
            Composition(atoms[1], reference=words['wrong'])), operation='sum')
        operand = Clause(ConceptualMeaning.from_description(point), point=point,
            refs=(('clause', 0), -1, -1), children=(child,), composition=term, evidence=(1., 0.))
    pred = predicate_point('equal', atoms[0])
    meaning = ConceptualMeaning(torch.stack((atoms[1], pred, operand.point)),
        torch.ones(3, dtype=torch.bool), bindings={'_equality': True})
    parent = Clause(meaning, relation='equal',
        refs=(words['wrong'], ClausePredicate(pred, 'equal'), ('clause', 0)),
        children=(operand,), evidence=(1., 0.))
    written = {}
    row = store.write_clause(parent, document_key='unresolved-operand', written_rows=written)
    raw = store.raw_meaning_of(row)
    marks = bindings(raw).get('_unresolved_operands', ())
    if pending and wrapped:
        occurrence = written[('occurrence', id(operand))]
        assert marks == ((2, occurrence),) and raw.role_refs[2] == occurrence
        assert occurrence[0] == 'sym' and store.parts.contains(occurrence[1])
        from ChainRead import point
        from LTM1 import RELATION
        assert point(store, (RELATION, occurrence[1])) is None
        retained = snapshot(resolver)
        _, restored, _, _, _ = world()
        restored.load_state_dict(copy.deepcopy(store.state_dict()))
        restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
        restored.parts.load_state_dict(copy.deepcopy(store.parts.state_dict()))
        from Spaces import _concept_alloc_of
        _concept_alloc_of(restored._concept_owner()).singletons = copy.deepcopy(
            _concept_alloc_of(store._concept_owner()).singletons)
        assert bindings(restored.raw_meaning_of(row))['_unresolved_operands'] == marks
        unchanged(resolver, retained)
    else:
        assert marks == ()
        if not pending:
            assert row[0] == 'sym' and store.parts.contains(row[1])


@pytest.mark.parametrize('phase', ('commit', 'resolved'))
@pytest.mark.parametrize('cached', (False, True))
def test_bind_training_never_updates_or_creates_a_concept_region(monkeypatch, phase, cached):
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    from Layers import Error
    from test_reference_close import fixture, close, form_exposure
    from test_medium_term_memory import snapshot, unchanged
    resolver, store, atoms = fixture()
    resolver.components.weight = 1.
    frames = []
    for index in range(3):
        frame, _, _ = close(resolver, atoms[index], atoms[index+6], 'mint', frames,
                            document='bank-isolation', sentence=index)
        frames.append(frame)
    owner = resolver.components.owner
    if cached:
        form_exposure(resolver, atoms[4], atoms[0])
    else:
        word = owner.interpret.lookup_word([101], (), form='fixture-cue')
        with torch.no_grad():
            owner.similarity_codebook.W[owner._csw_row_of(word)].copy_(atoms[4])
    monkeypatch.setattr(resolver, 'select', lambda choices, logits, **kwargs: 1)
    request = dict(mode='bind', leaves=(0,), cue=atoms[4], predicate=atoms[6], role=0)
    clause = Clause(ConceptualMeaning.from_description(atoms[1]), point=atoms[1],
                    order=1, reference_request=request)
    plan = resolver.preview(clause, chunk_context(resolver, frames))
    selected = plan['items'][0]['selected']
    assert selected == plan['items'][0]['candidates'][1]
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = store.write_clause(prepared, document_key='bank-isolation', sentence_index=3,
                             written_rows=written)
    retained = snapshot(resolver)
    alloc = owner._concept_allocator
    allocation, regions = alloc.next_id, dict(alloc.form_regions)
    resolver.commit(plan, row, written, document='bank-isolation', position=3,
                    training=phase == 'commit')
    if phase == 'resolved':
        errors = Error()
        resolver.resolved(row, 0, ('sym', selected), errors)
        loss = errors.total(objective='expectation')
        assert loss is not None and loss.requires_grad
    assert alloc.next_id == allocation and alloc.form_regions == regions
    unchanged(resolver, retained)


@pytest.mark.parametrize('readmit', (False, True))
@pytest.mark.parametrize('dummy', (False, True))
def test_relative_determination_keeps_its_introduction_and_later_bind(monkeypatch, readmit, dummy):
    from ClauseJournal import finish_clause
    from ReferenceReconstruction import cost
    from test_clause_acceptance import SentenceFixture
    from test_medium_term_memory import snapshot, unchanged
    f = SentenceFixture(monkeypatch)
    relative = ('definite_unary', ('implies', 'left', 'right'))
    if dummy:
        relative = ('bind', relative)
    tree = ('sum', ('equal', ('equal', relative, 'changed'), 'latest'), ('bind', 'alias'))
    entry, inputs, handle = recorded_identity_reading(f, tree, dummy_binds=(0,) if dummy else ())
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    intro = next(item for item in plan['items'] if item['node'].identity['introduction']
                 and item['node'].identity['handle'] == handle)
    assert intro['node'].subject_word_id == -1
    assert intro['node'].children[0].relation == 'implies'
    opened = next(value for name, _, value in inputs if name == 'definite_unary')[0]
    torch.testing.assert_close(intro['content'], opened)
    assert intro['own_object'] and intro['selected'] == ('new', 0)
    assert plan['items'][-1]['selected'] == intro['selected']
    assert all(item['selected'] == intro['selected'] for item in plan['items']
               if item['node'].identity['handle'] == handle)
    if dummy:
        wrapper = next(item for item in plan['items'] if item.get('compose_dummy') and item is not intro)
        assert not wrapper['own_object']
        torch.testing.assert_close(cost(resolver, [dict(plan, items=[wrapper])], torch.zeros(1)), torch.zeros(1))
    if readmit:
        clause = finish_clause(f.language, entry, registry=f.registry)
        sites = resolver.sites(clause, pending_names=plan['pending_names'])
        plan = dict(plan, items=[dict(item, **site) for item, site in zip(plan['items'], sites)])
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='relative', sentence_index=0)
    retained = snapshot(resolver)
    resolver.commit(plan, row, written, document='relative', position=0, training=False)
    mentions = [written[('occurrence', id(item['written_node']))] for item in plan['items']
                if item['written_node'].identity['handle'] == handle]
    token = f.cs.mtm.token_of(mentions[0])
    assert all(f.cs.mtm.token_of(mention) == token for mention in mentions)
    assert f.cs.mtm.entries[token].count == len(mentions)
    assert len([event for event in f.cs.mtm.entries[token].events if event.role == 0]) == 2
    torch.testing.assert_close(f.cs.mtm.value_of(token), entry.identity_values[handle-1])
    unchanged(resolver, retained)


def test_two_relative_determinations_keep_distinct_objects(monkeypatch):
    from ClauseJournal import finish_clause
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    relative = ('definite_unary', ('implies', 'left', 'right'))
    entry, _, _ = recorded_identity_reading(f, ('sum', relative, relative))
    resolver = f.cs.components.reference_close
    clause = finish_clause(f.language, entry, registry=f.registry)
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    introductions = [item for item in plan['items'] if item['node'].identity['introduction']]
    assert len(introductions) == 2
    assert introductions[0]['node'].identity['handle'] != introductions[1]['node'].identity['handle']
    assert [item['selected'] for item in introductions] == [('new', 0), ('new', 1)]
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='separate', sentence_index=0)
    resolver.commit(plan, row, written, document='separate', position=0, training=False)
    tokens = [f.cs.mtm.token_of(written[('occurrence', id(item['written_node']))]) for item in introductions]
    assert tokens[0] != tokens[1]
    assert all(f.cs.mtm.entries[token].count == 1 for token in tokens)


def test_missing_determined_event_introduction_fails_loudly():
    from ClauseRow import Clause
    from Meaning import ConceptualMeaning
    from test_reference_close import fixture
    resolver, store, atoms = fixture()
    value = ConceptualMeaning.from_description(atoms[0])
    node = Clause(value, point=atoms[0], identity=dict(handle=1, action=0,
        event=(1, 1, 0, 101, -1, 1), origins=torch.tensor([[3, 0, 1]]), values=atoms[:1]), evidence=(1., 0.))
    row = store.write_clause(Clause(value, point=atoms[0]), document_key='lost-introduction')
    plan = dict(items=[], prepared_root=node)
    written = {('occurrence', id(node)): store.occurrence_of(row)}
    with pytest.raises(RuntimeError, match='lost its determined introduction'):
        resolver._commit_events(plan, written, when=1)
    assert not resolver.components.owner.mtm.entries


@pytest.mark.parametrize('evict', (False, True))
@pytest.mark.parametrize('pool_aliases', (0, 2))
def test_journal_native_bind_without_a_pool_token_keeps_its_symbol_origin(monkeypatch, evict, pool_aliases):
    from dataclasses import replace
    from ChunkJournal import capture
    from ClauseJournal import finish_clause
    from IdentityRows import mint
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    entry = f.program(('bind', 'alias'))
    content = entry.leaves[0].clone()
    symbol = mint(f.cs, content)
    for index in range(pool_aliases):
        f.cs.mtm.open(address(index+1), kind=symbol, native=symbol, when=index, value=content)
    # No token and an ambiguous pair both leave the durable symbol usable.
    assert f.cs.mtm.token_of(symbol) is None
    state = chunks.empty(1, 8, len(content))
    state, _ = chunks.deposit(state, content[None], torch.tensor([True]), identity=symbol,
        refs=torch.tensor([[[chunks.SYMBOL, symbol], [0, 0]]]), order=1)
    walk = Walk(f.binary, f.unary)
    state = walk.empty(content[None], 2, state=state)
    state = walk.shift(state, content[None], word=int(entry.concept_ids[0]), index=0)
    local = next(i for i, rule in enumerate(f.binary) if rule.method_name == 'bind')
    offered, ids, indices, valid = chunks.bind_menu(state[0], content[None, None])
    column = int(((ids[0, 0] == symbol) & valid[0, 0]).long().argmax())
    selected = choice(content[None], local=local, identity=symbol,
                      chunk=int(indices[0, 0, column]), bind=True)
    operands = torch.stack((content[None], offered[:, 0, column]), 1)
    state = walk.apply(state, selected, operands, 1)
    stm, scope, records, values = state
    if evict:
        # Closing must retain the journal's durable address even after its
        # numerical working chunk has left the shared eight-slot cache.
        stm = chunks.empty(1, 8, len(content))
    identity = capture(f.cs, stm, scope, records, values, batch=0,
        positions=torch.tensor([0]), columns=torch.tensor([0, 1]), actions=entry.actions)
    handle = int(identity['identity_actions'][1, 4])
    assert identity['identity_origins'][handle-1].tolist() == [1, symbol, 1]
    entry = replace(entry, operation_values=values[0].reshape(2, 3, -1),
        operation_refs=records[0, :, :2], leaf_orders=torch.ones(1, dtype=torch.long),
        close_references=True, **identity)
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, state[0])
    assert plan['items'][0]['selected'] == symbol
    assert not plan['items'][0]['own_object']
    assert f.cs.mtm.token_of(symbol) is None
