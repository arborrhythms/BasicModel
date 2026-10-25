"""Fixed-operation timeline certificates; no claim of learned modal attachment."""
from copy import deepcopy

import pytest
import torch

from MediumTermMemory import MediumTermMemory
from MeaningKernels import MeaningAction
from ModalLearning import observe, register
from Layers import Error


def same(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            same(actual[key], expected[key])
    elif isinstance(actual, (tuple,list)):
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected):
            same(a,b)
    else:
        assert actual == expected


def address(number):
    return 'ltm', 'timeline-unit', number


def pool():
    result = MediumTermMemory()
    result.open(address(1), kind=11, when=1, value=torch.full((8,), .2))
    return result


def test_no_modifier_keeps_one_actual_timeline_and_legacy_checkpoints_load():
    p = pool()
    p.event(address(1), address(2), verb=12, role=0, when=2)
    p.update(address(1), torch.full((8,), .3), when=2)
    entry = p.entries[address(1)]
    assert len(entry.timelines) == 1 and entry.timelines[0].events == entry.events
    assert entry.timelines[0].value is entry.value
    saved = p.state()
    for version in (1, 2, 3, 4):
        legacy = deepcopy(saved)
        legacy['version'] = version
        for row in legacy['entries']:
            events, value = row.pop('timelines')[0]
            row.update(events=tuple(event[:7 if version == 4 else 4] for event in events), value=value)
        restored = MediumTermMemory()
        restored.restore(legacy)
        assert len(restored.entries[address(1)].timelines) == 1
        torch.testing.assert_close(restored.value_of(address(1)), entry.value)
        assert restored.entries[address(1)].events == entry.events


def test_alternative_continues_one_identity_and_preserves_the_actual_prefix():
    p = pool()
    p.event(address(1), address(2), verb=12, role=0, when=2)
    actual = p.entries[address(1)]
    branch = p.alternative(address(1), when=3)
    assert branch == 1 and p.entries[address(1)].timelines[branch].events == actual.events
    p.event(address(1), address(3), verb=13, role=0, when=3, timeline=branch)
    p.update(address(1), torch.full((8,), .4), when=3, timeline=branch)
    assert p.alternative(address(1), when=4) == branch
    p.event(address(1), address(4), verb=14, role=0, when=4, timeline=branch)
    p.update(address(1), torch.full((8,), .5), when=4, timeline=branch)
    entry = p.entries[address(1)]
    assert len(p.entries) == 1 and entry.count == actual.count and entry.occurrences == actual.occurrences
    assert entry.events == actual.events and len(entry.timelines[1].events) == 3
    torch.testing.assert_close(entry.value, actual.value)
    torch.testing.assert_close(entry.timelines[1].value, torch.full((8,), .5))
    assert p.token_of(address(4)) == address(1)
    p.forget(address(3))
    assert [event.occurrence for event in p.entries[address(1)].timelines[1].events] == [address(2), address(4)]
    p.forget(address(1))
    assert not p.entries and not p.names


def test_timeline_checkpoint_copies_and_validates_the_whole_pool_atomically():
    p = pool()
    branch = p.alternative(address(1), when=2)
    p.event(address(1), address(2), verb=12, role=0, when=2, timeline=branch,
        before=torch.full((8,), .2), after=torch.full((8,), .3), modifiers=(torch.full((8,), .4),))
    saved = p.state()
    restored = MediumTermMemory()
    restored.restore(saved)
    same(restored.state(), saved)
    saved['entries'][0]['timelines'][1][1].zero_()
    assert p.entries[address(1)].timelines[1].value.sum() > 0
    bad = p.state()
    bad['entries'][0]['timelines'][1][1][0] = float('nan')
    before = restored.state()
    with pytest.raises(ValueError):
        restored.restore(bad)
    same(restored.state(), before)


def test_equal_time_teacher_uses_only_defined_coordinates_once_and_defers_its_graph():
    p = pool()
    action = MeaningAction(8, modifier=True)
    actual = torch.tensor([.2, .3, 0., .4, .2, .3, 0., .4])
    after = torch.tensor([.3, .4, .8, .5, .3, .4, .8, .5])
    modifier = torch.full((8,), .2)
    p.event(address(1), address(2), verb=12, role=0, when=7, after=actual)
    branch = p.alternative(address(1), when=7)
    p.event(address(1), address(3), verb=13, role=0, when=7, timeline=branch,
        after=after, modifiers=(modifier,))
    records, errors = [], Error()
    report = observe(action, p, (address(1),), errors=errors, deferred=records)
    assert len(report) == 1 and report[0]['defined'] == 6 and report[0]['when'] == 7
    assert errors.total(objective='reconstruction') is None and len(records) == 1
    register(records, errors)
    errors.total(objective='reconstruction').backward()
    assert action.readout.grad.abs().sum() > 0
    assert torch.count_nonzero(action.readout.grad[2]) == 0
    assert observe(action, p, (address(1),)) == []
    assert not records


def test_a_modifier_without_an_equal_time_alternative_teaches_nothing():
    p = pool()
    action = MeaningAction(8, modifier=True)
    p.event(address(1), address(2), verb=12, role=0, when=7, after=torch.full((8,), .2))
    assert observe(action, p, (address(1),)) == []
    branch = p.alternative(address(1), when=8)
    p.event(address(1), address(3), verb=13, role=0, when=8, timeline=branch,
        after=torch.full((8,), .4), modifiers=(torch.full((8,), .2),))
    assert observe(action, p, (address(1),)) == []


def recording(f, tree, frames=()):
    """Fixed selected operations through the same numerical branch helpers."""
    from dataclasses import replace
    from MTMOverlay import MTMOverlay
    from TimelineOverlay import prepare, choice as timeline_choice
    from ReferenceContext import result_order
    from test_identity_threading import choice
    entry = f.program(tree)
    orders = torch.tensor([1 if word == f.words['subject'][0] else 0 for word in entry.word_ids])
    overlay = MTMOverlay(f.binary, f.unary)
    words, columns = len(entry.leaves), len(entry.actions)
    state = overlay.stage(f.cs.components.reference_close, [frames],
        entry.leaves.new_zeros(1, 8, 8), words=words, journal=columns, timelines=True)
    base = state[0].shape[1] - words - columns
    stack, stamps, recorded, refs = [], [], [], []
    for column, (kind, local, word) in enumerate(entry.actions.tolist()):
        if kind == 0:
            value = entry.leaves[word][None]
            state = overlay.push(state, torch.tensor([True]), value,
                word=entry.concept_ids[word:word+1], order=orders[word:word+1],
                index=torch.tensor(word), words=words, resolved=torch.tensor([True]))
            stack.insert(0, value); stamps.insert(0, int(orders[word]))
            recorded.append(value.new_zeros(3, 8)); refs.append((-1, -1))
            continue
        live = overlay.refresh(state, torch.stack(stack,1), torch.arange(len(stack))[None])
        rule = (f.binary if kind == 1 else f.unary)[local]
        operands = live[:, [1,0]] if kind == 1 else torch.stack((live[:,0], live[:,0]*0),1)
        if kind == 1:
            data = prepare(dict(binary_ops=(local,), left=operands[:,0:1,None], right=operands[:,1:2,None]),
                state, torch.tensor([[1,0]]), f.binary, meaning_start=0)
            operands = torch.stack((data['left'][:,0,0],data['right'][:,0,0]),1)
        handle = 0
        if rule.method_name == 'bind':
            points, records, _, _, available = overlay.menu(state, live[:,:1], torch.tensor([[0]]))
            handle = int(available[0,0,1:].nonzero()[0])+1
            result = points[:,0,handle]
            operands = torch.stack((live[:,0],records[:,0,handle]),1)
        else:
            values = (operands[:,0],operands[:,1]) if kind == 1 else (operands[:,0],)
            result = f.operation(rule).compose(*values).reshape(1,-1)
        arity = 2 if kind == 1 else 1
        order = result_order(rule, tuple(reversed(stamps[:arity])))
        selected = choice(result, kind=kind, identity=handle)._replace(
            local_op=torch.tensor([local]),order=torch.tensor([order]))
        def evaluate(a,b,active):
            values = (a,b) if kind == 1 else (a,)
            return f.operation(rule).compose(*values).reshape(1,-1)
        selected = timeline_choice(overlay,state,selected,operands,
            torch.tensor([stamps+[-1]*(8-len(stamps))]),evaluate,meaning_start=0)
        state = overlay.apply(state,selected,operands,column=torch.tensor(column),words=words)
        del stack[:arity]; del stamps[:arity]
        stack.insert(0,result);stamps.insert(0,int(selected.order))
        recorded.append(torch.cat((operands,result[:,None]),1)[0]);refs.append((-1,-1))
    return replace(entry,leaf_orders=orders,close_references=True,
        operation_values=torch.stack(recorded),operation_refs=torch.tensor(refs),
        identity_origins=state[3][0],identity_actions=state[4][0],identity_values=state[0][0],
        identity_leaves=torch.arange(base+1,base+words+1),timeline_actions=state[11][0],timeline_values=state[5][0],
        end_state=torch.stack((stack[0][0],stack[0][0]*0,stack[0][0]*0)))


def fixture(monkeypatch):
    from types import SimpleNamespace
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    verb, modifier = MeaningAction(8,capacity=2), MeaningAction(8,modifier=True)
    with torch.no_grad():
        verb.readout.fill_(.2)
        modifier.readout.fill_(.15)
    for word,value in (('subject',.2),('move',.3),('shift',.4)):
        symbol=f.noun(word)
        with torch.no_grad():
            f.cs.similarity_codebook.W[f.cs._csw_row_of(symbol)].fill_(value)
    f.ops['lift']=SimpleNamespace(compose=lambda a,b:verb(a,b))
    f.ops['adverb']=SimpleNamespace(compose=lambda a,b:modifier(a,b))
    f.ops['lower_unary']=SimpleNamespace(compose=lambda a:a)
    return f,verb,modifier


def close(f, program, frames=(), *, position=0, actions=(None,None), question=False, evidence=None):
    from dataclasses import replace
    from ClauseJournal import finish_clause
    from ReferenceContext import SituationAddress
    from DocumentCover import open_cover
    if not hasattr(f, 'cover'):
        f.cover = open_cover(f.cs, f.store, stream=0, document='timeline')
    clause=finish_clause(f.language,program,registry=f.registry)
    if question:
        clause=replace(clause,meaning=replace(clause.meaning,mode='interrogative'))
    if evidence is not None:
        clause=replace(clause,evidence=evidence)
    resolver=f.cs.components.reference_close
    plan=resolver.preview(clause,frames)
    kept=resolver.prepare(clause,plan)
    written={}
    row=f.store.write_clause(kept,written_rows=written,document_key='timeline',sentence_index=position,stream=0)
    resolver.commit(plan,row,written,document='timeline',position=position,training=False,
                    verb_action=actions[0],modifier_action=actions[1])
    frame=SituationAddress(f.store.occurrence_of(row),symbols=written.get(('symbols',id(kept)),()),
        activations=written.get(('activations',id(kept)),())).read(f.store)
    return kept,frame,row,written


@pytest.mark.parametrize('post', (False,True))
def test_modal_bracketings_write_only_an_idea_and_preserve_the_actual(monkeypatch,post):
    f,verb,modifier=fixture(monkeypatch)
    subject=('lower_unary','subject')
    tree=('adverb',('lift',subject,'move'),'shift') if post else ('lift',subject,('adverb','move','shift'))
    program=recording(f,tree)
    clause,frame,row,written=close(f,program,actions=(verb,modifier))
    pool=f.cs.mtm
    assert len(pool.entries)==1
    token,entry=next(iter(pool.entries.items()))
    assert clause.order>=1 and int(f.store.order[row])>=1
    assert not entry.events and len(entry.timelines)==2
    assert len(entry.timelines[1].events)==1
    assert not any(int(f.store.order[f.store.location(event.occurrence)])==0 for event in entry.timelines[1].events)
    torch.testing.assert_close(entry.value,torch.full((8,),.2))
    subject_value=torch.full((8,),.2)
    expected=(modifier(verb(subject_value,torch.full((8,),.3)),torch.full((8,),.4)) if post else
              verb(subject_value,modifier(torch.full((8,),.3),torch.full((8,),.4))))
    torch.testing.assert_close(entry.timelines[1].value,expected)
    assert pool.token_of(entry.timelines[1].events[0].occurrence)==token
    assert any(pool.token_of(use)==token for use in entry.occurrences)


@pytest.mark.parametrize('post',(False,True))
def test_second_modal_acts_on_the_open_branch_and_bind_still_reads_actual(monkeypatch,post):
    f,verb,modifier=fixture(monkeypatch)
    def tree(subject):
        return ('adverb',('lift',subject,'move'),'shift') if post else ('lift',subject,('adverb','move','shift'))
    _,frame,_,_=close(f,recording(f,tree(('lower_unary','subject'))),actions=(verb,modifier))
    token,entry=next(iter(f.cs.mtm.entries.items()))
    before=entry.timelines[1].value.clone()
    program=recording(f,tree(('bind','subject')),(frame,))
    bind=next(i for i,(kind,local,_) in enumerate(program.actions.tolist())
              if kind==2 and f.unary[local].method_name=='bind')
    torch.testing.assert_close(program.operation_values[bind,1],entry.value)
    _,_,_,_=close(f,program,(frame,),position=1,actions=(verb,modifier))
    updated=f.cs.mtm.entries[token]
    assert len(f.cs.mtm.entries)==1 and len(updated.timelines)==2
    assert len(updated.timelines[1].events)==2 and not updated.events
    torch.testing.assert_close(updated.value,entry.value)
    expected=(modifier(verb(before,torch.full((8,),.3)),torch.full((8,),.4)) if post else
              verb(before,modifier(torch.full((8,),.3),torch.full((8,),.4))))
    torch.testing.assert_close(updated.timelines[1].value,expected)


@pytest.mark.parametrize('post',(False,True))
def test_fixed_query_reads_event_order_and_never_asserts_the_question(monkeypatch,post):
    from ThoughtClosing import closing_question
    from ThoughtReferences import bindings
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    f,verb,modifier=fixture(monkeypatch)
    actual=('lift',('lower_unary','subject'),'move')
    _,first,actual_row,_=close(f,recording(f,actual),evidence=(.8,0.))
    subject=('bind','subject')
    tree=('adverb',('lift',subject,'move'),'shift') if post else ('lift',subject,('adverb','move','shift'))
    _,second,idea_row,_=close(f,recording(f,tree,(first,)),(first,),position=1,evidence=(.7,0.))
    token,entry=next(iter(f.cs.mtm.entries.items()))
    reader=ThoughtLTMCapability(store=lambda:f.store,equal=lambda a,b:1.,tau_id=.9)
    for n,query_tree,expected in ((2,('lift',subject,'move'),actual_row),(3,tree,idea_row)):
        c,_,row,written=close(f,recording(f,query_tree,(second,)),(second,),position=n,question=True)
        question=closing_question(c.meaning,c,store=f.store,written_rows=written)
        assert bindings(question)['_query_event'][2]==int(f.store.order[row])
        found=reader.best_match(question,max_records=64,work=QueryWorkBudget(256))
        assert found['witnesses']==(f.store.occurrence_of(expected),)
        current=f.cs.mtm.entries[token]
        assert len(current.events)==1 and len(current.timelines[1].events)==2
        torch.testing.assert_close(current.value,entry.value)
        torch.testing.assert_close(current.timelines[1].value,entry.timelines[1].value)
    # Equal initial state values do not make different predicates witnesses.
    from dataclasses import replace
    data=bindings(question)
    subject_ref,_,order=data['_query_event']
    data['_query_event']=(subject_ref,f.words['shift'][1],order)
    missing=reader.best_match(replace(question,bindings=data),max_records=64,work=QueryWorkBudget(256))
    assert not missing['witnesses'] and missing['support_true']==missing['support_false']==0
    unknown=reader.truth_evidence(question,max_records=0,work=QueryWorkBudget(0))
    assert unknown['support_true']==0 and unknown['incomplete']==('timeline_read_limit',)


def test_some_every_include_actual_and_use_both_lanes(monkeypatch):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    f,_,_=fixture(monkeypatch)
    _,frame,row,_=close(f,recording(f,('lift',('lower_unary','subject'),'move')),evidence=(.8,.2))
    token,entry=next(iter(f.cs.mtm.entries.items()))
    # Open an alternative at the same subject value with no event yet.
    branch=f.cs.mtm.alternative(token,when=10)
    from MediumTermMemory import Timeline
    f.cs.mtm.entries[token]=replace(f.cs.mtm.entries[token],timelines=(entry.timelines[0],Timeline((),entry.value)))
    q=replace(ConceptualMeaning.from_description(entry.value),bindings={
        '_query_event':(token,entry.events[0].verb,1)})
    reader=ThoughtLTMCapability(store=lambda:f.store,equal=lambda a,b:1.,tau_id=.9)
    possible=reader.existence_evidence(q,max_records=32,work=QueryWorkBudget(64))
    necessary=reader.truth_evidence(q,max_records=32,work=QueryWorkBudget(64))
    assert possible['support_true']==pytest.approx(.8) and possible['support_false']==0
    assert necessary['support_true']==0 and necessary['support_false']==pytest.approx(.2)
    f.cs.mtm.event(token,f.store.occurrence_of(row),verb=entry.events[0].verb,role=0,when=10,timeline=branch)
    necessary=reader.truth_evidence(q,max_records=32,work=QueryWorkBudget(64))
    assert necessary['support_true']==pytest.approx(.8) and necessary['support_false']==pytest.approx(.2)


def test_promotion_folds_only_actual_events_and_document_end_keeps_ideas(monkeypatch):
    from MemoryConsolidation import consolidate
    f,verb,modifier=fixture(monkeypatch)
    _,frame,actual_row,_=close(f,recording(f,('lift',('lower_unary','subject'),'move')),evidence=(1.,0.))
    token,entry=next(iter(f.cs.mtm.entries.items()))
    actual_value=f.store.slots[actual_row,0].clone()
    tree=('lift',('bind','subject'),('adverb','move','shift'))
    ideas=[]
    for position in (1,2):
        _,frame,row,_=close(f,recording(f,tree,(frame,)),(frame,),position=position)
        ideas.append(f.store.occurrence_of(row))
    entry=f.cs.mtm.entries[token]
    assert entry.symbol is not None and len(entry.occurrences)==3
    torch.testing.assert_close(f.cs.similarity_codebook.W[f.cs._csw_row_of(entry.symbol)],actual_value)
    assert entry.timelines[1].value.ne(actual_value).any()
    rows,bank=f.store.slots.clone(),f.cs.similarity_codebook.W.clone()
    consolidate(f.cs,f.store)
    f.cs.mtm.end_document(f.cover,store=f.store)
    assert not f.cs.mtm.entries and not f.cs.mtm.names
    assert all(int(f.store.order[f.store.location(ref)])>=1 for ref in ideas)
    torch.testing.assert_close(f.store.slots,rows)
    torch.testing.assert_close(f.cs.similarity_codebook.W,bank)


@pytest.mark.parametrize('compiled',(False,True))
def test_post_attachment_rolls_back_resolution_and_preserves_pathwise_derivatives(compiled):
    from types import SimpleNamespace
    from MTMOverlay import MTMOverlay
    from TimelineOverlay import choice as annotate
    from test_identity_threading import choice,push
    overlay=MTMOverlay((SimpleNamespace(method_name='lift',clause_form='S',head_role=2),
                        SimpleNamespace(method_name='adverb',head_role=1)),
                       (SimpleNamespace(method_name='bind',head_role=1),))
    def run(first,scale,modifier):
        state=overlay.empty(first[:,None].expand(-1,8,-1),journal=16,words=8,timelines=True)
        state=push(overlay,state,first,word=101,index=0,resolved=False)
        def apply(state,selected,operands,column,evaluate,orders):
            selected=annotate(overlay,state,selected,operands,orders,evaluate,meaning_start=0)
            return overlay.apply(state,selected,operands,column=torch.tensor(column),words=8)
        state=apply(state,choice(first),torch.stack((first,first*0),1),0,
                    lambda a,b,mask:a,torch.ones(1,8,dtype=torch.long))
        state=push(overlay,state,first*0+.3,word=202,index=1,order=0)
        state=apply(state,choice(first*scale,kind=1),torch.stack((first,first*0+.3),1),1,
                    lambda a,b,mask:a*scale,torch.tensor([[0,1,0,0,0,0,0,0]]))
        state=push(overlay,state,first*0+.4,word=303,index=2,order=0)
        selected=choice(first*scale*modifier,kind=1)._replace(local_op=torch.ones(1,dtype=torch.long))
        return apply(state,selected,torch.stack((first*scale,first*0+.4),1),2,
                    lambda a,b,mask:a*modifier,torch.zeros(1,8,dtype=torch.long))
    if compiled:
        run=torch.compile(run,backend='aot_eager',fullgraph=True)
    first=torch.tensor([[.2,.3]],requires_grad=True)
    scale,modifier=torch.tensor(.6,requires_grad=True),torch.tensor(.4,requires_grad=True)
    state=run(first,scale,modifier)
    torch.testing.assert_close(state[0][:,0],first)
    assert state[3][0,0,2]==0 and state[11][0,1,3]==1 and state[11][0,2,0]==1
    torch.testing.assert_close(state[5][:,0],first*scale*modifier)
    state[5][:,0].sum().backward()
    torch.testing.assert_close(first.grad,torch.full_like(first,.24))
    torch.testing.assert_close(scale.grad,first.detach().sum()*modifier.detach())
    torch.testing.assert_close(modifier.grad,first.detach().sum()*scale.detach())


def test_post_attachment_preserves_a_later_independent_actual_write(monkeypatch):
    f,verb,modifier=fixture(monkeypatch)
    tree=('adverb',('lift',('lower_unary','subject'),'move'),
                    ('lift',('bind','subject'),'move'))
    program=recording(f,tree)
    lifts=[i for i,(kind,op,_) in enumerate(program.actions.tolist())
           if kind==1 and f.binary[op].method_name=='lift']
    expected=program.operation_values[lifts[-1],2]
    close(f,program,actions=(verb,modifier))
    entry=next(iter(f.cs.mtm.entries.values()))
    assert len(entry.events)==1 and len(entry.timelines[1].events)==1
    torch.testing.assert_close(entry.value,expected)


def test_retained_idea_keeps_its_subject_reference_after_token_expiry(monkeypatch):
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    f,_,_=fixture(monkeypatch)
    _,frame,row,_=close(f,recording(f,('lift',('lower_unary','subject'),('adverb','move','shift'))),
                         evidence=(.9,0.))
    token,entry=next(iter(f.cs.mtm.entries.items()))
    query=replace(ConceptualMeaning.from_description(f.store.slots[row,0]),
        bindings={'_query_event':(token,entry.timelines[1].events[-1].verb,1)})
    f.cs.mtm.end_document(f.cover,store=f.store)
    reader=ThoughtLTMCapability(store=lambda:f.store,equal=lambda a,b:1.,tau_id=.9)
    found=reader.best_match(query,max_records=64,work=QueryWorkBudget(256))
    assert found['witnesses']==(f.store.occurrence_of(row),)
    assert found['support_true']==pytest.approx(.9)


@pytest.mark.parametrize('modal',(False,True))
def test_bound_idea_has_both_value_and_retained_identity_reference(monkeypatch,modal):
    from PartRead import leaves
    f,_,_=fixture(monkeypatch)
    _,frame,_,_=close(f,recording(f,('lift',('lower_unary','subject'),'move')))
    token,entry=next(iter(f.cs.mtm.entries.items()))
    vp=('adverb','move','shift') if modal else 'move'
    program=recording(f,('lift',('bind','subject'),vp),(frame,))
    bind=next(i for i,(kind,op,_) in enumerate(program.actions.tolist())
              if kind==2 and f.unary[op].method_name=='bind')
    torch.testing.assert_close(program.operation_values[bind,1],entry.value)
    _,_,row,_=close(f,program,(frame,),position=1)
    updated=f.cs.mtm.entries[token]
    mention=updated.occurrences[-1]
    reading=f.store.raw_meaning_of(f.store.location(mention)).role_refs[0]
    root=f.store.raw_meaning_of(row).role_refs[0]
    assert f.cs.mtm.token_of(mention)==token and reading[1] in leaves(f.store.parts,root[1])
    assert (int(f.store.order[row])>0)==modal


def test_modal_object_role_keeps_its_own_state(monkeypatch):
    f,_,_=fixture(monkeypatch)
    f.noun('object')
    tree=('lift',('lower_unary','subject'),('adverb',('verb','move',('lower_unary','object')),'shift'))
    close(f,recording(f,tree))
    objects=[entry for entry in f.cs.mtm.entries.values()
             if any(event.role==2 for timeline in entry.timelines for event in timeline.events)]
    assert len(objects)==1
    entry=objects[0]
    assert not entry.events and len(entry.timelines)==2
    torch.testing.assert_close(entry.timelines[1].value,entry.value)
