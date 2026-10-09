"""The 6.2 certificates use native rows, the grammar scorer and real episodes."""
from dataclasses import replace
from types import SimpleNamespace
import ast
from pathlib import Path
import pytest
import torch

from Language import Grammar, OperationSelectionLayer
from Layers import TernaryTruthStore, WhatInteractionMemory, Error
from Meaning import ConceptualMeaning, ClosingImage
from Models import BasicModel
from Queries import GrammaticalThoughtRegistry, THOUGHT_EXECUTORS
from ThoughtReferences import bindings, evidence_pair, open_slots, question, with_slots
from test_cs_symbol_table import _cs


def world():
    cs = _cs()
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    refs = tuple(('sym', cs.new_concept()) for _ in range(3))
    for ref in refs:
        cs._csw_concept_row(0, ref[1])
    store = TernaryTruthStore(cs.outputShape[-1], capacity=128)
    memory = WhatInteractionMemory(batch=1, capacity=128, detach_mode='episode')
    model = BasicModel()
    model.spaces = []
    model.eval()
    model.attention_budget = 64
    object.__setattr__(model, 'conceptualSpace', cs)
    object.__setattr__(model, 'grammatical_thoughts', registry)
    object.__setattr__(model, 'symbolSpace', SimpleNamespace(ltm_store=store,
        what_memory=memory, grammatical_thoughts=registry))
    model.shared_grammar = OperationSelectionLayer(d_model=cs.outputShape[-1], chooser='mlp')
    object.__setattr__(model, 'languageSpace', SimpleNamespace(
        language_layer=SimpleNamespace(operation_layer=model.shared_grammar)))
    return model, registry, store, refs


def run(model, q, **kwargs):
    with model._query_boundary_scope((0,)):
        return model.run_selected_thought(q, **kwargs)


def chain_world(monkeypatch):
    model, registry, store, (a,b,c) = world()
    for left,right in ((a,b),(b,c)):
        store.append_meaning(registry.form('part',left,right,mode='assertive'),
            kind='fact', rel_type=store.REL_PARTOF, evidence=(1.,0.),trust=1.)
    chosen=[]
    def choose(root,active,actions,**kwargs):
        if None in actions:
            return None
        for action in actions:
            if action.semantic_id=='query' and action.request.constituents:
                q=action.request.constituents[0]
                if q.role_refs[0] == (a if not chosen else b) and (
                        q.role_refs[2] is None if not chosen else q.role_refs[2]==c):
                    chosen.append(action)
                    return action
        raise AssertionError('missing serial chain candidate')
    monkeypatch.setattr(model,'_choose_selected_thought_action',choose)
    return model,registry,store,(a,b,c),chosen


def test_chain_two_queries_one_conclusion_and_both_witnesses(monkeypatch):
    model,registry,store,(a,b,c),chosen=chain_world(monkeypatch)
    result=run(model,registry.form('isPart',a,c),work_budget=64)
    assert [r.operation for r in result.records if r.kind=='thought']==['query','query','conclude']
    assert len(chosen)==2 and not open_slots(result.meaning)
    assert evidence_pair(result.meaning)==(1.,0.)
    saved=store.row(2)
    assert saved['kind']=='inference'
    assert set(bindings(saved['meaning'])['_thought_witnesses'])=={store.occurrence_of(0),store.occurrence_of(1)}
    assert saved['meaning'].role_refs[0]==a and saved['meaning'].role_refs[2]==c


def test_open_reference_kinds_and_bound_surface():
    model,registry,store,(a,b,c)=world()
    q=registry.form('isPart',a,c)
    assert open_slots(q)==(('evidence',-1),)
    bound=replace(with_slots(q,(),pair=(1.,0.)),mode='interrogative')
    assert not open_slots(bound)
    assert not run(model,bound).records
    relation=question(replace(q,role_refs=(a,None,c)),(('relation',1),))
    assert open_slots(relation)==(('relation',1),)
    referent=registry.form('isPart',a,open_roles=('I2',))
    assert open_slots(referent)==(('referent',2),)


def test_conclude_gate_and_exhaustion_question(monkeypatch):
    model,registry,store,(a,b,c)=world()
    def illegal(*args,**kwargs):return None
    monkeypatch.setattr(model,'_choose_selected_thought_action',illegal)
    with pytest.raises(ValueError,match='conclude'):
        run(model,registry.form('isPart',a,c),work_budget=16)
    model._end_finished_selected_thought_episodes()
    result=run(model,registry.form('isPart',a,c),work_budget=0)
    assert open_slots(result.meaning)
    assert store.row(len(store)-1)['kind']=='question'


def test_thought_uses_existing_compose_scorer_and_owner_credit():
    model,registry,store,(a,b,c)=world()
    model.train()
    model.errors.clear()
    result=run(model,registry.form('equal',a,a),work_budget=20,
               score=lambda result:float(bool(open_slots(result.meaning))))
    assert model._selected_thought_chooser(result.meaning) is model.shared_grammar
    assert not hasattr(model,'selected_thought_choosers')
    assert model._last_thought_comparison is not None
    value=model._last_thought_score_function['surrogate']
    if value is not None:
        value.backward()
        assert any(p.grad is not None for p in model.shared_grammar.chooser.parameters())


def test_credit_sign_and_exact_tie():
    from ThoughtCredit import surrogate
    parameter=torch.nn.Parameter(torch.tensor([.2,-.1]))
    probability=parameter.softmax(0)
    trace=dict(eligible=[True,True])
    other=dict(choices=[dict(probability=probability[1],alternatives=1)])
    value=surrogate(trace,other,0,(2.,.1))
    value.backward()
    assert parameter.grad[1]<0 and parameter.grad[0]>0
    assert surrogate(trace,other,0,(1.,1.)) is None


def test_not_exchanges_both_poles_and_preserves_content():
    model,registry,store,(a,b,c)=world()
    source=with_slots(registry.form('part',a,b,mode='assertive'),(),pair=(.3,.8))
    with model._query_boundary_scope((0,)):
        from QueryWork import QueryWorkBudget
        work=QueryWorkBudget(32)
        request=registry.form('not',source)
        context=model._thought_grammar_context(request,row=0,work=work,continuation=None)
        result=registry.execute(request,context)
    assert result.evidence['support_true']==.8 and result.evidence['support_false']==.3
    assert torch.equal(result.evidence['meaning'].roles,source.roles)


def test_operations_all_declare_pairs_and_ast_never_returns_scalar():
    expected={'ask','query','not','isTrue','exist','isPart','part','isEqual','equal','isImplied','implies','gain'}
    assert set(THOUGHT_EXECUTORS)==expected
    assert all(item.evidence_pair for item in THOUGHT_EXECUTORS.values())
    source=Path('bin/ThoughtFaces.py').read_text()
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Return):
            assert not isinstance(node.value,(ast.Constant,ast.BinOp,ast.UnaryOp))
            assert not (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                        and node.value.func.id in ('float','int','bool'))


def test_nested_ask_returns_the_binding_and_shares_budget(monkeypatch):
    model,registry,store,(a,b,c)=world()
    model.conceptualSpace.add_whole(a[1],b)
    child=registry.form('isPart',a,b)
    outer=registry.form('ask',child)
    def choose(root,active,actions,**kwargs):
        if None in actions:return None
        name='ask' if kwargs['level']==0 else 'isPart'
        return next(action for action in actions if action.semantic_id==name and not action.open_roles)
    monkeypatch.setattr(model,'_choose_selected_thought_action',choose)
    result=run(model,outer,work_budget=64)
    assert not open_slots(result.meaning)
    assert evidence_pair(result.meaning)==(1.,0.)
    assert result.work.spent==model._what_memory().thought_state().work_spent
    assert [r.kind for r in result.records].count('descend')==1
    assert [r.kind for r in result.records].count('return')==1
    assert bindings(result.meaning)['_thought_witnesses']


def test_negative_image_concludes_absence_with_confidence(monkeypatch):
    from ThoughtClosing import absence
    from Layers import MeaningExpectation
    model,registry,store,(a,b,c)=world()
    predicted=registry.form('part',a,b,mode='assertive')
    roles=torch.ones_like(predicted.roles)
    prior=MeaningExpectation(roles, torch.zeros(3))
    observed=replace(predicted,roles=torch.zeros_like(roles))
    model.expectation_gain=.6
    model._closing_images={0:ClosingImage.form(observed.roles,roles,
        torch.full((3,),.5),gain=.6,form_width=0)}
    source=store.append_meaning(predicted,kind='observation',evidence=(1.,0.))
    witness=store.occurrence_of(source)
    model.symbolSpace.expectation=SimpleNamespace(last_expectation_comparison=lambda row:SimpleNamespace(estimate=prior,source_occurrences=(witness,)))
    index=absence(model,0,observed)
    assert index==1 and store.row(index)['kind']=='inference'
    assert store.row(index)['evidence']==pytest.approx((0.,.3))
    assert bindings(store.meaning_of(index))['_thought_witnesses']==(witness,)
    model._closing_images={0:ClosingImage.form(roles,roles,
        torch.full((3,),.5),gain=.6,form_width=0)}
    assert absence(model,0,predicted) is None


def test_modus_ponens_requires_premise_and_implication_witnesses(monkeypatch):
    model,registry,store,(p,q,_)=world()
    premise=replace(ConceptualMeaning.from_description(registry._payload(p)),role_refs=(p,None,None))
    store.append_meaning(premise,kind='fact',evidence=(.8,.2))
    store.append_meaning(registry.form('implies',p,q,mode='assertive'),kind='fact',
        rel_type=store.REL_IMPLIES,evidence=(.9,0.))
    chosen=[]
    def choose(root,active,actions,**kwargs):
        if None in actions:return None
        for action in actions:
            if action.semantic_id=='query' and action.request.constituents:
                relation=bindings(action.request.constituents[0]).get('_query_relation')
                if relation==('truth' if not chosen else 'implies'):
                    chosen.append(action)
                    return action
        raise AssertionError('missing modus ponens query')
    monkeypatch.setattr(model,'_choose_selected_thought_action',choose)
    result=run(model,registry.form('isImplied',p,q),work_budget=64)
    assert len(chosen)==2
    assert evidence_pair(result.meaning)==pytest.approx((.8,0.))
    assert set(bindings(result.meaning)['_thought_witnesses'])=={store.occurrence_of(0),store.occurrence_of(1)}
    assert store.row(2)['kind']=='inference'


def test_expectation_only_credit_uses_held_next_sentence_cost():
    from ThoughtCredit import PendingCredit, observe
    from Layers import MeaningExpectation
    from Occurrence import source_at
    model,registry,store,(a,b,c)=world()
    model.errors.clear()
    parameter=torch.nn.Parameter(torch.tensor([.2,-.1]))
    probability=parameter.softmax(0)
    trace=dict(eligible=[True,True])
    other=dict(choices=[dict(probability=probability[1],alternatives=1)])
    target=registry.form('part',a,b,mode='assertive')
    first=MeaningExpectation(torch.zeros_like(target.roles), torch.zeros(3))
    second=MeaningExpectation(target.roles,torch.zeros(3))
    doc=source_at(model,0,0)[0]
    model._pending_thought_credit={0:PendingCredit(trace,other,0,(0.,0.),(first,second),doc)}
    observe(model,[target],sentence=1)
    cost=model._last_thought_score_function
    assert cost['costs'][1]<cost['costs'][0]
    cost['surrogate'].backward()
    assert parameter.grad[1]<0 and parameter.grad[0]>0
    assert not model._pending_thought_credit


def credit_chain_world(monkeypatch):
    """One binary departure; the rest is the same metered serial mechanism."""
    from ThoughtStream import query_pattern, slots
    from Queries import ThoughtOperationCandidate
    model, registry, store, (a,b,c) = world()
    for left,right in ((a,b),(b,c)):
        store.append_meaning(registry.form('part',left,right,mode='assertive'),
            kind='fact', rel_type=store.REL_PARTOF, evidence=(1.,0.))
    goal=registry.form('isPart',a,c)
    def candidate(name, meaning):
        return ThoughtOperationCandidate(registry.operation_spec(name),meaning,())
    direct=candidate('isPart',goal)
    first=candidate('query',registry.form('query',query_pattern(goal,'part',a,None)))
    second=candidate('query',registry.form('query',query_pattern(goal,'part',b,c)))
    def menu(registry,root,active,current,records,descriptions=()):
        if not open_slots(current):return ()
        results=slots(records)
        if not results:return direct,first
        return (second,) if any(r.evidence.get('frames') for r in results) else (direct,)
    monkeypatch.setattr('ThoughtStream.candidates',menu)
    with torch.no_grad():
        model.shared_grammar.chooser.mlp[-1].weight.zero_()
        model.shared_grammar.chooser.mlp[-1].bias.zero_()
    model.train()
    model.errors.clear()
    return model,registry,store,goal,(direct.request,first.request)


def test_answer_credit_moves_shared_chooser_toward_the_completed_chain(monkeypatch):
    model,registry,store,goal,menu=credit_chain_world(monkeypatch)
    chooser=model.shared_grammar
    before=chooser.thought_logits(goal,menu).softmax(-1)[0,1].detach()
    result=run(model,goal,work_budget=64,
        score=lambda result:float(bool(open_slots(result.meaning))))
    assert model._last_thought_comparison['explore_kept']
    assert [r.operation for r in result.records if r.kind=='thought']==['query','query','conclude']
    assert not open_slots(result.meaning)
    value=model._last_thought_score_function['surrogate']
    assert value is not None
    optimizer=torch.optim.SGD(chooser.parameters(),lr=.1)
    optimizer.zero_grad();value.backward();optimizer.step()
    assert chooser.thought_logits(goal,menu).softmax(-1)[0,1]>before
    assert len(store)==3 and store.row(2)['kind']=='inference'


def test_expectation_credit_moves_shared_chooser_toward_the_completed_chain(monkeypatch):
    from Layers import BracketExpectation
    from ThoughtCredit import observe, forecast
    from test_sentence_expectation import observe as expose
    model,registry,store,goal,menu=credit_chain_world(monkeypatch)
    # Equal work prices leave only the following observation as policy credit.
    model.WHAT_STEP_COST=0.
    discourse=BracketExpectation(4,8,8,concept_dim=8,expectation_scope='structured')
    model.symbolSpace.expectation=discourse
    with torch.no_grad():
        head=discourse._inter_predictor.network[-1]
        head.weight[8::9].zero_();head.bias[8::9].zero_()
    expose(discourse,goal.roles*.25)
    before=model.shared_grammar.thought_logits(goal,menu).softmax(-1)[0,1].detach()
    run(model,goal,work_budget=64)
    held=model._pending_thought_credit[0]
    assert held.costs==(0.,0.)
    assert not torch.equal(held.forecasts[0].roles,held.forecasts[1].roles)
    target=ConceptualMeaning(held.forecasts[1].roles.detach(),
        torch.ones(3,dtype=torch.bool),
        sentence_kind='relation' if float(held.forecasts[1].kind_logit)>0 else 'idea')
    choice=held.other['choices'][held.departure]
    before=model.shared_grammar.thought_logits(choice['active'],choice['requests'])[0].detach()
    observe(model,[target],sentence=1)
    value=model._last_thought_score_function['surrogate']
    optimizer=torch.optim.SGD(model.shared_grammar.parameters(),lr=.1)
    optimizer.zero_grad();value.backward();optimizer.step()
    assert model._last_thought_score_function['costs'][1]<model._last_thought_score_function['costs'][0]
    after = model.shared_grammar.thought_logits(choice['active'],choice['requests'])[0]
    assert (after[1]-after[0]) > (before[1]-before[0])


def test_pair_argument_and_conceptual_extent_keep_both_lanes():
    model,registry,store,(a,b,c)=world()
    from QueryWork import QueryWorkBudget
    with model._query_boundary_scope((0,)):
        q=registry.form('not',(.2,.7))
        result=registry.execute(q,model._thought_grammar_context(q,row=0,
            work=QueryWorkBudget(16),continuation=None))
    assert (result.support_true,result.support_false)==pytest.approx((.7,.2))
    from Queries import ThoughtConceptualCapability
    from reasoning import TruthGroundedReasoner
    cap=ThoughtConceptualCapability(model.conceptualSpace,TruthGroundedReasoner.equal)
    object.__setattr__(cap,'_ThoughtConceptualCapability__form_width',4)
    value=torch.tensor([1.,0.,0.,0.,.3,0.,.8,0.])
    assert cap.extent(value)['support_true']==pytest.approx(.3)
    assert cap.extent(value)['support_false']==pytest.approx(.8)
    torch.testing.assert_close(cap.negate(value),torch.tensor([1.,0.,0.,0.,.8,0.,.3,0.]))


@pytest.mark.parametrize('surface', ['?', 'who', 'which', 'unicorn'])
def test_open_closing_and_later_binding_are_reference_driven(surface):
    from ThoughtClosing import closing_question, close
    from ClauseRow import Clause
    from Understanding import SentenceEndState
    model,registry,store,(a,b,c)=world()
    source=ConceptualMeaning.from_description(torch.arange(8.).float())
    clause=Clause(source,point=source.roles[0],evidence=(0.,0.))
    parsed=replace(source,mode='interrogative') if surface!='unicorn' else None
    q=closing_question(source,clause,parsed,SimpleNamespace(lexical_forms=(surface,)))
    assert q is not None and open_slots(q)
    model.attention_budget=0
    model._sentence_fields={}
    fields=[SentenceEndState(source,query=q)]
    with model._query_boundary_scope((0,)):
        close(model,fields,sentence=0,ready=(0,))
    assert store.row(0)['kind']=='question'
    assert len(model._last_closing_thoughts)==1
    # A new occurrence can fill the earlier null. Its source address remains
    # distinct from the question's address and from the new inference.
    supplied=with_slots(replace(source,role_refs=(a,None,None)),(),pair=(1.,0.))
    index=store.append_meaning(supplied,kind='observation',evidence=(1.,0.),sentence_index=1)
    fields=[SentenceEndState(supplied,row_id=int(store.row_ids[index]),evidence=(1.,0.))]
    model._open_sentence_slot=1
    with model._query_boundary_scope((0,)):
        close(model,fields,sentence=1,ready=(0,))
    assert model._last_closing_thoughts==()
    inference=store.row(2)
    assert inference['kind']=='inference' and not open_slots(inference['meaning'])
    assert set(bindings(inference['meaning'])['_thought_witnesses'])=={store.occurrence_of(0),store.occurrence_of(1)}
    # Reusing the same surface after a successful binding opens no episode.
    bound_clause=Clause(supplied,point=supplied.roles[0],refs=(a[1],-1,-1),evidence=(1.,0.))
    assert closing_question(supplied,bound_clause,replace(supplied,mode='interrogative'),
        SimpleNamespace(lexical_forms=(surface,))) is None


def test_document_end_flushes_answer_work_credit_without_a_future_target():
    from ThoughtCredit import PendingCredit, finish_documents
    model,registry,store,(a,b,c)=world()
    parameter=torch.nn.Parameter(torch.zeros(2))
    probability=parameter.softmax(-1)[1]
    model._pending_thought_credit={0:PendingCredit(dict(eligible=[True]),
        dict(choices=[dict(probability=probability,alternatives=1)]),0,(1.,.5),(),('doc',))}
    finish_documents(model,(0,))
    assert not model._pending_thought_credit
    model._last_thought_score_function['surrogate'].backward()
    assert parameter.grad[1]<0


@pytest.mark.parametrize('positions,known', [((1,2),True),((2,1),False)])
def test_event_implication_reads_endpoint_when_order(monkeypatch,positions,known):
    model,registry,store,(a,b,c)=world()
    events=[]
    for ref,position in zip((a,b),positions):
        value=replace(ConceptualMeaning.from_description(registry._payload(ref)),role_refs=(ref,None,None))
        index=store.append_meaning(value,kind='fact',evidence=(.8,.1),document_key='events',sentence_index=position)
        events.append(store.occurrence_of(index))
    rule=registry.form('implies',a,b,mode='assertive')
    rule=replace(rule,role_refs=(events[0],rule.role_refs[1],events[1]))
    store.append_meaning(rule,kind='fact',rel_type=store.REL_IMPLIES,evidence=(1.,0.))
    goal=registry.form('isImplied',a,b)
    goal=replace(goal,role_refs=(events[0],goal.role_refs[1],events[1]))
    def choose(root,active,actions,**kwargs):
        return None if None in actions else next(action for action in actions if action.semantic_id=='isImplied')
    monkeypatch.setattr(model,'_choose_selected_thought_action',choose)
    result=run(model,goal,work_budget=24)
    assert (not open_slots(result.meaning)) is known
    assert store.row(len(store)-1)['kind']==('inference' if known else 'question')
    if known:
        assert evidence_pair(result.meaning)==pytest.approx((.8,0.))
        assert set(events).issubset(bindings(result.meaning)['_thought_witnesses'])


def test_symbolic_open_neighbor_binds_its_reference_and_conceptual_face_cannot_read(monkeypatch):
    model,registry,store,(a,b,c)=world()
    model.conceptualSpace.add_whole(a[1],b)
    goal=registry.form('isPart',a,open_roles=('I2',))
    def choose(root,active,actions,**kwargs):
        return None if None in actions else next(action for action in actions
            if action.semantic_id=='isPart' and action.open_roles==('I2',))
    monkeypatch.setattr(model,'_choose_selected_thought_action',choose)
    result=run(model,goal,work_budget=32)
    assert not open_slots(result.meaning) and result.meaning.role_refs[2]==b
    assert evidence_pair(result.meaning)==(1.,0.)
    assert bindings(result.meaning)['_thought_witnesses']
    with pytest.raises(ValueError,match='open'):
        registry.form('part',a,open_roles=('I2',))
    descriptor=THOUGHT_EXECUTORS['part']
    assert not any(name in ('ltm','taxonomy') for name,_ in descriptor.method_grants)


@pytest.mark.parametrize('mode', ['bind', 'pronoun'])
def test_null_binder_candidate_survives_native_journal_and_closing(mode):
    from ReferenceContext import ReferenceBank, prepare_operands
    from test_sentence_references import language_and_program
    from reading_fixtures import finish_reading
    from ThoughtClosing import closing_question
    language, entry = language_and_program(mode)
    window=entry.leaves[None]
    ids=entry.concept_ids[None]
    bank=ReferenceBank(ids.new_empty(1,0),window.new_empty(1,0,4),
        torch.zeros(1,0,dtype=torch.bool),torch.zeros(1,0,dtype=torch.bool),
        window[:,0],torch.zeros(1,dtype=torch.bool),column_ids=ids.new_empty(0))
    proposed=prepare_operands(window,ids,torch.zeros_like(ids),torch.arange(2)[None],
        rules=language._compose_binary_rules,unary_rules=(),bank=bank,live=None,
        active=torch.ones(1,2,dtype=torch.bool))
    assert proposed['binary_valid'][0,0,0]
    assert proposed['binary_refs'][0,0,0,0] == -1
    refs=proposed['binary_refs'][0,0,0]
    entry=replace(entry,reference_ids=refs,reference_values=torch.stack((
        proposed['left'][0,0,0],proposed['right'][0,0,0])))
    clause=finish_reading(language,entry)
    assert clause.refs[0] == -1
    assert ('referent',0) in open_slots(clause.meaning)
    assert closing_question(clause.meaning,clause,program=entry) is not None


def test_thought_nested_writer_is_restricted_and_preflights_capacity():
    from ThoughtStream import write
    model,registry,store,(a,b,c)=world()
    child=registry.form('isPart',a,b)
    parent=registry.form('ask',child)
    write(model,parent,row=0)
    assert len(store)==2
    assert all(store.row(i)['kind']=='question' for i in range(len(store)))
    assert store.meaning_of(1).role_refs[0]==store.occurrence_of(0)
    store.reset()
    store.capacity=1
    with pytest.raises(OverflowError,match='forgetting'):
        write(model,parent,row=0)
    assert len(store)==0
