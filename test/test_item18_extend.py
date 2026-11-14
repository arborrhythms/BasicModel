"""The extension certificates: operators act on sets, not code similarities."""
from types import SimpleNamespace
import pytest
import torch
from test_item18_finish import named_scalar_model


def sets(*values):
    from Extension import ConceptSet
    return tuple(ConceptSet(torch.arange(1, len(v)+1), torch.tensor(v, dtype=torch.float32))
                 for v in values)


def test_34_each_shipped_structural_operator_has_both_extension_faces():
    from pathlib import Path
    from Language import Grammar, GRAMMAR_LAYER_CLASSES
    for path in Path('data').rglob('*.grammar'):
        grammar = Grammar()
        grammar.load_from_grammar_file(str(path.resolve()))
        for rule in grammar.rules:
            cls = GRAMMAR_LAYER_CLASSES.get(rule.method_name)
            if cls is not None:
                assert callable(getattr(cls, 'compose_extend', None)), (path, rule)
                assert callable(getattr(cls, 'generate_extend', None)), (path, rule)


def test_34_missing_extension_face_fails_at_load(monkeypatch):
    from Language import Grammar, IntersectionLayer
    monkeypatch.setattr(IntersectionLayer, 'generate_extend', None, raising=False)
    with pytest.raises(ValueError, match='generate_extend'):
        Grammar().load_from_grammar_file('field.grammar')


@pytest.mark.parametrize('name,expected', [
    ('intersection', [[.4,.7],[0.,.2]]),
    ('union', [[.8,.1],[1.,0.]]),
    ('not', [[.1,.4],[0.,1.]]),
    ('non', [[0.,0.],[0.,0.]]),
    ('conjunction', [[.4,.7],[0.,.2]]),
])
def test_35_operator_sets_use_the_common_two_lane_logic(name, expected):
    from Language import GRAMMAR_LAYER_CLASSES
    a,b = sets([[.4,.1],[1.,0.]], [[.8,.7],[0.,.2]])
    op = GRAMMAR_LAYER_CLASSES[name]()
    inputs = (a,) if name in ('not','non') else (a,b)
    out = op.compose_extend(*inputs)
    torch.testing.assert_close(out.values, torch.tensor(expected))


def test_35_missing_membership_is_zero_in_an_intersection():
    from Extension import ConceptSet
    from Language import IntersectionLayer, UnionLayer
    a = ConceptSet(torch.tensor([1]), torch.tensor([[1.,0.]]))
    b = ConceptSet(torch.tensor([2]), torch.tensor([[1.,0.]]))
    assert not IntersectionLayer().compose_extend(a,b).values.any()
    torch.testing.assert_close(UnionLayer().compose_extend(a,b).values[:,0], torch.ones(2))


def test_36_determiners_read_the_stored_members_and_mtm(monkeypatch):
    from Extension import Context
    from Language import LowerLayer
    from FieldProjection import key
    from Understanding import ConceptualField
    import ConceptHierarchy
    monkeypatch.setattr(ConceptHierarchy,'members',lambda owner,cid: (1,2,3) if cid == 4 else ())
    codes = torch.eye(3)
    owner = SimpleNamespace(_csw_row_of=lambda cid: cid-1,
        similarity_codebook=SimpleNamespace(lookup_rows=lambda rows: codes[rows]))
    kind, = sets([[0.,0.],[0.,0.],[0.,0.],[1.,0.]])
    op = LowerLayer()
    results = {}
    for name in ('universal','lower','some','definite'):
        context = Context(owner=owner, name=name, identity=('sym',2), trust=.7)
        results[name] = op.compose_extend(kind, context=context)
    torch.testing.assert_close(results['universal'].values[:,0], torch.ones(3))
    torch.testing.assert_close(results['lower'].values[:,0], torch.full((3,), 1/3))
    assert int(results['some'].values[:,0].gt(0).sum()) == 2
    definite = results['definite']
    assert definite.ids[definite.values[:,0] > 0].tolist() == [2]
    assert float(definite.values[definite.ids == 2,0]) == pytest.approx(.7)
    absent = op.compose_extend(kind,context=Context(owner=owner,name='definite'))
    assert not absent.values.any()
    generic = results['lower']
    field = ConceptualField(generic.ids,generic.values[:,None,None],
                            torch.zeros(1,1,2),torch.zeros(1,1,2),None)
    torch.testing.assert_close(key(owner,field)[0,0], codes.mean(0))


def test_36_some_is_declared_in_the_complete_grammar():
    from Language import Grammar
    grammar=Grammar(); grammar.load_from_grammar_file('complete.grammar')
    assert any(rule.surface_name == 'some' for rule in grammar.rules_upward)
    assert any(rule.surface_name == 'some' for rule in grammar.rules_downward)


def test_36_definite_reads_the_held_identity_and_both_its_lanes(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from Extension import held_context,ConceptSet
    f=SentenceFixture(monkeypatch)
    members=tuple(f.noun(word) for word in ('a','b','c'))
    from ConceptIndex import index_part_row
    from Memory import SYMBOL
    common=f.noun('shared')
    kinds=[index_part_row(f.cs,(SYMBOL,cid),(SYMBOL,common)) for cid in members]
    assert len(set(kinds)) == 1
    kind=kinds[0]
    rule=next(r for r in f.binary if r.surface_name == 'definite')
    op=f.operation(rule)
    source=ConceptSet(torch.tensor([kind]),torch.tensor([[1.,0.]]))
    reference=('sym',members[1])
    assert not op.compose_extend(source,context=held_context(f.cs,rule,reference)).values.any()
    token=f.cs.mtm.open(reference,kind=kind,native=members[1],when=1,
        value=f.registry._payload(reference))
    f.cs.mtm.record_reading(token,0,(.7,.2))
    result=op.compose_extend(source,context=held_context(f.cs,rule,reference))
    assert result.ids[result.values.any(-1)].tolist() == [members[1]]
    torch.testing.assert_close(result.values[result.ids == members[1]],torch.tensor([[.7,.2]]))


def test_35_native_compose_journal_extends_in_order_without_stimulate(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from Extension import reading
    from Understanding import ConceptualField
    from Language import AdverbLayer
    import FieldProjection
    f=SentenceFixture(monkeypatch)
    f.ops['adverb']=AdverbLayer(f.cs.nDim,f.cs.nDim)
    program=f.program(('verb',('adverb','subject','modifier'),'predicate'))
    calls=[]
    for name in ('adverb','verb'):
        op=f.ops[name]
        original=op.compose_extend
        def checked(*args,_name=name,_original=original,**kwargs):
            calls.append(_name)
            return _original(*args,**kwargs)
        monkeypatch.setattr(op,'compose_extend',checked)
    def forbidden(*args,**kwargs):
        raise AssertionError('compose_extend must read the existing field')
    monkeypatch.setattr(FieldProjection,'stimulate',forbidden)
    model=SimpleNamespace(_concept_owner=lambda:f.cs,languageSpace=f.language)
    field=ConceptualField(program.concept_ids,
        torch.tensor([[[[.9,0.],[.8,0.],[.7,0.]]],
                      [[[.8,0.],[.4,0.],[0.,0.]]],
                      [[[1.,0.],[0.,0.],[1.,0.]]]]),
        torch.tensor([[[0,7],[8,16],[17,26]]]),torch.tensor([[[0,1],[0,1],[0,1]]]),None)
    root,=reading(model,program,field,0,torch.arange(3))
    assert calls == ['adverb','verb']
    torch.testing.assert_close(root.field.values[:,0],torch.tensor([.7,0.,0.]))


def recall_tree(name, *children):
    from Extension import Node,Context
    from Language import GRAMMAR_LAYER_CLASSES
    op=GRAMMAR_LAYER_CLASSES[name]()
    context=Context(name=name)
    return Node(op.compose_extend(*(child.field for child in children),context=context),children,op,context)


def reference_node(cid, strength=1.):
    from Extension import ConceptSet,Node
    return Node(ConceptSet(torch.tensor([cid]),torch.tensor([[strength,0.]])))


def test_38_query_operators_combine_row_results():
    from test_item18_finish import memory_brick
    from EpisodicMemory import query
    store=memory_brick()
    store.episodic_concepts[:3,0]=10
    store.episodic_values[:3,0]=torch.tensor([1.,0.,1.])
    store.episodic_concepts[:3,1]=20
    store.episodic_values[:3,1]=torch.tensor([0.,1.,1.])
    a,b=reference_node(10),reference_node(20)
    def rows(root):
        return [frame['index'] for frame in query(store,(root,),max_records=8)['value']]
    assert rows(recall_tree('union',a,b)) == [2,1,0]
    assert rows(recall_tree('intersection',a,b)) == [2]
    assert rows(recall_tree('not',recall_tree('union',a,b))) == []
    store.episodic_concepts[3,0]=10
    store.episodic_values[3,0]=-1.
    assert rows(recall_tree('not',a)) == [3]


def test_39_open_operand_reads_only_the_same_generate_operator_path():
    from Extension import ConceptSet,Context,Node
    from Language import AskLayer,UnionLayer,IntersectionLayer
    a,b=reference_node(10),reference_node(20)
    ask=AskLayer()
    context=Context(name='ask',open=True)
    hole=Node(ask.compose_extend(b.field,context=context),(b,),ask,context)
    question=recall_tree('union',a,hole)
    recalled=recall_tree('union',a,b)
    path,=question.open_paths()
    assert path == (('union',1),)
    found=recalled.operand(path)
    assert found.ids[found.values[:,0]>0].tolist() == [20]
    assert recall_tree('intersection',a,b).operand(path) is None
    # The open operand contributes no constraint; non is a real withheld
    # operand and must not be mistaken for that missing constraint.
    assert question.field.ids[question.field.values[:,0]>0].tolist() == [10]
    assert not recall_tree('intersection',a,recall_tree('non',b)).field.values.any()


def test_39_cued_rows_preserves_extension_relevance_without_role_matching():
    import inspect
    from EntryRead import EntryRead
    from test_item18_finish import memory_brick
    from Meaning import ConceptualMeaning
    store=memory_brick()
    # Opposite vectors do not alter the relevance that the question's
    # operators already computed. Nor does an explicit address grant it one.
    frame=dict(store.row(0),index=0,match=.7)
    cue=ConceptualMeaning.from_description(-torch.ones(12))
    result=store.cued_rows(cue,episodic_frames=(frame,),max_records=8)
    assert result['value'][0]['match'] == .7
    assert 'normalize' not in inspect.getsource(EntryRead.cued_rows)
    assert '_cue_roles' not in inspect.getsource(EntryRead.cued_rows)


def test_39_sentence_object_is_the_generated_operand_and_not_the_witness_row(monkeypatch):
    from dataclasses import replace
    from test_clause_acceptance import SentenceFixture
    from Extension import ConceptSet,Context,Node,generated,held_context,stored
    from ExtendedRecall import answer
    from Language import AskLayer
    from ThoughtReferences import fill,with_slots,open_slots
    from Understanding import ConceptualField
    f=SentenceFixture(monkeypatch)
    clause=f.clause(('lift','I',('verb','ate','rice')))
    subject,predicate,obj=(f.noun(word) for word in ('I','ate','rice'))
    # The predicate's declared extension includes the object. Nothing is
    # inferred from a cosine or from a surface word in the implementation.
    f.cs._concept_allocator.store_of(predicate).add_constituent(predicate,'part',('sym',obj))
    row=f.store.write_clause(clause,trust=.7)
    retained=f.store.meaning_of(row)
    assert not retained.extension
    binary=tuple(r for r in f.grammar.rules_downward if ',' in r.lhs and r.space_role == 'CS')
    unary=tuple(r for r in f.grammar.rules_downward if ',' not in r.lhs)
    f.language._generate_binary_rules=binary;f.language._generate_unary_rules=unary
    f.language._generate_binary_ops=tuple(f.operation(r) if r.method_name in ('lift','verb') else None for r in binary)
    f.language._generate_unary_ops=(None,)*len(unary)
    lift=next(i for i,r in enumerate(binary) if r.method_name == 'lift')
    verb=next(i for i,r in enumerate(binary) if r.method_name == 'verb')
    stop=len(binary)+len(unary)
    # A controlled independent generate trace. Certificate 37 separately
    # exercises acquisition of this trace through the native generate walk.
    ids=torch.tensor([subject,predicate,obj])
    leaves=ConceptualField(ids,torch.eye(3)[:,None,:,None]*torch.tensor([1.,0.]),
        torch.tensor([[[0,1],[2,5],[6,10]]]),torch.tensor([[[1,2],[1,2],[1,2]]]),None)
    model=SimpleNamespace(_concept_owner=lambda:f.cs,languageSpace=f.language)
    trees,_field=generated(model,torch.tensor([[lift,verb,stop,stop,stop]]),torch.tensor([3]),
        torch.tensor([1]),leaves,
        ((('sym',subject),('sym',predicate),('sym',obj)),))
    root,=trees[0]
    predicate_node=root.children[1].children[0]
    object_node=root.children[1].children[1]
    ask=AskLayer();context=Context(name='ask',open=True)
    hole=Node(ask.compose_extend(object_node.field,context=context),(object_node,),ask,context)
    vp=root.children[1]
    question_vp=replace(vp,children=(predicate_node,hole),field=vp.operator.compose_extend(
        predicate_node.field,hole.field,context=vp.context))
    question_tree=replace(root,children=(root.children[0],question_vp),field=root.operator.compose_extend(
        root.children[0].field,question_vp.field,context=root.context))
    request=with_slots(replace(clause.meaning,role_refs=(('sym',subject),('sym',predicate),None),
        extension=(question_tree,)),(('referent',2),),pair=(0.,0.))
    result=answer(f.cs,request,replace(retained,extension=trees[0]))
    assert result is not None and result.role_refs[0] == ('sym',obj)
    assert result.extension[0].field.ids[result.extension[0].field.values[:,0]>0].tolist() == [obj]
    bound=fill(request,dict(meaning=result,answer_operand=result,frames=(f.store.row(row),),
        support_true=.7,support_false=0.),witnesses=(f.store.occurrence_of(row),),operation='query')
    assert not open_slots(bound)
    assert bound.role_refs[2] == ('sym',obj)
    assert bound.role_refs[2] != f.store.occurrence_of(row)
    torch.testing.assert_close(bound.roles[2],f.registry._payload(('sym',obj)))


def test_40_signed_existence_is_maximum_and_analogy_is_separate():
    from Memory import LTM0
    from Extension import ConceptSet
    from ExtendedRecall import existence
    store=LTM0(12,capacity=8,percept_width=12)
    for i,trust in enumerate((.3,.4,-.6,0.)):
        store.append_idea(torch.ones(12),trust=trust)
    results=ConceptSet(torch.arange(4),torch.tensor([[1.,0.],[.5,0.],[1.,0.],[1.,0.]]))
    luminous,exact=existence(store,results)
    torch.testing.assert_close(luminous,torch.tensor([.3,.6]))
    torch.testing.assert_close(exact,torch.tensor([.3,.6]))
    # Two like examples do not sum into a stronger exact statement.
    results=ConceptSet(torch.tensor([0,1]),torch.tensor([[.5,0.],[.5,0.]]))
    luminous,exact=existence(store,results)
    torch.testing.assert_close(luminous,torch.tensor([.2,0.]))
    assert not exact.any()


@pytest.mark.parametrize('allowance', (128, 1, 0))
def test_40_deduction_carries_exact_premises_and_keeps_analogy_apart(allowance):
    from dataclasses import replace
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    from Queries import ThoughtLTMCapability,QueryWorkBudget
    from test_item18_finish import memory_brick
    owner=memory_brick()._concept_owner()
    store=LTM0(12,capacity=8,percept_width=12,episodic_concepts=4)
    store._concept_owner=lambda:owner
    for i,trust in enumerate((.9,.3,-.6,0.)):
        store.append_idea(torch.ones(12),trust=trust)
    store.episodic_concepts[:,0]=10
    store.episodic_values[:4,0]=torch.tensor([.8,1.,1.,1.])
    question=replace(ConceptualMeaning.from_description(torch.ones(12)),extension=(reference_node(10),))
    reader=ThoughtLTMCapability(store=lambda:store,equal=lambda a,b:0.,tau_id=.5)
    work=QueryWorkBudget(allowance)
    result=reader.best_match(question,max_records=8,work=work)
    if allowance < 128:
        expected = .3 if allowance else 0.
        assert result['support_true'] == pytest.approx(expected)
        assert result['support_false'] == 0.
        assert result['existence'] == pytest.approx((expected,0.))
        assert result['exact'] == pytest.approx((expected,0.))
        assert result['records_scanned'] == work.spent == allowance
        assert result['witnesses'] == ((store.occurrence_of(1),) if allowance else ())
        assert 'work_budget' in result['incomplete']
        return
    assert result['existence'] == pytest.approx((.72,.6))
    assert result['pervasion'] == (0.,0.)
    assert result['support_true'] == pytest.approx(.3)
    assert result['support_false'] == pytest.approx(.6)
    assert result['analogy'] == pytest.approx((.72,.6))
    assert result['witnesses'] == (store.occurrence_of(1),store.occurrence_of(2))


def test_40_pervasion_comes_from_the_kind_relation_not_the_members():
    from test_taxonomy_view import _world
    from Meaning import ConceptualMeaning
    from ExtendedRecall import pervasion
    from QueryWork import QueryWorkBudget
    from order_fixtures import force_order
    from Memory import SYMBOL
    owner,(member,kind,property_kind)=_world()
    alloc=owner._concept_allocator
    force_order(owner,kind,1);force_order(owner,property_kind,1)
    for cid in (kind,property_kind):
        alloc.symbol_types.setdefault(cid,set()).add('class')
    alloc.parts.add(member,kind,truth_plus=.8)
    def check(subject):
        meaning=ConceptualMeaning.from_description(torch.ones(owner.nDim))
        meaning=__import__('dataclasses').replace(meaning,roles=torch.ones(3,owner.nDim),
            role_mask=torch.ones(3,dtype=torch.bool),role_refs=(('sym',subject),None,('sym',property_kind)))
        return pervasion(owner,meaning,work=QueryWorkBudget(1024),max_records=128)
    assert check(member)['support_true'] == 0.
    # Every known member may have its own property statement. None of
    # these is the missing kind-to-property relation.
    more=[member,owner.new_concept(),owner.new_concept()]
    for cid in more[1:]:
        force_order(owner,cid,0);owner._csw_concept_row(0,cid)
        alloc.parts.add(cid,kind,truth_plus=1.)
    for cid in more:
        alloc.parts.add(cid,property_kind,truth_plus=.9)
    assert check(kind)['support_true'] == 0.
    relation=alloc.parts.add(kind,property_kind,truth_plus=.6)
    proof=check(kind)
    assert proof['support_true'] == pytest.approx(.6)
    assert proof['path'][-1].owner == ('sym',relation)
    # A separate member has only the path, at the least of its two edges.
    last=owner.new_concept();force_order(owner,last,0);owner._csw_concept_row(0,last)
    alloc.parts.add(last,kind,truth_plus=.4)
    assert check(last)['support_true'] == pytest.approx(.4)


def test_37_recalled_idea_extends_without_its_stm_sentence(named_scalar_model):
    from ExtendedRecall import forest
    from Extension import as_field,joined,ConceptSet,reading
    from Understanding import Reading,ConceptualField
    from ClauseJournal import finish_clause
    model,objects=named_scalar_model
    owner=model._concept_owner()
    word=owner.interpret.word_for_form('alpha')
    obj=owner.interpret.forward(word)
    point=owner.similarity_codebook.lookup_rows(owner._csw_row_of(obj)).detach()
    # A controlled, fully identified sentence. Its selected interpretation,
    # numerical leaf and perceptual membership all come from the same word.
    # This certifies recall's mechanism, without requiring an untrained free
    # compose policy to rediscover the known interpretation.
    program=Reading(rows=torch.tensor([owner._csw_row_of(obj)]),
        word_rows=torch.tensor([owner._csw_row_of(word)]),word_ids=torch.tensor([word]),
        concept_ids=torch.tensor([obj]),activations=torch.ones(1),leaves=point[None],
        actions=torch.tensor([[0,-1,0]]),targets=torch.tensor([0]),
        end_state=torch.stack((point,point*0,point*0)),
        operation_values=point.new_zeros(1,3,len(point)),leaf_orders=torch.tensor([0]),
        leaf_evidence=torch.tensor([[1.,0.]]))
    program=model.conceptualSpace.stm.strata.capture(program,model.languageSpace,0)
    clause=finish_clause(model.languageSpace,program,registry=model.grammatical_thoughts)
    source=ConceptualField(torch.tensor([word]),torch.tensor([[[[1.,0.]]]]),
        torch.tensor([[[0,5]]]),torch.tensor([[[1,2]]]),None)
    expected=reading(model,program,source,0,torch.tensor([0]))[0].field
    store=model.symbolSpace.ltm_store
    row=store.write_clause(clause,trust=.7)
    recalled=store.meaning_of(row)
    assert not recalled.extension
    model.conceptualSpace.stm.strata.clear()
    with torch.no_grad():
        roots=forest(owner,recalled)
    empty=ConceptSet(expected.ids.new_empty(0),expected.values.new_empty(0,2))
    result=joined(roots,empty)
    ids=torch.cat((expected.ids,result.ids)).unique(sorted=True)
    torch.testing.assert_close(result.aligned(ids),expected.aligned(ids))


@pytest.mark.parametrize('operation,trust', [('part',.7),('equal',.7),('part',-.6),('equal',-.6)])
def test_56_kept_relative_tree_closes_in_both_relation_stores(monkeypatch,operation,trust):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import record_reading
    from SentenceStrata import SentenceStrata
    from ClauseJournal import finish_clause
    from ExtendedRecall import pervasion
    from Queries import QueryWorkBudget
    from Meaning import ConceptualMeaning
    from dataclasses import replace
    from order_fixtures import force_order
    from Memory import SYMBOL
    f=SentenceFixture(monkeypatch)
    program=record_reading(f.language,f.program((operation,'cats','furry-things')))
    kind,prop=map(int,program.concept_ids)
    for cid in (kind,prop):
        force_order(f.cs,cid,1)
        f.cs._concept_allocator.symbol_types.setdefault(cid,set()).add('class')
    strata=SentenceStrata(1,8,8,f.cs.nDim,relation_capacity=f.store.parts.capacity)
    program=strata.capture(program,f.language,0)
    clause=finish_clause(f.language,program,registry=f.registry)
    written={}
    row=f.store.write_clause(clause,trust=trust,written_rows=written)
    assert isinstance(row,tuple) and row[0] == 'sym'
    assert len(f.store) == 0
    assert f.store.parts.references(row[1]) == ((SYMBOL,kind),(SYMBOL,prop))
    expected=(max(trust,0.),max(-trust,0.))
    assert (float(f.store.parts.truth_plus[row[1]]),float(f.store.parts.truth_minus[row[1]])) == pytest.approx(expected)
    strata.keep_close(0,program.end_state[0],torch.tensor([1.,0.,0.,1.]),
        relations=written['relations'],source=f.store.parts)
    held=strata.recent_stores[0].parts
    assert held.contains(row[1])
    assert held.references(row[1]) == f.store.parts.references(row[1])
    assert (float(held.truth_plus[row[1]]),float(held.truth_minus[row[1]])) == pytest.approx(expected)
    if operation == 'part' and trust > 0:
        member=f.noun('Felix')
        f.store.parts.add(member,kind,truth_plus=.8)
        question=replace(clause.meaning,role_refs=(('sym',member),clause.meaning.role_refs[1],('sym',prop)))
        proof=pervasion(f.cs,question,work=QueryWorkBudget(1024),max_records=128)
        assert proof['support_true'] == pytest.approx(.7)


def test_public_existence_reads_the_question_extension_and_signed_trust():
    from dataclasses import replace
    from Memory import LTM0
    from Meaning import ConceptualMeaning
    from reasoning import TruthGroundedReasoner, QuerySpec
    from test_item18_finish import memory_brick
    from test_item18_extend import reference_node
    owner = memory_brick()._concept_owner()
    store = LTM0(12,capacity=8,percept_width=12,episodic_concepts=4)
    store._concept_owner = lambda: owner
    for trust in (.8,-.6,0.):
        store.append_idea(torch.zeros(12),trust=trust)
    store.episodic_concepts[:3,0]=10
    store.episodic_values[:3,0]=torch.tensor([.5,1.,1.])
    question=replace(ConceptualMeaning.from_description(torch.ones(12)),extension=(reference_node(10),))
    result=TruthGroundedReasoner(store=store).evaluate(QuerySpec.from_surface('exist',question))
    assert result['support_true'] == pytest.approx(.4)
    assert result['support_false'] == pytest.approx(.6)
    assert {item['occurrence'] for item in result['candidates']} == {store.occurrence_of(0),store.occurrence_of(1)}


@pytest.mark.parametrize('form,expected', [('positive',(.6,.2)),('not',(.2,.6)),('non',(0.,0.))])
def test_40_question_form_applies_to_the_higher_order_proof(form, expected):
    from dataclasses import replace
    from test_taxonomy_view import _world
    from order_fixtures import force_order
    from Queries import ThoughtLTMCapability
    from QueryWork import QueryWorkBudget
    from Memory import STM0
    from Meaning import ConceptualMeaning
    owner,(member,kind,prop)=_world()
    alloc=owner._concept_allocator
    for cid in (kind,prop):
        force_order(owner,cid,1)
        alloc.symbol_types.setdefault(cid,set()).add('class')
    alloc.parts.add(member,kind,truth_plus=.8)
    alloc.parts.add(kind,prop,truth_plus=.6,truth_minus=.2)
    store=STM0(owner.nDim,capacity=8)
    store._concept_owner=lambda:owner
    reader=ThoughtLTMCapability(store=lambda:store,equal=lambda *_:0.,tau_id=.5)
    leaf=reference_node(member)
    root=leaf if form=='positive' else recall_tree(form,leaf)
    question=replace(ConceptualMeaning.from_description(torch.ones(owner.nDim)),
        sentence_kind='idea',roles=torch.ones(3,owner.nDim),
        role_mask=torch.ones(3,dtype=torch.bool),
        role_refs=(('sym',member),None,('sym',prop)),extension=(root,))
    result=reader.best_match(question,max_records=128,work=QueryWorkBudget(2048))
    assert result['pervasion'] == pytest.approx(expected)
    assert result['existence'] == (0.,0.)
    assert (result['support_true'],result['support_false']) == pytest.approx(expected)
    if form=='non':
        assert not result['witnesses']


def test_40_thought_negation_transforms_its_existing_extension():
    from dataclasses import replace
    from Meaning import ConceptualMeaning
    from ThoughtFaces import negate
    original=replace(ConceptualMeaning.from_description(torch.ones(8)),
        extension=(reference_node(10, .7),))
    result=negate(None,dict(argin1=original))['meaning']
    torch.testing.assert_close(result.extension[0].field.values,
                               original.extension[0].field.values.flip(-1))
    restored=negate(None,dict(argin1=result))['meaning']
    torch.testing.assert_close(restored.extension[0].field.values, original.extension[0].field.values)
    assert restored.role_refs == original.role_refs


@pytest.mark.parametrize('allowance', [0,1,3])
def test_39_recalled_generation_obeys_the_shared_allowance_before_walking(named_scalar_model, monkeypatch, allowance):
    from Meaning import ConceptualMeaning
    from ExtendedRecall import forest
    from QueryWork import QueryWorkBudget, QueryWorkExhausted
    model,objects=named_scalar_model
    owner=model._concept_owner()
    ids=tuple(objects[name] for name in ('alpha','beta','gamma'))
    roles=owner.similarity_codebook.lookup_rows([owner._csw_row_of(cid) for cid in ids]).detach()
    meaning=ConceptualMeaning(roles,torch.ones(3,dtype=torch.bool),
        sentence_kind='idea',role_refs=tuple(('sym',cid) for cid in ids))
    native=model._compiled_output_walk()
    budgets=[]
    def tracked(event, steps, *args, **kwargs):
        budgets.append(steps)
        return native(event,steps,*args,**kwargs)
    monkeypatch.setattr(model,'_compiled_output_walk',lambda:tracked)
    meter=QueryWorkBudget(allowance)
    with torch.no_grad():
        if allowance < 3:
            with pytest.raises(QueryWorkExhausted):
                forest(owner,meaning,work=meter)
        else:
            result=forest(owner,meaning,work=meter)
            assert len(result) == 3
    assert budgets == ([] if allowance==0 else [allowance])
    assert meter.spent == allowance
    if allowance:
        assert int((model._last_output_walk_trace[0]>=0).sum()) == allowance
