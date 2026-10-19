"""§12 mechanism certificates. No learned grammar or identity gate is claimed."""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLOCKS = (0, 25, 50, 100, 200, 400, 777, 1500, 3000)


def _serial_read(tmp_path, clock=0, offset=0):
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    from reading_fixtures import capture_readings
    from What import What
    # Same serial fixture, ordinary chooser. The older provisioning helper's
    # controlled chooser is installed only for its original pathname.
    config = tmp_path / 'serial.xml'
    config.write_text(Path(_SERIAL_CONFIG).read_text())
    torch.manual_seed(20261010)
    model = _make_model(str(config))
    model.when_time.fill_(clock)
    model._tensor_peer_while_eager = True
    model.checkpoint_every_batches = 0
    try:
        programs = []
        for text in ('cat sat','cat ran'):
            raw = model.inputSpace.prepInput([' '*offset+text])
            with capture_readings(model) as captured, torch.no_grad():
                model.runBatch(train=False, split='validation', batchSize=1,
                    batch_override=(raw, torch.empty(1,0)), document_keys=('item55',),
                    questions=(What.present(0,split='validation'),))
            programs.extend(p for batch in captured.values() for p in batch if p is not None)
        assert programs, 'certificate needs a kept native reading'
        store = model.symbolSpace.ltm_store
        assert len(store), 'certificate needs durable rows'
        # The answer's public tensor is the completed conceptual value.
        answer = model._last_what_actual[0].what
        assert torch.is_tensor(answer), 'certificate needs an actual model answer'
        return dict(programs=[p.actions.detach().clone() for p in programs],
            rows={k:v.detach().clone() for k,v in store.state_dict().items()},
            content=store.slots[:len(store)].detach().clone(), answer=answer.detach().clone())
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def _stream_code_banks(tmp_path,monkeypatch):
    """A genuine meaning complement makes the two stream contexts differ."""
    from identity_experiment import make_model
    from CaseSelection import stage_case_bank,select_cases,fold_cases,search_cases
    from Meaning import ConceptualMeaning
    torch.manual_seed(20261010)
    model=make_model(tmp_path,monkeypatch)
    model.candidate_attention=None
    try:
        with torch.no_grad():
            raw=model.inputSpace.prepInput(['cat','cat'])
            model.runBatch(train=False,split='validation',batchSize=2,
                batch_override=(raw,torch.empty(2,0)),document_keys=('a','b'))
            owner=model._concept_owner();book=owner.similarity_codebook
            cid=owner.definitions.deref(owner.definitions.word(form='cat'))
            row=owner._csw_row_of(cid)
            point=book.lookup_rows(torch.tensor(row))
            meaning=ConceptualMeaning(torch.stack((point,point*0,point*0)),
                torch.tensor([True,False,False]),role_refs=(('sym',cid),None,None))
            model.symbolSpace.ltm_store.append_meaning(meaning,kind='observation',
                stream=1,document_key='c',sentence_index=1)
            book.mereology.begin_forward()
            leaves=book.lookup_rows(torch.tensor([[row],[row]]),streams=True)
            assert not torch.equal(leaves[0],leaves[1]),'fixture must exercise different occurrence contexts'
            bank=stage_case_bank(owner,torch.tensor([[cid],[cid]]),limit=16)
            position=(bank.ids==cid).nonzero().item()
            assert torch.equal(bank.codes[:,position],leaves[:,0])
            assert torch.equal(model._grammar_target_word('cat',stream=1),leaves[1,0])
            # Both the batched fold and its free inverse use their own codes.
            from test_compound_cases import _bank
            fixture=_bank()
            codes=torch.stack((fixture.codes,fixture.codes*3))
            fixture=fixture._replace(codes=codes,
                eligible=torch.tensor([[True,False,False,True,False],[True,False,False,True,False]]))
            parent=fold_cases(select_cases(fixture,torch.tensor([40,40]),torch.tensor([10,10])))
            assert torch.equal(parent[1],parent[0]*3)
            left,right,valid=search_cases(fixture,parent)
            assert valid.all()
            assert torch.equal(left,codes[:,3]) and torch.equal(right,codes[:,0])
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.usefixtures('eager_reading')
def test_clock_independence(tmp_path):
    baseline = None
    for clock in CLOCKS:
        value = _serial_read(tmp_path, clock=clock)
        if baseline is None:
            baseline = value
            continue
        assert len(value['programs']) == len(baseline['programs'])
        for a,b in zip(value['programs'],baseline['programs']):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert value['rows'].keys() == baseline['rows'].keys()
        for name in value['rows']:
            torch.testing.assert_close(value['rows'][name],baseline['rows'][name],rtol=0,atol=0,msg=name)
        torch.testing.assert_close(value['answer'],baseline['answer'],rtol=0,atol=0)


@pytest.mark.usefixtures('eager_reading')
def test_offset_independence(tmp_path):
    first, second = (_serial_read(tmp_path,offset=offset) for offset in (0,7))
    same = (len(first['programs']) == len(second['programs'])
            and all(torch.equal(a,b) for a,b in zip(first['programs'],second['programs']))
            and torch.equal(first['content'],second['content']))
    # §12 explicitly requires recording this finding, without chooser tuning.
    if not same:
        pytest.xfail('§12 offset finding: absolute .where changes the kept program or row content')


@pytest.mark.usefixtures('eager_reading')
def test_stream_chain(tmp_path,monkeypatch):
    from Layers import TernaryTruthStore, BracketExpectation
    from Meaning import ConceptualMeaning
    from Occurrence import sentence_key
    store = TernaryTruthStore(4,capacity=8)
    history = SimpleNamespace(_ltm_store=store)
    rows = {}
    for index in (1,2):
        for stream,document in enumerate(('a','b')):
            text = f'{document}{index}'.encode()
            with store.at_address(document,index,sentence_key((text,))):
                rows[stream,index] = store.append_meaning(ConceptualMeaning.from_description(
                    torch.full((4,),float(10*stream+index))),kind='observation',stream=stream)
    for stream in (0,1):
        chain = BracketExpectation.get_stm_chain(history,b=stream)
        assert [int(item[1][0,0]) for item in chain] == [10*stream+1,10*stream+2]
        assert store.stream_rows(stream).tolist() == [rows[stream,1],rows[stream,2]]
    before = store.when.clone()
    for stream in (0,1):
        for index in (1,2):
            document = 'ab'[stream]
            with store.at_address(document,index,sentence_key((f'{document}{index}'.encode(),))):
                assert store.append_meaning(ConceptualMeaning.from_description(
                    torch.full((4,),float(10*stream+index))),kind='observation',stream=stream) == rows[stream,index]
    assert len(store) == 4 and store.witness_count[:4].tolist() == [2]*4
    torch.testing.assert_close(store.when,before,rtol=0,atol=0)
    assert store.sentence_index[:4].tolist() == [1,1,2,2]
    torch.testing.assert_close(store.recency_weights(0),torch.tensor([.5,.5,1.,.5]))
    # The same source can be experienced in both streams without relocating it.
    store.witness_stream(rows[0,1],1)
    assert store.stream_rows(0).tolist() == [0,2]
    assert store.stream_rows(1).tolist() == [0,1,3]
    restored = TernaryTruthStore(4,capacity=8)
    restored.load_state_dict(store.state_dict())
    assert torch.equal(restored.stream,store.stream)
    # The production closing supplies stream witnesses and document-local time.
    from test_native_attention_field import _model
    model = _model(tmp_path,monkeypatch)
    model.candidate_attention = None
    try:
        with torch.no_grad():
            for texts,documents in ((['cat','dog'],('a','b')),(['dog','cat'],('b','a'))):
                raw=model.inputSpace.prepInput(texts)
                model.runBatch(train=False,split='validation',batchSize=2,
                    batch_override=(raw,torch.empty(2,0)),document_keys=documents)
                live=model.symbolSpace.ltm_store
                from Occurrence import document_digest
                roots=torch.stack([live.document_keys[:len(live)].eq(
                    torch.tensor(list(document_digest(doc)))).all(-1) for doc in ('a','b')]).any(0)
                observations=roots.nonzero().flatten()
                assert observations.numel()==2
                assert torch.equal(model.when_encoding.decode_index(live.when[observations]),
                                   live.sentence_index[observations])
                if documents==('a','b'):
                    before={key:getattr(live,key)[observations].clone()
                            for key in ('row_ids','when','sentence_index')}
                    witnesses=live.witness_count[observations].clone()
                    first=[live.stream_rows(b).tolist() for b in (0,1)]
                    assert set(first[0]).isdisjoint(first[1])
                else:
                    for key,value in before.items():
                        assert torch.equal(getattr(live,key)[observations],value)
                    assert (live.witness_count[observations]>witnesses).all()
                    assert live.stream[observations].all()
    finally:
        model.End()
    _stream_code_banks(tmp_path,monkeypatch)


def test_interleaving(tmp_path,monkeypatch,eager_reading):
    from test_interleave_schedule import test_one_native_read_retains_shared_owner
    test_one_native_read_retains_shared_owner(tmp_path,monkeypatch,False,eager_reading)


def test_preposition_idempotence(monkeypatch):
    from Language import PrepositionLayer
    op = PrepositionLayer(4,4)
    marker = torch.tensor([.2,.3,.4,.5])
    for phrase in (torch.tensor([.1,.4,.6,.2]),torch.arange(12.).reshape(3,4)):
        once = op.compose(marker.expand_as(phrase),phrase)
        assert torch.equal(op.compose(marker.expand_as(phrase),once),once)
        leaf, back = op.reverse(once,marker=marker.expand_as(phrase))
        assert torch.equal(leaf,marker.expand_as(phrase)) and torch.equal(back,phrase)
    from test_clause_acceptance import SentenceFixture
    fixture=SentenceFixture(monkeypatch)
    for phrase in (('lower','a','cat'),('verb','chased',('lower','a','cat'))):
        once=('preposition','in',phrase)
        first=fixture.clause(once)
        second=fixture.clause(('preposition','in',once))
        torch.testing.assert_close(first.meaning.roles,second.meaning.roles,rtol=0,atol=0)
        assert first.meaning.metadata()==second.meaning.metadata()
        assert first.refs==second.refs
        torch.testing.assert_close(first.point,second.point,rtol=0,atol=0)


@pytest.mark.usefixtures('eager_reading')
def test_surface_inverses(tmp_path, monkeypatch):
    from Language import (SurfaceLayer, LowerLayer, PrepositionLayer, MarkerUnaryLayer,
        LowerUnaryLayer, DefiniteUnaryLayer, PrepositionUnaryLayer, SurfaceUnaryLayer,
        TransposeLayer, OperationSelectionLayer, _BinaryGrammarOpAdapter)
    left,right = torch.randn(2,8),torch.randn(2,8)
    surface = SurfaceLayer(8,8)
    parent = surface.compose(left,right)
    for side,witness in (('left',left),('right',right)):
        a,b = surface.reconstruct(parent,witness,side)
        torch.testing.assert_close(a,left)
        torch.testing.assert_close(b,right)
    compiled = torch.compile(surface.generate, backend='eager', fullgraph=True)
    a,b = compiled(parent)
    torch.testing.assert_close(surface.compose(a,b),parent,atol=1e-5,rtol=1e-5)
    for face,base in ((LowerUnaryLayer,LowerLayer(8,8)),
                      (DefiniteUnaryLayer,LowerLayer(8,8)),
                      (PrepositionUnaryLayer,PrepositionLayer(8,8)),
                      (SurfaceUnaryLayer,surface)):
        unary = face(base)
        if isinstance(base,SurfaceLayer):
            with torch.no_grad(): unary.marker.fill_(.125)
        else:
            assert not list(unary.parameters())
            assert torch.equal(unary.marker, torch.zeros_like(unary.marker))
        assert unary.base is base
        value = unary.compose(right)
        restored = unary.reverse(value)
        assert torch.is_tensor(restored) and restored.shape == right.shape
        torch.testing.assert_close(restored,right)
        if not isinstance(base,SurfaceLayer):
            a,b = base.reverse(base.compose(left,right),marker=left)
            assert torch.equal(a,left) and torch.equal(b,right)
    transpose = TransposeLayer()
    a,b = transpose.reverse(*transpose.compose(left,right))
    assert torch.equal(a,left) and torch.equal(b,right)
    layer = OperationSelectionLayer(d_model=8,ops=[_BinaryGrammarOpAdapter(transpose)],chooser='mlp')
    def binary(x,candidates,*args,**kwargs):
        return x.new_zeros((*x.shape[:2],1)), x.new_full(candidates.shape[:-1],100.)
    monkeypatch.setattr(layer.chooser,'score_binary',binary)
    slab = torch.stack((left,right),1)
    _,value,route = layer(slab,slots=2,allowance=3,rounds_left=3)
    assert route['kind'].tolist() == [3,3] and route['depth'].tolist() == [2,2]
    assert torch.equal(value,slab.flip(1))
    _,_,again = layer(value,slots=2,allowance=3,rounds_left=3,previous_unary=route['previous_unary'])
    assert again['kind'].tolist() == [0,0]
    from GrammarLessons import ComposeLessonStates
    lesson = ComposeLessonStates()
    lesson.record(3,slab[:1],dict(load_depth=torch.tensor([2])),
        dict(valid=torch.tensor([True]),kind=torch.tensor([3]),position=torch.tensor([0]),probabilities=torch.ones(1,1)))
    lesson.record(4,slab[:1].flip(1),dict(load_depth=torch.tensor([2])),
        dict(valid=torch.tensor([True]),kind=torch.tensor([0]),position=torch.tensor([0]),probabilities=torch.ones(1,1)))
    attempted = torch.zeros(1,6,dtype=torch.bool);attempted[:,3:5]=True
    states = lesson.sentence(0,torch.tensor([0,1]),width=2,closing_base=6,closing_width=0,attempted=attempted)
    assert states[0]['spans'] == ((0,),(1,)) and states[1]['spans'] == ((1,),(0,))
    from Spaces import ConceptualSpace, LanguageOperationChoice
    columns = tuple(torch.arange(4).reshape(2,2)+10*i for i in range(4))
    state=(slab,torch.tensor([2,2]),*columns)
    choice=LanguageOperationChoice(route['candidate'],route['kind'],torch.zeros(2,dtype=torch.long),
        route['op'],route['valid'],route['probability'],route['action'],route['valid'])
    swapped=ConceptualSpace.apply_language_choice(state,choice)
    assert torch.equal(swapped[0],slab.flip(1)) and torch.equal(swapped[1],state[1])
    for a,b in zip(swapped[2:],columns):assert torch.equal(a,b.flip(1))
    for a,b in zip(ConceptualSpace.apply_language_choice(swapped,choice),state):
        assert torch.equal(a,b)
    _native_step_comparison(tmp_path, monkeypatch)


def _native_step_comparison(tmp_path, monkeypatch):
    """Two real paired walks differ by one controlled, content-preserving step.

    The fixture controls the route, not the cost or comparison. Replaying both
    walks from their start makes the additional step independently auditable.
    """
    import SentenceCredit
    from test_compiled_word_chunk import _tiny_canonical_model
    from ReferenceContext import reference_requests
    for extra in ('transpose','preposition_unary','question'):
        with monkeypatch.context() as patch:
            torch.manual_seed(20261010)
            paired=extra!='question'
            batch=1 if paired else 3
            model = _tiny_canonical_model(tmp_path, patch, input_width=32,
                word_buckets='8', batch_size=batch, stm_capacity=8, concept_rows=64,
                training_overrides={'intraLossWeight':0., 'reconstructionPlacement':'eager'})
            model._tensor_peer_while_eager = True
            model.candidate_attention = None
            model.inputSpace.data.has_supervised_outputs = False
            language = model.languageSpace
            operation = language.language_layer.operation_layer
            if not paired:
                surface=next(op.gl for op in operation.ops if op.gl.rule_name=='surface')
                with torch.no_grad():surface.marker_map.weight.mul_(2.)
            ordinary = language.choose_operation
            selected = {}
            observed = []
            def choose(state, gate, **kwargs):
                slot = int(kwargs['lesson_slot'].flatten()[0])
                width = model.inputSpace._word_active_mask.shape[1]
                name = (extra if model._sentence_trial == 'explore' and slot == 6
                                    else 'sum' if slot >= 3*width and bool((state[1]>1).any())
                                    else None)
                if paired and name == extra:
                    assert state[1].tolist() == [3]
                selected['names']=[name]*batch
                if not paired:
                    if slot==3:selected['names']=['transpose','surface','surface']
                    elif slot==4:selected['names']=['surface']*batch
                    elif slot>=3*width:
                        closing=slot-3*width
                        if closing==0:
                            assert state[1].tolist()==[3]*batch
                            assert not torch.equal(state[0][0,0],state[0][0,1])
                        selected['names']=[{0:'transpose',1:'preposition',2:'surface'}.get(closing)]*batch
                    else:selected['names']=[None]*batch
                return ordinary(state, gate, **kwargs)
            patch.setattr(language,'choose_operation',choose)
            original_binary = operation.chooser.score_binary
            original_unary = operation.chooser.score_unary
            def scores(x,candidates,*args,op_indices=None,unary=False,**kwargs):
                if op_indices is None:
                    return (original_unary if unary else original_binary)(x,candidates,*args,**kwargs)
                values = x.new_full(candidates.shape[:-1],-1e6)
                names = operation.unary_names if unary else operation.op_names
                stop=x.new_full((*x.shape[:2],1),-1e6)
                for b,name in enumerate(selected['names']):
                    for column,index in enumerate(op_indices.tolist()):
                        if names[index]==name:values[b,...,column]=1e6
                    if name is None:stop[b]=1e6
                return stop,values
            patch.setattr(operation.chooser,'score_binary',scores)
            patch.setattr(operation.chooser,'score_unary',lambda *a,**k:scores(*a,unary=True,**k))
            departure = SentenceCredit.departure
            def replay(*a,**k):
                draw = departure(*a,**k)
                draw.update(narrowing=torch.ones_like(draw['narrowing']),
                    attention_round=torch.full_like(draw['attention_round'],-1),
                    compose_round=torch.full_like(draw['compose_round'],-1),
                    candidate_round=torch.full_like(draw['candidate_round'],-1))
                return draw
            patch.setattr(SentenceCredit,'departure',replay)
            observe = model._sentence_observation
            def capture(*a,**k):
                result = observe(*a,**k)
                observed.append(result)
                return result
            patch.setattr(model,'_sentence_observation',capture)
            try:
                inputs = model.inputSpace.prepInput(['alpha alpha alpha'] if paired else
                    ['is Felix here?','Felix is here.','is Felix here?'])
                model.runBatch(train=paired, optimizer=model.getOptimizer(lr=0.) if paired else None,
                    batchSize=batch,batch_override=(inputs,torch.empty(batch,0)),
                    document_keys=('cost',) if paired else ('question','declarative','control'))
                if not paired:
                    programs=observed[0]['entries']
                    assert int((programs[0].actions[:,0]==3).sum())==2
                    torch.testing.assert_close(programs[0].end_state,programs[1].end_state,rtol=0,atol=0)
                    assert not torch.equal(programs[0].end_state,programs[2].end_state)
                    live=model.symbolSpace.ltm_store
                    from Occurrence import source_at,document_digest
                    roots=[]
                    for b in range(batch):
                        document,_=source_at(model,b,0)
                        roots.append(live.document_keys[:len(live)].eq(torch.tensor(list(document_digest(document)))).all(-1).nonzero().flatten()[-1])
                    torch.testing.assert_close(live.slots[roots[0]],live.slots[roots[1]],rtol=0,atol=0)
                    for program in programs:
                        reference_requests(language,program.actions,leaf_orders=program.leaf_orders)
                    from DerivationReconstruction import reconstruct
                    record=observed[0]['record']
                    recovered,unavailable,defined=reconstruct(language,record,programs,
                        lambda b:model._sentence_leaf_positions(record,b))
                    assert not unavailable.any()
                    torch.testing.assert_close(recovered[record.word_valid],record.word_values[record.word_valid])
                    continue
                audit = model._last_sentence_credit
                torch.testing.assert_close(audit['keep_costs'][:,1]-audit['keep_costs'][:,0],
                    torch.tensor([model.WHAT_STEP_COST]),rtol=1e-4,atol=2e-7)
                trials = model._last_sentence_field['trials']
                assert trials[1]['cost']['compose_steps'].item() == trials[0]['cost']['compose_steps'].item()+1
                programs = [entry['entries'][0] for entry in observed[:2]]
                assert int((programs[1].actions[:,0] == (3 if extra=='transpose' else 2)).sum()) == 1
                for program in programs:
                    reference_requests(language, program.actions,leaf_orders=program.leaf_orders)
                torch.testing.assert_close(programs[0].end_state,programs[1].end_state,rtol=0,atol=0)
                assert len(model.symbolSpace.ltm_store)
            finally:
                model.End()


@pytest.mark.usefixtures('eager_reading')
def test_nested_wholes():
    from Layers import RadixLayer
    from WordIdentity import WordIdentity
    from Meronomy import pin_word_spans
    from constituent_corpus import corpus
    store = RadixLayer(96, initial_cap=256)
    identity = WordIdentity(store)
    words = (b'walk',b'walked',b'walking',b'walks',b'cat',b'cats',b'went')
    vocabulary = tuple(sorted({word.encode() for _,text,_,_ in corpus() for word in text['text'].split()}))
    assert len(vocabulary) == 17
    for word in (vocabulary+words)*2:
        identity.observe_nested(word)
    assert not identity.nested
    before = {word: identity.form(word).clone() for word in vocabulary+words}
    with identity.nested_lesson(b'ed'):
        for word in vocabulary+words:
            identity.observe_nested(word)
        parts = identity.observe_nested(b'walked')
        with pytest.raises(ValueError, match='only one'):
            with identity.nested_lesson(b's'):
                pass
    assert set(identity.nested) == {b'walked'}
    assert identity._marker_pattern is None
    assert all(torch.equal(identity.form(word), value) for word,value in before.items() if word != b'walked')
    # Merely allocating a stem is not witnessing it in a lesson's input.
    late = WordIdentity(RadixLayer(96))
    late.admit(b'walk')
    with late.nested_lesson('ed'):
        assert not late.observe_nested(b'walked')
        late.observe_nested(b'walk')
        assert late.nested[b'walked'] == ((b'walk',0,4),(b'ed',4,6))
    named = [(store.bytes_for(row),lo,hi) for row,lo,hi in parts]
    assert named == [(b'walk',0,4),(b'ed',4,6)]
    row = identity.word_rows[b'ed']
    assert sum(pid == row for pid,_,_ in parts) == 1
    assert identity.observe_nested(b'walked') == parts
    assert not identity.nested.get(b'went')
    assert pin_word_spans(b'walked',[(0,4),(4,6)]) == [(0,6)]
    assert bool((identity.form(b'walked') >= identity.form(b'walk')).all())
    assert identity.read_parts(identity.parts(b'walked')) == b'walked'
    # Each part's .where is contained by the predefined word's .where.
    from Spaces import WhereEncoding
    encoding = WhereEncoding(nWhere=4).set_capacity(128)
    assert encoding.decode_index(encoding.encode(torch.tensor([lo for _,lo,_ in parts]))).tolist() == [0,4]
    assert all(0 <= lo < hi <= 6 for _,lo,hi in parts)
    # The native field reads these parts at their actual sub-word locations.
    from Spaces import PartSpace
    store.identity=identity
    store.ensure_atomic_bytes([bytes([v]) for v in b'walked'])
    ids=[store.get_id(bytes([v])) for v in b'walked']
    native=torch.tensor([ids])
    spans=torch.tensor([[[i,i+1] for i in range(6)]])
    locations=torch.tensor([[[0,6],[0,4],[4,6]]])
    values=PartSpace.part_memberships(store,[identity.word_rows[b'walk'],row],
                                      native,spans,locations,6)
    assert values[...,0].tolist()==[[[True,True,False]],[[True,False,True]]]


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.phase('stage3')
def test_nested_wholes_from_stage3_artifact(tmp_path, monkeypatch, restore_phase):
    # This phase's certificate restores the preceding phase's trained artifact.
    from constituent_corpus import corpus
    words = (b'walk',b'walked',b'walking',b'walks',b'cat',b'cats',b'went')
    vocabulary = tuple(sorted({word.encode() for _,text,_,_ in corpus() for word in text['text'].split()}))
    from Models import BaseModel
    from identity_experiment import make_model, restore_warmup
    restore = BaseModel.load_weights
    model = make_model(tmp_path,monkeypatch)
    try:
        restore_phase(lambda path: restore_warmup(model, path, restore))
        current=model.perceptualSpace.percept_store.identity
        assert not current.nested and current._marker_pattern is None
        for _ in range(2):
            with torch.no_grad():
                model._lex_embed_stem(model.inputSpace.prepInput([' '.join(w.decode() for w in vocabulary+words)]))
            staged=model.perceptualSpace._forward_input
            assert not current.nested
            mask=staged['word_part_mask'][0,0]
            assert staged['word_part_indices'][0,0][mask].tolist()==list(current.parts(vocabulary[0]))
        with current.nested_lesson('ed'), torch.no_grad():
            model._lex_embed_stem(model.inputSpace.prepInput(['walked went']))
        staged = model.perceptualSpace._forward_input
        identity = model.perceptualSpace.percept_store.identity
        ids, spans = staged['indices'][0],staged['part_spans'][0]
        for raw,expected in ((b'walk',[0,4]),(b'ed',[4,6])):
            assert spans[ids == identity.word_rows[raw]].tolist() == [expected]
        assert staged['word_texts'][0][0] == 'walked'
        assert not identity.nested.get(b'went')
        assert set(identity.nested) == {b'walked'}
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_absences():
    import Language
    from ClauseRow import Clause
    from Layers import TernaryTruthStore
    store = TernaryTruthStore(4,8)
    assert not {'timestamp','_next_ts'} & set(store.state_dict())
    assert not any(hasattr(store,name) for name in ('timestamp','_next_ts','write_timestamp','recent','_timestamp_source'))
    assert 'eternal' not in {f.name for f in fields(Clause)}
    for name in ('_lift_when','_lower_when','TenseLayer','AspectLayer','MorphologyLayer'):
        assert not hasattr(Language,name)
    for name in ('surface_tense.py','surface_morphology.py'):
        assert not (ROOT/'bin'/name).exists()
    assert not {'tense','aspect','morphology','split','insert'} & Language.GRAMMAR_LAYER_CLASSES.keys()
    assert not {'tense','aspect','morphology'} & Language._OPERATOR_SURFACE_SCHEMAS.keys()
    import subprocess
    grammars = subprocess.check_output(['git','ls-files','data/*.grammar','test/*.grammar'],cwd=ROOT,text=True).splitlines()
    for name in grammars:
        import xml.etree.ElementTree as ET
        for rule in ET.parse(ROOT/name).iter('rule'):
            assert not any(f'{op}.' in (rule.text or '') for op in ('tense','aspect','morphology'))
