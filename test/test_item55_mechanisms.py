"""§12 mechanism certificates. No learned grammar or identity gate is claimed."""
from Memory import STM0
from dataclasses import fields
import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLOCKS = (0, 25, 50, 100, 200, 400, 777, 1500, 3000)


def _serial_read(tmp_path, clock=0, offset=0):
    # Collect the preceding completed read before allocating another model.
    gc.collect()
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
    store = model.symbolSpace.ltm_store
    initial_rows, initial_clock = len(store), int(store.model_clock)
    omission_rows, forecast_rows = [], []
    import ThoughtStream
    patches = pytest.MonkeyPatch()
    native_write, native_append = ThoughtStream.write, store.append_meaning
    def thought_write(host, meaning, **options):
        start=len(store)
        result=native_write(host,meaning,**options)
        if '_omitted_symbol' in dict(meaning.bindings):
            omission_rows.extend(range(start,len(store)))
        return result
    def append(value, **options):
        start=len(store)
        result=native_append(value,**options)
        if options.get('surprise',-1.) >= 0:
            forecast_rows.extend(range(start,len(store)))
        return result
    patches.setattr(ThoughtStream,'write',thought_write)
    patches.setattr(store,'append_meaning',append)
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
        addresses = {store.row_name(i):i for i in range(len(store))}
        referents = store.referent.clone()
        from Memory import OCCURRENCE
        for i in range(len(store)):
            if int(referents[i,0]) == OCCURRENCE:
                referents[i,1] = addresses[int(referents[i,1])]
        return dict(programs=[p.actions.detach().clone() for p in programs],
            rows={k.removeprefix('addressing.'):v.detach().clone() for k,v in store.state_dict().items()},
            forecast_errors=tuple(forecast_rows),
            time_start=store.when_start,
            omissions=tuple(omission_rows), forecasts=tuple(omission_rows),
            semantics=referents,
            initial_rows=initial_rows, initial_clock=initial_clock, count=len(store),
            content=store.slots[:len(store)].detach().clone(), answer=answer.detach().clone())
    finally:
        patches.undo()
        model.End()
        model.symbolSpace.soft_reset()
        from Layers import GrammarLayer
        GrammarLayer.set_chart_authority(None)


def _stream_code_banks(tmp_path,monkeypatch):
    """Both streams read the same online word context and preserve their bank axes."""
    from identity_experiment import make_model
    from CaseSelection import stage_case_bank,select_cases,fold_cases,search_cases
    from Meaning import ConceptualMeaning
    torch.manual_seed(20261010)
    model=make_model(tmp_path,monkeypatch)
    model.candidate_attention=None
    try:
        with torch.no_grad():
            owner=model._concept_owner();book=owner.similarity_codebook
            # Admit the form this placement certificate observes. A free
            # sentence may instead bind it to an accessible cover, whose
            # native identity cache deliberately does not vary by stream.
            word=owner.interpret.lookup_word((), (), form='cat')
            cid=owner.interpret.forward(word)
            row=owner._csw_row_of(cid)
            point=book.lookup_rows(torch.tensor(row))
            meaning=ConceptualMeaning(torch.stack((point,point*0,point*0)),
                torch.tensor([True,False,False]),role_refs=(('sym',cid),None,None))
            # The certificate supplies two distinct situated observations.
            # Bare objectless readings need not have closed point-valued rows.
            from dataclasses import replace
            from ClauseRow import Clause
            for stream, value in enumerate((replace(meaning,roles=meaning.roles*.5),meaning)):
                model.symbolSpace.ltm_store.write_clause(
                    Clause(value, point=value.roles[0], refs=(cid,-1,-1), order=0),
                     )
            # Placement now consumes the kept reading online, not stored
            # memberships. One word seat accumulates contexts across streams.
            for form in ('context-a', 'context-b'):
                other=owner.interpret.lookup_word((), (), form=form)
                other_row=owner._csw_row_of(other)
                owner._sentence_seen_rows[0].fill_(-1)
                owner._sentence_seen_rows[0, :2] = torch.tensor([row,other_row])
                owner.prime_seen(torch.tensor([[row,other_row]]))
                book.mereology.observe_close(SimpleNamespace(
                    word_rows=torch.tensor([row,other_row]), concept_ids=torch.tensor([cid,other])))
            assert int(book.mereology.placement_witnesses[row]) == 2
            assert bool(book.mereology.placement[row].ne(0).any())
            book.mereology.begin_forward()
            leaves=book.lookup_rows(torch.tensor([[row],[row]]),streams=True)
            torch.testing.assert_close(leaves[0],leaves[1],rtol=0,atol=0)
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
    from RowTime import encoder
    baseline = None
    measurements = []
    for clock in CLOCKS:
        value = _serial_read(tmp_path, clock=clock)
        # Entering chunks now carry current time (§1.6). The existing predictor
        # and answer reader see that band; their numeric forecasts are allowed
        # to vary. Keep their values in the certificate, not in content equality.
        measurements.append(dict(clock=clock,answer=value['answer'].tolist(),
            omission_rows=value['omissions'],
            omission_trust=value['rows']['trust'][list(value['omissions'])].tolist(),
            forecast_content=value['rows']['slots'][list(value['forecasts'])].tolist(),
            activity=value['rows']['activity'].tolist()))
        assert torch.isfinite(value['answer']).all()
        start = value['time_start']
        stamps = value['rows']['slots'][value['initial_rows']:value['count'],start:start+4]
        # Reading pending identities can close into a Part row. Both durable
        # row stores use the same model clock and carry its percept stamp.
        stamped = stamps.ne(0).any(-1)
        times = encoder().decode_index(stamps[stamped])
        assert times.numel() and int(value['rows']['model_clock']) == max(
            value['initial_clock'], int(times.max())+1)
        if baseline is None:
            baseline = value
            continue
        assert len(value['programs']) == len(baseline['programs'])
        for a,b in zip(value['programs'],baseline['programs']):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert value['rows'].keys() == baseline['rows'].keys()
        assert (value['initial_rows'],value['count']) == (baseline['initial_rows'],baseline['count'])
        # Located content has different native hashes when its time changes;
        # compare the retained references by the same physical-row isomorphism.
        torch.testing.assert_close(value['semantics'],baseline['semantics'],rtol=0,atol=0)
        assert value['omissions'] == baseline['omissions']
        assert times.unique().numel() == times.numel()
        for name in value['rows']:
            if name == 'model_clock':
                continue  # Verified against the actual write times above.
            if name == 'referent':
                continue  # Typed references are compared by seat isomorphism above.
            if name in ('activity','trust'):
                first,second = value['rows'][name].clone(),baseline['rows'][name].clone()
                assert torch.isfinite(first).all()
                rows = list(value['forecast_errors'] if name == 'activity' else value['omissions'])
                if name == 'activity':
                    assert rows == list(baseline['forecast_errors'])
                    assert first[rows].le(1).all() and first[rows].ge(0).all()
                else:
                    assert first[rows].le(0).all() and first[rows].ge(-1).all()
                first[rows] = second[rows] = 0
                torch.testing.assert_close(first,second,rtol=0,atol=0,msg=name)
                continue
            if name == 'slots':
                before, after = value['initial_rows'], value['count']
                torch.testing.assert_close(value['rows'][name][:before],baseline['rows'][name][:before],rtol=0,atol=0)
                torch.testing.assert_close(value['rows'][name][after:],baseline['rows'][name][after:],rtol=0,atol=0)
                earlier = baseline['rows'][name][before:after,start:start+4]
                assert torch.equal(stamped,earlier.ne(0).any(-1))
                expected = encoder().decode_index(earlier[stamped])+clock
                torch.testing.assert_close(times,expected,rtol=0,atol=0)
                first, second = value['rows'][name].clone(), baseline['rows'][name].clone()
                first[before:after,start:start+4] = second[before:after,start:start+4] = 0
                first[list(value['forecasts'])] = second[list(baseline['forecasts'])] = 0
                torch.testing.assert_close(first,second,rtol=0,atol=0)
                continue
            torch.testing.assert_close(value['rows'][name],baseline['rows'][name],rtol=0,atol=0,msg=name)
    import json
    (tmp_path/'certificate.json').write_text(json.dumps(dict(
        exact_content_and_programs=True, time_sensitive_reads=measurements,
        differing_binding_row=None, clock_dependent_binding=False),indent=2))


@pytest.mark.usefixtures('eager_reading')
def test_offset_independence(tmp_path):
    first, second = (_serial_read(tmp_path,offset=offset) for offset in (0,7))
    same = (len(first['programs']) == len(second['programs'])
            and all(torch.equal(a,b) for a,b in zip(first['programs'],second['programs']))
            and torch.equal(first['content'],second['content']))
    # §12 explicitly requires recording this finding, without chooser tuning.
    if not same:
        pytest.xfail('§12 offset finding: absolute .where changes the kept program or row content')


def test_stream_code_banks_keep_their_independent_batch_axes(tmp_path,monkeypatch):
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
    left,right = torch.rand(2,8),torch.rand(2,8)
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
    from chunk_fixtures import from_stack
    from Memory import STMChunks as chunks
    state=from_stack(slab,torch.tensor([2,2]),*columns)
    choice=LanguageOperationChoice(route['candidate'],route['kind'],torch.zeros(2,dtype=torch.long),
        route['op'],route['valid'],route['probability'],route['action'],route['valid'])
    swapped=ConceptualSpace.apply_language_choice(state,choice)
    view = chunks.stack_view(swapped)
    assert torch.equal(view[0],slab.flip(1)) and torch.equal(view[1],state[1])
    for a,b in zip(view[2:],columns):assert torch.equal(a,b.flip(1))
    torch.testing.assert_close(swapped[chunks.BUFFER], state[chunks.BUFFER], rtol=0, atol=0)
    restored = ConceptualSpace.apply_language_choice(swapped,choice)
    for field,(a,b) in enumerate(zip(restored,state)):
        if field == chunks.READ:
            assert (a > b).all()  # Both operand reads refresh their recency.
        else:
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
                training_overrides={ 'reconstructionPlacement':'eager'})
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
                names = operation.unary_names if unary else operation.op_names
                if op_indices is None or bool((op_indices >= len(names)).any()):
                    return (original_unary if unary else original_binary)(
                        x,candidates,*args,op_indices=op_indices,**kwargs)
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
                for key in ('bind_round', 'reconstruction_round', 'thought_round', 'output_round'):
                    draw[key] = torch.full_like(draw[key], -1)
                draw.update(walk=torch.zeros_like(draw['walk']),
                    narrowing=torch.ones_like(draw['narrowing']),
                    attention_round=torch.full_like(draw['attention_round'],-1),
                    compose_round=torch.full_like(draw['compose_round'],-1),
                    candidate_round=torch.full_like(draw['candidate_round'],-1))
                return draw
            patch.setattr(SentenceCredit,'departure',replay)
            observe = model._sentence_observation
            def capture(*a,**k):
                result = observe(*a,**k)
                pool = model._concept_owner().mtm
                result['reading_tokens'] = {word: (token.pending, token.symbol, pool.value_of(key).clone())
                    for program in result['entries'] if program is not None
                    for word in program.word_ids.tolist()
                    if word > 0 and (key := pool.token_of(('name',word))) is not None
                    for token in (pool.entries[key],)}
                # A test observation is taken before the kept close expires
                # STM's tree. It is never fed back into the production model.
                from dataclasses import replace
                result['observed_entries'] = tuple(
                    None if program is None else replace(program, tree=None,
                        actions=program.actions, leaves=program.leaves)
                    for program in result['entries'])
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
                    programs=observed[0]['observed_entries']
                    assert int((programs[0].actions[:,0]==3).sum())==2
                    torch.testing.assert_close(programs[0].end_state,programs[1].end_state,rtol=0,atol=0)
                    assert not torch.equal(programs[0].end_state,programs[2].end_state)
                    live=model.symbolSpace.ltm_store
                    roots = [live.location(field.row_id) for field in model._sentence_fields[0]]
                    # Native occurrence times now distinguish the two reads;
                    # their content coordinates must still be identical.
                    coordinates = torch.ones(live.slots.shape[-1], dtype=torch.bool)
                    if live.when_start is not None:
                        coordinates[live.when_start:live.when_start+4] = False
                    torch.testing.assert_close(live.slots[roots[0],coordinates],live.slots[roots[1],coordinates],rtol=0,atol=0)
                    for program in programs:
                        reference_requests(language,program.actions,leaf_orders=program.leaf_orders)
                    from DerivationReconstruction import reconstruct
                    record=observed[0]['record']
                    recovered,unavailable,defined=reconstruct(language,record,programs,
                        lambda b:model._sentence_leaf_positions(record,b))
                    assert not unavailable.any()
                    # D6 interprets each identified word before the surface
                    # operations. Their exact inverse returns the pending
                    # token's value, whose magnitude is not form attention.
                    expected = record.word_values.clone()
                    for b, program in enumerate(programs):
                        positions = model._sentence_leaf_positions(record,b)
                        for position, word in zip(positions,program.word_ids.tolist()):
                            if word <= 0:
                                continue
                            pending, symbol, point = observed[0]['reading_tokens'][word]
                            assert pending and symbol is None
                            expected[b,position] = point
                    torch.testing.assert_close(recovered[record.word_valid],expected[record.word_valid])
                    continue
                audit = model._last_sentence_credit
                # Compare the two totals in their own chart. Subtracting
                # two rounded sentence costs loses bits of the 0.01 step.
                # Round 6 retires the iterations surcharge. The changed
                # order-zero face now supplies reconstruction cost directly.
                assert "reconstruction.iterations" not in model._sentence_cost_registry._terms
                observations = observed[:2]
                programs = [entry['observed_entries'][0] for entry in observations]
                steps = [(program.actions[:, 0] > 0).sum() for program in programs]
                assert (steps[1] - steps[0]).item() == 1
                programs = [entry['observed_entries'][0] for entry in observed[:2]]
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
    from Memory import LTM0
    store = STM0(4,8, relation_capacity=8192)
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
