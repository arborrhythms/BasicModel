"""Passive, stepped-walk assessment and native-lane provenance (§53).

All patches live in this observer process. No choice, input, learning update or
production definition is replaced. Assessment's transaction restores probe reads.
"""
from contextlib import contextmanager
from collections import defaultdict
import json
from pathlib import Path
import torch


@contextmanager
def word_deposits(sink):
    """Observe actual deposit payloads and signed reads without replacing either."""
    import inspect
    from Memory import ShortTermMemory
    from Spaces import ConceptualSpace
    push = ShortTermMemory.functional_push_step_masked
    reduce = ConceptualSpace.reduce_aligned_word_peers
    def reading(owner, *args, **kwargs):
        result = reduce(owner, *args, **kwargs)
        # The row's own scalar, before the payload takes its absolute value.
        rows, values = result[2], result[3]
        sink(dict(kind='read', negative=int(((rows >= 0) & (values < 0)).sum()),
                  count=int((rows >= 0).sum())))
        return result
    def deposit(state, value, active, *args, **kwargs):
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == 'stage_cs_lang':
            local = caller.f_locals
            model = local['self']
            live = active.reshape(-1)
            walk = getattr(model, '_attention_words', None) is not None
            sink(dict(kind='deposit', walk=walk, count=int(live.sum()),
                negative=int((value[live] < 0).sum()),
                pair=local['leaf_evidence'][live].detach().tolist(),
                word=int(local['index']),
                own_code=torch.is_tensor(getattr(model.inputSpace,'_ar_word_concept_atoms',None))))
        return push(state,value,active,*args,**kwargs)
    ConceptualSpace.reduce_aligned_word_peers = reading
    ShortTermMemory.functional_push_step_masked = staticmethod(deposit)
    try:
        yield
    finally:
        ConceptualSpace.reduce_aligned_word_peers = reduce
        ShortTermMemory.functional_push_step_masked = staticmethod(push)


@torch.no_grad()
def idea_read_measurements(owner, samples):
    """Final read weights on the fields and semantic roots actually kept."""
    from FieldProjection import key
    from Understanding import ConceptualField
    width = owner._order_caps()[0]
    ids = torch.tensor([owner.concept_id_at_row(row) or -1 for row in range(width)],
                       device=owner._expected_knowing.device)
    form_width = owner.similarity_codebook.mereology.percept_event_width
    records = []
    for sample in samples:
        observed, expected, target = (sample[name] for name in ('observed', 'expected', 'target'))
        read = owner.idea_stack(observed.flatten(1))
        anticipated = owner.idea_stack(expected.flatten(1))
        field = ConceptualField(ids, observed.transpose(0, 1).unsqueeze(2),
            observed.new_zeros(len(observed), 1, 2), observed.new_zeros(len(observed), 1, 2), None)
        centroid = key(owner, field, order=0)[:, 0]
        blocks = {}
        for name, region in (('form', slice(0, form_width)), ('meaning', slice(form_width, None))):
            count = target[:, region].numel()
            blocks[name] = dict(read_mse=float((read[:, region]-target[:, region]).square().mean()) if count else None,
                centroid_mse=float((centroid[:, region]-target[:, region]).square().mean()) if count else None,
                coordinates=count)
        records.append(dict(presentation=sample['presentation'], blocks=blocks,
            no_omission=sample['no_omission'],
            field_mse=float((expected-observed).square().mean()),
            field_mean_absolute=float((expected-observed).abs().mean()),
            point_mse=float((anticipated-read).square().mean()),
            point_mean_absolute=float((anticipated-read).abs().mean())))
    return dict(protocol='final idea-read and centroid on saved open fields; targets are their kept semantic roots',
                form_width=form_width, records=records)


@contextmanager
def look_lane_counts(sink):
    """Observe the whole properties and basic percepts actually read at each extent."""
    from unittest.mock import patch
    import FamiliarTier, PerceptField, SentenceLooks
    from BasicPercepts import BasicPercepts

    def measured(original, kind):
        def measure(*args, **kwargs):
            whole_read = PerceptField.read_percepts
            basic_read = BasicPercepts.read
            counts = {}

            def count(field, family):
                keys = [i for i, key in enumerate(field.keys) if key[0] == family]
                # Native and basic reads identify their features by distinct keys.
                if len({field.keys[i] for i in keys}) != len(keys):
                    raise ValueError('look census found duplicate feature keys')
                both = (field.values.detach()[keys] > 0).all(-1).sum(0).tolist()
                for b, row in enumerate(field.where.tolist()):
                    for place, (start, stop) in enumerate(row):
                        if stop > start:
                            extent = counts.setdefault((b, place, start, stop), {})
                            extent[family] = both[b][place]

            def whole(*a, **kw):
                result = whole_read(*a, **kw)
                count(result, 'ws')
                return result

            def basic(*a, **kw):
                result = basic_read(*a, **kw)
                count(result, 'bp')
                return result

            with patch.object(PerceptField, 'read_percepts', whole), patch.object(BasicPercepts, 'read', basic):
                result = original(*args, **kwargs)
            sink(dict(kind=kind, extents=[
                dict(batch=b, place=p, where=[start, stop], whole_both=poles['ws'],
                     basic_both=poles['bp'])
                for (b, p, start, stop), poles in counts.items()]))
            return result
        return measure

    with patch.object(FamiliarTier, 'open_sentence', measured(FamiliarTier.open_sentence, 'opening')), \
         patch.object(SentenceLooks, 'perceive', measured(SentenceLooks.perceive, 'selected')):
        yield


@contextmanager
def native_provenance(sink):
    """Read actual native-return values, including the pre-definition field."""
    import ModelAttention, PerceptField
    original = ModelAttention.native_word_poles
    def observed(model, spans, forms, known):
        owner = model._concept_owner()
        read = owner.cs_read_memberships
        native_read = PerceptField.read_percepts
        originals = {name: (name in vars(owner), vars(owner).get(name))
                     for name in ('cs_read_memberships',)}
        captured, natives = [], []
        def native(*args, **kwargs):
            result = native_read(*args, **kwargs)
            natives.append(result)
            return result
        def reading(*args, **kwargs):
            result = read(*args, **kwargs)
            # The order-zero concepts as the Sigma of order zero read them,
            # before the orders above.
            direct = owner._cs_order0_raw
            perception = owner._percept_field
            ids = owner._cs_field_concept_ids[:result.shape[0]]
            record = owner._concept_allocator.order_zero_members
            basic = owner.basic_percepts
            native = natives[-1] if natives else None
            entries=[]
            for b, row in enumerate(forms):
                identities = ids if ids.ndim == 1 else ids[:, b]
                for p, form in enumerate(row[:result.shape[2]]):
                    if not bool(known[b,p]): continue
                    word = owner.interpret.word_for_form(form)
                    concepts=[]
                    for cid in dict.fromkeys((word, *owner.definitions.objects(word))):
                        physical = owner._csw_row_of(cid)
                        indices = (identities == cid).nonzero().flatten()
                        evidence = result[indices,b,p].amax(0) if len(indices) else result.new_zeros(2)
                        order0 = indices[indices < direct.shape[0]]
                        before = direct[order0,b,p].amax(0) if len(order0) else evidence.new_zeros(2)
                        edges=[]
                        for (r,bp,negated), weight in record.items():
                            if r != physical: continue
                            neg = int(negated)
                            key=('bp',bp)
                            if key not in perception.keys: continue
                            i=perception.keys.index(key)
                            parts,wholes=basic.occurrences[bp]
                            components=[]
                            if native is not None:
                                # Include every native component; ancestral part IDs are
                                # expanded by the real BasicPercepts reader, not guessed here.
                                for j,k in enumerate(native.keys):
                                    if (k[0]=='ws' and k[1] in wholes) or k[0]=='ps':
                                        value=native.values[j,b,p]
                                        if bool(value.any()):
                                            events=native.events[j,b,p]
                                            live=events.any(-1).nonzero().flatten()
                                            components.append(dict(key=k,pair=value.detach().tolist(),
                                                events=[dict(span=native.spans[b,int(e)].tolist(),pair=events[e].detach().tolist()) for e in live]))
                            edges.append(dict(bp=bp,negated=neg,weight=weight,
                                parts=parts,wholes=wholes,pair=perception.values[i,b,p].detach().tolist(),
                                native=components))
                        store=owner._concept_allocator.layer(0)
                        definitions={name:[dict(column=c,weight=float(m.values[e].detach()))
                            for (r,c),e in m._index.items() if r==physical]
                            for name,m in [('union',store)]}
                        concepts.append(dict(concept=cid,row=physical,evidence=evidence.detach().tolist(),
                            before_definitions=before.tolist(),features=edges,definitions=definitions))
                    entries.append(dict(form=form,position=p,span=spans[b,p].tolist(),concepts=concepts))
            captured.extend(entries)
            return result
        owner.cs_read_memberships=reading
        PerceptField.read_percepts=native
        try:
            result=original(model,spans,forms,known)
            sink(dict(words=captured,poles=result.detach().tolist(),forms=forms))
            return result
        finally:
            PerceptField.read_percepts=native_read
            for name,(had,value) in originals.items():
                if had:setattr(owner,name,value)
                else:delattr(owner,name)
    ModelAttention.native_word_poles=observed
    try:yield
    finally:ModelAttention.native_word_poles=original


def main():
    import argparse, os, random
    import numpy as np
    from random_streams import write
    from corpus_lessons import measure
    import Assessment
    from Models import BasicModel
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--timeline',action='store_true',help='Preserved two-text probe after each training close')
    parser.add_argument('--assessment-texts',type=Path,
                        help='Replay these assessment JSONL texts for a like-for-like table')
    args=parser.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    state=torch.load(args.replay,weights_only=False)
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    write(args.output/'initial.pt',scope='160-sentence postmerge lane observer')
    count=0; phase='training'
    idea_samples, choices = [], {}
    from Comprehension import Evaluation
    finish = Evaluation.finish
    from Comprehension import Result
    detach = Result.detached
    def finished(evaluation, *a, **kw):
        result = finish(evaluation, *a, **kw)
        choices[id(result)] = tuple(evaluation.choices)
        return result
    Evaluation.finish = finished
    def detached(result):
        saved = detach(result)
        if id(result) in choices:
            choices[id(saved)] = choices[id(result)]
        return saved
    Result.detached = detached
    get_optimizer=BasicModel.getOptimizer
    def observed_optimizer(model,*a,**kw):
        optimizer=get_optimizer(model,*a,**kw)
        step=optimizer.step
        def observed_step(*a,**kw):
            owner=model._concept_owner(); sigma=owner.order_zero; weight=sigma.value
            before=weight.detach().clone()
            gradient=None if weight.grad is None else weight.grad.detach().clone()
            result=step(*a,**kw)
            changes=[]
            for entry in (weight.detach()!=before).nonzero().flatten().tolist():
                changes.append(dict(row=int(sigma.entry_input[entry]),column=int(sigma.entry_row[entry]),
                    before=float(before[entry]),after=float(weight[entry].detach()),
                    gradient=None if gradient is None else float(gradient[entry])))
            with (args.output/'writes.jsonl').open('a') as f:
                f.write(json.dumps(dict(presentation=count,writer=type(optimizer).__name__+'.step',changes=changes))+'\n')
            return result
        optimizer.step=observed_step
        return optimizer
    BasicModel.getOptimizer=observed_optimizer
    def sink(record):
        record.update(presentation=count+1 if phase=='training' else count,phase=phase)
        with (args.output/'native.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    commit=BasicModel._commit_sentence
    def dense(field,b):
        ids=field.concept_ids if field.concept_ids.ndim==1 else field.concept_ids[:,b]
        values=field.evidence[:,b].amax(1) if field.evidence.shape[2] else field.evidence.new_zeros(len(ids),2)
        return {int(cid):v.detach() for cid,v in zip(ids,values) if int(cid)>=0}
    def kept(model,*a,**kw):
        nonlocal count,phase
        result=commit(model,*a,**kw)
        if model.training:
            count+=1
            state,sid,active,observations,wins=a
            owner=model._concept_owner()
            orders=defaultdict(int)
            for row in owner._row_to_concept():orders[owner._row_order(row)]+=1
            comparisons=[]
            for b in active.nonzero().flatten().tolist():
                selected=observations[int(wins[b])]
                comprehension = selected['comprehensions'][b]
                selected_choices = () if comprehension is None else choices.get(id(comprehension))
                with (args.output/'objects.jsonl').open('a') as f:
                    f.write(json.dumps(dict(presentation=count,batch=b,available=selected_choices is not None,
                        choices=selected_choices))+'\n')
                width = owner._order_caps()[0]
                idea_samples.append(dict(presentation=count,
                    observed=model._idea_observation[b:b+1].detach().clone(),
                    expected=owner._expected_knowing[b:b+1,:width].detach().clone(),
                    target=model._last_sentence_understanding.meaning_root[b:b+1].detach().clone(),
                    no_omission=not bool(model._concept_prediction.active[b])))
                understood=selected['record'].understanding_field
                generated=selected.get('generation')
                if understood is None or generated is None:
                    comparisons.append(dict(batch=b,unavailable=True));continue
                left,right=dense(understood,b),dense(generated.field,b)
                zero=understood.evidence.new_zeros(2)
                ids=sorted(left.keys()|right.keys())
                delta=torch.stack([(left.get(cid,zero)-right.get(cid,zero)).abs() for cid in ids]) if ids else zero.new_empty(0,2)
                comparisons.append(dict(batch=b,exact=not bool(delta.any()),
                    mean_absolute=float(delta.mean()) if delta.numel() else 0.,
                    max_absolute=float(delta.max()) if delta.numel() else 0.,
                    coordinates=delta.numel(),
                    understood={cid:value.tolist() for cid,value in left.items()},
                    generated={cid:value.tolist() for cid,value in right.items()}))
            with (args.output/'fields.jsonl').open('a') as f:
                f.write(json.dumps(dict(presentation=count,comparisons=comparisons,
                    orders=[dict(order=k,occupancy=orders[k],capacity=cap)
                            for k,cap in enumerate(owner._order_caps())]))+'\n')
            # A preserved read after the close dates the first nonzero lane;
            # it adds no presentation and cannot alter the training stream.
            phase='after-close'
            for text in (('This is Felix.','What is this?') if args.timeline else ()):
                record=Assessment.read_sentence(model,text,trace=True)
                with (args.output/'timeline.jsonl').open('a') as f:f.write(json.dumps(dict(presentation=count,record=record))+'\n')
            phase='training'
            if count%16==0: print('sentences',count,flush=True)
            choices.clear()
        return result
    BasicModel._commit_sentence=kept
    def assess(model):
        nonlocal phase
        phase='assessment'
        with Assessment.preserved(model):
            measurements = idea_read_measurements(model._concept_owner(), idea_samples)
        (args.output/'idea-read.json').write_text(json.dumps(measurements,indent=2)+'\n')
        curriculum=json.loads(Path('data/corpus-lessons.json').read_text())
        texts=list(dict.fromkeys(text for stage in curriculum['stages'] for doc in stage['documents'] for record in doc['records'] for text in record['sentences']))
        if args.assessment_texts:
            texts=[json.loads(line)['text'] for line in args.assessment_texts.read_text().splitlines()]
        # All permutations preserve the source words exactly. The extra places
        # are assessment only, with no capitalization or spelling rewrite.
        records=[Assessment.read_sentence(model,text,trace=True) for text in texts]
        targets={word['form'] for record in records for word in record['words']
                 if word['native_field'] is not None and word['attention']=='left'
                 and all(word['native_field']['lanes'][lane]['value']>0 for lane in ('for','against'))}
        targets.update(('is','this'))
        extra=list(dict.fromkeys(text for word in sorted(targets)
                   for text in (f'is {word} Felix.',f'Felix is {word}.')))
        extra.append('is Felix This.')
        if not args.assessment_texts:
            records.extend(Assessment.read_sentence(model,text,trace=True) for text in extra if text not in texts)
        for record in records:
            with (args.output/'assessment.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print('assessment complete',flush=True)
        phase='ladder'
        from AssessmentProbe import known_probe
        from util import parse
        words=list(dict.fromkeys(form.strip() for text in texts
                   for form,_ in parse(text,lex='words') if form.strip()))
        report=known_probe(model,dict(word='Felix',words=words,name='Felix',noun='cat',
                           verb='is',pronoun='This'),source='explore-1 after four epochs',trace=True)
        (args.output/'ladder.json').write_text(json.dumps(report,indent=2)+'\n')
    def deposit_sink(record):
        record.update(presentation=count+1 if phase=='training' else count,phase=phase)
        with (args.output/'deposits.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    def look_sink(record):
        record.update(presentation=count+1 if phase=='training' else count, phase=phase)
        with (args.output/'look-lanes.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')

    from unittest.mock import patch
    from SentenceLooks import Looks
    run_looks = Looks.run
    def measured_looks(path, *a, **kw):
        reading = run_looks(path, *a, **kw)
        spans = path.walk.state.spans
        with (args.output/'surprises.jsonl').open('a') as stream:
            for step, (field, decision) in enumerate(zip(path.fields, path.decisions, strict=True)):
                extents = spans if step == 0 else field['where'][:, None]
                for b in field['active'].nonzero().flatten().tolist():
                    for j, (lo, hi) in enumerate(extents[b].tolist()):
                        if hi <= lo:
                            continue
                        positions = ((spans[b, :, 0] == lo) & (spans[b, :, 1] == hi)).nonzero().flatten()
                        outcome = ('several words' if not len(positions) else
                                   'interpreted' if reading.accepted[b, int(positions[0])] else 'left')
                        stream.write(json.dumps(dict(phase=phase, presentation=count+1, step=step,
                            where=[lo,hi], outcome=outcome, pair=field['surprise'][b,j].tolist()))+'\n')
        return reading

    with patch.object(Looks, 'run', measured_looks), native_provenance(sink), word_deposits(deposit_sink), look_lane_counts(look_sink):
        measure('doc/benchmarks/2026-11-11-whole-paths/configurations/explore-1.xml',
                'data/corpus-lessons.json',args.output/'native-lesson',eager=True,after_training=assess)
    print('completed',count,flush=True)

if __name__=='__main__':main()
