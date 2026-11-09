"""Measure staged text learning without supplying an order or a reading.

The JSON's measurement forms are read only after each epoch. They never become
grammar choices or definitions; supplied answers use the existing loss path.
Every failed learning criterion remains in the report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

LESSON_CONFIG = Path(__file__).resolve().parents[1] / 'data/MM_corpus_lessons.xml'


def readiness_report(owner):
    """The same measured readiness in presentation and lexical records."""
    return dict(trust_measurements=owner.trust_measurements(), gates=owner.gate_measurements())


def membership_report(alloc, identity):
    """The durable row and both evidence poles for every named membership."""
    from Memory import SYMBOL
    whole=(SYMBOL,identity)
    result=[]
    for member in alloc.parts.parts(whole):
        row=alloc.parts.find(member,whole)
        if row is None:raise RuntimeError('membership index has no durable witness')
        result.append(dict(reference=list(member),row=row,
            for_evidence=float(alloc.parts.truth_plus[row]),
            against_evidence=float(alloc.parts.truth_minus[row])))
    return result


def lexical_report(model, forms):
    from Spaces import _concept_alloc_of
    from Memory import SYMBOL
    cs = model._concept_owner()
    alloc = _concept_alloc_of(cs)
    speech = model.languageSpace.parts_of_speech
    result = {}
    for form in forms:
        word = cs.interpret.word_for_form(form)
        objects = () if word is None else cs.interpret.denotations(word)
        result[form] = dict(word=word, objects=[dict(identity=obj, order=alloc.order_of(obj),
            members=list(alloc.parts.parts((SYMBOL, obj))),
            membership_witnesses=membership_report(alloc,obj)) for obj in objects],
            arguments=[] if word is None else [dict(argument=arg, concept=speech.concepts[arg],
                score=float(score), member=word in speech.members(arg))
                for arg, score in zip(speech.arguments, speech.similarities(word))])
    return dict(forms=result, **readiness_report(cs))


def stage_documents(data, stage, *, split='train'):
    """Math records supply loss-side answers; document addresses stay grouped."""
    import torch
    from math_chain_corpus import ChainDocument, flatten
    records, sources = [], []
    for document in stage['documents']:
        offset = 0
        for record in document['records']:
            doc = ChainDocument(**dict(record, sentences=tuple(record['sentences'])))
            records.append(doc)
            sources.extend((document['document'], offset+i) for i in range(len(doc.sentences)))
            offset += len(doc.sentences)
    texts, targets, addresses = flatten(records, supplied=True)
    setattr(data,split+'_input',texts)
    setattr(data,split+'_output',[torch.zeros(1) for _ in texts])
    if data.text_answers is None:
        data.text_answers = {}
    data.text_answers[split] = targets
    data.has_supervised_outputs = any(target is not None for target in targets)
    data.source_addresses[split] = [dict(address, split=split, document=key, sentence=sentence)
        for address, (key, sentence) in zip(addresses, sources)]
    return texts, targets


def acquisition(model, measurement, *, relations):
    from Memory import SYMBOL
    lexical = lexical_report(model, [*measurement['names'], measurement['kind']])['forms']
    members = [{obj['identity'] for obj in lexical[name]['objects'] if obj['order'] == 1}
               for name in measurement['names']]
    ideas = all(members) and (len(members)<2 or any(a!=b for a in members[0] for b in members[1]))
    kinds = [obj for obj in lexical[measurement['kind']]['objects'] if obj['order'] == 2
             and all(any([SYMBOL, member] in [witness['reference'] for witness in obj['membership_witnesses']
                                             if witness['for_evidence']>0]
                         for member in choices) for choices in members)] if ideas else []
    return dict(ideas_acquired=ideas, relations_acquired=bool(kinds),
                passed=bool(kinds) if relations else ideas,
                names=lexical, qualifying_classes=[obj['identity'] for obj in kinds])


def observed_sides(owner, references):
    """Resolve the kept operands in sentence order, preserving an unbound side."""
    from Memory import SYMBOL, OCCURRENCE, RELATION
    store = owner._closed_clause_store()
    result = []
    for ref in (*references, None, None)[:2]:
        current, visited, symbol = ref, set(), None
        while current is not None and current not in visited:
            visited.add(current)
            tag, identity = current
            if tag == SYMBOL:
                symbol = identity
                break
            if tag != OCCURRENCE:
                break
            location = store.index_of_row(identity)
            if location is None:
                break
            token = owner.mtm.token_of(store.occurrence_of(location))
            if token is not None:
                entry = owner.mtm.entries[token]
                symbol = entry.symbol if entry.symbol is not None else entry.native
                if symbol is not None:
                    break
            native = dict(store.raw_meaning_of(location).bindings).get('_identity')
            if native is None:
                break
            if native[0] == 'sym':
                current = (RELATION if store.parts.contains(native[1]) else SYMBOL, native[1])
            elif native[0] == 'ltm' and native[1] == store.address_domain:
                current = (OCCURRENCE, native[-1])
            else:
                break
        result.append(dict(reference=ref, symbol=symbol,
                           order=None if symbol is None else owner._concept_source_order(symbol)))
    return result


def observed_field(model, field):
    """Read the kept denotation and its actual witnesses, never a POS guess."""
    if field is None:
        return dict(order=None,witnessed=False,references=[],meaning=None)
    owner=model._concept_owner()
    def references(meaning):
        result=[ref for ref in meaning.role_refs if ref is not None]
        for child in meaning.constituents:result.extend(references(child))
        return result
    refs=references(field.meaning)
    orders=[]
    for ref in refs:
        if ref[0]=='sym':orders.append(dict(reference=ref,order=owner._concept_source_order(ref[1])))
    store=owner._closed_clause_store()
    witnessed=store is not None and store.location(field.row_id) is not None
    return dict(order=field.order,row_id=field.row_id,witnessed=witnessed,
                references=orders,meaning=field.meaning.metadata(),evidence=field.evidence)


def phase_acquisition(model, measurement, observations, *, relations=False):
    mode=measurement.get('mode','kind' if relations else 'proper')
    if mode in ('kind','proper'):
        return acquisition(model,measurement,relations=mode=='kind')
    if mode!='observations':raise ValueError('unknown measurement mode')
    tests=[]
    for requirement in measurement['utterances']:
        matches=[row for row in observations if row['text']==requirement['text']]
        passed=any(row['order']==requirement['order']
            and (row['witnessed'] or not requirement.get('witness_required',True))
            and ('operand_order' not in requirement or any(ref['order']==requirement['operand_order']
                                                          for ref in row['references'])) for row in matches)
        tests.append(dict(requirement=requirement,readings=matches,passed=passed))
    return dict(passed=bool(tests) and all(row['passed'] for row in tests),tests=tests)


def heldout(model, stage, data):
    """Untrained held-out recombinations; their answers never admit forms."""
    from MathChainTraining import present
    from BindingAnswers import matches
    texts,targets=stage_documents(data,dict(documents=stage['heldout']),split='validation')
    rows=[]
    def scored(model,split,indices,result):
        fields=model._sentence_fields.get(0,())
        answers=getattr(model,'_last_what_answers',())
        for b,index in enumerate(indices):
            field=fields[b] if b<len(fields) else None
            target=targets[index];actual=answers[b] if b<len(answers) else None
            rows.append(dict(text=texts[index],supplied=target,field=observed_field(model,field),
                answer=None if actual is None else actual.text,
                text_correct=None if target is None else actual is not None and actual.text==str(target),
                bound_correct=None if target is None else field is not None and bool(matches(model,field.meaning,str(target)))))
    # Evaluation may observe the input but may not grow its vocabulary from
    # the held-out records. The records reuse admitted training forms.
    spaces=(model.inputSpace,model.perceptualSpace,model.conceptualSpace,model.wholeSpace)
    old=[getattr(space,'_online_learning_frozen',False) for space in spaces]
    try:
        for space in spaces:space._online_learning_frozen=True
        present(model,split='validation',optimizer=None,batch_size=1,after_batch=scored,epoch_boundary=False)
    finally:
        for space,value in zip(spaces,old):space._online_learning_frozen=value
    return rows


def measure(config, lessons, output, *, eager=False, presentations=None, curriculum_name='lesson-13', gradient_report=False):
    import torch
    from data import TheData
    from Models import BaseModel
    from MathChainTraining import present
    from Forgetting import model_boundary
    from BindingAnswers import matches, cost
    from Memory import SYMBOL
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    curriculum = json.loads(Path(lessons).read_text())
    if curriculum_name=='final':curriculum=curriculum['final_round']
    elif curriculum_name!='lesson-13':raise ValueError('unknown curriculum')
    if curriculum['version'] != 2 or curriculum['epochs'] < 1:
        raise ValueError('unsupported or empty lesson protocol')
    if presentations is not None:
        curriculum['epochs'] = presentations
    data = TheData
    data.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                    for split in ('train', 'validation', 'test')})
    data.source_manifest = dict(dataset='bounded_corpus_lesson',
        lessons_sha256=hashlib.sha256(Path(lessons).read_bytes()).hexdigest())
    if eager:
        import util
        util.TheCompileBackend = 'none'
        def eager_while(condition, body, values):
            while bool(condition(*values)):
                values = body(*values)
            return values
        torch.while_loop = eager_while
    started = time.monotonic()
    model = BaseModel.from_config(str(config), data=data)[0]
    if model.inputSpace.data is not data:
        raise RuntimeError('lesson data differs from the native input owner')
    optimizer = model.getOptimizer(lr=.001)
    model._measure_sentence_gradients = bool(gradient_report)
    model._observe_lesson_sides = True
    records, phases = [], []
    forms = list(dict.fromkeys(form for stage in curriculum['stages'] for form in stage['measure']))
    metadata = dict(torch_version=str(torch.__version__), config=str(config),
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        lessons=str(lessons), curriculum=curriculum_name,manifest=data.source_manifest, eager=eager,
        epochs=curriculum['epochs'], before=lexical_report(model, forms),
        separate_chooser=model.separate_chooser, reconstruction_jitter=model.reconstruction_jitter,
        gradient_report=bool(gradient_report))
    (output / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        for stage in curriculum['stages']:
            texts, targets = stage_documents(data, stage)
            from Corpus import admit_training_forms
            admitted=admit_training_forms(model,texts,targets)
            (output/(stage['name']+'-admission.json')).write_text(json.dumps([
                dict(torch_version=str(torch.__version__), form=raw.decode('utf8','surrogateescape'),word=word)
                for raw,word in admitted],indent=2)+'\n')
            for repetition in range(curriculum['epochs']):
                model_boundary(model, epoch_boundary=True)
                answers, fold_counts, observations, attention, placements, step_times = [], [], [], [], [], []
                training_times, diagnostic_times = [], []
                last_step = time.monotonic()
                last_diagnostic = getattr(model, '_gradient_diagnostic_seconds', 0.)
                def scored(model, split, rows, result):
                    nonlocal last_step, last_diagnostic
                    now = time.monotonic()
                    step_times.append(now-last_step)
                    diagnostic = getattr(model, '_gradient_diagnostic_seconds', 0.) - last_diagnostic
                    diagnostic_times.append(diagnostic)
                    training_times.append(now-last_step-diagnostic)
                    owner = model._concept_owner()
                    opened = getattr(model, '_last_open_attention', None)
                    if opened is not None:
                        expected = owner._expected_knowing.detach()
                        attention.append(dict(brackets=opened['brackets'].tolist(),
                            seen=opened['seen'], expected=[dict(rows=(value.amax(-1)>0).nonzero().flatten().tolist(),
                                pairs=value[value.amax(-1)>0].tolist()) for value in expected],
                            accepted=(None if not hasattr(model, '_attention_words') else
                                      model._attention_words.accepted.detach().tolist())))
                    codes = getattr(owner.similarity_codebook, 'mereology', None)
                    if codes is not None:
                        occupied = codes.placement_witnesses > 0
                        placements.append(dict(rows=occupied.nonzero().flatten().tolist(),
                            directions=codes.placement[occupied].tolist(),
                            witnesses=codes.placement_witnesses[occupied].tolist(),
                            valid=codes.placement_valid[occupied].tolist(),
                            shape=list(codes.placement.shape), pointer=codes.placement.data_ptr()))
                    fold=getattr(model._concept_owner(),'_last_definition_fold',None)
                    if fold is not None:
                        fold_counts.append((fold['both'],fold['count']))
                    fields = model._sentence_fields.get(0, ())
                    thoughts = dict(getattr(model, '_last_closing_thoughts', ()))
                    actual = getattr(model, '_last_what_answers', ())
                    for batch, source in enumerate(rows):
                        field = fields[batch] if batch < len(fields) else None
                        observed = observed_field(model, field)
                        observed.update(readiness_report(owner))
                        readings = getattr(model, '_last_leaf_readings', ())
                        observed['leaf_readings'] = readings[batch] if batch < len(readings) else None
                        refs = getattr(model, '_lesson_side_refs', ())
                        observed['sides'] = observed_sides(owner, refs[batch] if batch < len(refs) else ())
                        observations.append(dict(text=texts[source], **observed))
                        target = targets[source]
                        if target is None:
                            continue
                        answer = str(target)
                        field = fields[batch] if batch < len(fields) else None
                        thought = thoughts.get(batch)
                        meaning = thought.meaning if thought is not None else None if field is None else field.meaning
                        produced = actual[batch] if batch < len(actual) else None
                        answers.append(dict(source=data.source_addresses[split][source],
                            question=texts[source], supplied=answer,
                            text=None if produced is None else produced.text,
                            available=False if produced is None else produced.available,
                            bound_correct=False if meaning is None else bool(matches(model, meaning, answer)),
                            binding_cost=None if meaning is None else float(cost(model, meaning, answer)),
                            answer_trial_costs=model._last_sentence_credit['components'][batch, :, 2].tolist()))
                    last_diagnostic = getattr(model, '_gradient_diagnostic_seconds', 0.)
                    last_step = time.monotonic()
                summary = present(model, split='train', optimizer=optimizer, batch_size=1,
                                  after_batch=scored, epoch_boundary=False)
                if summary['sentences'] != len(texts):
                    raise RuntimeError('native lesson did not consume every supplied sentence')
                owner = model._concept_owner()
                alloc = owner._concept_allocator
                row = dict(torch_version=str(torch.__version__), phase=stage['name'], epoch=repetition + 1,
                    elapsed_seconds=time.monotonic() - started, **summary,
                    lexical=lexical_report(model, forms), **readiness_report(owner),
                    null_concept=owner.null_concept_read(),
                    raised_classes=[dict(symbol=cid, order=alloc.order_of(cid),
                        members=list(alloc.parts.parts((SYMBOL, cid))),
                        membership_witnesses=membership_report(alloc,cid)) for cid in sorted(alloc.raised)],
                    candidates=[dict(key=list(key), order=entry.order, count=entry.count,
                        symbol=entry.symbol, support=list(entry.support.items()),readings=entry.readings,
                        both=all(any(reading[pole] > 0 for reading in entry.readings) for pole in (2,3)))
                        for key, entry in owner.mtm.entries.items() if entry.order is not None],
                    answers_scored=answers, observations=observations,
                    ltm1_rows=[dict(row=i, references=alloc.parts.references(i),
                        evidence=[float(alloc.parts.truth_plus[i]), float(alloc.parts.truth_minus[i])],
                        percept=alloc.parts.percept[i].tolist()) for i in alloc.parts.indices().tolist()],
                    attention=attention, placements=placements, step_seconds=step_times,
                    step_training_seconds=training_times, step_diagnostic_seconds=diagnostic_times,
                    class_both=dict(both=sum(x for x,_ in fold_counts),
                        readings=sum(n for _,n in fold_counts),
                        fraction=(sum(x for x,_ in fold_counts)/sum(n for _,n in fold_counts)
                                  if sum(n for _,n in fold_counts) else None)),
                    acquisition=phase_acquisition(model, stage.get('measurement',curriculum['measurement']),observations,
                        relations=stage.get('produces') is not None),
                    native_forms=len(model.perceptualSpace.percept_store.identity.words),
                    rung0_audit=getattr(model, '_last_rung0_identity_audit', None))
                if gradient_report:
                    from SentenceTraining import gradient_receipt
                    row['gradient_report'] = gradient_receipt(model, clear=True)
                records.append(row)
                with (output / 'epochs.jsonl').open('a') as target:
                    target.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            last = records[-1]
            phase = dict(phase=stage['name'], exit=last['acquisition'],
                lexical=last['lexical'], raised_classes=last['raised_classes'],
                candidates=last['candidates'], answers_scored=[answer for record in records
                    if record['phase'] == stage['name'] for answer in record['answers_scored']])
            if stage.get('measurement',{}).get('mode')=='proper':
                histories=[row['lexical']['forms'] for row in records if row['phase']==stage['name']]
                stable={name:sorted(set.intersection(*[{obj['identity'] for obj in report[name]['objects']
                    if obj['order']==1} for report in histories])) for name in stage['measurement']['names']}
                phase['stable_individuals']=stable
                phase['exit']['passed'] &= all(stable.values())
            if stage.get('heldout'):
                phase['heldout']=heldout(model,stage,data)
            if stage.get('produces'):
                phase['produces'] = stage['produces']
                phase['artifact_written'] = False
                if phase['exit']['passed']:
                    path = output / (stage['produces'] + '.pt')
                    model.save_weights(str(path))
                    phase.update(artifact_written=True, path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            phase['torch_version'] = str(torch.__version__)
            phases.append(phase)
            (output / 'phases.json').write_text(json.dumps(phases, indent=2) + '\n')
    finally:
        model.End()
        model.symbolSpace.soft_reset()
    (output / 'result.json').write_text(json.dumps(dict(metadata=metadata, records=records,
        phases=phases, passed=all(phase['exit']['passed'] for phase in phases)), indent=2) + '\n')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=LESSON_CONFIG)
    parser.add_argument('--lessons', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--gradient-report', action='store_true')
    parser.add_argument('--presentations', '--epochs', type=int)
    parser.add_argument('--curriculum',choices=('lesson-13','final'),default='lesson-13')
    args = parser.parse_args()
    measure(args.config, args.lessons, args.output, eager=args.eager,
            presentations=args.presentations,curriculum_name=args.curriculum,
            gradient_report=args.gradient_report)


if __name__ == '__main__':
    main()
