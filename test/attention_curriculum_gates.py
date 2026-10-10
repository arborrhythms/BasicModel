"""Section 7.14: one fresh unseeded generated curriculum per mode, no retries."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import traceback
import pytest
import torch
import util
from AttentionLesson import reading_lesson, prompt_need
from attention_field_gates import observation, serial, words_from_leaves
from bounded_tests import source_snapshot
from test_compiled_word_chunk import _tiny_canonical_model

PERCEPTS = ['luma', 'nemi', 'tavo', 'suri']
INTERRUPTION = ['pako', 'feni', 'balo', 'runi']
NOVEL = ['dax', 'wug', 'nep', 'zot']
ORDER_CE_MAX = .01  # nats per read, on every row at presentation 64
STAGES = [
 dict(name='reading_two', stage=3, lesson=True, fields=['red blue','blue red','green gold','gold green']),
 dict(name='reading_four', stage=3, lesson=True, fields=['red blue green gold','gold green blue red','blue gold red green','green red gold blue']),
 dict(name='identities', stage=4, fields=['cat is animal','dog is animal','x is three','y is four'],
      questions=['what is cat ?','what is dog ?','what is x ?','what is y ?'],answers=['animal','animal','three','four']),
 dict(name='noun_phrase', stage=4, fields=['the red ball','the blue box','the green ball','the gold box'],
      questions=['what is red ?','what is blue ?','what is green ?','what is gold ?'],answers=['ball','box','ball','box']),
 dict(name='noun_verb', stage=4, fields=['the cat sleeps','the dog runs','the dog sleeps','the cat runs'],
      questions=['what sleeps ?','what runs ?','what sleeps ?','what runs ?'],answers=['cat','dog','dog','cat']),
 dict(name='relations', stage=4, fields=['cat is on mat','dog is on rug','x is bigger than y','y is bigger than x'],
      questions=['what is on mat ?','what is on rug ?','what is bigger than y ?','what is bigger than x ?'],answers=['cat','dog','x','y'])]


def present(model, stage, *, train=False, optimizer=None):
    from What import What
    fields = stage['fields']; B=len(fields)
    data=model.inputSpace.data
    supervised='targets' in stage
    data.has_supervised_outputs=supervised
    kwargs={}
    if supervised:
        data.train_input=fields
        data.train_output=[torch.zeros(1) for _ in fields]
        data.text_answers={'train':stage['targets']}
        data.source_addresses={'train':[dict(document=b, sentence=0, document_key=f'curriculum:{b}',sentence_index=0,split='train') for b in range(B)]}
        factory=What.supervised if train else What.inference
        kwargs=dict(questions=tuple(factory(b,split='train',prompt=stage['prompts'][b]) for b in range(B)),source_rows=list(range(B)))
    inputs=model.inputSpace.prepInput(fields)
    lesson=reading_lesson(model,[text.split() for text in fields]) if train and stage.get('lesson') else nullcontext()
    need=prompt_need(model,stage['prompts']) if supervised else nullcontext()
    with torch.set_grad_enabled(train),lesson,need:
        model.runBatch(train=train,optimizer=optimizer,batchSize=B,split='train',
            batch_override=(inputs,torch.zeros(B,1) if supervised else torch.empty(B,0)),**kwargs)
        result=observation(model)
        raw=model._last_sentence_field
        record=model._last_sentence_understanding
        trials=raw['trials'];wins=raw['wins']
        recon=torch.where(wins[:,None,None],trials[-1]['reconstructed'],trials[0]['reconstructed'])
        bad=torch.where(wins[:,None],trials[-1]['unavailable'],trials[0]['unavailable'])
        orders=[];counts=[];compact=torch.zeros_like(recon)
        for b in range(B):
            selected=trials[-1 if bool(wins[b]) else 0]
            order=[int(selected['positions'][read['action'][b]]) for read in selected['reads'] if bool(read['admitted'][b].any())]
            orders.append(order);counts.append(len(order))
            if order:compact[b,:len(order)]=recon[b,order]
        readbacks=[' '.join(row) for row in words_from_leaves(compact,torch.tensor(counts),record.primed)]
        free=model._last_decoder_trace
        free_text=[' '.join(row) for row in words_from_leaves(*free[:2],record.primed)]
        result.update(readbacks=readbacks,exact_readbacks=sum(a==b for a,b in zip(readbacks,fields)),
            free_readbacks=free_text,free_exact=sum(a==b for a,b in zip(free_text,fields)),
            inverse_unavailable=serial(bad),orders=orders,
            word_rows=serial(record.word_rows),word_ids=serial(model.inputSpace._ar_word_concept_ids),
            lesson_cross_entropy=serial(trials[0]['lesson_cross_entropy']))
        single_reads=all(n==int(active) for trial in result['trials'] for read in trial['reads']
                    for n,active in zip(read['supported_words'],read['active'],strict=True))
        complete=counts==[len(text.split()) for text in fields]
        result['read_gate_passed']=single_reads and complete and all(
            order==list(range(len(text.split()))) for order,text in zip(orders,fields,strict=True))
        result['reconstruction_passed']=result['exact_readbacks']==B
        result['passed']=(result['read_gate_passed'] if stage.get('lesson') else
                          result['reconstruction_passed'] and complete and single_reads)
        if supervised:
            actual=getattr(model,'_last_what_actual',())
            construction=getattr(model,'_last_answer_construction',None)
            answer=list(getattr(construction,'texts',()) or ())
            result['answer_values']=serial([getattr(value,'what',None) for value in actual])
            # The output adapter is numeric even for a text lesson. Use the
            # existing lexical inverse's text report, never the target or
            # input reconstruction, for an answer readback. Training omits
            # that no-grad report; the final free evaluation supplies it.
            result['answer_readbacks']=answer if answer else None
            result['answer_exact']=(sum(a==b for a,b in zip(answer,stage['targets']))
                                    if answer else None)
            result['passed'] &= result['answer_exact']==B
    model.flush_word_buffers();model.dispatch_soft_reset()
    return result


def identities(model, fields):
    index=model._concept_owner().definitions
    return [([] if (word:=index.word(form=form)) is None else
             list(map(int,index.objects(word)))) for form in fields]


def run(mode, folder, epochs, control=None, *, through_stage=4):
    folder.mkdir(parents=True,exist_ok=False);results=[]
    def save(): (folder/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(util,'TheCompileBackend','none')
        model=_tiny_canonical_model(folder,patch,word_buckets='16',input_width=128,concept_rows=128,batch_size=4,
            training_overrides={'intraLossWeight':0.,'reconstructionPlacement':'eager'})
        model._tensor_peer_while_eager=True
        model.inputSpace.data.has_supervised_outputs=False
        if mode=='disabled':model.candidate_attention=None
        optimizer=model.getOptimizer(lr=.01)
        def control_passed(name):
            return control is None or next((row.get('passed',False) for row in control if row['name']==name),False)
        try:
            presentations=[]
            for label,fields in [('first',PERCEPTS),('repeat',PERCEPTS),('interruption',INTERRUPTION),('after_gap',PERCEPTS)]:
                report=present(model,dict(fields=fields));report['identities']=identities(model,fields)
                presentations.append(dict(label=label,fields=fields,report=report))
            original=presentations[0]['report']['identities']
            same=original==presentations[1]['report']['identities']==presentations[3]['report']['identities']
            # Resolve the retained symbolic references through the ordinary
            # read-only capability after the interruption; it cannot mint.
            from Queries import _existing_row,_payload_row
            recalled=[]
            for refs in original:
                recalled.append([dict(identity=cid,row=_existing_row(model._concept_owner(),('sym',cid)),
                    finite=bool(torch.isfinite(_payload_row(model._concept_owner(),_existing_row(model._concept_owner(),('sym',cid)))).all())) for cid in refs])
            passed=same and all(refs for refs in original) and all(r['finite'] for row in recalled for r in row) and all(presentations[i]['report']['passed'] for i in [0,1,3])
            results.append(dict(name='identity_permanence',stage=1,presentations=presentations,recall=recalled,
                same_rows=same,passed=passed,interpretable=control_passed('identity_permanence'),departures_attempted=0,departures_kept=0,lesson_cross_entropy=None));save()
            prior=identities(model,NOVEL);presentations=[]
            for label,fields in [('known',PERCEPTS),('novel_first',NOVEL),('novel_second',NOVEL)]:
                report=present(model,dict(fields=fields));report['identities']=identities(model,fields)
                presentations.append(dict(label=label,fields=fields,report=report))
            minted=all(not row for row in prior) and presentations[1]['report']['identities']==presentations[2]['report']['identities']
            passed=minted and all(p['report']['passed'] for p in presentations)
            results.append(dict(name='identifying_words',stage=2,presentations=presentations,novel_before=prior,
                minted_once=minted,passed=passed,interpretable=control_passed('identifying_words'),departures_attempted=0,departures_kept=0,lesson_cross_entropy=None));save()
            pending=[stage for stage in STAGES if stage['stage']<=through_stage]
            while pending:
                stage=pending.pop(0);name=stage['name'];start=time.monotonic()
                initial=present(model,stage);attempted=kept=0;ce=[]
                with (folder/(name+'.jsonl')).open('w') as log:
                    for epoch in range(epochs):
                        report=present(model,stage,train=True,optimizer=optimizer)
                        backup=report['backup']
                        if backup is not None:
                            attempted+=sum(backup['rows']);kept+=sum(a and b for a,b in zip(backup['rows'],report['wins']))
                        if report['lesson_cross_entropy'] is not None:ce.append(report['lesson_cross_entropy'])
                        log.write(json.dumps(dict(epoch=epoch+1,**report))+'\n');log.flush()
                final=present(model,stage)
                ce_passed=(bool(ce) and max(ce[-1])<=ORDER_CE_MAX
                           and sum(ce[-1])<sum(ce[0])) if stage.get('lesson') and mode=='enabled' else None
                row=dict(name=name,stage=stage['stage'],corpus=stage,initial=initial,final=final,passed=final['passed'],
                    interpretable=control_passed(name),departures_attempted=attempted,departures_kept=kept,
                    lesson_cross_entropy=ce,elapsed_seconds=time.monotonic()-start)
                row['lesson_ce_passed']=ce_passed
                if ce_passed is not None:row['passed'] &= ce_passed
                results.append(row);save()
                print(mode,name,final['exact_readbacks'],'/4', 'answers',final.get('answer_exact'), 'departures',attempted,kept,flush=True)
                if stage['stage']==4 and 'questions' in stage:
                    if final['passed'] and control_passed(name):
                        fields=stage['fields']
                        pending[:0]=[
                            dict(name=name+'_asked_parts',stage=4,fields=fields,prompts=['the second word','the middle phrase']*2,
                                targets=[' '.join(text.split()[1:2 if b%2==0 else 3]) for b,text in enumerate(fields)]),
                            dict(name=name+'_supplied_answer',stage=4,fields=fields,prompts=stage['questions'],targets=stage['answers'])]
                    else:
                        for suffix in ['asked_parts','supplied_answer']:
                            results.append(dict(name=name+'_'+suffix,stage=4,not_run='Sentence reconstruction control or enabled reconstruction did not pass.',passed=False))
                        save()
        except Exception as error:
            (folder/'error.txt').write_text(traceback.format_exc());raise
        finally:
            save();model.End();model.symbolSpace.soft_reset();torch._dynamo.reset()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path);parser.add_argument('--epochs',type=int,default=64)
    parser.add_argument('--through-stage',type=int,choices=[3,4],default=4)
    args=parser.parse_args();torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];folder=args.destination.resolve();folder.mkdir(parents=True,exist_ok=False)
    source=source_snapshot(root)
    plan=dict(source=source,epochs=args.epochs,seed=None,retries=0,modes=['disabled','enabled'],fresh_models=2,
        learning_rate=.01,attention_optimizer='Adam',order_ce_max=ORDER_CE_MAX,
        through_stage=args.through_stage,percepts=PERCEPTS,interruption=INTERRUPTION,novel=NOVEL,
        stages=[stage for stage in STAGES if stage['stage']<=args.through_stage],
        selection='Only stage 3 training names and teacher-forces each word; every measurement is a hard free read. Disabled is source order.',
        reconstruction='Actual derivation with occurrence co-operands; undefined coordinates uncharged and reported. List reconstruction, free decoder and byte identity are separate from the stage 3 reads-and-CE gate.',
        answers='Asked parts and supplied answers use the existing output-owned supervised answer path. Targets do not enter the scorer.',
        protocol='Each mode proceeds once through the declared corpus. Failed controls make the corresponding results uninterpretable; asked output gates require sentence reconstruction to pass in both modes.')
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    try:
        disabled=run('disabled',folder/'disabled',args.epochs,through_stage=args.through_stage)
        run('enabled',folder/'enabled',args.epochs,disabled,through_stage=args.through_stage)
    finally:(folder/'source_check.json').write_text(json.dumps(dict(source_matched=source==source_snapshot(root)),indent=2)+'\n')

if __name__=='__main__':main()
