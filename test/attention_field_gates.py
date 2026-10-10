"""Support-mask receipt: one fresh unseeded campaign per mode, no retries."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time
import pytest
import torch
import util
from AttentionLesson import part_lesson
from bounded_tests import source_snapshot
from test_compiled_word_chunk import _tiny_canonical_model

STAGES = (
    dict(name='one_word', fields=['red', 'blue', 'green', 'gold']),
    dict(name='two_words_in_turn', fields=['red blue', 'blue red', 'green gold', 'gold green']),
    dict(name='sentence_four_words', fields=['red blue green gold', 'gold green blue red', 'blue gold red green', 'green red gold blue']),
    dict(name='asked_parts', fields=['red blue green', 'gold green blue', 'blue gold red', 'green red gold'],
         prompts=['second word', 'middle phrase', 'second word', 'middle phrase'],
         targets=['blue', 'green blue', 'gold', 'red gold']),
)
THREE_WORDS = dict(name='sentence_three_words',
    fields=['red blue green', 'gold green blue', 'blue gold red', 'green red gold'])


def serial(value):
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value, dict):return {key:serial(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [serial(item) for item in value]
    return value


def words_from_leaves(leaves, count, bank):
    from SentenceUnderstanding import readback_scores
    scores = torch.stack([readback_scores(leaves[:, i], bank.codes, bank.weights,
        percept_width=bank.percept_width, meaning_start=bank.meaning_start) for i in range(leaves.shape[1])], 1)
    scores = scores.masked_fill(~bank.valid[:, None], -torch.inf)
    chosen = scores.argmax(-1)
    rows=[]
    for b in range(len(leaves)):
        words=[]
        for i in range(int(count[b])):
            j=int(chosen[b,i])
            words.append('<unresolved>' if float(scores[b,i,j]) <= 0 else
                bytes(bank.bytes[b,j][bank.byte_valid[b,j]].tolist()).decode('utf8'))
        rows.append(words)
    return rows


def observation(model):
    report=serial(model._last_sentence_field)
    if report is None:raise RuntimeError('native support traversal did not run')
    record=model._last_sentence_understanding
    bank=record.primed
    report['cost']=serial(model._last_field_cost)
    report['meter']=[dict(m.counts) for m in model._field_meters]
    report['bracket_meter']=[dict(m.counts) for m in model._attention_meters]
    report['identity_audit']=serial(model._sentence_reconstructions[-1][2]/math.log(256))
    staged=getattr(model,'_attention_forms',None)
    forms=None if staged is None else staged[0]
    for source, trial in zip(model._last_sentence_field['trials'],report['trials']):
        for read, item in zip(source['reads'],trial['reads']):
            live=read['admitted'].any(-1)
            candidates=[]; sizes=[]
            for b in range(len(live)):
                if not bool(live[b]):candidates.append(None);sizes.append(0);continue
                action=int(read['action'][b]);position=int(source['positions'][action])
                form=forms[b][position] if forms is not None else str(position)
                candidates.append(form.decode('utf8') if isinstance(form,bytes) else str(form))
                sizes.append(int(source['support'][b,action].ne(0).any(-1).sum()))
            item['candidate']=candidates
            item['support_size']=sizes
            item['supported_words']=read['admitted'].sum(-1).tolist()
            if 'decoded' in read:
                item['decoded_word']=words_from_leaves(read['decoded'][:,None],live.long(),bank)
            item.pop('decoded',None)
        trial.pop('support',None)
    return report


def lesson(model, stage):
    return part_lesson(model,stage['prompts'],stage['targets']) if 'prompts' in stage else nullcontext()


def measure(model,stage):
    model.eval()
    inputs=model.inputSpace.prepInput(stage['fields'])
    with torch.no_grad(),lesson(model,stage):
        model.runBatch(train=False,batchSize=4,batch_override=(inputs,torch.empty(4,0)))
        report=observation(model)
        record=model._last_sentence_understanding;bank=record.primed
        decoded=model._decode_conceptual_sentence(record.root,record.end_slots,record.end_depth,
            bank.codes,bank.valid,bank.weights,case_bank=bank.case_bank,
            terminal_valid=bank.terminal_valid, constituents=record.constituents,
            constituent_valid=record.constituent_valid, constituent_families=record.constituent_families)
        texts=[' '.join(row) for row in words_from_leaves(*decoded[:2],bank)]
    target=stage.get('targets',stage['fields'])
    report.update(readbacks=texts,exact_readbacks=sum(a==b for a,b in zip(texts,target)),
        end_depth=serial(record.end_depth),decoded_count=serial(decoded[1]),
        decoder_truncated=serial(decoded[2]),decoder_actions=serial(decoded[4]))
    report['passed']=report['exact_readbacks']==4
    if stage['name']=='two_words_in_turn':
        report['passed'] &= report['iterations']==[2]*4 and all(
            all(n==1 for n in read['supported_words']) for read in report['trials'][0]['reads'])
    model.flush_word_buffers();model.dispatch_soft_reset()
    return report


def run(mode,destination,epochs,control=None):
    destination.mkdir(parents=True,exist_ok=False)
    results=[];start=time.monotonic()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(util,'TheCompileBackend','none')
        model=_tiny_canonical_model(destination,patch,word_buckets='8',input_width=64,concept_rows=64,batch_size=4,
            training_overrides={'intraLossWeight':0.,'reconstructionPlacement':'eager'})
        model._tensor_peer_while_eager=True
        model.inputSpace.data.has_supervised_outputs=False
        if mode=='disabled':model.candidate_attention=None
        optimizer=model.getOptimizer(lr=.01)
        try:
            pending = list(STAGES)
            while pending:
                stage = pending.pop(0)
                if stage['name']=='asked_parts' and (not results[-1]['final']['passed'] or
                        (control is not None and not control[-1]['final']['passed'])):
                    results.append(dict(stage=stage,not_run='Stage 3 did not pass in both modes.'))
                    break
                if stage['name']=='asked_parts':
                    fields = results[-1]['stage']['fields']
                    stage = dict(stage, fields=fields, targets=[
                        ' '.join(text.split()[1:2 if i % 2 == 0 else 3]) for i,text in enumerate(fields)])
                initial=measure(model,stage);attempted=kept=0
                with (destination/(stage['name']+'.jsonl')).open('w') as log:
                    inputs=model.inputSpace.prepInput(stage['fields'])
                    for epoch in range(epochs):
                        model.train()
                        with lesson(model,stage):
                            model.runBatch(train=True,optimizer=optimizer,batchSize=4,
                                batch_override=(inputs,torch.empty(4,0)))
                        report=observation(model);backup=report['backup']
                        if backup is not None:
                            attempted+=sum(backup['rows'])
                            kept+=sum(a and b for a,b in zip(backup['rows'],report['wins']))
                        log.write(json.dumps(dict(epoch=epoch+1,**report))+'\n');log.flush()
                        model.flush_word_buffers();model.dispatch_soft_reset()
                final=measure(model,stage)
                result=dict(stage=stage,initial=initial,final=final,departures_attempted=attempted,
                    departures_kept=kept,elapsed_seconds=time.monotonic()-start)
                results.append(result)
                if stage['name']=='sentence_four_words':
                    shorten = (not final['passed'] if control is None else
                        any(row['stage']['name']=='sentence_three_words' for row in control))
                    if shorten:
                        pending.insert(0, THREE_WORDS)
                (destination/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                print(mode,stage['name'],final['exact_readbacks'],'/4',final['iterations'],
                      'departures',attempted,'kept',kept,flush=True)
        finally:
            (destination/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            model.End();model.symbolSpace.soft_reset();torch._dynamo.reset()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--epochs',type=int,default=64)
    args=parser.parse_args();torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];folder=args.destination.resolve()
    folder.mkdir(parents=True,exist_ok=False);source=source_snapshot(root)
    plan=dict(stages=STAGES,three_word_fallback=THREE_WORDS,epochs_per_stage=args.epochs,seed=None,retries=0,modes=['disabled','enabled'],
        fresh_models=2,learning_rate=.01,source=source,
        control='Same support, field objective and independent meter; source-order candidate selection.',
        decoder_diagnosis='Run the four-word sentence first. The free lexical binary inverse chooses children from the word bank; composed children are absent. If the disabled four-word control fails, also run the three-word sentence in both modes to test the existing one/three-slot ending. This does not assert that every four-leaf tree is impossible.',
        stopping='No extra stop alternative; exhaustion or the field allowance ends reading.',
        when_inventory_sha256=hashlib.sha256((root/'test/fixtures/when-readers-round4a0.json').read_bytes()).hexdigest())
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    try:
        control=run('disabled',folder/'disabled',args.epochs)
        run('enabled',folder/'enabled',args.epochs,control=[r for r in control if 'final' in r])
    finally:
        (folder/'source_check.json').write_text(json.dumps(dict(source_matched=source==source_snapshot(root)),indent=2)+'\n')


if __name__=='__main__':main()
