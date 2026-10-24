"""One first-draw corpus gate per surface form; no rescue or parse labels."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test'),
    str(ROOT/'doc/benchmarks/2026-10-23-item3-5-step2c/development'),
    str(ROOT/'doc/benchmarks/2026-10-21-item3-5-reference-reconstruction')]
import numpy as np
import pytest
import torch
from arrival_capture import plain
from bounded_tests import source_snapshot
from identity_experiment import make_model, reset
from constituent_budget import paired_updates_only


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2)+'\n')


def rng():
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())


def corpus(protocol, form):
    rows = []
    for i, subject in enumerate(protocol['subjects']):
        for j, verb in enumerate(protocol['verbs']):
            other = protocol['verbs'][1-j]
            modal = (f'{subject} {other} {protocol["modifier"]}' if form == 'free'
                     else f'{subject} {other}{protocol["modifier"]}')
            for view in ('actual', 'alternative'):
                rows.append(dict(id=f'{form}-{i}-{j}-{view}', subject=subject,
                    sentences=[f'{subject} {verb}', modal], view=view,
                    prompt=protocol['prompts'][view].format(subject=subject),
                    answer=verb if view == 'actual' else other,
                    heldout=(i+j)%4 == 3))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--form',choices=('free','affix'),required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('the corpus gate requires a clean frozen source')
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((HERE/'corpus-protocol.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    audit=source_snapshot(ROOT)
    save(args.out/'protocol.json',dict(protocol,source=source,form=args.form))
    save(args.out/'source-audit.json',audit)
    rows=corpus(protocol,args.form)
    save(args.out/'corpus.json',rows)
    train,evaluation=([row for row in rows if not row['heldout']],
                      [row for row in rows if row['heldout']])
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    torch.set_num_threads(1)
    torch.save(rng(),args.out/'initial-rng.pt')
    progress,results,keeps=[],[],[]
    started=time.monotonic()
    from Models import BasicModel
    from What import What
    from AttentionLesson import prompt_need
    from identity_keep_probe import clause_graph
    with pytest.MonkeyPatch.context() as patch:
        model=make_model(args.out,patch)
        optimizer=model.getOptimizer(lr=protocol['learning_rate'])
        def snapshot():
            return dict(state_dict=model.state_dict(),vocab_extras=model._collect_vocab_extras(),
                structural_extras=model._collect_structural_extras(),rng=rng(),optimizer=optimizer.state_dict())
        torch.save(snapshot(),args.out/'initial.pt')
        original=BasicModel._commit_sentence
        def keep(current,state,sid,active,observations,predictions,wins):
            record=dict(batch=len(progress),sentence=int(sid),active=active,
                credit=current._last_sentence_credit,
                trials=[dict(clauses=[clause_graph(c) for c in o['clauses']],
                             plans=o.get('reference_plans')) for o in observations])
            value=original(current,state,sid,active,observations,predictions,wins)
            record['pool']=current._concept_owner().mtm.state()
            store=current.symbolSpace.ltm_store
            record['orders']={str(store.occurrence_of(i)):int(store.order[i]) for i in range(len(store))}
            with gzip.open(args.out/'keep.jsonl.gz','at') as stream:
                stream.write(json.dumps(plain(record))+'\n')
            keeps.append(record)
            return value
        patch.setattr(BasicModel,'_commit_sentence',keep)
        try:
            with paired_updates_only(model,optimizer) as updates:
                def present(batch,training,epoch):
                    split='train' if training else 'validation'
                    data=model.inputSpace.data
                    data.grammar_lessons={}
                    texts=[text for row in batch for text in (*row['sentences'],row['prompt'])]
                    data.train_input=texts if training else []
                    data.train_output=[torch.zeros(1) for _ in texts] if training else []
                    data.source_addresses={split:[dict(document=row['id'],sentence=sid,
                        document_key=row['id'],sentence_index=sid,split=split)
                        for row in batch for sid in range(3)]}
                    torch.save(snapshot(),args.out/'last-batch-start.pt')
                    torch.save(rng(),args.out/f'batch-{len(progress):03d}-rng.pt')
                    keeps.clear()
                    # The two premises have no supplied target of any kind.
                    data.has_supervised_outputs=False
                    data.text_answers={}
                    with torch.set_grad_enabled(training):
                        inputs=model.inputSpace.prepPackedInput([row['sentences'] for row in batch])
                        model.runBatch(train=training,optimizer=optimizer if training else None,
                            batchNum=2*len(progress),batchSize=len(batch),split=split,
                            batch_override=(inputs,torch.empty(len(batch),0)),
                            source_rows=[[3*b,3*b+1] for b in range(len(batch))],
                            questions=tuple(What.present(3*b+1,split=split) for b in range(len(batch))))
                    # Ask in a separate closing, keeping this document's frames.
                    prompts=[row['prompt'] for row in batch]
                    data.has_supervised_outputs=training
                    data.text_answers={'train':[answer for row in batch for answer in
                        (None,None,row['answer'])]} if training else {}
                    factory=What.supervised if training else What.inference
                    questions=tuple(factory(3*b+2,split=split,prompt=prompt) for b,prompt in enumerate(prompts))
                    with torch.set_grad_enabled(training),prompt_need(model,prompts):
                        inputs=model.inputSpace.prepPackedInput([[prompt] for prompt in prompts])
                        model.runBatch(train=training,optimizer=optimizer if training else None,
                            batchNum=2*len(progress)+1,batchSize=len(batch),split=split,
                            batch_override=(inputs,torch.zeros(len(batch),1) if training else torch.empty(len(batch),0)),
                            source_rows=[[3*b+2] for b in range(len(batch))],questions=questions)
                    answers=list(getattr(getattr(model,'_last_answer_construction',None),'texts',()) or ())
                    if not training:
                        # Grade the kept event histories, never supply their
                        # roles, operations, orders or references to the model.
                        store=model.symbolSpace.ltm_store
                        for b,row in enumerate(batch):
                            histories=[]
                            for record in keeps:
                                owner=model._concept_owner()
                                subject_word=owner.definitions.word(form=row['subject'])
                                actual_form=row['sentences'][0].split()[-1]
                                other=protocol['verbs'][1-protocol['verbs'].index(actual_form)]
                                def concepts(form):
                                    word=owner.definitions.word(form=form)
                                    return set() if word is None else {word,*owner.definitions.objects(word)}
                                actual_verbs,alternative_verbs=concepts(actual_form),concepts(other)
                                named={token for mention,token,reading in record['pool']['names']
                                       if reading==subject_word}
                                for entry in record['pool']['entries']:
                                    origin=store.location(entry['key'])
                                    if origin is None or not bool(store.stream[origin,b]) or entry['key'] not in named:
                                        continue
                                    timelines=entry['timelines']
                                    actual=[event[0] for event in timelines[0][0] if event[2]==0 and event[1] in actual_verbs
                                        and record['orders'].get(str(event[0]))==0]
                                    alternative=[event[0] for timeline in timelines[1:] for event in timeline[0]
                                        if event[2]==0 and event[1] in alternative_verbs and record['orders'].get(str(event[0]),0)>0]
                                    histories.append(dict(token=entry['key'],actual=actual,alternative=alternative))
                            answer=answers[b] if b<len(answers) else None
                            results.append(dict(id=row['id'],view=row['view'],expected=row['answer'],answer=answer,
                                answer_exact=answer==row['answer'],histories=histories,
                                timeline_exact=any(h['actual'] and h['alternative'] for h in histories)))
                    progress.append(dict(batch=len(progress),training=training,epoch=epoch,
                        ids=[row['id'] for row in batch],updates=dict(updates),elapsed=time.monotonic()-started))
                    save(args.out/'progress.json',progress)
                    print(json.dumps(progress[-1]),flush=True)
                    reset(model,len(batch))
                for epoch in range(protocol['epochs']):
                    for offset in range(0,len(train),protocol['batch_size']):
                        present(train[offset:offset+protocol['batch_size']],True,epoch)
                if updates['paired'] != protocol['paired_updates_per_form']:
                    raise RuntimeError(f'paired budget mismatch: {updates}')
                torch.save(snapshot(),args.out/'trained.pt')
                model.languageSpace.parts_of_speech.frozen=True
                model._concept_owner()._online_learning_frozen=True
                for offset in range(0,len(evaluation),protocol['batch_size']):
                    present(evaluation[offset:offset+protocol['batch_size']],False,None)
                report=dict(completed=True,source=source,form=args.form,results=results,updates=dict(updates),
                    exact_answers=sum(row['answer_exact'] for row in results),
                    exact_timelines=sum(row['timeline_exact'] for row in results),total=len(results),
                    passed=all(row['answer_exact'] and row['timeline_exact'] for row in results),
                    elapsed_seconds=time.monotonic()-started)
                save(args.out/'results.json',report)
                print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
        except BaseException:
            torch.save(snapshot(),args.out/'failure.pt')
            save(args.out/'failure.json',dict(error=traceback.format_exc(),progress=progress))
            raise
        finally:
            save(args.out/'end.json',dict(source=source,unchanged=source_snapshot(ROOT)==audit,
                finished=datetime.now(timezone.utc).isoformat()))
            model.End()


if __name__=='__main__':
    main()
