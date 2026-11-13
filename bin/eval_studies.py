"""Frozen causal/learning study harnesses; unavailable controls never masquerade as arms."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def load_tasks(manifest,split,*,family=None,limit=None):
    from eval_operators import digest
    manifest=Path(manifest)
    meta=json.loads(manifest.read_text())
    entry=meta['files'][split]; path=manifest.parent/entry['path']
    if digest(path)!=entry['sha256']:
        raise ValueError('study task file differs from its frozen hash')
    rows=json.loads(path.read_text())['rows']
    if len(rows)!=entry['rows']:
        raise ValueError('study task count differs from its manifest')
    rows=sorted((row for row in rows if family is None or row['family']==family),key=lambda r:r['key'])
    return rows if limit is None else rows[:limit]


def documents(rows,control):
    if control not in ('ordinary','shuffled_order','renamed_vocabulary','shuffled_determiners'):
        raise ValueError('unsupported native study control: '+control)
    return [dict(document=row['key'],records=[dict(key=row['key'],
        sentences=row['sentences'] if control=='ordinary' else row['controls'][control],
        question=row['question'],answer=row['controls']['renamed_answer'] if control=='renamed_vocabulary' else row['answer'])])
        for row in rows]


def evaluation_documents(rows, control):
    """Training-order controls share ordinary held-out text; renaming is bijective."""
    return documents(rows, 'renamed_vocabulary' if control=='renamed_vocabulary' else 'ordinary')


@contextmanager
def observe_query_work():
    """Count successful runtime charges, including discarded trials; do not alter a meter."""
    from QueryWork import QueryWorkBudget
    original=QueryWorkBudget.consume
    result=dict(charged_units=0,attempts=0,exhausted=0,counts=Counter())
    def observed(meter,kind,units=1):
        accepted=original(meter,kind,units)
        result['attempts']+=1
        if accepted:
            result['charged_units']+=units; result['counts'][kind]+=units
        else:result['exhausted']+=1
        return accepted
    QueryWorkBudget.consume=observed
    try:yield result
    finally:QueryWorkBudget.consume=original


def validate_catalogs(catalogs,implementations):
    previous=None; opaque=None
    for row in catalogs:
        structural=set(row['structural']); current=set(row['opaque'])
        if not current or (previous is not None and not previous<structural):
            raise ValueError('routing study requires strict nested structure and an opaque candidate')
        if structural & current or (opaque is not None and current!=opaque):
            raise ValueError('opaque inventory must be fixed and disjoint')
        previous,opaque=structural,current
    from GrammarPreference import operator_is_structural
    absent=sorted(set().union(*(set(row['structural'])|set(row['opaque']) for row in catalogs))-set(implementations))
    if absent:return dict(status='unavailable',missing_implementations=absent)
    for name in opaque:
        if operator_is_structural(implementations[name]):
            raise ValueError('declared opaque candidate has a structural implementation')
    for row in catalogs:
        if not all(operator_is_structural(implementations[name]) for name in row['structural']):
            raise ValueError('structural catalog contains an opaque implementation')
    return dict(status='ready')


def compare_gate(protocol,name,treatment,control):
    gate=protocol['gates'][name]
    if treatment['status']!='complete' or control['status']!='complete':
        return dict(status='unavailable',passed=False,reason='both independent arm readings are required')
    if treatment['task_keys']!=control['task_keys']:
        raise ValueError('gate arms require identical held-out task identities')
    if treatment['exposure']!=control['exposure']:
        return dict(status='unmatched',passed=False,reason='actual training exposure differs')
    if treatment['initialization_sha256']==control['initialization_sha256']:
        raise ValueError('gate arms must be independently trained')
    metric=gate['metric']
    values=[arm['metrics'].get(metric) for arm in (treatment,control)]
    if any(value is None for value in values):
        return dict(status='unavailable',passed=False,metric=metric,values=values)
    thresholds=protocol['thresholds']
    if thresholds is None or name not in thresholds:
        return dict(status='undeclared',passed=False,metric=metric,values=values)
    declaration=thresholds[name]
    difference=(values[1]-values[0]) if declaration['direction']=='lower' else (values[0]-values[1])
    passed=difference>=declaration['minimum_difference']
    return dict(status='passed' if passed else 'failed',passed=passed,metric=metric,values=values,difference=difference)


def causal_readiness(protocol,arms,training_states):
    """Validate provenance before the later qualified comparison can claim utility."""
    from LearningEvaluation import fineweb_readiness
    if set(arms)!=set(protocol['causal_arms']):
        raise ValueError('the causal study requires all four named arms')
    unavailable=[name for name,row in arms.items() if row['status']!='complete']
    if unavailable:return dict(status='unavailable',arms=unavailable,learned_utility='unproven')
    if set(training_states)!=set(arms):
        raise ValueError('every arm needs its own training-state qualification')
    readiness={name:fineweb_readiness(training_states[name]) for name in arms}
    if not all(value.eligible for value in readiness.values()):
        return dict(status='unqualified',reasons={name:value.reason for name,value in readiness.items() if not value.eligible},learned_utility='unproven')
    first=next(iter(arms.values()))
    if any(row['task_keys']!=first['task_keys'] for row in arms.values()) or len(first['task_keys'])!=256:
        raise ValueError('qualified causal arms must read the same 256 tasks')
    if any(row['updates_after_qualification']!=protocol['updates_after_qualification'] for row in arms.values()):
        raise ValueError('qualified arms did not complete the declared update budget')
    return dict(status='undeclared' if protocol['thresholds'] is None else 'ready',learned_utility='unproven')


def evaluate_saved(config,checkpoint,manifest,*,family,control,limit):
    import torch
    from eval_operators import restore_model
    from corpus_lessons import heldout
    from eval_lessons import fraction
    rows=load_tasks(manifest,'learning-heldout',family=family,limit=limit)
    model,data=restore_model(config,checkpoint)
    try:
        contexts={}; costs=[]; reference_rows={}
        expected_references=referent_expectations(model,rows)
        def capture(model,split,indices,result):
            costs.append(float(result.lossIn.detach()))
            for batch,index in enumerate(indices):
                address=data.source_addresses[split][index]
                reference_rows.setdefault(address['document'],[]).extend(
                    dict(event,sentence=address['sentence']) for event in pending if event['batch']==batch)
            pending.clear()
            owner=model.symbolSpace.expectation
            if owner is None:return
            for batch,index in enumerate(indices):
                if len(owner._inter_context)<=batch or not owner._inter_context[batch]:continue
                _depth,roles,mask=owner._read_context(batch)[0][-1]
                document=data.source_addresses[split][index]['document']
                contexts.setdefault(document,[]).append((roles.detach().cpu(),mask.detach().cpu()))
        with observe_query_work() as work, observe_referents(model) as pending:
            answers=heldout(model,dict(heldout=evaluation_documents(rows,control)),data,after_batch=capture)
        prediction={}
        if contexts:
            from bench_sentence_expectation import meaning_windows,_controlled_inputs,_score_head
            owner=model.symbolSpace.expectation
            views=meaning_windows([torch.stack([v for v,m in values]) for values in contexts.values()],
                owner._inter_chain_window,[torch.stack([m for v,m in values]) for values in contexts.values()])
            if views is not None:
                for context in ('ordered','shuffled','context_free'):
                    prediction[context]=[]
                    for seed in (0,1,2):
                        values,masks=_controlled_inputs(views[0],views[1],context,seed)
                        with torch.no_grad():
                            prediction[context].append(_score_head(owner._inter_predictor,values,masks,views[2],
                                bank=owner._bank_keys,target_masks=views[4]))
        referents=score_referents(expected_references,reference_rows)
        return dict(status='complete',task_keys=[row['key'] for row in rows],answers=answers,
            metrics=dict(answer_accuracy=fraction(answers,'text_correct'),
                reconstruction_error=None if not costs else sum(costs)/len(costs),
                prediction_error=None if not prediction else prediction['ordered'][0]['activation_mse'],
                referent_accuracy=referents['value']),query_work=work,prediction_controls=prediction,
            reference_readings=referents,unavailable_metrics={} if referents['status']=='complete' else
                dict(referent_accuracy='no complete native occurrence reading for the selected reference tasks'))
    finally:model.End()


def smoke(protocol_path,output,*,arm=None,gate=None,side='treatment'):
    import numpy as np
    import torch
    import Language,util
    from data import TheData
    from Models import BaseModel
    from Corpus import admit_training_forms
    from corpus_lessons import stage_documents
    from MathChainTraining import present
    from eval_operators import digest
    sys.path.insert(0,str(ROOT/'test'))
    from bounded_tests import source_snapshot,run_guarded
    protocol_path=Path(protocol_path).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(protocol_path.read_text());budget=protocol['small_run']
    if (arm is None)==(gate is None):raise ValueError('select exactly one causal arm or learning gate')
    family=None
    if arm is not None:
        if arm not in protocol['causal_arms']:raise ValueError('unknown causal arm')
        control='ordinary' if arm=='learned_thought' else arm
    else:
        entry=protocol['gates'][gate];family=entry['family'];control=entry[side]
    result=dict(schema=1,arm=arm,gate=gate,side=side,control=control,budget=budget,
        protocol_sha256=digest(protocol_path),status='running',learned_utility='unproven')
    if control in protocol['unavailable_controls']:
        result.update(status='unavailable',reason=protocol['unavailable_controls'][control])
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    torch.set_num_threads(1); util.init_device('cpu');util.TheCompileBackend='none'
    rng=output/'initial-rng.pt'
    torch.save(dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),rng)
    result['initialization_sha256']=digest(rng)
    source=source_snapshot(ROOT)
    result['source_sha256']=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (output/'source.json').write_text(json.dumps(source,sort_keys=True)+'\n')
    manifest=ROOT/protocol['manifest'];config=ROOT/protocol['config']
    rows=load_tasks(manifest,'training',family=family,limit=budget['train_documents'])
    selected=dict(documents=documents(rows,control))
    result['training_keys']=[row['key'] for row in rows]
    result['task_keys']=[row['key'] for row in load_tasks(manifest,'learning-heldout',family=family,limit=budget['heldout_documents'])]
    model=None;start=time.perf_counter()
    try:
        TheData.processLM({split:dict(text=['.'],label=[]) for split in ('train','validation','test')})
        TheData.source_manifest=dict(dataset='study-smoke',manifest_sha256=digest(manifest),control=control)
        Language.TheGrammar._configured=False
        model=BaseModel.from_config(str(config),data=TheData)[0];model._tensor_peer_while_eager=True
        texts,targets=stage_documents(TheData,selected)
        admit_training_forms(model,texts,targets)
        optimizer=model.getOptimizer(lr=budget['learning_rate']);costs=[]
        def observed(model,split,indices,value):
            costs.append(dict(rows=indices,input=float(value.lossIn.detach()),output=float(value.lossOut.detach())))
        with observe_query_work() as work:
            counts=present(model,split='train',optimizer=optimizer,batch_size=budget['batch_size'],after_batch=observed)
        result.update(exposure=counts['sentences'],updates=counts['batches'],training_costs=costs,query_work=work,
            query_work_scope='charged runtime query units across all executed trials; not total compute')
        checkpoint=output/'trained.ckpt';model.save_weights(str(checkpoint))
        result['checkpoint_sha256']=digest(checkpoint)
        model.End();model=None
        command=[sys.executable,str(Path(__file__).resolve()),'evaluate','--config',str(config),
            '--checkpoint',str(checkpoint),'--manifest',str(manifest),'--control',control,
            '--limit',str(budget['heldout_documents']),'--output',str(output/'evaluation.json')]
        if family:command+=['--family',family]
        process=run_guarded(command,cwd=ROOT,env=dict(__import__('os').environ),log_path=output/'evaluation.log',memory_bytes=8*1024**3,timeout=1800)
        result['evaluation_process']=process
        if process['exit_code']:
            result.update(status='evaluation_failed',metrics={},reason=(output/'evaluation.log').read_text()[-12000:])
        else:result.update(json.loads((output/'evaluation.json').read_text()))
    except Exception:result.update(status='failed',error=traceback.format_exc())
    finally:
        if model is not None:model.End()
        result['elapsed_seconds']=time.perf_counter()-start
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def referent_expectations(model, rows):
    """Locate the repeated form specified by a frozen same-referent task."""
    import re
    expected=[]
    for document,row in enumerate(rows):
        if not row.get('expectations',{}).get('same_referent'):
            continue
        repeated=[word for word,count in Counter(re.findall(r'\w+',row['sentences'][0].casefold())).items()
                  if count==2]
        if len(repeated)!=1:
            raise ValueError('same-referent task must identify one twice-occurring form')
        expected.append(dict(document=document,key=row['key'],
            word_id=model._concept_owner().interpret.word_for_form(repeated[0])))
    return expected


@contextmanager
def observe_referents(model):
    """Copy native occurrence audits; never infer a referent from operator use."""
    absent=object();local=model.__dict__.get('_commit_sentence',absent)
    original=model._commit_sentence;pending=[]
    def observed(state,sid,active,observations,predictions,wins):
        owner=model._concept_owner();before=set(owner.mtm.entries);selected=[]
        old=getattr(model,'_reference_close_audit',{})
        for batch in range(len(active)):
            if not bool(active[batch]):continue
            program=observations[int(wins[batch])]['entries'][batch]
            words=None if program is None or program.word_ids is None else program.word_ids.detach().cpu().tolist()
            selected.append((batch,words,old.get((int(sid),batch))))
        result=original(state,sid,active,observations,predictions,wins)
        for batch,words,previous in selected:
            audit=getattr(model,'_reference_close_audit',{}).get((int(sid),batch),())
            values=[]
            for item in (() if audit is previous else audit):
                if 'leaves' not in item:continue
                reference=item.get('reference')
                token=None if reference is None else owner.mtm.token_of(reference)
                resolved=reference if token is None else token
                values.append(dict(leaves=list(item['leaves']),occurrence=item['occurrence'],
                    reference=resolved,new_reference=token is not None and token not in before,
                    operation=item.get('operation'),missing_reason=item.get('missing_reason')))
            pending.append(dict(batch=batch,slot=int(sid),word_ids=words,audit=values))
        return result
    model._commit_sentence=observed
    try:yield pending
    finally:
        if local is absent:del model._commit_sentence
        else:model._commit_sentence=local


def score_referents(expected, observations):
    """Fresh first referent, distinct mention occurrences, and actual reuse."""
    readings=[]
    for task in expected:
        row=dict(key=task['key'],correct=None)
        events=[event for event in observations.get(task['document'],[]) if event['sentence']==0]
        if len(events)!=1 or events[0]['word_ids'] is None:
            row['reason']='no unique observed kept program for the probe sentence'
        elif task['word_id'] is None:
            row.update(correct=False,reason='the repeated form was not admitted')
        else:
            event=events[0]
            positions=[i for i,word in enumerate(event['word_ids']) if word==task['word_id']]
            if len(positions)!=2:
                row.update(correct=False,reason='the two presented mentions were not retained as that form')
            else:
                matches=[[entry for entry in event['audit'] if position in entry['leaves']
                          and positions[1-i] not in entry['leaves']] for i,position in enumerate(positions)]
                if any(len(group)>1 for group in matches):
                    row['reason']='ambiguous native occurrence alignment'
                elif not all(matches):
                    row.update(correct=False,reason='a required mention has no native occurrence audit')
                else:
                    first,second=matches[0][0],matches[1][0]
                    row.update(correct=bool(first['new_reference'] and first['reference'] is not None
                        and first['reference']==second['reference'] and first['occurrence']!=second['occurrence']),
                        mentions=[first,second])
        readings.append(row)
    complete=bool(readings) and all(row['correct'] is not None for row in readings)
    return dict(status='complete' if complete else 'unavailable',count=len(readings),
        value=sum(row['correct'] for row in readings)/len(readings) if complete else None,
        definition='fresh first referent, distinct mention occurrences, same resolved referent',rows=readings)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('smoke');run.add_argument('--protocol',default=str(ROOT/'data/eval/studies/protocol.json'))
    run.add_argument('--output',required=True);run.add_argument('--arm');run.add_argument('--gate')
    run.add_argument('--side',choices=['treatment','control'],default='treatment')
    evaluate=sub.add_parser('evaluate')
    for name in ('config','checkpoint','manifest','control','output'):evaluate.add_argument('--'+name,required=True)
    evaluate.add_argument('--family');evaluate.add_argument('--limit',required=True,type=int)
    args=parser.parse_args()
    if args.command=='smoke':
        result=smoke(args.protocol,args.output,arm=args.arm,gate=args.gate,side=args.side)
        print(json.dumps({key:result[key] for key in ('status','arm','gate','side')}))
        raise SystemExit(0 if result['status'] in ('complete','unavailable') else 1)
    result=evaluate_saved(args.config,args.checkpoint,args.manifest,family=args.family,control=args.control,limit=args.limit)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
