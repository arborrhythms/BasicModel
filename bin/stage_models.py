"""Bounded part-A rehearsal. Nulls and unattempted successors remain explicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def select_documents(documents, count):
    if count < 1:
        raise ValueError('the declared rehearsal needs a positive document budget')
    chosen = sorted(documents, key=lambda d:d['document'])[:count]
    if not chosen:
        raise ValueError('empty stage cannot count as a rehearsal')
    return chosen


def math_stages(root, count):
    manifest = json.loads((root/'data/math/manifest.json').read_text())
    for grade in manifest['grades']:
        rows = [json.loads(line) for line in (root/'data/math'/grade['file']).read_text().splitlines()]
        documents = [dict(document=row['document']['key'], records=[row['document']])
                     for row in rows if row['split'] == 'train']
        heldout = [dict(document=row['document']['key'], records=[row['document']])
                   for row in rows if row['split'] == 'validation']
        yield dict(name=f"math-{grade['grade']:02d}",
            documents=select_documents(documents,count) if documents else [], heldout=heldout,
            evaluation_only=not documents)


def criterion_process(config, checkpoint, protocol, lesson, output):
    command = [sys.executable, str(ROOT/'bin/eval_operators.py'), '--config', str(config),
        '--checkpoint', str(checkpoint), '--manifest', str(protocol), '--lesson', lesson,
        '--output', str(output)]
    with output.with_suffix('.log').open('w+') as log:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            result = None
        if result is None or result.returncode:
            log.seek(0)
            return dict(status='unavailable', passed=False, reason='isolated criterion reader failed',
                detail=log.read()[-12000:])
    value = json.loads(output.read_text())
    return value['criteria'][lesson]['criterion']


def run(protocol, output, *, only_lesson=None):
    import numpy as np
    import torch
    import Language
    import util
    from data import TheData
    from Models import BaseModel
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    from MathChainTraining import present
    from eval_operators import operator_parameters, parameter_change, digest
    sys.path.insert(0, str(ROOT/'test'))
    from bounded_tests import source_snapshot
    protocol, output = Path(protocol).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(protocol.read_text())
    config = ROOT/spec['config']
    source = source_snapshot(ROOT)
    torch.save(dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate()), output/'initial-rng.pt')
    write(output/'source.json', source)
    source_hash = hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    report = dict(schema=1, protocol_sha256=digest(protocol), config=str(config.relative_to(ROOT)),
        config_sha256=digest(config), source_sha256=source_hash,
        training_manifest_sha256=digest(ROOT/spec['training_manifest']),
        seed_policy=spec['seed_policy'], only_lesson=only_lesson,
        lessons=[], stages={}, checkpoints={}, status='running')
    write(output/'result.json', report)
    torch.set_num_threads(1); util.init_device('cpu'); util.TheCompileBackend='none'
    TheData.processLM({split:dict(text=['.'],label=[]) for split in ('train','validation','test')})
    TheData.source_manifest = dict(dataset='part-a-rehearsal', protocol_sha256=report['protocol_sha256'],
        training_manifest_sha256=report['training_manifest_sha256'])
    Language.TheGrammar._configured = False
    model = None
    start = time.perf_counter()
    try:
        model = BaseModel.from_config(str(config), data=TheData)[0]
        model._tensor_peer_while_eager = True
        optimizer = model.getOptimizer(lr=spec['learning_rate'])
        baseline = operator_parameters(model)
        def save(label):
            path = output/(label+'.ckpt')
            model.save_weights(str(path))
            parameters = operator_parameters(model)
            delta = {name:parameter_change(values,baseline[name]) for name,values in parameters.items()}
            report['checkpoints'][label] = dict(path=str(path), sha256=digest(path),
                parameters=delta)
            write(output/'result.json',report)
            return path
        save('M0')
        lessons = json.loads((ROOT/spec['lessons']).read_text())['stages']
        manifest = json.loads((ROOT/spec['training_manifest']).read_text())
        for stage_name, artifact in (('A1','stage_words'), ('A2','stage_sentences'), ('A3','stage_math')):
            selected = (list(math_stages(ROOT,spec['training_documents'])) if stage_name=='A3' else
                [dict(stage,documents=select_documents(stage['documents'],spec['training_documents']))
                 for stage in lessons if stage['name'] in manifest['stages'][stage_name]['lessons']])
            if only_lesson is not None:
                selected=[lesson for lesson in selected if lesson['name']==only_lesson]
                if not selected:
                    report['stages'][stage_name]=dict(status='unattempted',registration=None,reason='standalone lesson run')
                    continue
            stage_result = dict(status='running', phase=artifact, registration=None,
                incomplete_components=[spec['games_training']] if stage_name=='A2' else [], lessons=[])
            report['stages'][stage_name] = stage_result
            for lesson in selected:
                record = dict(phase=lesson['name'], stage=stage_name, status='running', epochs=[],
                    documents=[doc['document'] for doc in lesson['documents']], artifact=None)
                report['lessons'].append(record); stage_result['lessons'].append(lesson['name'])
                write(output/'result.json',report)
                if lesson.get('evaluation_only'):
                    record.update(status='evaluation_only', reason='frozen grade has no training split')
                    continue
                texts, targets = stage_documents(TheData, lesson)
                admit_training_forms(model,texts,targets)
                for epoch in range(spec['max_epochs']):
                    costs=[]
                    epoch_record=dict(epoch=epoch+1, status='running', completed_batches=0,
                        completed_sentences=0, costs=costs)
                    record['epochs'].append(epoch_record)
                    write(output/'result.json',report)
                    def observe(model,split,rows,result):
                        costs.append(dict(rows=rows, input=float(result.lossIn.detach()), output=float(result.lossOut.detach())))
                        epoch_record['completed_batches'] += 1
                        epoch_record['completed_sentences'] += len(rows)
                        write(output/'result.json',report)
                    measured=present(model,split='train',optimizer=optimizer,batch_size=spec['batch_size'],after_batch=observe)
                    if measured['sentences'] != len(texts):
                        raise RuntimeError('rehearsal omitted a supplied sentence')
                    checkpoint=save(lesson['name'])
                    criterion=criterion_process(config,checkpoint,protocol,lesson['name'],output/(lesson['name']+'.evaluation.json'))
                    epoch_record.update(status='complete', **measured, criterion=criterion)
                    write(output/'result.json',report)
                    if criterion['passed']:
                        # Registration remains a separate verified artifact operation.
                        record['artifact']=report['checkpoints'][lesson['name']]
                        break
                record['status']='passed' if record['artifact'] else 'null'
            save('M'+stage_name)
            stage_result['status']='null'
            stage_result['reason']='Rehearsal requires declared criteria, complete components and a reproducible saved-model table.'
        report['status']='complete'
        if only_lesson is not None and not report['lessons']:
            raise ValueError('unknown standalone lesson: '+only_lesson)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        for record in report['lessons']:
            if record['status']=='running':record.update(status='failed',error=report['error'])
        for stage_name in ('A1','A2','A3'):
            if stage_name not in report['stages']:
                report['stages'][stage_name]=dict(status='unattempted',registration=None,reason='predecessor failed')
            elif report['stages'][stage_name]['status']=='running':
                report['stages'][stage_name]['status']='failed'
    finally:
        if model is not None:model.End()
        report['elapsed_seconds']=time.perf_counter()-start
        write(output/'result.json',report)
    return report


def register_stage(report, table, stage, registry_path, receipt_path):
    """Persist only a verified stage; successful training alone is insufficient."""
    from eval_lessons import require_criteria
    from eval_operators import digest
    sys.path.insert(0,str(ROOT/'test'))
    from phase_artifacts import PhaseRegistry, PhaseArtifact
    record=report['stages'][stage]
    if report['status'] != 'complete' or report.get('only_lesson') is not None or record.get('incomplete_components'):
        raise ValueError('stage rehearsal has incomplete components')
    label='M'+stage
    reading=table['readings'][label]
    if reading['status'] != 'complete' or table['readings']['M0']['status'] != 'complete':
        raise ValueError('stage and baseline require reproducible saved-model readings')
    expected=['math'] if stage=='A3' else record['lessons']
    required=[name for name in expected if name in reading['criteria']]
    if len(required)!=len(expected):
        raise ValueError('stage table omitted lesson criteria')
    require_criteria([reading['criteria'][name]['criterion'] for name in required])
    checkpoint=report['checkpoints'][label]
    if digest(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('stage checkpoint changed after measurement')
    config=ROOT/report['config']
    if digest(config)!=report['config_sha256']:
        raise ValueError('stage configuration changed after measurement')
    if digest(ROOT/'data/training-corpus.json')!=report['training_manifest_sha256']:
        raise ValueError('stage corpus changed after measurement')
    for name in ('M0',label):
        entry=report['checkpoints'][name]
        if digest(entry['path'])!=entry['sha256'] or table['readings'][name]['checkpoint_sha256']!=entry['sha256']:
            raise ValueError('operator table refers to a different saved model')
    phase=record['phase']
    # Relabel the tested column consistently; M0 remains the comparison origin.
    rows=[dict(rule=row['rule'],models={('M0' if key=='M0' else phase):value
        for key,value in row['models'].items() if key in ('M0',label)}) for row in table['rows']]
    receipt=dict(config_sha256=report['config_sha256'], source_sha256=report['source_sha256'],
        training_manifest_sha256=report['training_manifest_sha256'], required_lessons=required,
        baseline=dict(config=str(config),checkpoint=report['checkpoints']['M0']['path']),operator_rows=rows)
    write(receipt_path,receipt)
    registry=PhaseRegistry.read(registry_path)
    registry.phases[phase]=PhaseArtifact(checkpoint['path'],checkpoint['sha256'],str(receipt_path),config=report['config'])
    registry.write(registry_path)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default=str(ROOT/'data/eval/stage-models.json'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--only-lesson')
    args=parser.parse_args()
    result=run(args.protocol,args.output,only_lesson=args.only_lesson)
    print(json.dumps({key:result[key] for key in ('status','elapsed_seconds')}))
    raise SystemExit(0 if result['status']=='complete' else 1)
