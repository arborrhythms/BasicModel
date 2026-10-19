#!/usr/bin/env python3
"""Bounded native Item 4 curriculum comparison; see native_protocol.md."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
OUTPUT = ROOT/'doc/benchmarks/2026-10-19-item4-repair/seed23-historical-retention'
CORPUS = [
    ['red cat', 'blue dog'], ['blue cat', 'red dog'],
    ['red cat runs', 'blue dog walks'], ['blue cat walks', 'red dog runs'],
    ['red cat runs fast', 'blue dog walks slowly'],
    ['blue cat walks fast', 'red dog runs slowly']]
HELDOUT = ['red dog walks fast', 'blue cat runs slowly']


def fingerprint():
    digest=hashlib.sha256()
    for folder in ('bin','data','test'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.xml','.grammar'):
                digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def run():
    import torch
    import pytest
    import util
    from test_compiled_word_chunk import _tiny_canonical_model, _stage_fullgraph_tensor_peer
    torch.set_num_threads(1)
    import Expectation
    def historical_surprise(residual, expected):
        error = residual.detach().square().mean()
        return error/(1.+error)
    Expectation.retention_surprise = historical_surprise
    report=dict(protocol='native_protocol.md', source=fingerprint(),
                corpus=CORPUS, heldout=HELDOUT, runs=[], status='running')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    destination=OUTPUT/'measurement.json'
    def save():destination.write_text(json.dumps(report,indent=2)+'\n')
    save()
    started=time.monotonic()
    for seed in ((19,23) if '--full-replay' in sys.argv else (23,)):
        for control in (('graded','shuffled') if '--full-replay' in sys.argv else ('graded',)):
            torch.manual_seed(seed);random.seed(seed)
            with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
                patch.setattr(util,'TheCompileBackend','none')
                model=_tiny_canonical_model(Path(directory),patch,input_width=32,
                    word_buckets='8',stm_capacity=4,chooser_depth=1,
                    training_overrides={'intraLossWeight':0., **({level+'ExpectationGain': .5
                        for level in ('percept','symbol','row','document')} if '--reviewed-gains' in sys.argv else {})})
                model._tensor_peer_while_eager=True
                optimizer=model.getOptimizer(lr=.001)
                predictor=model.symbolSpace.expectation._inter_predictor
                chooser=model._stm_reducer()
                before_head=[p.detach().clone() for p in predictor.parameters()]
                before_choice=[p.detach().clone() for p in chooser.parameters()]
                item=dict(seed=seed,control=control,batches=[],order=list(range(6)), gains={level:getattr(model,attr) for level,attr in [('P','percept_expectation_gain'),('S','word_expectation_gain'),('R','expectation_gain'),('D','document_expectation_gain')]})
                if control=='shuffled':random.Random(seed+100).shuffle(item['order'])
                report['runs'].append(item)
                original=model._sentence_path_cost
                current=None
                def cost(*args):
                    result=original(*args)
                    registry=model._sentence_cost_registry
                    def scalar(objective):
                        value=registry.total(objective=objective)
                        return None if value is None else float(value.detach().mean())
                    observation=result[2]
                    current['trials'].append(dict(R=scalar('reconstruction'),E=scalar('expectation'),
                        depths=list(map(int,observation['observed_depths'])),
                        rounds=[None if e is None else len(e.actions) for e in observation['entries']],
                        symbol_reads=len(observation.get('symbol_expectation',()))))
                    return result
                model._sentence_path_cost=cost
                surrogate = model._compose_score_function_loss
                def policy(path, costs):
                    result = surrogate(path, costs)
                    audit = model._last_compose_score_function
                    draw = model._sentence_departure
                    current.setdefault('departures', []).append(dict(
                        walk=draw['walk'].tolist(), walk_count=draw['walk_count'].tolist(),
                        walk_rounds=draw['walk_rounds'].tolist(),
                        compose_round=draw['compose_round'].tolist(),
                        path_scale=path[1][26].detach().tolist(),
                        selected_scale=audit['scale'][audit['mask']].tolist(),
                        mask=audit['mask'].tolist(), costs=costs.tolist(),
                        probabilities=audit['probabilities'][audit['mask']].tolist(),
                        chooser_loss=float(result.detach()), requires_grad=result.requires_grad))
                    return result
                model._compose_score_function_loss=policy
                step = model._sentence_train_step
                def train_step(loss):
                    before = [p.detach().clone() for p in chooser.parameters()]
                    gradients = []
                    handles = [p.register_hook(lambda g: gradients.append(float(g.detach().abs().sum())))
                               for p in chooser.parameters() if p.requires_grad]
                    try:
                        step(loss)
                    finally:
                        for handle in handles: handle.remove()
                    current.setdefault('optimizer', []).append(dict(loss=float(loss.detach()),
                        chooser_gradient_l1=sum(gradients),
                        chooser_delta=sum(float((p.detach()-v).abs().sum()) for p,v in zip(chooser.parameters(),before))))
                model._sentence_train_step=train_step
                def present(samples,training,phase):
                    nonlocal current
                    current=dict(phase=phase,samples=samples,training=training,trials=[])
                    item['batches'].append(current)
                    model.train(training)
                    grad=torch.enable_grad() if training else torch.no_grad()
                    with grad:
                        raw=_stage_fullgraph_tensor_peer(model,samples)
                        with model._sentence_run(training=training,optimizer=optimizer,split='train' if training else 'runtime'):
                            result=model._forward_with_compiled_sentence_state(raw)
                        model._publish_compiled_sentence_state(result)
                        model._end_step()
                    current['elapsed']=time.monotonic()-started
                    save()
                    model.End();model.symbolSpace.soft_reset()
                try:
                    present(HELDOUT,False,'before')
                    for index in item['order']:present(CORPUS[index],True,f'train-{index}')
                    present(HELDOUT,False,'after')
                    item['predictor_delta']=sum(float((p.detach()-v).abs().sum()) for p,v in zip(predictor.parameters(),before_head))
                    item['chooser_delta']=sum(float((p.detach()-v).abs().sum()) for p,v in zip(chooser.parameters(),before_choice))
                    item['paired_trials']=all(len(b['trials'])==2 for b in item['batches'] if b['training'])
                    item['before_R']=item['batches'][0]['trials'][0]['R']
                    item['after_R']=item['batches'][-1]['trials'][0]['R']
                finally:
                    model.End();model.symbolSpace.soft_reset()
                save()
    report['execution_gate']=all(r['paired_trials'] and r['predictor_delta']>0 and r['chooser_delta']>0 for r in report['runs'])
    report['learning_gate']=False
    report['diagnostic_only']=True
    report['execution_gate']=False  # remains closed until the cause is documented
    report['status']='complete';report['seconds']=time.monotonic()-started;save()
    print(json.dumps({k:report[k] for k in ('status','execution_gate','learning_gate','seconds')}))


if __name__=='__main__':
    if '--worker' in sys.argv:
        run()
    else:
        from bounded_tests import run_guarded
        import inspect
        OUTPUT.mkdir(parents=True,exist_ok=True)
        kwargs=dict(cwd=str(ROOT),env=dict(os.environ,BASICMODEL_DEVICE='cpu'),
            log_path=OUTPUT/'run.log',memory_bytes=8*1024**3,timeout=600.)
        print(inspect.signature(run_guarded))
        result=run_guarded([sys.executable,str(Path(__file__).resolve()),'--worker',*sys.argv[1:]],**kwargs)
        (OUTPUT/'process.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps(result,default=str))
