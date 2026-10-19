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
OUTPUT = ROOT/'doc/benchmarks/2026-10-19-item4-repair/native-repaired'
CORPUS = [
    ['red cat', 'blue dog'], ['blue cat', 'red dog'],
    ['red cat runs', 'blue dog walks'], ['blue cat walks', 'red dog runs'],
    ['red cat runs fast', 'blue dog walks slowly'],
    ['blue cat walks fast', 'red dog runs slowly']]
HELDOUT = ['red dog walks fast', 'blue cat runs slowly']



def run():
    import torch
    import pytest
    import util
    from test_compiled_word_chunk import _tiny_canonical_model, _stage_fullgraph_tensor_peer
    torch.set_num_threads(1)
    report=dict(protocol='native-repaired-protocol.md',
                corpus=CORPUS, heldout=HELDOUT, runs=[], status='running')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    destination=OUTPUT/'measurement.json'
    def save():destination.write_text(json.dumps(report,indent=2)+'\n')
    save()
    started=time.monotonic()
    for seed in (19,23):
        for control in ('graded','shuffled'):
            torch.manual_seed(seed);random.seed(seed)
            with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
                patch.setattr(util,'TheCompileBackend','none')
                model=_tiny_canonical_model(Path(directory),patch,input_width=32,
                    word_buckets='8',stm_capacity=4,chooser_depth=1,
                    training_overrides={'intraLossWeight':0., **({level+'ExpectationGain': .5
                        for level in ('percept','symbol','row','document')})})
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
                        row_R=registry.total(objective='reconstruction').detach().tolist(),
                        P_residual=None if getattr(model,'_percept_residual',None) is None else float(model._percept_residual.detach().mean()),
                        S_MSE=[float((r['observed']-r['expected']).square().mean()) for r in observation.get('symbol_expectation',())],
                        raw_norms=[None if v is None else float(v.detach().norm()) for v in observation['observed']],
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
                        chooser_loss=float(result.detach()), requires_grad=result.requires_grad,
                        owner_exposures={name:int((audit['mask'] & audit['walk'].eq(walk) & audit['advantage'].ne(0)[:,None]).sum()) for walk,name in ((0,'narrowing'),(1,'compose'))},
                        loss_per_owner={name:float(torch.where(audit['mask'] & audit['walk'].eq(walk),audit['probabilities']*audit['scale']*audit['advantage'][:,None],0.).sum()) for walk,name in ((0,'narrowing'),(1,'compose'))}))
                    return result
                model._compose_score_function_loss=policy
                step = model._sentence_train_step
                def train_step(loss):
                    before = [p.detach().clone() for p in chooser.parameters()]
                    candidate_before=[p.detach().clone() for p in model.candidate_attention.parameters()]
                    candidate_gradients=[]
                    gradients = []
                    handles = [p.register_hook(lambda g: gradients.append(float(g.detach().abs().sum())))
                               for p in chooser.parameters() if p.requires_grad]
                    handles += [p.register_hook(lambda g: candidate_gradients.append(float(g.detach().abs().sum()))) for p in model.candidate_attention.parameters() if p.requires_grad]
                    try:
                        step(loss)
                    finally:
                        for handle in handles: handle.remove()
                    current.setdefault('optimizer', []).append(dict(loss=float(loss.detach()),
                        chooser_gradient_l1=sum(gradients), candidate_gradient_l1=sum(candidate_gradients),
                        candidate_delta=sum(float((p.detach()-v).abs().sum()) for p,v in zip(model.candidate_attention.parameters(),candidate_before)),
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
                    audit=getattr(model,'_last_sentence_credit',None)
                    if audit is not None:
                        current['judge']=dict(costs=audit['costs'].tolist(),components=audit['components'].tolist(),wins=audit['wins'].tolist(),walk=None if audit['departure'] is None else audit['departure']['walk'].tolist())
                    field=getattr(model,'_last_sentence_field',None)
                    backup=None if field is None else field.get('backup')
                    current['candidate_departure']=None if backup is None else {k:v.tolist() for k,v in backup.items()}
                    comparisons=[model.symbolSpace.expectation.last_expectation_comparison(b) for b in range(len(samples))]
                    current['D_residual_MSE']=[None if c is None else float(c.residual.square().mean()) for c in comparisons]
                    current['D_retention_surprise']=[None if c is None else float(c.surprise) for c in comparisons]
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
    report['paired_execution']=all(r['paired_trials'] for r in report['runs'])
    report['learning_gate']=False
    report['diagnostic_only']=True
    report['original_failed_run_unchanged']=True
    report['execution_gate']=report['paired_execution']
    report['derivation_optimality_gate']=False
    report['derivation_optimality_rate']=None
    report['optimality_limitation']='No exhaustive native derivation oracle is supplied by this fixed six-batch corpus. R is not relabeled global optimality.'
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
