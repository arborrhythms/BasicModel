"""Run the delivered MM_query_reasoning configuration once, without a forced reading."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'bin'),str(ROOT/'test')]


def child(folder):
    from contextlib import ExitStack
    from unittest.mock import patch
    import torch
    import Models
    import ThoughtCredit
    from thinking_observer import observe
    from rng_replay import entry
    entry(folder)
    result=dict(config='data/MM_query_reasoning.xml',configured_epochs=300,
        batch_size=6,seed=None,retries=0,forced_reading=False,training_epochs=0,
        thought_credit=[],chooser_movement=[],outcome='running',
        observation_scope='Raw thought-credit gradients and actual shared-scorer epoch changes. Other owner losses also update this scorer; total movement alone is not proof of learned chaining.')
    epoch=Models.BasicModel.runEpoch
    credit=ThoughtCredit.register
    complete=ThoughtCredit.complete
    surrogate=ThoughtCredit.surrogate
    initial={}
    epoch_gradients={}
    comparisons={}
    latest={}
    tracked=[None]

    def parameters(model):
        return {name:p for name,p in model._selected_thought_chooser(None).named_parameters()
                if p.requires_grad and p.numel()}

    def snapshot(model):
        return {name:p.detach().cpu().clone() for name,p in parameters(model).items()}

    def norm(values):
        return sum(float(value.double().square().sum()) for value in values)**.5

    def run_epoch(model,*args,**kwargs):
        training=kwargs.get('optimizer') is not None and kwargs.get('split','train')=='train'
        if training:
            tracked[0]=model
            before=snapshot(model)
            if not initial:
                initial.update(before)
                torch.save(initial,folder/'chooser-before.pt')
            epoch_gradients.clear()
        value=epoch(model,*args,**kwargs)
        if training:
            result['training_epochs']+=1
            after=snapshot(model)
            assert before.keys()==after.keys()==initial.keys()
            delta={name:after[name]-before[name] for name in before}
            result['chooser_movement'].append(dict(epoch=result['training_epochs'],
                epoch_change_l2=norm(delta.values()),
                initial_change_l2=norm(after[name]-initial[name] for name in after),
                maximum_absolute_change=max(float(value.abs().max()) for value in delta.values()),
                raw_thought_gradient_l2=norm(epoch_gradients.values()),
                negative_thought_gradient_dot_epoch_change=-sum(float(
                    grad.double().mul(delta[name].double()).sum()) for name,grad in epoch_gradients.items())))
        (folder/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        return value

    def completed(model,greedy,explore,trace,other,departure,costs,row,**kwargs):
        from ThoughtReferences import open_slots
        comparisons[id(other)]=dict(
            greedy_operations=[r.operation for r in greedy.records if r.kind=='thought'],
            explore_operations=[r.operation for r in explore.records if r.kind=='thought'],
            greedy_bound=not bool(open_slots(greedy.meaning)),
            explore_bound=not bool(open_slots(explore.meaning)),
            work=[greedy.work.spent,explore.work.spent])
        return complete(model,greedy,explore,trace,other,departure,costs,row,**kwargs)

    def compared(trace,other,departure,costs):
        latest.clear()
        latest.update(comparisons.pop(id(other),{}),departure=departure,
                      eligible_rounds=sum(trace['eligible']))
        return surrogate(trace,other,departure,costs)

    def register(model,value,*,costs,source):
        observed=dict(epoch=result['training_epochs']+1,costs=list(map(float,costs)),source=source,
            requires_grad=value is not None and value.requires_grad,
            surrogate=None if value is None else float(value.detach()),**latest)
        named=parameters(model)
        gradients=(torch.autograd.grad(value,tuple(named.values()),retain_graph=True,allow_unused=True)
                   if value is not None and value.requires_grad else (None,)*len(named))
        detached={name:g.detach().cpu() for name,g in zip(named,gradients) if g is not None}
        observed['raw_chooser_gradient_l2']=norm(detached.values())
        observed['chooser_parameters_with_nonzero_gradient']=sum(bool(g.count_nonzero()) for g in detached.values())
        for name,grad in detached.items():
            epoch_gradients[name]=epoch_gradients.get(name,torch.zeros_like(grad))+grad
        result['thought_credit'].append(observed)
        return credit(model,value,costs=costs,source=source)
    try:
        with ExitStack() as stack:
            stack.enter_context(observe(folder/'thinking.json'))
            stack.enter_context(patch.object(Models.BasicModel,'runEpoch',run_epoch))
            stack.enter_context(patch.object(ThoughtCredit,'register',register))
            stack.enter_context(patch.object(ThoughtCredit,'complete',completed))
            stack.enter_context(patch.object(ThoughtCredit,'surrogate',compared))
            results=Models.ModelFactory.run(str(ROOT/'data/MM_query_reasoning.xml'))
        assert result['training_epochs']==300, 'configured training did not complete all 300 epochs'
        result['outcome']='completed'
        result['results']=[dict(name=name,correct=correct.detach().cpu().tolist() if torch.is_tensor(correct) else correct) for name,correct,model in results]
    except BaseException as error:
        result.update(outcome='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if tracked[0] is not None:
            final=snapshot(tracked[0])
            torch.save(final,folder/'chooser-after.pt')
            result['chooser_initial_to_final_l2']=norm(final[name]-initial[name] for name in initial)
        result['credit_summary']=dict(observations=len(result['thought_credit']),
            nonzero_chooser_gradients=sum(row['raw_chooser_gradient_l2']>0 for row in result['thought_credit']),
            exact_cost_ties=sum(row['costs'][0]==row['costs'][1] for row in result['thought_credit']),
            two_query_explore_chains=sum(row.get('explore_operations',[]).count('query')>=2 for row in result['thought_credit']))
        (folder/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    import bounded_tests as bounded
    from verification import validate
    from campaign import environment
    source=bounded.source_snapshot(ROOT);validate(source)
    helpers=json.loads((HERE/'measured-source/measurement-helpers.json').read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in helpers.items())
    folder=HERE/'mm-query-configured';folder.mkdir(exist_ok=False)
    bounded.write_json(folder/'plan.json',dict(config='data/MM_query_reasoning.xml',epochs=300,
        seed=None,retries=0,forced_reading=False,timeout_seconds=3600,memory_bytes=8*bounded.GIB,source=source))
    process=bounded.GuardedProcess([sys.executable,str(Path(__file__).resolve()),'child',str(folder)],
        cwd=ROOT,env=environment(),log_path=folder/'run.log',memory_bytes=8*bounded.GIB,timeout=3600).start()
    try:
        while True:
            assert source==bounded.source_snapshot(ROOT),'source changed during MM query measurement'
            result=process.poll()
            if result is not None:break
            time.sleep(.5)
    finally:
        if process.proc.poll() is None:process.stop(exit_code=130,reason='measurement_stopped')
    bounded.write_json(folder/'process.json',result)
    if not (folder/'outcome.json').exists():
        progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {}
        bounded.write_json(folder/'outcome.json',dict(progress,outcome='resource_cutoff',error=result['reason']))
    print(json.dumps(result))
    return result['exit_code']

if __name__=='__main__':
    if len(sys.argv)>1:child(Path(sys.argv[2]))
    else:raise SystemExit(main())
