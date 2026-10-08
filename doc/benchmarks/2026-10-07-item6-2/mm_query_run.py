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
    from unittest.mock import patch
    import torch
    import Models
    import ThoughtCredit
    from thinking_observer import observe
    from rng_replay import entry
    entry(folder)
    result=dict(config='data/MM_query_reasoning.xml',configured_epochs=300,
        batch_size=6,seed=None,retries=0,forced_reading=False,training_epochs=0,
        thought_credit=[],outcome='running')
    epoch=Models.BasicModel.runEpoch
    credit=ThoughtCredit.register
    def run_epoch(model,*args,**kwargs):
        value=epoch(model,*args,**kwargs)
        if kwargs.get('split','train')=='train':result['training_epochs']+=1
        (folder/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        return value
    def register(model,value,*,costs,source):
        result['thought_credit'].append(dict(costs=list(map(float,costs)),source=source,
            requires_grad=value is not None and value.requires_grad))
        return credit(model,value,costs=costs,source=source)
    try:
        with observe(folder/'thinking.json'), patch.object(Models.BasicModel,'runEpoch',run_epoch), patch.object(ThoughtCredit,'register',register):
            results=Models.ModelFactory.run(str(ROOT/'data/MM_query_reasoning.xml'))
        result['outcome']='completed'
        result['results']=[dict(name=name,correct=correct.detach().cpu().tolist() if torch.is_tensor(correct) else correct) for name,correct,model in results]
    except BaseException as error:
        result.update(outcome='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
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
