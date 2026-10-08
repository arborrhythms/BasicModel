"""One frozen repair measurement; independent jobs reserve at most 24 GiB."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    result=HERE/'final-sweep/result.json'
    while True:
        if result.exists():
            try:
                sweep=json.loads(result.read_text())
                if sweep['reason']=='passed':
                    break
                if sweep.get('elapsed_seconds') is not None and sweep['reason'] not in ('running','collecting','incomplete'):
                    raise RuntimeError('the final sweep did not pass: '+sweep['reason'])
            except json.JSONDecodeError:
                pass
        time.sleep(1)
    assert sweep['exit_code']==0 and sorted(sweep['completed'])==sorted(sweep['selected'])
    assert not (HERE/'measurement-stages.json').exists(), 'measurement already launched'
    env=os.environ.copy()
    for name in ('BASIC_SEED','BASIC_NUM_EPOCHS','BASIC_BATCH_SIZE','BASIC_EPOCHS',
                 'BASIC_BATCH','BASIC_DATASET','PYTEST_PLUGINS','RUN_SLOW',
                 'OPERATORS_REPLAY_RNG','OPERATORS_DISABLE','OWNERSHIP_OBSERVER_OUTPUT'):
        env.pop(name,None)
    env.update(MODEL_COMPILE='none',BASICMODEL_DEVICE='cpu',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1')
    stages=[]
    def record(stage,status,**extra):
        row=dict(stage=stage,status=status,**extra);stages.append(row)
        (HERE/'measurement-stages.json').write_text(json.dumps(stages,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    def run(stage):
        record(stage,'started')
        with (HERE/f'{stage}-driver.log').open('w') as log:
            code=subprocess.call([sys.executable,str(HERE/f'{stage}.py')],cwd=ROOT,
                                 env=env,stdout=log,stderr=subprocess.STDOUT)
        record(stage,'finished',exit_code=code)
        return code
    assert run('freeze')==0
    with (HERE/'mm_query_run-driver.log').open('w') as log:
        record('mm_query_run','started')
        mm=subprocess.Popen([sys.executable,str(HERE/'mm_query_run.py')],cwd=ROOT,
                            env=env,stdout=log,stderr=subprocess.STDOUT)
        run('thinking_gate')
        campaign=run('campaign')
        record('mm_query_run','finished',exit_code=mm.wait())
    if campaign==0:
        run('summarize')


if __name__=='__main__':main()
