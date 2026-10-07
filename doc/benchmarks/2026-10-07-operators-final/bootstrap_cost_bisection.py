"""Three-epoch prefix bisection of observed nonzero reconstruction at bootstrap.

The same recorded unseeded sum-01 entry, original 400-epoch schedule, explicit
stop after epoch three. These are diagnostics, never standing replacements.
"""
from pathlib import Path
import json,os,sys,hashlib,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'bootstrap-cost-bisection'
PRIOR=HERE.parent/'2026-10-03-operators-attention'

def child(part):
    sys.path[:0]=[str(HERE),str(PRIOR),str(ROOT/'bin'),str(ROOT/'test')]
    from unittest.mock import patch
    import torch
    import Models
    from Models import BasicModel,ModelFactory
    import test_explicit_dimensions as gates
    from rng_replay import entry,switches
    from operators_run_audit import observe_run,json_value
    folder=OUT/part;folder.mkdir()
    os.environ['OPERATORS_REPLAY_RNG']=str(HERE/'measurements/sum-01/unseeded-entry.pt')
    if part!='on':os.environ['OPERATORS_DISABLE']=part
    entry(folder)
    class PrefixComplete(BaseException):pass
    run=ModelFactory.run
    count=0
    with switches(), observe_run() as audit:
        epoch=BasicModel.runEpoch
        def limited(model,*args,**kwargs):
            nonlocal count
            value=epoch(model,*args,**kwargs)
            if kwargs.get('optimizer') is not None:
                count+=1
                if count==3:raise PrefixComplete()
            return value
        with patch.object(BasicModel,'runEpoch',limited),patch.object(ModelFactory,'run',lambda ignored:run(str(HERE/'measurements/XOR_grammar_sum_control.xml'))):
            try:gates._run_xor_grammar_in_process()
            except PrefixComplete:pass
    assert count==3
    (folder/'run-audit.json').write_text(json.dumps(audit,indent=2,default=json_value)+'\n')

def main():
    OUT.mkdir(exist_ok=False)
    summary=[]
    original=json.loads((HERE/'measurements/sum-01/run-audit.json').read_text())
    target=[t['components'] for t in original['sentence_trials'] if t['training'] and t['epoch']<=3]
    for part in ('on','B','C','D'):
        env=dict(os.environ,BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env.pop('BASIC_SEED',None)
        with (OUT/(part+'.log')).open('w') as log:
            process=subprocess.run([sys.executable,str(Path(__file__).resolve()),part],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        result=json.loads((OUT/part/'run-audit.json').read_text())
        trials=[t for t in result['sentence_trials'] if t['training']]
        summary.append(dict(part=part,exit_code=process.returncode,epochs=[t['epoch'] for t in trials],
            components=[t['components'] for t in trials],
            reproduction_matches_original=([t['components'] for t in trials]==target) if part=='on' else None,
            nonzero_reconstruction_epochs=[t['epoch'] for t in trials if any(c[0]!=0 for row in t['components'] for c in row)]))
    report=dict(seed=None,source_run='sum-01',original_schedule_epochs=400,diagnostic_prefix_epochs=3,
        variants=summary,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='components'} for row in summary]),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[1])
    else:main()
