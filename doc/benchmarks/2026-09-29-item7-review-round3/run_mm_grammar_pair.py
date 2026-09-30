"""Twenty unseeded full-900-epoch runs, alternating trees, at most three workers."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
sys.path.insert(0,str(MAIN/'test'))
from bounded_tests import GIB,run_guarded,source_snapshot,ProcessTree
roots={'head':Path(sys.argv[1]).resolve(),'candidate':MAIN}
outputs={label:HERE/(label+'-mm-grammar') for label in roots}
sources={label:source_snapshot(root) for label,root in roots.items()}
for label,out in outputs.items():
    out.mkdir(exist_ok=False)
    (out/'source-manifest.json').write_text(json.dumps(dict(validated_source=sources[label],
        driver_sha256=hashlib.sha256((HERE/'measure_mm_grammar.py').read_bytes()).hexdigest()),indent=2)+'\n')
env=os.environ.copy();env.pop('BASIC_SEED',None)
env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
processes={label:{} for label in roots}
def command(label,trial):
    out=outputs[label];root=roots[label]
    args=[str(MAIN/'.venv/bin/python'),str(HERE/'measure_mm_grammar.py'),str(out/f'run-{trial:02}.json')]
    child_env=dict(env,PYTHONPATH=os.pathsep.join(str(root/p) for p in ('bin','test')))
    return args,child_env

def run(label,trial):
    root=roots[label];out=outputs[label]
    args,child_env=command(label,trial)
    result=run_guarded(args,cwd=root,env=child_env,log_path=out/f'run-{trial:02}.log',
                       memory_bytes=8*GIB,timeout=1800)
    assert source_snapshot(root)==sources[label]
    return label,trial,result

with ProcessPoolExecutor(max_workers=3,mp_context=get_context('fork')) as pool:
    jobs=[pool.submit(run,label,trial) for trial in range(10) for label in roots]
    for job in as_completed(jobs):
        label,trial,result=job.result()
        processes[label][str(trial)]=result
        (outputs[label]/'processes.json').write_text(json.dumps(processes[label],indent=2)+'\n')
        print(label,trial,result['reason'],result['exit_code'],flush=True)

for label in roots:
    for trial,result in processes[label].items():
        if result['reason'] not in ('memory','aggregate_memory'):continue
        args,child_env=command(label,int(trial));args[-1]=str(outputs[label]/f'run-{int(trial):02}-diagnostic.json')
        started=time.monotonic()
        with (outputs[label]/f'run-{int(trial):02}-diagnostic.log').open('w') as log:
            proc=subprocess.Popen(args,cwd=roots[label],env=child_env,stdout=log,stderr=log,start_new_session=True)
            tree,peak,reason=ProcessTree(proc.pid),0,'completed'
            while proc.poll() is None:
                peak=max(peak,tree.sample())
                if time.monotonic()-started>1800:
                    tree.terminate(proc,.5);reason='timeout';break
                time.sleep(.1)
        result['diagnostic_only']=dict(exit_code=proc.returncode,peak_memory_bytes=peak,memory_guard=None,
                                      elapsed_seconds=time.monotonic()-started,reason=reason)
        assert source_snapshot(roots[label])==sources[label]
        (outputs[label]/'processes.json').write_text(json.dumps(processes[label],indent=2)+'\n')
