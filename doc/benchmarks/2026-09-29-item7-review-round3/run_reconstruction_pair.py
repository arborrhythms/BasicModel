"""The 16 predeclared measurements in three fresh guarded workers at most.

Each process is limited to 8 GiB. No tested input or measurement changes.
Dispatch alternates trees; eight seeds per tree, regardless of outcomes.
Memory diagnostics run once, serially, after the guarded jobs complete.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,os,subprocess,sys,time
from multiprocessing import get_context
from pathlib import Path
HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
sys.path.insert(0,str(MAIN/'test'))
from bounded_tests import GIB,run_guarded,source_snapshot,ProcessTree
roots={'head':Path(sys.argv[1]).resolve(),'candidate':MAIN}
outputs={label:HERE/(label+'-reconstruction') for label in roots}
sources={label:source_snapshot(root) for label,root in roots.items()}
for label,out in outputs.items():
    out.mkdir(exist_ok=False)
    (out/'source-manifest.json').write_text(json.dumps(sources[label],indent=2)+'\n')
env=os.environ.copy();env.pop('BASIC_SEED',None)
env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
processes={label:{} for label in roots}
def command(label,seed):
    out=outputs[label]; root=roots[label]
    revision='1678ee1fb79c474ffaee5725fd035716de5f1913' if label=='head' else 'item7-round3-candidate'
    args=[str(MAIN/'.venv/bin/python'),str(HERE/'measure_reconstruction.py'),
        '--revision',revision,'--seed',str(seed),'--out',str(out/f'seed-{seed}.json')]
    child_env=dict(env,PYTHONPATH=os.pathsep.join(str(root/p) for p in ('bin','test')))
    return args,child_env

def run(label,seed):
    root=roots[label]; out=outputs[label]
    args,child_env=command(label,seed)
    result=run_guarded(args,cwd=root,env=child_env,log_path=out/f'seed-{seed}.log',
                       memory_bytes=8*GIB,timeout=1200)
    assert source_snapshot(root)==sources[label], 'measurement source changed'
    return label,seed,result

with ProcessPoolExecutor(max_workers=3, mp_context=get_context('fork')) as pool:
    jobs=[pool.submit(run,label,seed) for seed in range(8) for label in roots]
    for job in as_completed(jobs):
        label,seed,result=job.result()
        processes[label][str(seed)]=result
        (outputs[label]/'processes.json').write_text(json.dumps(processes[label],indent=2)+'\n')
        print(label,seed,result['reason'],result['exit_code'],flush=True)

for label in roots:
    for seed,result in processes[label].items():
        if result['reason'] not in ('memory','aggregate_memory'):continue
        args,child_env=command(label,int(seed));args[-1]=str(outputs[label]/f'seed-{seed}-diagnostic.json')
        started=time.monotonic()
        with (outputs[label]/f'seed-{seed}-diagnostic.log').open('w') as log:
            proc=subprocess.Popen(args,cwd=roots[label],env=child_env,stdout=log,stderr=log,start_new_session=True)
            tree,peak,reason=ProcessTree(proc.pid),0,'completed'
            while proc.poll() is None:
                peak=max(peak,tree.sample())
                if time.monotonic()-started>1200:
                    tree.terminate(proc,.5);reason='timeout';break
                time.sleep(.1)
        result['diagnostic_only']=dict(exit_code=proc.returncode,peak_memory_bytes=peak,memory_guard=None,
                                      elapsed_seconds=time.monotonic()-started,reason=reason)
        assert source_snapshot(roots[label])==sources[label]
        (outputs[label]/'processes.json').write_text(json.dumps(processes[label],indent=2)+'\n')
    (outputs[label]/'driver-hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),HERE/'measure_reconstruction.py')},indent=2)+'\n')
