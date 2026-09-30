"""Ten declared independent full-budget measurements, each under 8 GiB."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=Path(sys.argv[1]).resolve();label=sys.argv[2]
sys.path.insert(0,str(MAIN/'test'))
from bounded_tests import GIB,run_guarded,source_snapshot,ProcessTree
out=HERE/(label+'-mm-grammar');out.mkdir(exist_ok=False)
source=source_snapshot(ROOT)
(out/'source-manifest.json').write_text(json.dumps(dict(validated_source=source,driver_sha256=hashlib.sha256((HERE/'measure_mm_grammar.py').read_bytes()).hexdigest()),indent=2)+'\n')
env=os.environ.copy();env.pop('BASIC_SEED',None)
env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false',
           PYTHONPATH=os.pathsep.join((str(ROOT/'bin'),str(ROOT/'test'))),
           OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
results={}
for trial in range(10):
    cmd=[str(MAIN/'.venv/bin/python'),str(HERE/'measure_mm_grammar.py'),str(out/f'run-{trial:02}.json')]
    result=run_guarded(cmd,cwd=ROOT,env=env,log_path=out/f'run-{trial:02}.log',memory_bytes=8*GIB,timeout=1800)
    if result['reason'] in ('memory','aggregate_memory'):
        repeat=cmd[:-1]+[str(out/f'run-{trial:02}-diagnostic.json')]
        started=time.monotonic()
        with (out/f'run-{trial:02}-diagnostic.log').open('w') as log:
            proc=subprocess.Popen(repeat,cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
            tree,peak=ProcessTree(proc.pid),0
            while proc.poll() is None:
                peak=max(peak,tree.sample())
                if time.monotonic()-started>1800:
                    tree.terminate(proc,.5);break
                time.sleep(.1)
        result['diagnostic_only']=dict(exit_code=proc.returncode,peak_memory_bytes=peak,memory_guard=None)
    results[str(trial)]=result
    assert source_snapshot(ROOT)==source,'measurement source changed'
    (out/'processes.json').write_text(json.dumps(results,indent=2)+'\n')
    print(label,trial,result['reason'],result['exit_code'],flush=True)
