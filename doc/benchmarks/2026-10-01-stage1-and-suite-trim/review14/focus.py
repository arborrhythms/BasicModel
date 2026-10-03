from pathlib import Path
import os,sys,json
R=Path(__file__).resolve().parent;ROOT=R.parents[3]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as b
name=sys.argv[1];selectors=sys.argv[2:]
env=dict(RUN_SLOW='1',BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false')
os.environ.update(env);os.environ.pop('BASIC_SEED',None)
out=R/name
frozen=b.source_snapshot(ROOT)
result=b.run_suite(root=ROOT,selectors=selectors,run_dir=out,memory_bytes=16*b.GIB,worker_memory_bytes=8*b.GIB,workers=2,timeout=1800,suite_timeout=10800,batch_size=8,max_files=1,lock_path=R/'focus.lock')
b.write_json(out/'source-match.json',dict(source=frozen,matched=b.source_snapshot(ROOT)==frozen))
print(json.dumps({k:result.get(k) for k in ('exit_code','reason','counts','elapsed_seconds')},indent=2),flush=True)
