"""One measured run per slow case, preserving the bounded runner's 8 GiB cap."""
import json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded
items=json.loads((HERE/'prior-slowest-25.json').read_text())
selectors=[r['nodeid'].replace('test_item9b_schedule','test_interleave_schedule') for r in items]
os.environ['PYTEST_PLUGINS']='item69_profile'
os.environ['PYTHONPATH']=str(HERE)+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ['ITEM69_PROFILE_DIR']=str(HERE/'profiles-after')
os.environ['BASICMODEL_DEVICE']='cpu'
result=bounded.run_suite(root=ROOT,selectors=selectors,run_dir=HERE/'slowest-25-check',
 memory_bytes=24*bounded.GIB,worker_memory_bytes=8*bounded.GIB,workers=2,
 timeout=1800,suite_timeout=7200,batch_size=1,max_files=1)
print(json.dumps({k:result[k] for k in ('reason','exit_code','elapsed_seconds')},indent=2),flush=True)
raise SystemExit(result['exit_code'])
