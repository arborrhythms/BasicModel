"""Profile only repaired or unfinished slow cases; preserve the first attempt."""
import json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'test'));import bounded_tests as bounded
prior=json.loads((HERE/'slowest-25-check/result.json').read_text())
passed={r['nodeid'] for w in prior['workers'] for r in w['reports']
        if r['phase']=='call' and r['outcome']=='passed'}
selectors=[r['nodeid'].replace('test_item9b_schedule','test_interleave_schedule')
           for r in json.loads((HERE/'prior-slowest-25.json').read_text())]
selectors=[node for node in selectors if node not in passed]
os.environ.update(PYTEST_PLUGINS='item69_profile',
    PYTHONPATH=str(HERE)+os.pathsep+os.environ.get('PYTHONPATH',''),
    ITEM69_PROFILE_DIR=str(HERE/'profiles-after-repair'),BASICMODEL_DEVICE='cpu')
result=bounded.run_suite(root=ROOT,selectors=selectors,run_dir=HERE/'slowest-remaining-check',
 memory_bytes=16*bounded.GIB,worker_memory_bytes=8*bounded.GIB,workers=2,
 timeout=1800,suite_timeout=7200,batch_size=2,max_files=1)
print({k:result[k] for k in ('reason','exit_code','elapsed_seconds')},flush=True)
raise SystemExit(result['exit_code'])
