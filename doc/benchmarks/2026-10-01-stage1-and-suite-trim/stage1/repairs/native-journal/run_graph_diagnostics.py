from pathlib import Path
import json,sys
H=Path(__file__).resolve().parent
ROOT=H.parents[5]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as b
source=b.source_snapshot(ROOT)
for name,root in [('head',H/'head-snapshot'),('candidate',ROOT)]:
    result=b.run_guarded([sys.executable,str(H/'diagnose_graph.py'),'--root',str(root),
        '--output',str(H/(name+'-graph'))],cwd=ROOT,env=b.worker_environment(ROOT),
        log_path=H/(name+'-graph.log'),memory_bytes=12*b.GIB,timeout=1800)
    b.write_json(H/(name+'-graph-process.json'),result)
    print(name,json.dumps(result),flush=True)
    assert source==b.source_snapshot(ROOT)
