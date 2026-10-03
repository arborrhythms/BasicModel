import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded
frozen=bounded.source_snapshot(ROOT)
env=bounded.worker_environment(ROOT)
env.update(BASICMODEL_DEVICE='cpu',MPLBACKEND='Agg')
results=[]
for name in ('Legacy','etc/SPNN','etc/SigmaPi','etc/SymPercept'):
 label=name.replace('/','-')
 result=bounded.run_guarded([sys.executable,str(ROOT/'bin'/(name+'.py'))],cwd=ROOT,env=env,
    log_path=HERE/('inline-'+label+'.log'),memory_bytes=8*bounded.GIB,timeout=1800)
 assert bounded.source_snapshot(ROOT)==frozen
 bounded.write_json(HERE/('inline-'+label+'.json'),result)
 results.append(result)
 print(label,{k:result[k] for k in ('reason','exit_code','elapsed_seconds','peak_memory_bytes')},flush=True)
result=bounded.run_suite(root=ROOT,selectors=[
 'test/test_meronomy_utf8.py','test/test_within_whole_division.py','test/test_where_bracket.py',
 'test/test_conceptual_introspection.py','test/test_mereology.py','test/test_surface_schema.py',
 'test/test_sigmapi.py','test/test_retired_names.py'],run_dir=HERE/'items-1-5-check',
 memory_bytes=24*bounded.GIB,worker_memory_bytes=8*bounded.GIB,workers=1,
 timeout=1800,suite_timeout=3600,batch_size=128,max_files=8)
assert bounded.source_snapshot(ROOT)==frozen
raise SystemExit(int(any(x['exit_code'] for x in results) or result['exit_code']))
