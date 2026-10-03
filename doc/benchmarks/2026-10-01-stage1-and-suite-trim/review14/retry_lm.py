from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as b
R=Path(__file__).resolve().parent
out=R/'mode-repair-batches-2';out.mkdir(exist_ok=False)
frozen=b.source_snapshot(ROOT)
b.write_json(out/'manifest.json',dict(source=frozen,guard_gib=8,config='data/LM_5M.xml',seed=None))
env=b.worker_environment(ROOT);env.pop('BASIC_SEED',None)
env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',RUN_SLOW='1')
result=b.run_guarded([sys.executable,str(R/'mode_first_batches.py'),'--child','--config','data/LM_5M.xml','--output',str(out/'LM_5M.json')],cwd=ROOT,env=env,log_path=out/'LM_5M.log',memory_bytes=8*b.GIB,timeout=1800)
b.write_json(out/'LM_5M-process.json',result)
b.write_json(out/'complete.json',dict(source_matched=b.source_snapshot(ROOT)==frozen,result=result))
print(result,flush=True)
