"""Capped development commands, explicitly outside the declared measurement."""
from pathlib import Path
import json, os, sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import run_guarded, worker_environment
name,*command=sys.argv[1:]
env=worker_environment(ROOT)
env.pop('BASIC_SEED',None)
env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MODEL_COMPILE='none', BASICMODEL_DEVICE='cpu', BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false', PYTHONPATH='bin:test')
result=run_guarded([str(ROOT/'.venv/bin/python'),*command],cwd=ROOT,env=env,log_path=HERE/(name+'.log'),memory_bytes=8*1024**3,timeout=float(os.environ.get('DEVELOPMENT_TIMEOUT', '1800')))
print(json.dumps(result),flush=True)
sys.exit(result['exit_code'])
