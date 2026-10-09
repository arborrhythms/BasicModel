"""Development coverage of the standing thinking selectors; not a measurement."""
import json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PRIOR=HERE.parent/'2026-10-08-math-chain-repair'
sys.path[:0]=[str(ROOT/'test'),str(ROOT/'bin'),str(PRIOR)]
from bounded_tests import main
selectors=json.loads((PRIOR/'thinking-gate-plan.json').read_text())['selectors']
os.environ.pop('BASIC_SEED',None)
os.environ.update(MODEL_COMPILE='none',BASICMODEL_DEVICE='cpu',RUN_SLOW='1',OMP_NUM_THREADS='1',
    BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false',PYTHONDONTWRITEBYTECODE='1',
    PYTEST_PLUGINS='thinking_gate_observer',THINKING_GATE_OUTPUT=str(HERE/'development-thinking-observations'),
    PYTHONPATH=os.pathsep.join((str(PRIOR),str(ROOT/'bin'),str(ROOT/'test'))))
code,path=main([*selectors,'--workers','1','--memory-gib','8','--batch-size','128',
    '--run-dir',str(HERE/'development-thinking-01')])
x=json.loads(path.with_name('result.json').read_text())
print(json.dumps(dict(kind='development',exit_code=code,reason=x['reason'],
    selected=len(x['selected']),completed=len(x['completed']))))
raise SystemExit(code)
