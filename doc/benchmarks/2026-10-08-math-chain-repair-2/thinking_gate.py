"""The unchanged 57 thinking selectors, once on the declared source."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent/'2026-10-08-math-chain-repair'
sys.path[:0] = [str(ROOT/'test'), str(HERE), str(PRIOR)]
import bounded_tests as bounded


def main():
    from verification import validate
    source = bounded.source_snapshot(ROOT)
    validate(source)
    selectors = json.loads((PRIOR/'thinking-gate-plan.json').read_text())['selectors']
    path = HERE/'thinking-gate-plan.json'
    with path.open('x') as stream:
        json.dump(dict(selectors=selectors,source=source,seed=None,retries=0,
            scope='Unchanged 57 mechanism nodes, unseeded helpers; no learned-chain claim.'),stream,indent=2)
    os.environ.pop('BASIC_SEED',None)
    os.environ.update(MODEL_COMPILE='none',BASICMODEL_DEVICE='cpu',RUN_SLOW='1',OMP_NUM_THREADS='1',
        BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false',PYTHONDONTWRITEBYTECODE='1',
        PYTEST_PLUGINS='thinking_gate_observer',THINKING_GATE_OUTPUT=str(HERE/'thinking-observations'),
        PYTHONPATH=os.pathsep.join((str(HERE),str(PRIOR),str(ROOT/'bin'),str(ROOT/'test'))))
    code,path = bounded.main([*selectors,'--workers','1','--memory-gib','8','--batch-size','128',
        '--run-dir',str(HERE/'thinking-gate')])
    result = json.loads(path.with_name('result.json').read_text())
    print(json.dumps(dict(exit_code=code,reason=result['reason'],selected=len(result['selected']),
        completed=len(result['completed']))))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
