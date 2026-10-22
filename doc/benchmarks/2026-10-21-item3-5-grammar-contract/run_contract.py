"""Run once per frozen source, stopping at the first contract failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = Path(__file__).resolve().parent
PRIOR = RECEIPT.parent/'2026-10-21-item3-5-review11-landing'
SOURCE = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', 'bin', 'test', 'data'],
                                   cwd=ROOT, text=True).strip()
env = os.environ.copy()
for name in ('ARRIVAL_REPLAY', 'ARRIVAL_CLOSING_BASELINE', 'CLOSING_REPLAY'):
    env.pop(name, None)
env.update(BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=os.pathsep.join(map(str, (RECEIPT, PRIOR, ROOT/'bin', ROOT/'test'))),
    PYTEST_PLUGINS='contract_capture,arrival_capture,arrival_keep_probe,early_failure_probe',
    CONTRACT_CAPTURE=str(RECEIPT/'units-state'),
    ARRIVAL_CAPTURE=str(RECEIPT/'pending-state'),
    ARRIVAL_KEEP_DIR=str(RECEIPT/'pending-keep'), ARRIVAL_KEEP_SOURCE=SOURCE,
    ITEM35_BIND_RECEIPT=str(RECEIPT/'generated'))
groups = [('units', ['test/test_bind_thought.py', 'test/test_reference_participants.py',
                    'test/test_reference_arrivals.py', 'test/test_reference_close.py']),
          ('pending-forward', ['test/test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders[False]']),
          ('pending-reverse', ['test/test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders[True]'])]
results = []
for name, selectors in groups:
    log = RECEIPT/(name+'.log')
    assert not log.exists()
    command = [str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', '-x',
               '--tb=long', '--showlocals', '--basetemp='+str(RECEIPT/(name+'-tmp')),
               '--junitxml='+str(RECEIPT/(name+'.xml')), *selectors]
    start = time.time()
    (RECEIPT/(name+'-request.json')).write_text(json.dumps(dict(source=SOURCE,
        command=command, fresh=True, started=start), indent=2)+'\n')
    with log.open('x') as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=900)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    results.append(dict(name=name, source=SOURCE, exit=code, seconds=time.time()-start))
    (RECEIPT/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(name, code, flush=True)
    if code:
        sys.exit(code)
