"""The retained forward-five regression, using its complete original draw."""
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert source == json.loads((HERE/'frozen-source.json').read_text())['source']
assert not subprocess.check_output(['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'],
                                   cwd=ROOT, text=True).strip()
initial = HERE.parent/('2026-10-21-item3-5-review10-observer/fresh-final-forward-5-state/'
                       'test_forced_ordinary_pending_premise_in_both_orders[False]')
name = 'marker-replay-final'
assert not (HERE/name).exists() and not (HERE/(name+'.log')).exists()
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=os.pathsep.join(map(str, (HERE, ROOT/'bin', ROOT/'test'))),
    PYTEST_PLUGINS='arrival_capture,arrival_keep_probe,early_failure_probe',
    ARRIVAL_REPLAY=str(initial), ARRIVAL_CAPTURE=str(HERE/(name+'-state')),
    ARRIVAL_KEEP_FILE=str(HERE/(name+'-keep.jsonl')), ARRIVAL_KEEP_SOURCE=source, ARRIVAL_KEEP_CASE=name)
selector = 'test/test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders[False]'
command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py', selector,
    '--workers', '1', '--memory-gib', '8', '--timeout', '240', '--suite-timeout', '300',
    '--run-dir', str(HERE/name)]
with (HERE/(name+'.log')).open('w') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
(HERE/(name+'.json')).write_text(json.dumps(dict(source=source, receipt=name,
    selector=selector, exit=result.returncode, initial=str(initial.relative_to(ROOT))), indent=2)+'\n')
print(name, result.returncode, flush=True)
raise SystemExit(result.returncode)
