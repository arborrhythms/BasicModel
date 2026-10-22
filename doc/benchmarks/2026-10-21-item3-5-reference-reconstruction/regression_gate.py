"""The additional MTM, anonymous bind, PartClosing and occurrence-view gates."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert source == json.loads((HERE/'frozen-source.json').read_text())['source']
assert not subprocess.check_output(['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'],
                                   cwd=ROOT, text=True).strip()
selectors = [f'test/test_{name}.py' for name in (
    'medium_term_memory', 'reference_arrivals', 'reference_arrival_review', 'reference_choices',
    'part_closing', 'reference_occurrence_views', 'reference_review10', 'reference_participants')]
name = 'regressions-final'
assert not (HERE/name).exists() and not (HERE/(name+'.log')).exists()
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
           DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=os.pathsep.join(map(str, (HERE, ROOT/'bin', ROOT/'test'))),
    PYTEST_PLUGINS='contract_capture', CONTRACT_CAPTURE=str(HERE/(name+'-state')))
command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py', *selectors,
    '--workers', '1', '--memory-gib', '8', '--timeout', '600', '--suite-timeout', '900',
    '--batch-size', '16', '--max-files', '1', '--run-dir', str(HERE/name)]
with (HERE/(name+'.log')).open('w') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
(HERE/(name+'.json')).write_text(json.dumps(dict(source=source, receipt=name,
    selectors=selectors, exit=result.returncode), indent=2)+'\n')
print(name, result.returncode, flush=True)
sys.exit(result.returncode)
