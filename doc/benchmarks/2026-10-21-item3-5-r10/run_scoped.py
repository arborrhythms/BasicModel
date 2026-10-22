"""The five authorized R10 files, once on the frozen source, with captures."""
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert not subprocess.check_output(['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'],
                                   cwd=ROOT, text=True).strip()
selectors = ['test/test_bind_thought.py', 'test/test_reference_close.py',
    'test/test_reference_arrivals.py', 'test/test_reference_participants.py',
    'test/test_reference_reconstruction.py']
assert not (HERE/'scoped').exists() and not (HERE/'scoped.log').exists()
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=os.pathsep.join(map(str, (HERE.parent/'2026-10-21-item3-5-reference-reconstruction',
                                       ROOT/'bin', ROOT/'test'))),
    PYTEST_PLUGINS='contract_capture', CONTRACT_CAPTURE=str(HERE/'scoped-state'))
command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py', *selectors,
    '--workers', '1', '--memory-gib', '8', '--timeout', '600', '--suite-timeout', '900',
    '--batch-size', '16', '--max-files', '1', '--run-dir', str(HERE/'scoped')]
(HERE/'frozen-source.json').write_text(json.dumps(dict(source=source,
    parent=subprocess.check_output(['git', 'rev-parse', 'HEAD^'], cwd=ROOT, text=True).strip(),
    selectors=selectors, command=command), indent=2)+'\n')
with (HERE/'scoped.log').open('x') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
(HERE/'scoped.json').write_text(json.dumps(dict(source=source, selectors=selectors,
    command=command, exit=result.returncode), indent=2)+'\n')
print('scoped', result.returncode, flush=True)
raise SystemExit(result.returncode)
