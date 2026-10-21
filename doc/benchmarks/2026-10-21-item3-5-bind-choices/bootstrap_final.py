"""Replay finding C's exact initial RNG on the frozen source."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

RECEIPT = Path(__file__).resolve().parent
ROOT = RECEIPT.parents[2]
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert source == '575cab06cb7d2ecb94a0c02cf64cb71bc6de33b0'
assert not subprocess.check_output(['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'], cwd=ROOT, text=True).strip()
initial = RECEIPT.parent/'2026-10-21-item3-5-anonymous-bind-v2/bootstrap-diagnostic-01-state'
name = 'bootstrap-final'
assert not (RECEIPT/name).exists() and not (RECEIPT/(name+'.log')).exists()
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=os.pathsep.join(map(str, (RECEIPT, ROOT/'bin', ROOT/'test'))),
    PYTEST_PLUGINS='bootstrap_probe', BOOTSTRAP_REPLAY=str(initial),
    BOOTSTRAP_CAPTURE=str(RECEIPT/(name+'-state')))
selector = 'test/test_operators_final_b.py::test_both_training_trials_share_bootstrap_snapshot'
command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py', selector,
    '--workers', '1', '--memory-gib', '8', '--timeout', '300',
    '--suite-timeout', '360', '--run-dir', str(RECEIPT/name)]
with (RECEIPT/(name+'.log')).open('w') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
(RECEIPT/(name+'.json')).write_text(json.dumps(dict(source=source, receipt=name,
    selector=selector, exit=result.returncode, command=command,
    initial_rng=str((initial/'initial-rng.pt').relative_to(ROOT)),
    initial_rng_sha256=hashlib.sha256((initial/'initial-rng.pt').read_bytes()).hexdigest()), indent=2)+'\n')
print(name, result.returncode, flush=True)
raise SystemExit(result.returncode)
