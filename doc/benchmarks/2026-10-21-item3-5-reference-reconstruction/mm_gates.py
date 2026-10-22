"""Replay both declared eight-seat MM draws without replacing either state."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RECEIPT = Path(__file__).resolve().parent
ROOT = RECEIPT.parents[2]
ARM = sys.argv[1]
assert ARM in ('saved', 'fresh')
SOURCE = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert SOURCE == json.loads((RECEIPT/'frozen-source.json').read_text())['source']
assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', 'bin', 'test', 'data'],
                                   cwd=ROOT, text=True).strip()
initial = (RECEIPT.parent/'2026-10-20-item3-5-stm1/rng-119/'
           'test_mm_grammar_learns_xor_signal/initial-rng.pt' if ARM == 'saved' else
           RECEIPT.parent/'2026-10-20-item3-5-inline-lowering/mm-fresh-02-state/initial-rng.pt')
name = f'mm-{ARM}-final'
assert not (RECEIPT/name).exists() and not (RECEIPT/(name+'.log')).exists()
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=os.pathsep.join(map(str, (RECEIPT, ROOT/'bin', ROOT/'test'))),
    PYTEST_PLUGINS='capacity_probe', CAPACITY_REPLAY=str(initial),
    CAPACITY_RECEIPT=str(RECEIPT/(name+'-state')))
selector = 'test/test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal'
command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py', selector,
    '--workers', '1', '--memory-gib', '8', '--timeout', '900',
    '--suite-timeout', '1000', '--run-dir', str(RECEIPT/name)]
with (RECEIPT/(name+'.log')).open('w') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
record = dict(receipt=name, source=SOURCE, selector=selector, exit=result.returncode,
    initial_rng=str(initial.relative_to(ROOT)),
    initial_rng_sha256=hashlib.sha256(initial.read_bytes()).hexdigest())
(RECEIPT/(name+'.json')).write_text(json.dumps(record, indent=2)+'\n')
print(name, result.returncode, flush=True)
sys.exit(result.returncode)
