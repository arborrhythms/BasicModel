"""Six predeclared fresh initializations per order; retain every outcome."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
RECEIPT = Path(__file__).resolve().parent
ARM = sys.argv[1]
assert ARM in ('forward', 'reverse')
SOURCE = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
ENV = os.environ.copy()
for name in ('ARRIVAL_REPLAY', 'ARRIVAL_CLOSING_BASELINE'):
    ENV.pop(name, None)
ENV.update(BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=f'{RECEIPT}:{ROOT / "bin"}:{ROOT / "test"}',
    PYTEST_PLUGINS='arrival_capture')
CASES = []
for number in range(1, 7):
    name = f'fresh-both-{ARM}-{number}'
    run = RECEIPT / name
    ENV['ARRIVAL_CAPTURE'] = str(RECEIPT / (name + '-state'))
    selector = ('test/test_math_chain_section14.py::'
        f'test_forced_ordinary_pending_premise_in_both_orders[{ARM == "reverse"}]')
    command = [str(ROOT / '.venv/bin/python'), 'test/test_report.py', selector,
        '--workers', '1', '--memory-gib', '8', '--timeout', '240',
        '--suite-timeout', '300', '--run-dir', str(run)]
    with (RECEIPT / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    CASES.append(dict(number=number, source=SOURCE, arm=ARM, exit=result.returncode,
                      receipt=name, selector=selector))
    (RECEIPT / f'fresh-both-{ARM}.json').write_text(json.dumps(CASES, indent=2) + '\n')
    print(f'{name}: exit {result.returncode}', flush=True)
sys.exit(int(any(case['exit'] for case in CASES)))
