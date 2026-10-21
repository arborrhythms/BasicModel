"""Six predeclared fresh initializations per order, with every trial retained."""
import json
import os
from pathlib import Path
import subprocess
import sys

RECEIPT = Path(__file__).resolve().parent
ROOT = RECEIPT.parents[2]
PRIOR = RECEIPT.parent/'2026-10-20-item3-5-stm1'/'review3'
ARM = sys.argv[1]
assert ARM in ('forward', 'reverse')
SOURCE = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert not subprocess.check_output(['git', 'diff', '--name-only', '--', 'bin', 'test', 'data'],
                                   cwd=ROOT, text=True).strip()
ENV = os.environ.copy()
for key in ('ARRIVAL_REPLAY', 'ARRIVAL_CLOSING_BASELINE'):
    ENV.pop(key, None)
ENV.update(BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=f'{RECEIPT}:{PRIOR}:{ROOT / "bin"}:{ROOT / "test"}',
    PYTEST_PLUGINS='arrival_capture,arrival_keep_probe', ARRIVAL_KEEP_SOURCE=SOURCE)
cases = []
for number in range(1, 7):
    name = f'fresh-final-{ARM}-{number}'
    run = RECEIPT/name
    assert not run.exists() and not (RECEIPT/(name+'.log')).exists()
    ENV.update(ARRIVAL_CAPTURE=str(RECEIPT/(name+'-state')),
        ARRIVAL_KEEP_FILE=str(RECEIPT/(name+'-keep.jsonl')), ARRIVAL_KEEP_CASE=name)
    selector = ('test/test_math_chain_section14.py::'
        f'test_forced_ordinary_pending_premise_in_both_orders[{ARM == "reverse"}]')
    command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py', selector,
        '--workers', '1', '--memory-gib', '8', '--timeout', '240',
        '--suite-timeout', '300', '--run-dir', str(run)]
    with (RECEIPT/(name+'.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    cases.append(dict(number=number, source=SOURCE, arm=ARM,
                      exit=result.returncode, receipt=name, selector=selector))
    (RECEIPT/f'fresh-final-{ARM}.json').write_text(json.dumps(cases, indent=2)+'\n')
    print(f'{name}: exit {result.returncode}', flush=True)
sys.exit(int(any(case['exit'] for case in cases)))
