"""The requested eight-file set, both original decomposition arms, plus paired R."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = Path(__file__).resolve().parent
PRIOR = RECEIPT.parent/'2026-10-20-item3-5-stm1'
MODE = sys.argv[1]
assert MODE in ('native', 'decomposition')
SOURCE = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', 'bin', 'test', 'data'],
                                   cwd=ROOT, text=True).strip()
ENV = os.environ.copy()
for name in ('ARRIVAL_REPLAY', 'ARRIVAL_CLOSING_BASELINE', 'CLOSING_REPLAY'):
    ENV.pop(name, None)
ENV.update(BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=os.pathsep.join(map(str, (RECEIPT, PRIOR, PRIOR/'review3',
        ROOT/'doc/benchmarks/2026-10-19-item4-step0-round2', ROOT/'bin', ROOT/'test'))))
native = [f'test/test_{name}.py' for name in (
    'math_chain_section14', 'reference_close', 'reference_mentions', 'bind_thought',
    'identity_rows', 'identity_cube', 'part_store')]
native.append('test/test_operators_final_b.py::test_both_training_trials_share_bootstrap_snapshot')
selections = [('native-final', native)] if MODE == 'native' else [
    (f'decomposition-{arm}-final', ['test/test_math_chain_decomposition.py'])
    for arm in ('zero', 'enabled')]
results = []
for name, selectors in selections:
    env = dict(ENV, CLOSING_TRACE=str(RECEIPT/(name+'-trace.jsonl')),
        SCOPED_REPORTS=str(RECEIPT/(name+'-reports')),
        ITEM35_BIND_RECEIPT=str(RECEIPT/(name+'-generated')))
    if MODE == 'native':
        env.update(PYTEST_PLUGINS='closing_probe,arrival_capture,arrival_keep_probe,early_failure_probe,reference_probe,bootstrap_probe',
                   BOOTSTRAP_CAPTURE=str(RECEIPT/(name+'-bootstrap-state')),
                   ARRIVAL_KEEP_DIR=str(RECEIPT/(name+'-keep')),
                   ARRIVAL_KEEP_SOURCE=SOURCE,
                   REFERENCE_EVIDENCE=str(RECEIPT/(name+'-reference-evidence')),
                   ARRIVAL_CAPTURE=str(RECEIPT/(name+'-state')))
    else:
        env.update(PYTEST_PLUGINS='closing_probe,bind_gate', BIND_GAIN_ARM=name.split('-')[1])
    command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py', *selectors,
        '--workers', '1', '--memory-gib', '8', '--timeout', '900',
        '--suite-timeout', '1800', '--batch-size', '16', '--max-files', '1',
        '--run-dir', str(RECEIPT/name)]
    assert not (RECEIPT/name).exists() and not (RECEIPT/(name+'.log')).exists()
    with (RECEIPT/(name+'.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(receipt=name, source=SOURCE, selectors=selectors, exit=result.returncode))
    (RECEIPT/(MODE+'-final-summary.json')).write_text(json.dumps(results, indent=2)+'\n')
    print(name, result.returncode, flush=True)
sys.exit(int(any(result['exit'] for result in results)))
