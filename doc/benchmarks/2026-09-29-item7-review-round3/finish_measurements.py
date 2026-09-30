"""Finish the ordered receipt on the frozen source; never commit or fix code.

Waits for the already-running sixteen reconstruction measurements, then runs
the required XOR table, twenty full MM_grammar trials, explicit gates and one
complete sweep. Red tests are retained. Only unfinished sweep coverage resumes.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HEAD = Path(sys.argv[1]).resolve()
STATUS = HERE / 'finish-status.json'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot

source = source_snapshot(ROOT)
status = dict(stage='waiting_for_reconstruction', steps=[], finished=False)


def save():
    STATUS.write_text(json.dumps(status, indent=2) + '\n')


def run(stage, arguments, allowed=(0, 1, 137)):
    assert source_snapshot(ROOT) == source, 'frozen source changed'
    status['stage'] = stage
    save()
    started = time.monotonic()
    print('START', stage, flush=True)
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT)
    status['steps'].append(dict(stage=stage, exit_code=result.returncode,
                                seconds=time.monotonic() - started))
    save()
    assert source_snapshot(ROOT) == source, 'frozen source changed'
    if result.returncode not in allowed:
        raise RuntimeError(f'{stage}: infrastructure exit {result.returncode}')
    print('END', stage, result.returncode, flush=True)


save()
try:
    while not all((HERE / (label + '-reconstruction') / 'driver-hashes.json').exists()
                  for label in ('head', 'candidate')):
        time.sleep(5)
    assert source_snapshot(ROOT) == source
    for label in ('head', 'candidate'):
        processes = json.loads((HERE / (label + '-reconstruction') / 'processes.json').read_text())
        assert set(processes) == set(map(str, range(8)))
    os.environ['ITEM7_RUN_SLOW'] = '0'
    run('native_definition_context', [HERE / 'run_checks.py', 'native-definition-context',
        ROOT, HERE / 'probe_definition_context_native.py'])
    run('final_xor', [HERE / 'run_xor.py', 'final-xor', ROOT])
    run('summarize_xor', [HERE / 'summarize_xor.py', HERE / 'final-xor'], allowed=(0,))
    run('mm_grammar_twenty_runs', [HERE / 'run_mm_grammar_pair.py', HEAD], allowed=(0,))
    run('summarize_measurements', [HERE / 'summarize_measurements.py'], allowed=(0,))
    run('explicit_gates', [HERE / 'run_explicit.py'])
    run('full_sweep', [HERE / 'run_full_sweep.py'])
    result = json.loads((HERE / 'full-sweep/run/result.json').read_text())
    if set(result['selected']) != set(result['completed']):
        run('unfinished_sweep_coverage', [HERE / 'continue_sweep.py'], allowed=(137,))
    run('summarize_sweep', [HERE / 'summarize_sweep.py'], allowed=(0,))
    status.update(stage='ready_for_receipt_review', finished=True)
except BaseException as error:
    status.update(stage='infrastructure_error', error=repr(error))
    raise
finally:
    save()
