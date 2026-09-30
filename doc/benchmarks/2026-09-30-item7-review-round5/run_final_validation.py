"""One declared round-5 campaign, followed by one source-matched full sweep.

Measurement workers retain the existing guards, deadlines and environments.
This parent only schedules independent jobs and records their ownership.
An exclusive marker prevents a second parent launching duplicate measurements.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = str(ROOT / '.venv/bin/python')
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-28-item7-review')]
from bounded_tests import source_snapshot
from review_source import supporting_inputs


def read(path):
    return json.loads(path.read_text())


def save(value):
    value['updated_at'] = datetime.now(timezone.utc).isoformat()
    temp = HERE / 'active-final-validation.tmp'
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(HERE / 'active-final-validation.json')


def main():
    marker = HERE / 'final-validation-started.json'
    with marker.open('x') as handle:
        json.dump(dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()), handle)
    source = read(HERE / 'final-source.json')
    inputs = read(HERE / 'final-inputs.json')
    assert source_snapshot(ROOT) == source
    assert supporting_inputs(ROOT) == inputs
    plan = read(HERE / 'final-validation-plan.json')
    state = dict(pid=os.getpid(), phase='explicit', jobs={})
    running = {}

    def start(label, command, *, slow='1'):
        assert label not in state['jobs']
        env = os.environ.copy()
        env.pop('BASIC_SEED', None)
        env['ITEM7_RUN_SLOW'] = slow
        log = HERE / (label + '-driver.log')
        with log.open('x') as handle:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        state['jobs'][label] = dict(pid=proc.pid, command=command, log=log.name,
                                   status='running', started_at=datetime.now(timezone.utc).isoformat())
        running[label] = proc
        save(state)

    def checks(label, slow='0'):
        start(label, [PYTHON, str(HERE / 'run_checks_matched.py'), label, str(ROOT),
                      *plan[label]], slow=slow)

    # At most three bounded workers are active: the XOR series, the MM series,
    # and the affected files. Graph release gets its own phase afterward.
    start('final-xor-candidate', [PYTHON, str(HERE / 'run_xor_matched.py'), 'final-xor-candidate', str(ROOT)])
    start('final-mm-grammar-candidate', [PYTHON, str(HERE / 'run_final_mm_grammar.py'), str(ROOT), 'final-mm-grammar-candidate'])
    checks('final-item7')
    while running:
        for label, proc in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            state['jobs'][label].update(status='finished', exit_code=code,
                                       finished_at=datetime.now(timezone.utc).isoformat())
            del running[label]
            assert source_snapshot(ROOT) == source, 'tested source changed'
            assert supporting_inputs(ROOT) == inputs, 'supporting inputs changed'
            save(state)
            if label == 'final-item7':
                checks('final-thought-reasoning')
        time.sleep(5)
    required = ('final-item7', 'final-thought-reasoning', 'final-mm-grammar-candidate')
    if any(state['jobs'][label]['exit_code'] for label in required):
        state['phase'] = 'needs_inspection_before_sweep'
        save(state)
        return 1
    state['phase'] = 'final-graph-release'
    checks('final-graph-release', slow='1')
    proc = running.pop('final-graph-release')
    code = proc.wait()
    state['jobs']['final-graph-release'].update(status='finished', exit_code=code,
        finished_at=datetime.now(timezone.utc).isoformat())
    assert source_snapshot(ROOT) == source
    assert supporting_inputs(ROOT) == inputs
    if code:
        state['phase'] = 'needs_inspection_before_sweep'
        save(state)
        return 1
    subprocess.run([PYTHON, str(HERE / 'summarize_xor.py'), str(HERE / 'final-xor-candidate')],
                   cwd=ROOT, check=True)
    for section in ('gates', 'xor', 'mm'):
        subprocess.run([PYTHON, str(HERE / 'summarize_final_receipt.py'), section], cwd=ROOT, check=True)
    state['phase'] = 'full-sweep'
    start('full-sweep', [PYTHON, str(HERE / 'run_final_sweep.py')], slow='0')
    proc = running.pop('full-sweep')
    code = proc.wait()
    state['jobs']['full-sweep'].update(status='finished', exit_code=code,
        finished_at=datetime.now(timezone.utc).isoformat())
    assert source_snapshot(ROOT) == source
    assert supporting_inputs(ROOT) == inputs
    state['phase'] = 'measurements_finished_receipt_pending'
    save(state)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
