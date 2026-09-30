"""Adopt existing measurements; finish the unchanged selection with faster scheduling.

The stopped serial coordinator is never restarted. Completed measurements are
not repeated. This continuation owns the remaining affected files, graph gate,
and the one full sweep, with exclusive creation and durable process ownership.
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
from bounded_tests import source_snapshot, write_json
from review_source import supporting_inputs


def read(path):
    return json.loads(path.read_text())


def main():
    with (HERE / 'scheduled-validation-started.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()), handle)
    source, inputs = read(HERE / 'final-source.json'), read(HERE / 'final-inputs.json')
    harness = read(HERE / 'scheduling-harness-source.json')

    def verify():
        assert source_snapshot(ROOT) == source
        assert supporting_inputs(ROOT) == inputs
        assert all(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
                   for name, digest in harness['files'].items())

    verify()
    state = read(HERE / 'active-final-validation.json')
    assert state['phase'] == 'scheduling_hold_requested_by_user'
    assert not (HERE / 'full-sweep').exists()
    state.update(pid=os.getpid(), phase='scheduled_explicit',
                 continuation='resume_scheduled_validation.py')
    owned = {}

    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(HERE / 'active-final-validation.json', state)

    def start(label, command, slow='0'):
        verify()
        assert label not in state['jobs']
        env = os.environ.copy()
        env.pop('BASIC_SEED', None)
        env['ITEM7_RUN_SLOW'] = slow
        log = HERE / f'{label}-driver.log'
        with log.open('x') as handle:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        owned[label] = proc
        state['jobs'][label] = dict(pid=proc.pid, command=command, log=log.name,
            status='running', started_at=datetime.now(timezone.utc).isoformat())
        save()

    def durable_outcome(label):
        directory = HERE / label
        if label == 'final-mm-grammar-candidate':
            path = directory / 'processes.json'
            if not path.exists():
                return None
            trials = read(path)
            if set(trials) != set(map(str, range(10))):
                return None
            return int(any(t['reason'] != 'exit' or t['exit_code'] for t in trials.values()))
        path = directory / 'result.json'
        if not path.exists():
            return None
        result = read(path)
        if result['reason'] == 'running':
            return None
        return int(result['reason'] != 'passed')

    def wait_labels(labels):
        remaining = set(labels)
        while remaining:
            for label in tuple(remaining):
                job = state['jobs'][label]
                code = durable_outcome(label)
                if label in owned:
                    process_code = owned[label].poll()
                    if process_code is not None and code is None:
                        raise RuntimeError(f'{label} exited {process_code} without a complete receipt')
                    if code is not None and process_code is None:
                        continue
                elif code is None:
                    command = subprocess.run(['ps', '-p', str(job['pid']), '-o', 'command='],
                                             capture_output=True, text=True).stdout
                    assert command.strip() and str(HERE) in command, (label, 'lost owner')
                if code is None:
                    continue
                job.update(status='finished', exit_code=code,
                           finished_at=datetime.now(timezone.utc).isoformat())
                remaining.remove(label)
                verify()
                save()
            if remaining:
                time.sleep(5)

    assert durable_outcome('final-item7') == 0
    start('final-thought-reasoning', [PYTHON, str(HERE / 'run_scheduled_checks.py'),
                                    'final-thought-reasoning'])
    wait_labels(('final-item7', 'final-xor-candidate', 'final-mm-grammar-candidate',
                 'final-thought-reasoning'))
    if any(state['jobs'][label]['exit_code'] for label in
           ('final-item7', 'final-mm-grammar-candidate', 'final-thought-reasoning')):
        state['phase'] = 'needs_inspection_before_sweep'
        save()
        return 1
    state['phase'] = 'final-graph-release'
    selector = read(HERE / 'final-validation-plan.json')['final-graph-release']
    start('final-graph-release', [PYTHON, str(HERE / 'run_checks_matched.py'),
          'final-graph-release', str(ROOT), *selector], slow='1')
    wait_labels(('final-graph-release',))
    if state['jobs']['final-graph-release']['exit_code']:
        state['phase'] = 'needs_inspection_before_sweep'
        save()
        return 1
    subprocess.run([PYTHON, str(HERE / 'summarize_xor.py'), str(HERE / 'final-xor-candidate')],
                   cwd=ROOT, check=True)
    for section in ('gates', 'xor', 'mm'):
        subprocess.run([PYTHON, str(HERE / 'summarize_final_receipt.py'), section], cwd=ROOT, check=True)
    verify()
    state['phase'] = 'full-sweep'
    start('full-sweep', [PYTHON, str(HERE / 'run_final_sweep.py')])
    code = owned['full-sweep'].wait()
    state['jobs']['full-sweep'].update(status='finished', exit_code=code,
                                     finished_at=datetime.now(timezone.utc).isoformat())
    verify()
    state['phase'] = 'measurements_finished_receipt_pending'
    save()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
