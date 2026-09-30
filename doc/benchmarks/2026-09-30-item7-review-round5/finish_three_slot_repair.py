"""Wait for the initial frozen campaign, apply AK's saved-probe repair, validate.

This script never edits source while the initial validation owns it. It
preserves that campaign and starts the final campaign only if all focused
checks pass on the amended source. It cannot launch a duplicate attempt.
"""
import ast
from datetime import datetime, timezone
import difflib
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


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def status(phase, **fields):
    write(HERE / 'three-slot-repair-state.json', dict(pid=os.getpid(), phase=phase,
        updated_at=datetime.now(timezone.utc).isoformat(), **fields))


def main():
    with (HERE / 'three-slot-repair-started.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()), handle)
    status('waiting_for_initial_validation')
    while True:
        initial = read(HERE / 'active-validation.json')
        if initial['phase'] == 'needs_inspection_before_sweep':
            assert all(job['status'] == 'finished' for job in initial['jobs'].values())
            break
        assert initial['phase'] == 'explicit', initial['phase']
        time.sleep(5)
    assert not (HERE / 'full-sweep').exists()
    assert source_snapshot(ROOT) == read(HERE / 'frozen-source.json')
    assert supporting_inputs(ROOT) == read(HERE / 'frozen-inputs.json')
    # The one failed item-7 assertion and all four new probes precede repair.
    initial_item7 = read(HERE / 'item7/result.json')
    failures = []
    for group in initial_item7['groups']:
        for worker in read(HERE / 'item7' / group['receipt'])['workers']:
            failures.extend(r['nodeid'] for r in worker.get('reports', []) if r['outcome'] == 'failed')
    assert failures == ['test/test_item7_unindexed_relations.py::test_three_slot_relation_closes_unindexed_numerical_operands']
    assert read(HERE / 'thought-reasoning/result.json')['reason'] == 'passed'
    assert read(HERE / 'three-slot-before/result.json')['reason'] == 'failed'
    for action in ('xor', 'mm'):
        subprocess.run([PYTHON, str(HERE / 'summarize_receipt.py'), action], cwd=ROOT, check=True)

    file = ROOT / 'bin/ClauseJournal.py'
    before = file.read_text()
    old = "        operation = None\n        if registry is not None:\n            probe = meaning\n"
    replacement = """        operation = None
        if type(references[1]) is int:
            selected = predicate_relation(references[1])
            if selected != 'operator':
                # A first predicate occurrence carries its value into the
                # writer, just as a selected binary grammar operation does.
                # It has no earlier inventory or LTM row to read.
                operation = selected
                predicate = operation_concept(stack[1], selected)
                references[1] = predicate
                roles[1] = predicate.point
        if operation is None and registry is not None:
            probe = meaning
"""
    assert before.count(old) == 1
    after = before.replace('Clause, ClausePredicate, predicate_point',
                           'Clause, ClausePredicate, predicate_point, predicate_relation').replace(old, replacement)
    ast.parse(after)
    test_file = ROOT / 'test/test_item7_predicate_identity.py'
    test_before = test_file.read_text()
    probe = (HERE / 'test_three_slot_predicate_probe.py').read_text()
    addition = probe[probe.index("@pytest.mark.parametrize('relation'"):]
    test_after = test_before.replace('from ClauseRow import predicate_identity\n',
        'from ClauseRow import predicate_identity\nfrom reading_fixtures import finish_reading\n')
    test_after = test_after.replace('from Queries import GrammaticalThoughtRegistry\n',
        'from Queries import GrammaticalThoughtRegistry\nfrom Understanding import AnswerProgram\n')
    test_after += '\n\n' + addition
    ast.parse(test_after)
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                     fromfile='before/bin/ClauseJournal.py', tofile='after/bin/ClauseJournal.py'))
    patch += ''.join(difflib.unified_diff(test_before.splitlines(True), test_after.splitlines(True),
                      fromfile='before/test/test_item7_predicate_identity.py', tofile='after/test/test_item7_predicate_identity.py'))
    (HERE / 'three-slot-repair.patch').write_text(patch)
    file.write_text(after)
    test_file.write_text(test_after)
    source = source_snapshot(ROOT)
    write(HERE / 'final-source.json', source)
    write(HERE / 'final-inputs.json', supporting_inputs(ROOT))
    status('focused_checks_running')
    selectors = ['test/test_item7_predicate_identity.py', 'test/test_item7_unindexed_relations.py',
        'test/test_grammatical_query_vps.py', 'test/test_thought_operation_catalog.py',
        'test/test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm',
        'test/test_item9b_interpret.py::test_selected_generic_grammar_ends_the_interpreted_kinds']
    with (HERE / 'three-slot-after-driver.log').open('x') as log:
        code = subprocess.run([PYTHON, str(HERE / 'run_checks_matched.py'), 'three-slot-after',
                               str(ROOT), *selectors], cwd=ROOT, stdout=log, stderr=log).returncode
    assert source_snapshot(ROOT) == source
    if code:
        status('focused_checks_need_inspection', exit_code=code)
        return code
    subprocess.run([PYTHON, str(HERE / 'audit_final_source.py')], cwd=ROOT, check=True)
    write(HERE / 'final-harness-source.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sorted(HERE.glob('*.py'))})
    with (HERE / 'final-validation-driver.log').open('x') as log:
        proc = subprocess.Popen([PYTHON, str(HERE / 'run_final_validation.py')], cwd=ROOT,
            stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    write(HERE / 'final-validation-launch.json', dict(pid=proc.pid, driver='run_final_validation.py'))
    status('final_validation_started', final_pid=proc.pid)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        status('needs_inspection', error=repr(error))
        raise
