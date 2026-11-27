"""The gate of ``make push``: a log, an ended run, the same source, and a
written ok for every failure."""
import json

import pytest

import push_gate
from bounded_tests import source_snapshot

FAILED = 'test/test_a.py::test_fails'


def tree(tmp_path, *, failed=(FAILED,), reason='test_failure', completed=2, oks=''):
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'bin' / 'a.py').write_text('x = 1\n')
    (tmp_path / 'test').mkdir()
    (tmp_path / 'test' / 'failures_ok.txt').write_text(oks)
    run = tmp_path / 'output' / 'tests' / 'run'
    run.mkdir(parents=True)
    (run / 'source-manifest.json').write_text(json.dumps(dict(validated_source=source_snapshot(tmp_path))))
    reports = [dict(nodeid=case, outcome='failed') for case in failed] + [dict(nodeid='t::passes', outcome='passed')]
    result = dict(exit_code=int(bool(failed)), reason=reason, selected=['a', 'b'], completed=['a', 'b'][:completed],
                  workers=[dict(reports=reports)])
    push_gate.write_log(tmp_path, result, run)
    return tmp_path


def test_no_log_blocks(tmp_path):
    reasons, _ = push_gate.blocking(tmp_path)
    assert len(reasons) == 1 and 'no test log' in reasons[0]


def test_a_clean_run_of_this_source_opens_the_gate(tmp_path):
    assert push_gate.blocking(tree(tmp_path, failed=(), reason='passed'))[0] == []


def test_a_failure_without_a_written_ok_blocks(tmp_path):
    reasons, _ = push_gate.blocking(tree(tmp_path))
    assert len(reasons) == 1 and FAILED in reasons[0]


def test_a_written_ok_answers_its_failure_and_no_other(tmp_path):
    ok = f'# why\n{FAILED} | Alec, 2026-11-27: at integration\n'
    reasons, notes = push_gate.blocking(tree(tmp_path, oks=ok))
    assert reasons == [] and any('ok to fail' in note for note in notes)
    other = tmp_path / 'second'
    other.mkdir()
    reasons, _ = push_gate.blocking(tree(other, failed=(FAILED, 'test/test_a.py::test_other'), oks=ok))
    assert len(reasons) == 1 and 'test_other' in reasons[0] and FAILED not in reasons[0]


def test_an_ok_names_who_and_why(tmp_path):
    with pytest.raises(ValueError):
        push_gate.blocking(tree(tmp_path, oks=f'{FAILED}\n'))


def test_changed_source_blocks(tmp_path):
    root = tree(tmp_path, failed=(), reason='passed')
    (root / 'bin' / 'a.py').write_text('x = 2\n')
    reasons, _ = push_gate.blocking(root)
    assert len(reasons) == 1 and 'changed' in reasons[0]


@pytest.mark.parametrize('reason, completed', [('timeout', 2), ('source_changed', 2), ('passed', 1)])
def test_a_run_that_did_not_end_blocks(tmp_path, reason, completed):
    reasons, _ = push_gate.blocking(tree(tmp_path, failed=(), reason=reason, completed=completed))
    assert len(reasons) == 1 and 'did not end' in reasons[0]


def test_only_a_complete_run_writes_the_log():
    import inspect
    import bounded_tests
    source = inspect.getsource(bounded_tests.main)
    assert 'if requires_suite_lock(args.selectors, args.keyword, args.marker):' in source
