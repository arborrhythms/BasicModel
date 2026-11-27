#!/usr/bin/env python
"""The gate of ``make push``: the log of the last complete ``make test``.

``make test`` with no selection writes ``output/tests/latest.json`` when it
ends (:func:`write_log`): where its receipt is, the source it tested and
every case that failed. ``make push`` reads it (:func:`blocking`) and pushes
only if the log is there, the run ended, the tested source is the source
being pushed, and every failed case has a written ok in
``test/failures_ok.txt``.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = Path('output/tests/latest.json')
OKS = Path('test/failures_ok.txt')
ENDED = ('passed', 'test_failure')


def source_digest(source):
    """One name for a tested source: the digest of its files' digests."""
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def failed_cases(result):
    return sorted({report['nodeid'] for worker in result['workers']
                   for report in worker.get('reports', ()) if report['outcome'] == 'failed'})


def write_log(root, result, run_dir):
    """Record a complete run as the log ``make push`` reads."""
    from bounded_tests import write_json
    root, run_dir = Path(root), Path(run_dir)
    source = json.loads((run_dir / 'source-manifest.json').read_text())['validated_source']
    path = root / LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(
        written=time.strftime('%Y-%m-%d %H:%M:%S'), run_dir=str(run_dir), receipt=str(run_dir / 'result.json'),
        exit_code=result['exit_code'], reason=result['reason'],
        selected=len(result['selected']), completed=len(result['completed']),
        failed=failed_cases(result), source=source_digest(source)))
    return path


def written_oks(root):
    """``{case: who, when and why}`` from the lines ``case | Name, date: reason``."""
    path, oks = Path(root) / OKS, {}
    if not path.is_file():
        return oks
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        case, bar, word = (part.strip() for part in line.partition('|'))
        if not bar or not case or ':' not in word or not word.split(':', 1)[1].strip():
            raise ValueError(f'{OKS}:{number}: expected "case | Name, date: reason"')
        oks[case] = word
    return oks


def blocking(root=ROOT):
    """Why this source may not be pushed, as lines; none when it may.

    Returns ``(reasons, notes)``.
    """
    from bounded_tests import source_snapshot
    root = Path(root)
    path = root / LOG
    if not path.is_file():
        return [f'There is no test log ({LOG}). Run the complete suite: make test'], []
    log = json.loads(path.read_text())
    reasons, notes = [], [f"Test log of {log['written']}: {log['completed']}/{log['selected']} cases, "
                          f"{len(log['failed'])} failed ({log['receipt']})"]
    if log['reason'] not in ENDED or log['completed'] != log['selected']:
        reasons.append(f"The logged run did not end ({log['reason']}; {log['completed']} of "
                       f"{log['selected']} cases). Run it again: make test")
    if source_digest(source_snapshot(root)) != log['source']:
        reasons.append('The source has changed since the logged run (bin, test, data, Makefile, '
                       'pytest.ini, requirements.txt or README.md). Run it again: make test')
    oks = written_oks(root)
    unanswered = [case for case in log['failed'] if case not in oks]
    if unanswered:
        reasons.append(f'{len(unanswered)} failed with no written ok in {OKS}:\n'
                       + '\n'.join('    ' + case for case in unanswered))
    notes.extend(f'ok to fail: {case} ({oks[case]})' for case in log['failed'] if case in oks)
    notes.extend(f'written ok no longer needed, the case did not fail: {case}'
                 for case in oks if case not in log['failed'])
    return reasons, notes


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    reasons, notes = blocking(ROOT)
    for note in notes:
        print(note)
    if not reasons and 'push' in argv:
        dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'],
                               capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            reasons.append('There are uncommitted changes; a push sends commits. Commit them first.')
    if reasons:
        print('NOT PUSHED:' if 'push' in argv else 'WOULD NOT PUSH:')
        for reason in reasons:
            print('  ' + reason)
        return 1
    if 'push' not in argv:
        print('The gate is open: make push would push.')
        return 0
    return subprocess.run(['git', '-C', str(ROOT), 'push', '-u', 'origin', 'HEAD',
                           *[a for a in argv if a != 'push']]).returncode


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
