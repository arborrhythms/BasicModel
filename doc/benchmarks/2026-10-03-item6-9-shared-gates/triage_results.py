"""Join saved outcomes to the ownership round's failed/stopped case IDs."""
from collections import Counter, defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def all_outcomes():
    rows = []
    sweep = read(HERE / 'full-sweep/receipt.json', {})
    for name in sweep.get('segments', []):
        segment = read(name, {})
        for worker in segment.get('workers', []):
            if worker.get('phase') == 'collection':
                continue
            for report in worker.get('reports', []):
                rows.append(dict(report, scope='full sweep', log=worker['log']))
    for report in sweep.get('failures', []):
        if report.get('outcome') == 'process_failed':
            rows.append(dict(report, scope='full sweep'))
    extras = read(HERE / 'extra-cases/complete.json', {})
    for report in extras.get('reports', []):
        rows.append(dict(report, scope='moved cases'))
    for report in extras.get('process_failures', []):
        rows.append(dict(report, scope='moved cases', outcome='process_failed'))
    for path in sorted(list((HERE / 'measurements').glob('gate-*/run/result.json')) + list((HERE / 'measurements').glob('xor-01/run/result.json'))):
        # Only the predeclared first gate run enters the named table.
        if 'trial-' in path.parts[-3] and not path.parts[-3].endswith('trial-01'):
            continue
        segment = read(path, {})
        for worker in segment.get('workers', []):
            for report in worker.get('reports', []):
                rows.append(dict(report, scope='named XOR table', log=worker['log']))
    return rows


def current_id(node):
    moved = read(HERE / 'prior-case-scheduling.json', [])
    for row in moved:
        if row['original'] == node:
            return row['current'], row.get('scheduled')
    base, sep, parameters = node.partition('[')
    for row in read(HERE / 'ports.json', []):
        if not base.startswith(row['file'] + '::'):
            continue
        for case in row['cases']:
            if base.endswith('::' + case['old']):
                if case['new'] is None:
                    return None, row['reason']
                return base[:-len(case['old'])] + case['new'] + sep + parameters, row['reason']
    return node, 'case ID unchanged'


def status(reports):
    outcomes = {r['outcome'] for r in reports}
    for name in ('process_failed', 'error', 'failed', 'passed', 'xpassed', 'xfailed', 'skipped'):
        if name in outcomes:
            return name
    return 'not observed'


def main():
    reports = all_outcomes()
    by_id = defaultdict(list)
    for report in reports:
        by_id[report['nodeid']].append(report)
    saved_prior = read(HERE / 'prior-failing-cases.json', [])
    prior = [row for row in saved_prior if row['scope'] in ('full sweep', 'extra slow cases')
             and row.get('report', {}).get('outcome') != 'xpassed']
    dispositions = []
    known = {current_id(row['nodeid'])[0] for row in saved_prior}
    for row in prior:
        node, reason = current_id(row['nodeid'])
        if node:
            known.add(node)
        observations = by_id[node] if node else []
        dispositions.append(dict(original=row['nodeid'], previous_scope=row['scope'],
            current=node, reason=reason, outcome=status(observations) if node else 'retired',
            reports=observations))
    failures = []
    notes = read(HERE / 'failure-notes.json', {})
    for node, observations in by_id.items():
        if status(observations) not in ('error', 'failed', 'process_failed'):
            continue
        bad = [r for r in observations if r['outcome'] in ('error', 'failed', 'process_failed')]
        if node.endswith('TestMMXorConvergence::test_convergence'):
            classification = 'declared §17 MM_xor exception; unchanged proof'
        elif node.endswith('TestMMXorConvergence::test_mm_grammar_learns_xor_signal'):
            classification = 'check whether this is the declared MM_grammar .25 stop'
        elif 'TestXorGrammar' in node:
            classification = 'predeclared first grammar gate outcome'
        elif node in known:
            classification = 'prior failing/stopped case remains non-passing; inspect current cause'
        else:
            classification = 'new or previously unrecorded non-passing case; unwaived'
        investigation = notes.get(node)
        if investigation:
            classification = investigation['classification']
        failures.append(dict(nodeid=node, classification=classification, reports=bad,
                             investigation=investigation))
    output = dict(prior_dispositions=dispositions, failures=failures,
        prior_outcome_counts=dict(Counter(r['outcome'] for r in dispositions)),
        full_sweep=read(HERE / 'full-sweep/receipt.json'),
        extras=read(HERE / 'extra-cases/complete.json'))
    (HERE / 'triage.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Closing failure and port triage', '',
        'This index joins first saved outcomes. A retirement is not a passing port. '
        'A repeated case ID alone does not establish an identical cause; messages are retained in '
        '[triage.json](triage.json) for review. No failure is retried to select a passing result.', '',
        '## Previous failed or stopped cases', '',
        '| Prior case | Current case | Outcome | Disposition |', '|---|---|---|---|']
    for row in dispositions:
        lines.append('| `' + row['original'] + '` | ' + ('`' + row['current'] + '`' if row['current'] else 'retired')
                     + ' | ' + row['outcome'] + ' | ' + row['reason'].replace('|', ' / ') + ' |')
    lines += ['', 'Counts: `' + json.dumps(output['prior_outcome_counts']) + '`.', '',
              '## Current non-passing cases', '', '| Case | Status | First error summary |', '|---|---|---|']
    for row in failures:
        report = row['reports'][0]
        message = report.get('message', report.get('reason', ''))
        errors = [line.strip() for line in message.splitlines() if line.startswith('E ') or line.startswith('E\t')]
        detail = ' '.join(errors[:3]) if errors else message[-500:]
        detail = detail[:650].replace('|', ' / ').replace('\n', ' ')
        lines.append('| `' + row['nodeid'] + '` | ' + row['classification'] + ' | ' + detail + ' |')
    lines += ['', '## Cause comparisons', '']
    for row in failures:
        note = row.get('investigation')
        if note:
            lines += ['- `' + row['nodeid'] + '`: ' + note['cause'] + ' '
                      + note['comparison'] + ' ' + note['action']
                      + ' [Saved outcome](' + note['evidence'] + ').', '']
    lines += ['', 'A full sweep skip of a weekly case does not override its actual moved-case outcome. '
              'The full-sweep case count and wall time, worker guard stops and the moved-case coverage are '
              'reported separately in the main receipt.', '']
    (HERE / 'triage.md').write_text('\n'.join(lines))
    print(json.dumps(dict(prior=output['prior_outcome_counts'], nonpassing_cases=len(failures))))


if __name__ == '__main__':
    main()
