"""Summarize the one scoped run without promoting any earlier evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
source = json.loads((HERE/'frozen-source.json').read_text())['source']
assert not subprocess.check_output(
    ['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'],
    cwd=ROOT, text=True).strip()
runs = json.loads((HERE/'gate-runs.json').read_text())
groups, total = {}, Counter()
for run in runs:
    name = run['name']
    path = HERE/name/'result.json'
    data = json.loads(path.read_text())
    assert run['source'] == source
    reports = [report for worker in data['workers'] for report in worker['reports']]
    counts = Counter(report['outcome'] for report in reports
                     if report['phase'] == 'call' or report['outcome'] != 'passed')
    total.update(counts)
    groups[name] = dict(source=source, exit=run['exit'], reason=data['reason'],
        counts=dict(counts), selected=len(data['selected']), completed=len(data['completed']),
        elapsed_seconds=data['elapsed_seconds'], report=str(path.relative_to(HERE)),
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
certificates = []
for path in sorted((HERE/'native-reports').glob('test_forced_ordinary_pending_premise_in_both_orders*/certificate.json')):
    data = json.loads(path.read_text())
    certificates.append(dict(certificate=str(path.relative_to(HERE)),
        **{key: data[key] for key in ('reverse', 'bound', 'open', 'closure_forced', 'attention_budget')}))
passed = (set(groups) == {'modules', 'regressions', 'native'}
          and all(group['exit'] == 0 and group['reason'] == 'passed'
                  and group['selected'] == group['completed'] for group in groups.values())
          and not (set(total) - {'passed', 'skipped'}) and len(certificates) == 2)
summary = dict(source=source, groups=groups, counts=dict(total),
    all_requested_gates_pass=passed, pending_premise=certificates,
    full_sweep=False, fresh_series=False)
(HERE/'gate-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
