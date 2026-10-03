"""Read saved live sweep reports; never execute tests or mutate the candidate."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
status = json.loads((HERE / 'status.json').read_text())
parts = sorted((HERE / 'full-sweep').glob('part-*/result.json'))
rows, completed, active, elapsed, selected = [], set(), [], 0., 0
stops = []
for path in parts:
    part = json.loads(path.read_text())
    is_active = (path == parts[-1] and status['stage'] == 'full_sweep'
                 and bool(part.get('active_workers')))
    elapsed += part['elapsed_seconds']
    if is_active:
        elapsed += max(0., time.time() - path.stat().st_mtime)
    selected = max(selected, len(part['selected']))
    completed.update(part['completed'])
    rows.extend(r for w in part['workers'] for r in w['reports'])
    stops.extend(dict(log=w['log'], reason=w['reason']) for w in part['workers']
                 if w['reason'] != 'exit')
    if is_active:
        for worker in part.get('active_workers', []):
            progress = Path(worker['progress_file'])
            if not progress.exists():
                continue
            current = json.loads(progress.read_text())
            completed.update(current.get('completed', []))
            rows.extend(current.get('reports', []))
            active.append(current.get('active'))
by_case = defaultdict(list)
for row in rows:
    by_case[row['nodeid']].append(row)
severity = ('failed', 'xpassed', 'passed', 'xfailed', 'skipped')
counts = Counter(next(s for s in severity if any(r['outcome'] == s for r in rs))
                 for rs in by_case.values()
                 if any(r['outcome'] in severity for r in rs))
failures = {r['nodeid']: r.get('message', '')
            for r in rows if r['outcome'] in ('failed', 'xpassed')}
saved = HERE / 'live-progress.json'
previous = json.loads(saved.read_text()) if saved.exists() else {}
report = dict(stage=status['stage'], minutes=elapsed / 60, selected=selected,
              completed=len(completed), reported_case_counts=dict(counts),
              active=active, process_stops=stops, failures=failures)
saved.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'failures'}))
new = {k: v for k, v in failures.items() if k not in previous.get('failures', {})}
if new:
    print(json.dumps({'new_failures': new}))
