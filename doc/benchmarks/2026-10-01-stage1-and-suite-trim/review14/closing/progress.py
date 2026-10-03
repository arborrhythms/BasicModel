"""Read saved live sweep reports; never execute tests or mutate the candidate."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
status = json.loads((HERE / 'status.json').read_text())
receipt = json.loads((HERE / 'full-sweep/receipt.json').read_text())
parts = sorted((HERE / 'full-sweep').glob('part-*'))
rows, completed, active = [], set(), []
stops = []
for part in parts:
    accounted = part / 'accounted.json'
    if accounted.exists():
        saved = json.loads(accounted.read_text())
        for worker in saved['workers']:
            completed.update(worker['completed'])
            rows.extend(worker['reports'])
        continue
    result_path = part / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    active_files = {w['progress_file'] for w in result.get('active_workers', [])}
    for request_path in sorted(part.glob('worker-*.request.json')):
        request = json.loads(request_path.read_text())
        if request.get('collect'):
            continue
        raw_path = part / (request_path.name.removesuffix('.request.json') + '.json')
        if not raw_path.exists():
            continue
        raw = json.loads(raw_path.read_text())
        completed.update(raw.get('completed', []))
        for row in raw.get('reports', []):
            if row['outcome'] == 'compile_cache_retry':
                row['outcome'] = row.get('original_outcome', 'failed')
            rows.append(row)
        if (status['stage'] == 'full_sweep' and str(raw_path) in active_files
                and raw.get('active') and raw['active'] not in raw.get('completed', [])):
            active.append(raw['active'])
for failure in receipt['failures']:
    if failure['phase'] == 'process':
        rows.append(failure)
        stops.append(failure)
by_case = defaultdict(list)
for row in rows:
    by_case[row['nodeid']].append(row)
severity = ('failed', 'process_failed', 'xpassed', 'passed', 'xfailed', 'skipped')
counts = Counter(next(s for s in severity if any(r['outcome'] == s for r in rs))
                 for rs in by_case.values()
                 if any(r['outcome'] in severity for r in rs))
failures = {r['nodeid']: r.get('message', '')
            for r in rows if r['outcome'] in ('failed', 'xpassed')}
saved = HERE / 'live-progress.json'
previous = json.loads(saved.read_text()) if saved.exists() else {}
elapsed = (receipt['duration_seconds'] if receipt['complete'] else
           (datetime.now(timezone.utc) - datetime.fromisoformat(receipt['started'])).total_seconds())
report = dict(stage=status['stage'], minutes=elapsed / 60, selected=receipt['selected'],
              completed=len(completed), reported_case_counts=dict(counts),
              active=active, process_stops=stops, failures=failures,
              duplicate_cases=receipt.get('duplicate_cases', 0),
              extra_attempts=receipt.get('extra_attempts', 0),
              wall_time_includes_receipt_repair=True)
saved.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'failures'}))
new = {k: v for k, v in failures.items() if k not in previous.get('failures', {})}
if new:
    print(json.dumps({'new_failures': new}))
