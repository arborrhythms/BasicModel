"""Read progress from the bounded runner without altering its receipt."""
import gzip
import json
from pathlib import Path

base = Path(__file__).resolve().parent
completed, reports, active, reasons = set(), [], [], {}
selected = None
for name in ('full', 'full-memory-remainder', 'full-continuation'):
    root = base / name
    path = root / 'result.json'
    if not path.exists() and not path.with_suffix('.json.gz').exists():
        continue
    result = (json.loads(path.read_text()) if path.exists() else
              json.loads(gzip.decompress(path.with_suffix('.json.gz').read_bytes())))
    if name == 'full':
        selected = len(result['selected'])
    reasons[name] = result['reason']
    completed.update(result['completed'])
    reports.extend(report for worker in result['workers'] for report in worker.get('reports', []))
    for worker in result.get('active_workers', []):
        path = root / Path(worker['progress_file']).name
        if path.exists():
            progress = json.loads(path.read_text())
            completed.update(progress.get('completed', []))
            reports.extend(progress.get('reports', []))
            if progress.get('active'):
                active.append(progress['active'])
failures = {report['nodeid']: report.get('message', '')
            for report in reports if report['outcome'] == 'failed'}
print(json.dumps(dict(reasons=reasons, completed=len(completed),
    selected=selected, failures=failures, active=active), indent=2))
