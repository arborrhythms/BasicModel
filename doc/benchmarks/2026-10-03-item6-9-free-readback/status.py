"""Compact read-only process status; append observed MM progress to the receipt."""
import json
import statistics
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
stamp = datetime.now(timezone.utc).isoformat()
for phase in ('measurements', 'attribution'):
    path = HERE / phase / 'progress.json'
    if not path.exists():
        continue
    value = json.loads(path.read_text())
    print(phase, dict(seconds=round(value['seconds']), done=len(value['done']),
                      pending=value['pending'], active=value['active']))
samples = []
finished = {}
for path in sorted((HERE / 'measurements').glob('mm-*/measurement.json')):
    value = json.loads(path.read_text())
    compact = {k: value.get(k) for k in
               ('completed_epochs', 'ending_training_mse', 'after_900_updates_mse', 'elapsed_seconds', 'error')}
    process = path.parent / 'process.json'
    if process.exists() and json.loads(process.read_text())['exit_code'] == 0:
        finished[path.parent.name] = value.get('ending_training_mse')
    else:
        print(path.parent.name, compact)
    samples.append(dict(run=path.parent.name, observation=value))
if samples and len(finished) < 10:
    with (HERE / 'mm-progress-samples.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(observed_at=stamp, samples=samples)) + '\n')
if finished:
    print('completed MM runs', len(finished), 'median ending MSE', statistics.median(finished.values()))
for phase in ('extra-cases', 'full-sweep'):
    parts = sorted((HERE / phase).glob('part-*/result.json'))
    if not parts:
        continue
    part = json.loads(parts[-1].read_text())
    reports = [report for worker in part.get('workers', []) for report in worker.get('reports', [])]
    active = []
    active_completed = 0
    for worker in part.get('active_workers', []):
        path = Path(worker['progress_file'])
        if not path.exists():
            continue
        progress = json.loads(path.read_text())
        active_completed += len(progress.get('completed', []))
        reports.extend(progress.get('reports', []))
        active.append(progress.get('active'))
    age = datetime.now(timezone.utc).timestamp() - parts[-1].stat().st_mtime
    print(phase, dict(segment=parts[-1].parent.name,
                      last_saved_elapsed_seconds=round(part['elapsed_seconds']),
                      last_update_seconds_ago=round(age),
                      selected=len(part['selected']), completed=len(part['completed']) + active_completed,
                      counts=dict(Counter(r['outcome'] for r in reports)), active=active,
                      peak_aggregate_gib=part.get('peak_aggregate_memory_bytes', 0) / 2**30))
for name in ('extra-cases/complete.json', 'full-sweep/receipt.json', 'campaign-complete.json'):
    path = HERE / name
    if path.exists():
        value = json.loads(path.read_text())
        print(name, {k: value[k] for k in ('seconds', 'duration_seconds', 'counts', 'complete',
                     'completed', 'source_matched') if k in value})
