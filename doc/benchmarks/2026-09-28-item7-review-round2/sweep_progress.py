"""Read durable full-sweep progress, including the active batches."""
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / 'full-sweep/run'
result = json.loads((HERE / 'result.json').read_text())
cases = {}
for path in HERE.glob('worker-*.json'):
    if path.name.endswith(('.request.json', '.process.json', '.recycle.json')):
        continue
    for report in json.loads(path.read_text()).get('reports', []):
        if report['phase'] == 'call' or report['outcome'] != 'passed':
            if cases.get(report['nodeid']) != 'failed':
                cases[report['nodeid']] = report['outcome']
active = []
for worker in result.get('active_workers', []):
    request = json.loads((HERE / (worker['name'] + '.request.json')).read_text())
    selectors = request.get('selectors', request.get('selected'))
    try:
        progress = json.loads(Path(worker['progress_file']).read_text())
    except FileNotFoundError:
        progress = {}
    active.append(dict(name=worker['name'], file=selectors[0].split('::')[0],
        cases=len(selectors), active=progress.get('active')))
print(json.dumps(dict(reason=result['reason'], selected=len(result['selected']),
    completed_batches_cases=len(result['completed']), reported=len(cases),
    outcomes=dict(Counter(cases.values())), active=active,
    failed=[n for n,v in cases.items() if v == 'failed'])))
