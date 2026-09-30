"""Read every original/continued worker receipt, including interrupted peers."""
from collections import Counter
import json
from pathlib import Path

out = Path(__file__).resolve().parent / 'full-sweep'
paths = [out / 'run/result.json'] + sorted(out.glob('continuation-*/result.json'))
cases, active = {}, []
for path in paths:
    part = json.loads(path.read_text())
    for response in sorted(path.parent.glob('worker-*.json')):
        if '.' in response.stem:
            continue
        data = json.loads(response.read_text())
        for r in data.get('reports', []):
            if r['phase'] == 'call' or r['outcome'] != 'passed':
                if r['nodeid'] not in cases or r['outcome'] == 'failed':
                    cases[r['nodeid']] = r['outcome']
    for w in part.get('active_workers', []):
        p = Path(w['progress_file'])
        d = json.loads(p.read_text()) if p.exists() else {}
        active.append(dict(part=path.parent.name, worker=w['name'], active=d.get('active')))
state = json.loads((out / 'continuation-state.json').read_text())
print(json.dumps(dict(outcomes=dict(Counter(cases.values())), reported=len(cases),
    failed=[n for n,v in cases.items() if v == 'failed'], active=active,
    resource_cases=state['resource_cases'], combined=(out / 'combined-result.json').exists())))
