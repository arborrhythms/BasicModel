"""Produce compact, auditable tables from the retained XOR observations."""
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
result = json.loads((root/'result.json').read_text())
measurements = [json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines()]
summary=[]
for group in result['groups']:
    directory=root/Path(group['receipt']).parent
    reports=[]
    for file in directory.glob('worker-*.json'):
        reports.extend(r for r in json.loads(file.read_text()).get('reports',[]) if r.get('phase')=='call')
    observed=[]
    for m in measurements:
        if m.get('nodeid','').split('[')[0] != group['selector']:
            continue
        value={k:v for k,v in m.items() if k not in ('stdout','stderr')}
        output=m.get('stdout','')
        if output:
            value['predictions']=re.findall(r'label=([\d.\-]+) predicted=([\d.\-]+)',output)
        observed.append(value)
    summary.append(dict(gate=group['selector'],outcome=group['reason'],measurements=observed,reports=reports))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for row in summary:
    print(row['gate'].split('::')[-1],row['outcome'],row['measurements'])
