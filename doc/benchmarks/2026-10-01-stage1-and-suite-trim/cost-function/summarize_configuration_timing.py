"""Keep every timing attempt, including pre-existing and introduced failures."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
rows=[]
configs=json.loads((HERE/'configuration-timing-before/complete.json').read_text())['results']
lines=['# Universal-reconstruction prototype: first-batch timings', '',
'These compare the verified bank-only candidate with the saved mandatory-serial-reconstruction prototype. The prototype is NOT the current production source: its new failures remain unresolved, and its patch has been restored out. One fresh unseeded process per configuration and phase; unchanged XML batch, CPU, no-compile backend in both phases, one first training batch including cold work, no endpoint evaluation. This is a timing probe, not a gate campaign or a steady-state epoch benchmark. No failed run is replaced.', '',
'The worker guard is 8 GiB / 30 minutes throughout. A memory stop is censored, not a timing. Configuration/data failures before training are retained. The defaults file model.xml was included conservatively in the initial static audit, but its actual model is parallel and it does not gain a serial inverse; it is explicitly excluded from the gain claim.', '',
'| Configuration | Before seconds | Prototype seconds | Delta seconds | Before outcome | Prototype outcome |',
'|---|---:|---:|---:|---|---|']
for c in configs:
    name=Path(c['config']).stem;row=dict(config=c['config'])
    for phase in ('before','after'):
        folder=HERE/('configuration-timing-'+phase)
        path=folder/(name+'.json')
        data=json.loads(path.read_text()) if path.exists() else {}
        process=json.loads((folder/(name+'-process.json')).read_text())
        row[phase]=dict(training_seconds=data.get('training_seconds'),completed=data.get('completed',False),error=data.get('error'),process=process)
    def outcome(v):
        return 'completed' if v['completed'] else v['error'] or v['process']['reason']
    a,b=row['before']['training_seconds'],row['after']['training_seconds']
    row['delta_seconds']=b-a if a is not None and b is not None else None
    fmt=lambda v:'—' if v is None else f'{v:.4f}'
    lines.append('| '+name+' | '+' | '.join([fmt(a),fmt(b),fmt(row['delta_seconds']),outcome(row['before']).replace('|','/'),outcome(row['after']).replace('|','/')])+' |')
    rows.append(row)
(HERE/'configuration-timing-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
(HERE/'configuration-timing-comparison.md').write_text('\n'.join(lines)+'\n')
