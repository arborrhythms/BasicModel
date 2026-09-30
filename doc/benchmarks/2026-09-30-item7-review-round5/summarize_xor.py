"""Every named XOR proof, every trial, and separate resource diagnostics."""
import json
from collections import Counter
from pathlib import Path
import sys

p=Path(sys.argv[1])
d=json.loads((p/'result.json').read_text())
assert d['reason']!='running'
obs=[json.loads(line) for line in (p/'measurements.jsonl').read_text().splitlines()] if (p/'measurements.jsonl').exists() else []
groups=[]
diagnostics=iter(d['diagnostic_only'])
for group in d['groups']:
    run=json.loads((p/group['receipt']).read_text())
    reports=[report for worker in run['workers'] for report in worker.get('reports',()) if report['phase']=='call' or report['outcome']!='passed']
    outcomes={}
    for report in reports:
        if report['nodeid'] not in outcomes or report['outcome']=='failed':
            outcomes[report['nodeid']]=report['outcome']
    diag=[]
    if any(worker['reason'] in ('memory','aggregate_memory') for worker in run['workers']):
        value=next(diagnostics)
        assert value['selector']==group['selector']
        diag=[value]
    groups.append(dict(gate=group['gate'],selector=group['selector'],reason=group['reason'],
        outcomes=outcomes,reports=reports,
        selected=run['selected'],completed=run['completed'],
        peak_memory_bytes=max((w.get('peak_memory_bytes',0) for w in run['workers']),default=0),
        observations=[x for x in obs if x['gate']==group['gate']],
        diagnostic_only=diag))
roundtrips=[g for g in groups if g['selector'].endswith('::test_mm20m_xor_exact_roundtrip')]
assert len(roundtrips)==15
summary=dict(root=d['root'],reason=d['reason'],groups=groups,
    named_pytest_outcomes=dict(Counter(outcome for g in groups for outcome in g['outcomes'].values())),
    guarded_group_outcomes=dict(Counter(g['reason'] for g in groups)),
    roundtrip_attempts=15,
    guarded_roundtrip_outcomes=dict(Counter(g['reason'] for g in roundtrips)),
    diagnostic_roundtrip_outcomes=dict(Counter(x['returncode'] for g in roundtrips for x in g['diagnostic_only'])),
    diagnostic_only=d['diagnostic_only'])
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| Proof | Guarded result | Observations / diagnostic |','|---|---|---|']
for g in groups:
    for nodeid in g['outcomes'] or (g['selector'],):
        measures=[]
        for o in g['observations']:
            if o.get('nodeid') not in (None, nodeid):
                continue
            if o['kind']=='reconstruction':measures.append(f"exact {o['exact_match_rate']}; where {o['where_recovery']}; output {o['output_loss']}; reconstruction {o['recon_loss']}")
            elif o['kind']=='mm':measures.append(f"ending {o.get('last')}, calls {o.get('calls')}, predictions {o.get('predictions')}")
            elif o['kind']=='cli':measures.append(f"CLI exit {o['returncode']}; MSE {o['mse']}; reconstruction {o['reconstructed']}")
            elif o['kind']=='grammar':measures.append(f"class accuracy {o['accuracy']}")
        label=nodeid.removeprefix('test/')
        if g in roundtrips:label+=f" — trial {roundtrips.index(g)+1}/15"
        diagnostic=g['diagnostic_only']
        if diagnostic:
            measures=['UNGUARDED DIAGNOSTIC: '+value for value in measures]
            measures.extend(f"diagnostic exit {x['returncode']}; peak {x['peak_memory_bytes']/2**30:.2f} GiB" for x in diagnostic)
        result=g['outcomes'].get(nodeid, g['reason'])
        lines.append('| '+label+' | '+result+f"; peak {g['peak_memory_bytes']/2**30:.2f} GiB"+' | '+'; '.join(measures)+' |')
(p/'table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({key:summary[key] for key in ('named_pytest_outcomes','guarded_group_outcomes','guarded_roundtrip_outcomes')},indent=2))
