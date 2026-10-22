"""Summarize retained results and the already-produced keep probes only."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
frozen = json.loads((HERE/'frozen-source.json').read_text())
source = frozen['source']
expected = ['reference-final',
    *(f'fresh-final-{arm}-{i}' for arm in ('forward', 'reverse') for i in range(1, 7)),
    'native-final-02', 'decomposition-zero-final', 'decomposition-enabled-final',
    'mm-saved-final', 'mm-fresh-final', 'bootstrap-final', 'carrier-final', 'regressions-final']
summary, initializations, stamps, departures = [], [], [], []
for name in expected:
    path = HERE/name/'result.json'
    if not path.exists():
        summary.append(dict(receipt=name, status='unrun'))
        continue
    result = json.loads(path.read_text())
    manifest = json.loads((path.parent/'source-manifest.json').read_text())
    assert manifest['validated_source'] == frozen['source_manifest'], name
    reports = [report for worker in result['workers'] for report in worker.get('reports', ())]
    record = dict(receipt=name, source=source, exit=result['exit_code'], reason=result['reason'],
        outcomes=dict(Counter(report['outcome'] for report in reports)),
        selected=len(result['selected']), completed=result['completed'])
    failures = [dict(nodeid=r['nodeid'], message=r.get('message')) for r in reports if r['outcome']=='failed']
    if failures:
        record['failures'] = failures
    certificates = []
    for p in sorted((HERE/(name+'-state')).glob('*/certificate.json')):
        value = json.loads(p.read_text())
        if 'bound' in value:
            certificates.append(dict(case=p.parent.name, bound=value['bound'], open=value['open']))
    if certificates:
        record['certificates'] = certificates
    summary.append(record)
    if not name.startswith('fresh-final-'):
        continue
    folders = list((HERE/(name+'-state')).glob('*/initial-model.json'))
    if len(folders) != 1:
        continue
    p = folders[0]
    state = json.loads(p.read_text())
    parameters = hashlib.sha256(json.dumps(state['parameters'], sort_keys=True).encode()).hexdigest()
    initializations.append(dict(receipt=name, parameters_sha256=parameters,
        initial_rng_sha256=hashlib.sha256((p.parent/'initial-rng.pt').read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256((p.parent/'generator-0.pt').read_bytes()).hexdigest()))
    probe = HERE/(name+'-keep.jsonl')
    if not probe.exists():
        continue
    for line in probe.read_text().splitlines():
        item = json.loads(line)
        assert item['source'] == source
        readings = {r['trial']: r for r in item['readings']}
        words = item['words']['x']
        credit = item['credit']
        for row, decision in enumerate(credit['keep_decision']):
            trial = 'explore' if credit['wins'][row] else 'exploit'
            terms = readings[trial]['clauses'][row]['compositions']
            leaves = [t for t in terms if t['word_id'] in words and not t['children']]
            determining = [t for t in terms if t['formal_name'] in ('lower', 'lower_unary', 'definite')]
            stamps.append(dict(receipt=name, batch=item['batch'], row=row, trial=trial,
                keep_decision=decision, x=[{k:t[k] for k in ('word_id', 'reference', 'order', 'leaves')}
                                          for t in leaves], determining=determining))
            explore = readings['explore']
            menu = explore['reference_plans'][row]['decisions']
            if not any(d['departed'] for d in menu):
                continue
            terms = [readings[which]['registry'].get('reconstruction.reference')
                     for which in ('exploit', 'explore')]
            values = [0. if t is None else t['row_values'][row]*t['weight'] for t in terms]
            departures.append(dict(receipt=name, batch=item['batch'], row=row,
                components=credit['components'][row], reference_cost=values,
                reference_delta=values[1]-values[0], keep_decision=decision,
                deciding=credit['deciding'][row], advantage=credit['advantage'][row],
                decisions=[d for d in menu if d['departed']]))
distinct = len({row['parameters_sha256'] for row in initializations}) == len(initializations)
assert distinct
(HERE/'fresh-initializations.json').write_text(json.dumps(dict(source=source,
    draws=initializations, distinct_parameter_initializations=distinct), indent=2)+'\n')
(HERE/'x-reading-stamps.json').write_text(json.dumps(dict(source=source, rows=stamps,
    all_kept_x_leaves_order_one=all(s['x'] and all(t['order']==1 for t in s['x']) for s in stamps),
    no_determining_composition=all(not s['determining'] for s in stamps)), indent=2)+'\n')
(HERE/'bind-departures.json').write_text(json.dumps(dict(source=source,
    count=len(departures), tied=sum(d['advantage']==0 for d in departures),
    reference_distinguished=sum(d['reference_delta']!=0 for d in departures),
    departures=departures), indent=2)+'\n')
complete = all(row.get('exit') == 0 for row in summary)
(HERE/'gate-summary.json').write_text(json.dumps(dict(source=source,
    source_files=len(frozen['source_manifest']), all_requested_gates_pass=complete,
    gates=summary), indent=2)+'\n')
for row in summary:
    print(row['receipt'], row.get('status', row.get('reason')), row.get('outcomes', ''),
          row.get('certificates', ''))
print('bind departures', len(departures), 'ties', sum(d['advantage']==0 for d in departures),
      'reference differences', sum(d['reference_delta']!=0 for d in departures))
