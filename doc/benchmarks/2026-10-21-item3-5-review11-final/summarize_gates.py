"""Receipt-only summary; no model execution or source changes."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
frozen = json.loads((HERE/'frozen-source.json').read_text())
source = frozen['source']
expected = [*(f'fresh-final-{arm}-{i}' for arm in ('forward', 'reverse') for i in range(1, 7)),
    'marker-replay-final', 'native-final-02', 'decomposition-zero-final',
    'decomposition-enabled-final', 'mm-saved-final', 'mm-fresh-final',
    'bootstrap-final', 'carrier-final', 'regressions-final']
summary, initializations = [], []
for name in expected:
    path = HERE/name/'result.json'
    if not path.exists():
        summary.append(dict(receipt=name, status='unrun'))
        continue
    result = json.loads(path.read_text())
    manifest = json.loads((path.parent/'source-manifest.json').read_text())
    assert manifest['validated_source'] == frozen['source_manifest'], name
    reports = [report for worker in result['workers'] for report in worker.get('reports', ())]
    counts = dict(Counter(report['outcome'] for report in reports))
    record = dict(receipt=name, source=source, exit=result['exit_code'], reason=result['reason'],
                  outcomes=counts, selected=len(result['selected']), completed=result['completed'])
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
    if name.startswith('fresh-final-'):
        folders = list((HERE/(name+'-state')).glob('*/initial-model.json'))
        assert len(folders) == 1
        p = folders[0]
        state = json.loads(p.read_text())
        parameters = hashlib.sha256(json.dumps(state['parameters'], sort_keys=True).encode()).hexdigest()
        initializations.append(dict(receipt=name, parameters_sha256=parameters,
            initial_rng_sha256=hashlib.sha256((p.parent/'initial-rng.pt').read_bytes()).hexdigest(),
            generator_sha256=hashlib.sha256((p.parent/'generator-0.pt').read_bytes()).hexdigest()))
assert len({row['parameters_sha256'] for row in initializations}) == len(initializations)
(HERE/'fresh-initializations.json').write_text(json.dumps(dict(source=source,
    draws=initializations, distinct_parameter_initializations=True), indent=2)+'\n')
complete = all(row.get('exit') == 0 for row in summary)
(HERE/'gate-summary.json').write_text(json.dumps(dict(source=source,
    source_files=len(frozen['source_manifest']), all_requested_gates_pass=complete,
    gates=summary), indent=2)+'\n')
for row in summary:
    print(row['receipt'], row.get('status', row.get('reason')), row.get('outcomes', ''),
          row.get('certificates', ''))
