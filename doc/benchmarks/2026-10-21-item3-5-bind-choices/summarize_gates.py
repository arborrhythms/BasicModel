"""Summarize retained frozen-source runs; never overwrite raw run evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
frozen = json.loads((P/'frozen-source.json').read_text())
SOURCE = frozen['source']
names = [f'fresh-final-{arm}-{n}' for arm in ('forward', 'reverse') for n in range(1, 7)]
names += ['native-final', 'native-final-02', 'regressions-final', 'decomposition-zero-final',
          'decomposition-enabled-final', 'mm-saved-final', 'mm-fresh-final', 'bootstrap-final', 'carrier-final']
runs, manifests = [], []
for name in names:
    directory = P/name
    result = json.loads((directory/'result.json').read_text())
    manifest = json.loads((directory/'source-manifest.json').read_text())
    actual = manifest['validated_source']
    changed = sorted(k for k in set(actual) | set(frozen['validated_source'])
                     if actual.get(k) != frozen['validated_source'].get(k))
    manifests.append(dict(run=name, files=len(actual), differences=changed,
        phase_registry_override=manifest.get('phase_registry_override')))
    cases = {}
    failures = []
    for worker in sorted(directory.glob('worker-*.json')):
        if '.request.' in worker.name:
            continue
        for report in json.loads(worker.read_text()).get('reports', ()):
            node = report['nodeid']
            outcome = report['outcome']
            if outcome in ('failed', 'skipped') or node not in cases:
                cases[node] = outcome
            if outcome == 'failed':
                failures.append(dict(node=node, phase=report.get('phase'),
                    worker=str(worker.relative_to(P)), message=report.get('message')))
    recovery = None
    if name == 'fresh-final-forward-6':
        recovery = json.loads((P/'fresh-final-forward-6-recovery.json').read_text())['reason']
        assert cases and set(cases.values()) == {'passed'}
    runs.append(dict(receipt=name, source=SOURCE, runner_exit=result['exit_code'],
        runner_reason=result['reason'], recovery=recovery,
        selected=len(result.get('selected', ())), completed=len(cases),
        outcomes=dict(Counter(cases.values())), cases=cases, failures=failures,
        skipped_certificates_by_phase=result.get('skipped_certificates_by_phase')))
changed_actual = sorted(name for name, digest in frozen['validated_source'].items()
    if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest)
changed_git = subprocess.check_output(['git','diff',SOURCE,'--name-only','--',
    *frozen['validated_source']], cwd=ROOT, text=True).splitlines()
helpers = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
           for f in sorted(P.glob('*.py'))}
for relative in ('doc/benchmarks/2026-10-20-item3-5-stm1/bind_gate.py',
                 'doc/benchmarks/2026-10-20-item3-5-stm1/closing_probe.py'):
    helpers[relative] = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
audit = dict(source=SOURCE, base=frozen['base'], source_files=len(frozen['validated_source']),
    runs=manifests, current_source_differences=changed_actual,
    git_differences_from_frozen=changed_git, helper_hashes=helpers,
    assertion_audit=json.loads((P/'pre-freeze-audit.json').read_text()),
    harness_recovery=json.loads((P/'harness-recovery.json').read_text()))
assert not changed_actual and not changed_git and not any(m['differences'] for m in manifests)
(P/'source-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
summary = dict(source=SOURCE, runs=runs,
    fresh=json.loads((P/'fresh-final-summary.json').read_text()),
    bind_departures=json.loads((P/'bind-departures.json').read_text())['count'],
    mm={arm:json.loads((P/f'mm-{arm}-final-state/summary.json').read_text())
        for arm in ('saved','fresh')})
(P/'gate-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('Audited',len(manifests),'run manifests against',len(frozen['validated_source']),'frozen files.')
for run in runs:
    print(run['receipt'],run['runner_reason'],run['outcomes'])
