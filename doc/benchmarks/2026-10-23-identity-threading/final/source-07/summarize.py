"""Summarize only completed results for this frozen source; never promote a prior run."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
source = json.loads((HERE/'frozen-source.json').read_text())['source']
selections = json.loads((HERE/'selections.json').read_text())
assert not subprocess.check_output(['git','diff',source,'--name-only','--','bin','test','data'],
                                   cwd=ROOT,text=True).strip()
for name, digest in json.loads((HERE/'source-audit.json').read_text())['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
groups, totals, all_pass = {}, Counter(), True
for name in selections:
    manifest, path = HERE/(name+'-run.json'), HERE/name/'result.json'
    if not manifest.exists() or not path.exists():
        groups[name] = dict(status='not completed'); all_pass = False; continue
    run, result = json.loads(manifest.read_text()), json.loads(path.read_text())
    assert run['source'] == source
    reports = {}
    for worker in sorted((HERE/name).glob('worker-*.json')):
        if '.request.' in worker.name or '.process.' in worker.name:
            continue
        for report in json.loads(worker.read_text()).get('reports', ()):
            if report['phase'] == 'call' or report['outcome'] != 'passed':
                reports[report['nodeid']] = report
    counts = Counter(report['outcome'] for report in reports.values())
    complete = set(result['selected']) == set(result['completed']) == set(reports)
    passed = run['exit'] == result['exit_code'] == 0 and complete and not counts['failed']
    all_pass = all_pass and passed
    groups[name] = dict(status='passed' if passed else 'failed', counts=dict(counts),
        selected=len(result['selected']), completed=len(result['completed']), complete=complete,
        elapsed_seconds=result['elapsed_seconds'], exit=run['exit'], reason=result['reason'],
        failures=[dict(nodeid=r['nodeid'],message=r['message']) for r in reports.values() if r['outcome']=='failed'],
        skips=[dict(nodeid=r['nodeid'],phase=r.get('certificate_phase'),message=r['message'])
               for r in reports.values() if r['outcome']=='skipped'],
        compile_cache_retries=result['compile_cache_retries'])
    totals.update(counts)
certificates = {str(p.relative_to(HERE)):json.loads(p.read_text())
                for p in HERE.glob('native-arrival-state/*/certificate.json')}
summary = dict(source=source, generated=datetime.now(timezone.utc).isoformat(),
    all_requested_gates_pass=all_pass, counts=dict(totals), groups=groups,
    pending_premise_certificates=certificates, full_sweep=False, learning_claim=False,
    source_audit='source-audit.json', selections='selections.json')
(HERE/'gate-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(source=source, all_requested_gates_pass=all_pass,counts=dict(totals),
    groups={k:dict(status=v['status'],counts=v.get('counts',{})) for k,v in groups.items()}),indent=2))
