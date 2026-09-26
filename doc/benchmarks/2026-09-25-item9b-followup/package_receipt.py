"""Preserve final source-matched validation and the failing development probes."""
from collections import Counter
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def write(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2)+'\n')


def main():
    source = source_snapshot(ROOT)
    summaries = {}
    final = dict(affected='affected-clean', full='full-clean', slow='slow-clean', **{'doc-links':'doc-links'})
    final_runs = {ROOT/'output'/('item9b-followup-'+name) for name in final.values()}
    final_runs.update(ROOT/'output'/('item9b-followup-'+name+'-clean') for name in ('measurements','metric'))
    for label, suffix in final.items():
        run = ROOT/'output'/('item9b-followup-'+suffix)
        if label == 'doc-links' and not (run/'result.json').exists():
            continue
        result = json.loads((run/'result.json').read_text())
        manifest = json.loads((run/'source-manifest.json').read_text())
        assert manifest['validated_source'] == source, label
        assert result['exit_code'] == 0, (label, result['reason'])
        assert Counter(result['selected']) == Counter(result['completed'])
        assert len(result['selected']) == len(set(result['selected']))
        reports = [r for p in run.glob('worker-*.json')
                   for r in json.loads(p.read_text()).get('reports', [])]
        outcomes = Counter()
        for node in result['selected']:
            found = [r for r in reports if r['nodeid'] == node]
            calls = [r for r in found if r['phase']=='call']
            other = [r for r in found if r['outcome'] in ('skipped','xfailed','failed')]
            assert not any(r['outcome'] in ('failed','xpassed') for r in found), node
            outcomes[(calls or other)[-1]['outcome']] += 1
        summary = {k:result[k] for k in ('elapsed_seconds','limits','peak_aggregate_memory_bytes','exit_code')}
        summary['peak_worker_memory_bytes']=max(w['peak_memory_bytes'] for w in result['workers'])
        summary.update(selected=len(result['selected']), completed=len(result['completed']), outcomes=dict(outcomes))
        summaries[label] = summary
        write(label+'-summary.json', summary)
        for name in ('result.json','source-manifest.json'):
            (HERE/(label+'-'+name+'.gz')).write_bytes(gzip.compress((run/name).read_bytes(),mtime=0))
        logs=b''.join(p.name.encode()+b'\n'+p.read_bytes() for p in sorted(run.glob('worker-*.log')))
        (HERE/(label+'-workers.log.gz')).write_bytes(gzip.compress(logs,mtime=0))
    for label in ('measurements', 'metric'):
        run=ROOT/'output'/('item9b-followup-'+label+'-clean')
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['source'] == source and manifest['source_unchanged'], label
        completed=manifest.get('completed', [manifest.get('result')])
        assert all(c['exit_code']==0 for c in completed), label
        dest=HERE/label;dest.mkdir(exist_ok=True)
        for p in run.iterdir():
            if p.suffix=='.log':
                (dest/(p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
            elif p.is_file():
                shutil.copyfile(p,dest/p.name)
    assert json.loads((HERE/'measurements/comparison.json').read_text())['parity_demonstrated']
    prior=HERE.parent/'2026-09-25-item9b'
    before=json.loads((prior/'measurements/baseline.json').read_text())
    after=json.loads((HERE/'measurements/baseline.json').read_text())
    old={p['name']:p['reconstruction_mean'] for p in before['phases']}
    new={p['name']:p['reconstruction_mean'] for p in after['phases']}
    write('baseline.json',dict(source_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        reviewed_9b=old, followup=new, delta={k:new[k]-old[k] for k in old},
        cause='9b fixed incidence credit on written fractional feature edges; this changes optimizer dynamics.',
        scope='New comparison baseline; no numerical tolerance or improvement claim.'))
    write('review-source.json', source)
    previous=json.loads((prior/'review-source.json').read_text())
    delta={p:dict(before=previous.get(p),after=source.get(p))
           for p in sorted(set(previous)|set(source)) if previous.get(p)!=source.get(p)}
    write('source-delta.json',delta)
    patch=[]
    with tarfile.open(prior/'review-source.tar.gz') as archive:
        for name in delta:
            old=archive.extractfile(name).read() if name in previous else b''
            new=(ROOT/name).read_bytes() if name in source else b''
            if name in previous: assert hashlib.sha256(old).hexdigest()==previous[name]
            if name in source: assert hashlib.sha256(new).hexdigest()==source[name]
            patch.extend(difflib.unified_diff(old.decode().splitlines(keepends=True),
                new.decode().splitlines(keepends=True),
                fromfile='a/'+name if name in previous else '/dev/null',
                tofile='b/'+name if name in source else '/dev/null'))
    (HERE/'followup-source.patch').write_text(''.join(patch))
    with tarfile.open(HERE/'review-source.tar.gz','w:gz') as archive:
        for name in sorted(source):
            info=archive.gettarinfo(ROOT/name,arcname=name)
            info.mtime=info.uid=info.gid=0; info.uname=info.gname=''
            with (ROOT/name).open('rb') as f: archive.addfile(info,f)
    diagnostics=HERE/'diagnostics';diagnostics.mkdir(exist_ok=True)
    for run in (ROOT/'output').glob('item9b-followup-*'):
        if run in final_runs:
            continue
        for name in ('result.json','source-manifest.json','manifest.json','baseline.json','comparison.json','metric.json'):
            p=run/name
            if p.exists():
                (diagnostics/(run.name+'-'+name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
        result_path=run/'result.json'
        if result_path.exists():
            for worker in json.loads(result_path.read_text()).get('workers', []):
                if worker['exit_code'] == 0:
                    continue
                log=run/Path(worker['log']).name
                for p in (log, log.with_suffix('.json')):
                    if p.exists():
                        (diagnostics/(run.name+'-'+p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    write('validation-summary.json', summaries)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
