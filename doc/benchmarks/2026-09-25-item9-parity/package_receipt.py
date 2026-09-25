"""Package complete, source-matched checks; keep diagnostics separately."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import source_snapshot


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def receipt(label,source):
    run=ROOT/'output'/f'item9-{label}'
    result=json.loads((run/'result.json').read_text())
    manifest=json.loads((run/'source-manifest.json').read_text())
    assert result['exit_code']==0,(label,result['reason'])
    assert len(result['selected'])==len(set(result['selected']))
    assert Counter(result['selected'])==Counter(result['completed'])
    assert manifest['validated_source']==source,label
    reports=[r for p in sorted(run.glob('worker-*.json')) for r in json.loads(p.read_text()).get('reports',[])]
    counts=Counter()
    for node in result['selected']:
        found=[r for r in reports if r['nodeid']==node]
        calls=[r for r in found if r['phase']=='call']
        exceptions=[r for r in found if r['outcome'] in ('failed','skipped','xfailed')]
        final=(calls or exceptions)[-1]
        assert final['outcome'] not in ('failed','xpassed'),final
        counts[final['outcome']]+=1
    summary={k:result[k] for k in ('reason','exit_code','elapsed_seconds','limits','peak_aggregate_memory_bytes','compile_cache_retries')}
    summary.update(selected=len(result['selected']),completed=len(result['completed']),outcomes=dict(counts),
                   source_files=len(source),source_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest())
    write(HERE/f'{label}-summary.json',summary)
    (HERE/f'{label}-result.json.gz').write_bytes(gzip.compress((run/'result.json').read_bytes(),mtime=0))
    shutil.copyfile(run/'source-manifest.json',HERE/f'{label}-source-manifest.json')
    logs=b''.join(p.name.encode()+b'\n'+p.read_bytes() for p in sorted(run.glob('worker-*.log')))
    (HERE/f'{label}-workers.log.gz').write_bytes(gzip.compress(logs,mtime=0))
    return summary


def slow_audit(source):
    # The broad slow audit was split after a worker hit its unchanged memory
    # cap. Report every outcome, including the unresolved compiler failure;
    # never promote that diagnostic run into a green final receipt.
    audit={}; sources={}
    for label in ('affected-complete','affected-remaining','word-trace-final'):
        run=ROOT/'output'/f'item9-{label}'
        manifest=json.loads((run/'source-manifest.json').read_text())
        measured=manifest['validated_source']
        delta={name:dict(measured=measured.get(name),review=source.get(name))
               for name in sorted(set(measured)|set(source)) if measured.get(name)!=source.get(name)}
        assert set(delta)<= {'test/test_word_store.py'},(label,delta)
        sources[label]=dict(source_delta_from_review=delta)
        for p in sorted(run.glob('worker-*.json')):
            for r in json.loads(p.read_text()).get('reports',[]):
                if r['phase']=='call':audit[r['nodeid']]={**r,'run':label}
    selected=json.loads((ROOT/'output/item9-affected-complete/result.json').read_text())['selected']
    assert set(audit)==set(selected)
    write(HERE/'slow-audit.json',dict(outcomes=dict(Counter(r['outcome'] for r in audit.values())),
                                    sources=sources,reports=audit))


def main():
    source=source_snapshot(ROOT)
    checks={label:receipt(label,source) for label in ('affected-default','word-trace-final','full','doc-links')}
    measurements=ROOT/'output/item9-parity-review'
    manifest=json.loads((measurements/'manifest.json').read_text())
    assert manifest['source']==source and manifest['source_unchanged']
    assert len(manifest['completed'])==4 and all(r['exit_code']==0 for r in manifest['completed'])
    comparison=json.loads((measurements/'comparison.json').read_text())
    assert comparison['parity_demonstrated'],comparison
    destination=HERE/'measurements';destination.mkdir(exist_ok=True)
    for p in measurements.iterdir():
        if p.is_file():
            if p.suffix=='.log': (destination/(p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
            else: shutil.copyfile(p,destination/p.name)
    diagnostics=HERE/'diagnostics';diagnostics.mkdir(exist_ok=True)
    for name in ('parity-red','parity-red-corrected','parity-first-fix','affected','affected-final',
                 'baseline-trace','trace-fixed','parity-after','parity-final','affected-complete',
                 'affected-remaining','baseline-word-trace','word-trace-fixture','parity-complete'):
        for p in (ROOT/'output'/f'item9-{name}').glob('*'):
            if p.suffix in ('.json','.log'):
                (diagnostics/(name+'-'+p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    baseline_compile=ROOT/'output/item9-baseline-source/output/item9-baseline-compile'
    for p in baseline_compile.glob('*'):
        if p.suffix in ('.json','.log'):
            (diagnostics/('baseline-compile-'+p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    slow_audit(source)
    write(HERE/'review-source.json',source)
    previous=json.loads((HERE.parent/'2026-09-25-item10-forward/review-source.json').read_text())
    write(HERE/'source-delta.json',{name:dict(before=previous.get(name),after=source.get(name))
          for name in sorted(set(source)|set(previous)) if source.get(name)!=previous.get(name)})
    with tarfile.open(HERE/'review-source.tar.gz','w:gz') as archive:
        for name in sorted(source):
            info=archive.gettarinfo(ROOT/name,arcname=name)
            info.mtime=info.uid=info.gid=0;info.uname=info.gname=''
            with (ROOT/name).open('rb') as f:archive.addfile(info,f)
    (HERE/'tracked-source.patch').write_bytes(subprocess.check_output(['git','diff','--','bin','test','data'],cwd=ROOT))
    write(HERE/'validation-summary.json',checks)
    write(HERE/'heads.json',{name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
                            for name,path in (('basicmodel',ROOT),('wikioracle',ROOT.parent))})
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
