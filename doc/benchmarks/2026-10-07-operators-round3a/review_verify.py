"""Verify the completed receipt and preserve its final documents; no training or Git writes."""
import difflib,hashlib,json,subprocess,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests
def read(p):return json.loads(p.read_text())
def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def git(*args,root=ROOT):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def main():
    source=read(HERE/'delivered-source/source.json')
    assert source==bounded_tests.source_snapshot(ROOT)==read(HERE/'measurements/source.json')
    assert source==read(HERE/'full-sweep/source-manifest.json')['validated_source']
    with zipfile.ZipFile(HERE/'delivered-source/source.zip') as archive:
        assert set(archive.namelist())==set(source)
        assert all(hashlib.sha256(archive.read(n)).hexdigest()==v for n,v in source.items())
    helpers=read(HERE/'delivered-source/measurement-helpers.json')
    assert all(sha(ROOT/n)==v for n,v in helpers.items())
    preserved=read(HERE/'before/preserved-files.json')
    assert all(sha(ROOT/n)==v for n,v in preserved.items())
    ports=read(HERE/'delivered-source/test-ports.json')
    with zipfile.ZipFile(HERE/'before/source.zip') as old:
        for name,port in ports.items():
            before=(old.read(name).decode() if name in old.namelist() else '') if name.startswith('test/') else (HERE.parent/'2026-10-06-operators-round2e'/name).read_text()
            after=(ROOT/name if name.startswith('test/') else HERE/name).read_text()
            assert port==dict(before=before,after=after),name
    assert read(HERE/'delivered-source/seed-port-audit.json')['changed_seed_calls']==[]
    validation=read(HERE/'results-validation.json')
    assert validation['gate_trainings']==30 and validation['retries']==0
    assert validation['identity_audit_passed']
    files=['measurement-protocol.json','delivered-source/source.json','delivered-source/source.zip',
        'delivered-source/measurement-helpers.json','full-sweep/result.json','measurements/plan.json',
        'measurements/source.json','measurements/complete.json','measurements/sum-read-first.json',
        'measurements/summary.json','measurements/audit-summary.json','mm-first-forward/result.json',
        'paired-mm/result.json','runtime-contract-check.json','form-audit.json','identity-report.json',
        'results-validation.json','reader-diagnostics.json','aggregate-audit.json']
    write(HERE/'measurement-inputs.json',{name:sha(HERE/name) for name in files})
    documents=['doc/Architecture.md','doc/FutureWork.md','doc/GradientFlow.md',
               'doc/specs/2026-09-29-operator-catalogue.md']
    patch=[]
    with zipfile.ZipFile(HERE/'development/round2-working-baseline.zip') as old:
        for name in documents:
            before=old.read(name).decode()
            after=(ROOT/name).read_text()
            patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                fromfile='before-round3a/'+name,tofile='review-round3a/'+name))
    (HERE/'documentation-final.patch').write_text(''.join(patch))
    docs=[ROOT/name for name in documents]+sorted(HERE.glob('*.md'))+sorted(HERE.glob('reader-*.png'))
    with zipfile.ZipFile(HERE/'documentation-final.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in docs:archive.write(path,str(path.relative_to(ROOT)))
    write(HERE/'documentation-final.json',{str(p.relative_to(ROOT)):sha(p) for p in docs})
    assert git('rev-parse','HEAD')=='73cd7b71baeb64135c1e2e841b9bfc321339d8bc'
    assert git('rev-parse','HEAD',root=ROOT.parent)=='c9670b545ff88ee1b8176aa44f6c192ad84b597c'
    assert git('diff','--cached','--name-only')==git('diff','--cached','--name-only',root=ROOT.parent)==''
    state=dict(verified_at=datetime.now(timezone.utc).isoformat(),source_files=len(source),
        source_manifest_sha256=sha(HERE/'delivered-source/source.json'),
        source_archive_sha256=sha(HERE/'delivered-source/source.zip'),source_and_archive_matched=True,
        frozen_helpers=len(helpers),frozen_helpers_matched=True,preserved_files=len(preserved),
        earlier_receipts_and_construction_review_preserved=True,complete_test_ports=len(ports),
        changed_seed_calls=0,gate_trainings=30,retries=0,replacements=0,
        head=git('rev-parse','HEAD'),parent_head=git('rev-parse','HEAD',root=ROOT.parent),
        index_empty=True,parent_index_empty=True,committed=False,pushed=False,
        standing_gate_passed=validation['standing_gate_passed'],status=git('status','--short'))
    write(HERE/'review-state.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='status'}))
if __name__=='__main__':main()
