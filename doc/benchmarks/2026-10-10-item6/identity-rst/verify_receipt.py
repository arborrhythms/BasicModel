"""Verify the completed receipt against its source, corpus, and raw archives."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import source_snapshot
from identity_corpus import DEST


def read(path):return json.loads(path.read_text())
def digest(handle):return hashlib.file_digest(handle,'sha256').hexdigest()
def file_digest(path):
    with path.open('rb') as handle:return digest(handle)


def main():
    source=read(OUT/'source.json')
    assert source==source_snapshot(ROOT)
    with tarfile.open(OUT/'source.tar.gz') as archive:
        members=[member for member in archive.getmembers() if member.isfile()]
        assert len(members)==len(source) and {m.name for m in members}==set(source)
        for member in members:
            with archive.extractfile(member) as handle:assert digest(handle)==source[member.name],member.name
    plan=read(OUT/'experiment-plan.json')
    assert plan['source']==source
    previous=read(OUT.parent/'identity-first/experiment/plan.json')
    assert {k:v for k,v in plan.items() if k!='source'}=={k:v for k,v in previous.items() if k!='source'}
    for name,expected in plan['corpus_sha256'].items():assert file_digest(DEST/name)==expected,name
    manifests=('full/archive-manifest.json','experiment/archive-manifest.json',
        'validation/attention-archive-manifest.json','validation/observer-archive-manifest.json',
        'validation/recovery-archive-manifest.json','validation/keep-credit-archive-manifest.json')
    verified={}
    for name in manifests:
        manifest=read(OUT/name)
        for original,meta in manifest.items():
            path=OUT/meta['archive'];base=Path(original).name
            if path.name==base+'.tar.gz':
                with tarfile.open(path) as archive:
                    with archive.extractfile(base) as handle:actual=digest(handle)
            elif path.name==base+'.gz':
                with gzip.open(path,'rb') as handle:actual=digest(handle)
            else:actual=file_digest(path)
            assert actual==meta['sha256'],(name,original)
        verified[name]=dict(files=len(manifest),uncompressed_bytes=sum(v['bytes'] for v in manifest.values()))
    summary=read(OUT/'experiment/summary.json')
    assert summary['source_check']['source_matched'] and summary['source_check']['corpus_matched']
    assert set(summary['status'])==set(plan['branches'])
    result=dict(source_files=len(source),source_archive_matched=True,current_source_matched=True,
        corpus_files=len(plan['corpus_sha256']),corpus_matched=True,non_source_plan_identical=True,
        archives=verified,branch_exit_codes=summary['status'],
        all_branches_exit_zero=all(code==0 for code in summary['status'].values()),
        verifier_sha256=file_digest(Path(__file__)))
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
