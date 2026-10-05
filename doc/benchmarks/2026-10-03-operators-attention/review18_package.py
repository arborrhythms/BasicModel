"""Bind the §18 source and append-only evidence, preserving the earlier rounds."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

H=Path(__file__).resolve().parent; ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd)

def main():
    source=read(H/'review18-source/source.json')
    assert source==bounded.source_snapshot(ROOT)==read(H/'review18-measurements/source.json')
    assert source==read(H/'review18-sweep/source-manifest.json')['validated_source']
    before=read(H/'review18-before/source.json')
    assert sorted(k for k in source.keys()|before.keys() if source.get(k)!=before.get(k))==['bin/Layers.py']
    helpers=read(H/'review18-source/measurement-helpers.json')
    assert all(sha(ROOT/p)==s for p,s in helpers.items())
    preserved=read(H/'review18-before/preserved-evidence.json')
    mismatches=[p for p,s in preserved.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=s]
    assert not mismatches, mismatches
    write(H/'review18-preservation.json',dict(files=len(preserved),mismatches=mismatches,
        old_receipt_archive='README-before-review18.md',
        manifest='review18-before/preserved-evidence.json'))
    def sections(patch):
        return {s.splitlines()[0].decode():s for s in re.split(rb'(?=^diff --git )',patch,flags=re.M) if s}
    old=sections((H/'review18-before/prior-work.patch').read_bytes());new=sections(git('diff','--binary','HEAD'))
    allowed=(' a/bin/Layers.py b/bin/Layers.py','/README.md b/doc/benchmarks/2026-10-03-operators-attention/README.md')
    def filtered(parts):return {k:v for k,v in parts.items() if not any(p in k for p in allowed)}
    assert filtered(old)==filtered(new), 'Pre-existing work or documentation changed beyond the authorized initializer and receipt.'
    head=git('rev-parse','HEAD').decode().strip(); tag=git('rev-parse','6.8-s16.3-candidate^{}').decode().strip()
    assert head==tag=='eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d'
    assert not git('diff','--cached','--name-only')
    parent=git('ls-files','--stage','basicmodel',cwd=ROOT.parent).decode().strip()
    assert '802abb1acc95e1bddc8cb237b13230a336681c49' in parent
    summary=read(H/'review18-measurements/summary.json'); sweep=read(H/'review18-sweep/summary.json')
    assert summary['complete']['completed'] and sweep['exit_code']==0
    folder=H/'review18-delivery';folder.mkdir(exist_ok=False)
    for name in ('source.json','complete-source.json','source.zip','measurement-helpers.json'):
        shutil.copy2(H/'review18-source'/name,folder/name)
    shutil.copy2(H/'review18-only-change.patch',folder/'changes-from-review17.patch')
    write(folder/'test-port-status.json',dict(new_test_ports=0,tests_unchanged_from_review17=True,
        inherited_complete_old_new_bodies='../review17-delivery/test-ports.json'))
    # Include the saved probe failure, full collection/sweep, all thirty runs,
    # audits, postprocessing scripts, prior-observer helpers and the single receipt.
    files={p for p in H.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and any('review18' in part for part in p.relative_to(H).parts)
           and folder not in p.parents}
    files.add(H/'README.md')
    files.update(ROOT/p for p in helpers)
    files=sorted(files)
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    write(folder/'supplement-manifest.json',manifest)
    with zipfile.ZipFile(folder/'supplement.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
    for archive in ('source.zip','supplement.zip'):
        with zipfile.ZipFile(folder/archive) as z:assert z.testzip() is None
    # The forthcoming bridge link is intentional; create it before link validation.
    bridge=dict(status='§18 measured; all new work uncommitted and nothing pushed; awaiting review',
        candidate_commit=head,candidate_tag='6.8-s16.3-candidate',parent_index=parent,
        production_change='BytesFallbackEncoder.__init__ only: max-abs per-row normalization and [0,1] clamp',
        source_files=len(source),source_matches_sweep_and_measurement=True,new_test_ports=0,
        changed_seed_calls=0,assertions_bars_guards_xmls_unchanged=True,
        sweep_counts=sweep['counts'],full_sweep_green=True,gate_counts=summary['counts'],
        gate_trainings=30,gate_retries=0,historical_files_unchanged=len(preserved),supplement_files=len(manifest),
        source_zip_sha256=sha(folder/'source.zip'),supplement_zip_sha256=sha(folder/'supplement.zip'),
        supplement_manifest_sha256=sha(folder/'supplement-manifest.json'))
    write(folder/'bridge.json',bridge)
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',(H/'README.md').read_text()):
        if '://' in target or target.startswith('#'):continue
        path=target.split('#')[0]
        assert (H/path).exists(),target
        links.append(target)
    bridge['receipt_links_verified']=len(links);write(folder/'bridge.json',bridge)
    print(json.dumps(bridge))

if __name__=='__main__':main()
