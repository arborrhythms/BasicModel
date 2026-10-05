"""Package the exact §19 source, full failing sweep and zero-step norm probe."""
import hashlib,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'test'));import bounded_tests as bounded

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd)

def main():
    src=read(H/'review19-source/source.json')
    assert src==bounded.source_snapshot(ROOT)==read(H/'review19-sweep/source-manifest.json')['validated_source']
    before=read(H/'review19-before/source.json')
    assert sorted(k for k in src.keys()|before.keys() if src.get(k)!=before.get(k))==['bin/Layers.py']
    helpers=read(H/'review19-source/measurement-helpers.json')
    assert all(sha(ROOT/p)==v for p,v in helpers.items())
    oldfiles=read(H/'review19-before/preserved-evidence.json')
    mismatches=[p for p,v in oldfiles.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=v]
    assert not mismatches,mismatches
    write(H/'review19-preservation.json',dict(files=len(oldfiles),mismatches=mismatches,manifest='review19-before/preserved-evidence.json',old_receipt_archive='README-before-review19.md'))
    def sections(patch):return {s.splitlines()[0].decode():s for s in re.split(rb'(?=^diff --git )',patch,flags=re.M) if s}
    a=sections((H/'review19-before/prior-work.patch').read_bytes());b=sections(git('diff','--binary','HEAD'))
    allowed=(' a/bin/Layers.py b/bin/Layers.py','/README.md b/doc/benchmarks/2026-10-03-operators-attention/README.md')
    def filtered(d):return {k:v for k,v in d.items() if not any(x in k for x in allowed)}
    assert filtered(a)==filtered(b),'Preexisting work changed beyond the initializer and receipt.'
    assert git('branch','--show-current').decode().strip()=='main'
    head=git('rev-parse','HEAD').decode().strip();tag=git('rev-parse','6.8-s16.3-candidate^{}').decode().strip()
    assert head==tag=='eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d'
    assert not git('diff','--cached','--name-only')
    parent=git('ls-files','--stage','basicmodel',cwd=ROOT.parent).decode().strip()
    assert '802abb1acc95e1bddc8cb237b13230a336681c49' in parent
    sweep=read(H/'review19-sweep/summary.json');norms=read(H/'review19-start-norms.json')
    assert sweep['selected']==sweep['completed'] and sweep['exit_code']!=0
    assert not (H/'review19-measurements').exists()
    assert norms['optimizer_steps']==norms['training_runs']==norms['backward_calls']==0
    folder=H/'review19-delivery';folder.mkdir(exist_ok=False)
    for name in ('source.json','complete-source.json','source.zip','measurement-helpers.json'):shutil.copy2(H/'review19-source'/name,folder/name)
    shutil.copy2(H/'review19-only-change.patch',folder/'changes-from-review18.patch')
    write(folder/'test-port-status.json',dict(new_test_ports=0,tests_unchanged_from_review18=True,failures_retained=True))
    files={p for p in H.rglob('*') if p.is_file() and '__pycache__' not in p.parts and any('review19' in x for x in p.relative_to(H).parts) and folder not in p.parents}
    files.add(H/'README.md');files.update(ROOT/p for p in helpers)
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    write(folder/'supplement-manifest.json',manifest)
    with zipfile.ZipFile(folder/'supplement.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT))
    for name in ('source.zip','supplement.zip'):
        with zipfile.ZipFile(folder/name) as z:assert z.testzip() is None
    bridge=dict(status='§19 held at failing sweep; no campaign; uncommitted and unpushed',
        candidate_commit=head,candidate_tag='6.8-s16.3-candidate',parent_index=parent,
        production_change='RadixLayer.insert default admitted-row initialization only: max-abs normalization and [0,1] clamp',
        source_files=len(src),source_matches_full_sweep_and_norm_probe=True,new_test_ports=0,
        changed_seed_calls=0,assertions_bars_guards_xmls_unchanged=True,sweep_counts=sweep['counts'],full_sweep_green=False,
        gate_trainings=0,gate_retries=0,campaign_blocker='The campaign requires a green complete sweep of the frozen source.',
        first_forward_norm_probe=dict(optimizer_steps=0,training_runs=0,
            form_l2_range=[min(r['l2'] for r in norms['forms']),max(r['l2'] for r in norms['forms'])],
            root_l2_range=[min(r['l2'] for r in norms['roots']),max(r['l2'] for r in norms['roots'])]),
        historical_files_unchanged=len(oldfiles),supplement_files=len(manifest),
        source_zip_sha256=sha(folder/'source.zip'),supplement_zip_sha256=sha(folder/'supplement.zip'),
        supplement_manifest_sha256=sha(folder/'supplement-manifest.json'))
    write(folder/'bridge.json',bridge)
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',(H/'README.md').read_text()):
        if '://' in target or target.startswith('#'):continue
        assert (H/target.split('#')[0]).exists(),target
        links.append(target)
    bridge['receipt_links_verified']=len(links);write(folder/'bridge.json',bridge)
    print(json.dumps(bridge))

if __name__=='__main__':main()
