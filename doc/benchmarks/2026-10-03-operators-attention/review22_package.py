"""Bind the §22 delivered source, complete ports and append-only evidence."""
import hashlib,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'test'));import bounded_tests as bounded
read=lambda p:json.loads(p.read_text())
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd)

def main():
 source=read(H/'review22-delivered-source/source.json')
 assert source==bounded.source_snapshot(ROOT)==read(H/'review22-measurements/source.json')
 assert source==read(H/'review22-final-sweep/source-manifest.json')['validated_source']
 helpers=read(H/'review22-delivered-source/measurement-helpers.json')
 assert all(sha(ROOT/p)==v for p,v in helpers.items())
 before=read(H/'review22-before/source.json');changed=sorted(p for p in source.keys()|before.keys() if source.get(p)!=before.get(p))
 allowed={'bin/Layers.py','data/XOR_grammar.xml','data/MM_grammar.xml','test/test_review13_subspaces.py','test/test_review14_contracts.py'}
 assert set(changed)==allowed,changed
 oldfiles=read(H/'review22-before/preserved-evidence.json')
 mismatches=[p for p,v in oldfiles.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=v]
 assert not mismatches,mismatches
 write(H/'review22-preservation.json',dict(files=len(oldfiles),mismatches=mismatches,manifest='review22-before/preserved-evidence.json',old_receipt='README-before-review22.md'))
 def sections(patch):return {s.splitlines()[0].decode():s for s in re.split(rb'(?=^diff --git )',patch,flags=re.M) if s}
 a=sections((H/'review22-before/prior-work.patch').read_bytes());b=sections(git('diff','--binary','HEAD'))
 protected=read(H/'review22-before/protected-docs.json')
 assert all(sha(ROOT/p)==v for p,v in protected.items()),'Protected design doc changed'
 allowed_docs={'doc/plans/2026-09-29-item-6-9-xor-grammar.md','todo.md',str((H/'README.md').relative_to(ROOT))}
 allowed_headers={'diff --git a/'+p+' b/'+p for p in allowed|allowed_docs}
 assert {k:v for k,v in a.items() if k not in allowed_headers}=={k:v for k,v in b.items() if k not in allowed_headers},'Unrelated incoming work changed'
 head=git('rev-parse','HEAD').decode().strip();assert head==git('rev-parse','6.8-s16.3-candidate^{}').decode().strip()=='eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d'
 assert git('branch','--show-current').decode().strip()=='main';assert not git('diff','--cached','--name-only')
 parent=git('ls-files','--stage','basicmodel',cwd=ROOT.parent).decode().strip();assert '802abb1acc95e1bddc8cb237b13230a336681c49' in parent
 summary=read(H/'review22-measurements/summary.json');sweep=read(H/'review22-final-sweep/summary.json')
 assert summary['complete']['completed'] and sweep['exit_code']==0 and len(summary['complete']['jobs'])==30
 assert not read(H/'review22-delivered-source/seed-port-audit.json')['changed_seed_calls']
 folder=H/'review22-delivery';folder.mkdir(exist_ok=False)
 for name in ('source.json','complete-source.json','source.zip','measurement-helpers.json','test-ports.json','seed-port-audit.json','changes-from-review20.patch'):shutil.copy2(H/'review22-delivered-source'/name,folder/name)
 files={p for p in H.rglob('*') if p.is_file() and '__pycache__' not in p.parts and any('review22' in x for x in p.relative_to(H).parts) and folder not in p.parents}
 files.add(H/'README.md');files.update(ROOT/p for p in helpers)
 docs=set(git('diff','--name-only','HEAD').decode().splitlines());files.update(ROOT/p for p in docs if p.startswith('doc/') or p=='todo.md')
 manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)};write(folder/'supplement-manifest.json',manifest)
 with zipfile.ZipFile(folder/'supplement.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(files):z.write(p,p.relative_to(ROOT))
 for name in ('source.zip','supplement.zip'):
  with zipfile.ZipFile(folder/name) as z:assert z.testzip() is None
 bridge=dict(status='§22 measured; uncommitted and unpushed; awaiting Claude review',candidate_commit=head,candidate_tag='6.8-s16.3-candidate',parent_index=parent,source_files=len(source),changed_runtime_files=changed,source_matches_sweep_and_campaign=True,whole_old_new_test_files=len(read(folder/'test-ports.json')),changed_seed_calls=0,bars_budgets_optimizers_guards_unchanged=True,declared_capacity_changes=read(H/'review22-change-verification.json')['capacity_changes'],full_sweep_green=True,sweep_counts=sweep['counts'],gate_counts=summary['counts'],gate_trainings=30,gate_retries=0,historical_files_unchanged=len(oldfiles),protected_design_docs_unchanged=protected,supplement_files=len(manifest),source_zip_sha256=sha(folder/'source.zip'),supplement_zip_sha256=sha(folder/'supplement.zip'),supplement_manifest_sha256=sha(folder/'supplement-manifest.json'))
 write(folder/'bridge.json',bridge)
 links=[]
 for link in re.findall(r'\]\(([^)]+)\)',(H/'README.md').read_text()):
  if '://' in link or link.startswith('#'):continue
  assert (H/link.split('#')[0]).exists(),link
  links.append(link)
 bridge['receipt_links_verified']=len(links);write(folder/'bridge.json',bridge)
 print(json.dumps(bridge))
if __name__=='__main__':main()
