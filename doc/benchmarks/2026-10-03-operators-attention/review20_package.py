"""Bind the §20 delivered source, complete ports and append-only evidence."""
import hashlib,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'test'));import bounded_tests as bounded
read=lambda p:json.loads(p.read_text())
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd)

def main():
 source=read(H/'review20-source/source.json')
 assert source==bounded.source_snapshot(ROOT)==read(H/'review20-measurements/source.json')
 assert source==read(H/'review20-sweep/source-manifest.json')['validated_source']
 helpers=read(H/'review20-source/measurement-helpers.json')
 assert all(sha(ROOT/p)==v for p,v in helpers.items())
 before=read(H/'review20-before/source.json');changed=sorted(p for p in source.keys()|before.keys() if source.get(p)!=before.get(p))
 allowed={'bin/Language.py','bin/Layers.py','bin/Models.py','test/test_grammar_word_learning.py','test/test_review14_addendum.py','test/test_review12_contracts.py','test/test_echoic_decoder.py','test/test_concept_binding_operators.py','test/test_review20_activation.py'}
 assert set(changed)==allowed,changed
 oldfiles=read(H/'review20-before/preserved-evidence.json')
 mismatches=[p for p,v in oldfiles.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=v]
 assert not mismatches,mismatches
 write(H/'review20-preservation.json',dict(files=len(oldfiles),mismatches=mismatches,manifest='review20-before/preserved-evidence.json',old_receipt='README-before-review20.md'))
 def sections(patch):return {s.splitlines()[0].decode():s for s in re.split(rb'(?=^diff --git )',patch,flags=re.M) if s}
 a=sections((H/'review20-before/prior-work.patch').read_bytes());b=sections(git('diff','--binary','HEAD'))
 external=read(H/'review20-external-docs.json')
 for p,record in external.items():
  assert sha(ROOT/p)==record['sha256'],'Concurrent doc changed again: '+p
  header='diff --git a/'+p+' b/'+p
  assert hashlib.sha256(b[header]).hexdigest()==record['diff_sha256']
  a[header]=b[header]
 allowed_docs={'doc/GradientFlow.md','doc/plans/2026-09-29-item-6-9-xor-grammar.md','todo.md',str((H/'README.md').relative_to(ROOT))}
 allowed_headers={'diff --git a/'+p+' b/'+p for p in allowed|allowed_docs}
 assert {k:v for k,v in a.items() if k not in allowed_headers}=={k:v for k,v in b.items() if k not in allowed_headers},'Unrelated incoming work changed'
 head=git('rev-parse','HEAD').decode().strip();assert head==git('rev-parse','6.8-s16.3-candidate^{}').decode().strip()=='eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d'
 assert git('branch','--show-current').decode().strip()=='main';assert not git('diff','--cached','--name-only')
 parent=git('ls-files','--stage','basicmodel',cwd=ROOT.parent).decode().strip();assert '802abb1acc95e1bddc8cb237b13230a336681c49' in parent
 summary=read(H/'review20-measurements/summary.json');sweep=read(H/'review20-sweep/summary.json')
 assert summary['complete']['completed'] and sweep['exit_code']==0 and len(summary['complete']['jobs'])==30
 assert not read(H/'review20-source/seed-port-audit.json')['changed_seed_calls']
 folder=H/'review20-delivery';folder.mkdir(exist_ok=False)
 for name in ('source.json','complete-source.json','source.zip','measurement-helpers.json','test-ports.json','seed-port-audit.json','changes-from-review19.patch'):shutil.copy2(H/'review20-source'/name,folder/name)
 files={p for p in H.rglob('*') if p.is_file() and '__pycache__' not in p.parts and any('review20' in x for x in p.relative_to(H).parts) and folder not in p.parents}
 files.add(H/'README.md');files.update(ROOT/p for p in helpers)
 docs=set(git('diff','--name-only','HEAD').decode().splitlines());files.update(ROOT/p for p in docs if p.startswith('doc/') or p=='todo.md')
 manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)};write(folder/'supplement-manifest.json',manifest)
 with zipfile.ZipFile(folder/'supplement.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(files):z.write(p,p.relative_to(ROOT))
 for name in ('source.zip','supplement.zip'):
  with zipfile.ZipFile(folder/name) as z:assert z.testzip() is None
 bridge=dict(status='§20 measured; uncommitted and unpushed; awaiting Claude review',candidate_commit=head,candidate_tag='6.8-s16.3-candidate',parent_index=parent,source_files=len(source),changed_runtime_files=changed,source_matches_sweep_and_campaign=True,whole_old_new_test_files=len(read(folder/'test-ports.json')),changed_seed_calls=0,bars_budgets_optimizers_guards_xmls_unchanged=True,full_sweep_green=True,sweep_counts=sweep['counts'],gate_counts=summary['counts'],gate_trainings=30,gate_retries=0,historical_files_unchanged=len(oldfiles),concurrent_docs_preserved=external,supplement_files=len(manifest),source_zip_sha256=sha(folder/'source.zip'),supplement_zip_sha256=sha(folder/'supplement.zip'),supplement_manifest_sha256=sha(folder/'supplement-manifest.json'))
 write(folder/'bridge.json',bridge)
 links=[]
 for link in re.findall(r'\]\(([^)]+)\)',(H/'README.md').read_text()):
  if '://' in link or link.startswith('#'):continue
  assert (H/link.split('#')[0]).exists(),link
  links.append(link)
 bridge['receipt_links_verified']=len(links);write(folder/'bridge.json',bridge)
 print(json.dumps(bridge))
if __name__=='__main__':main()
