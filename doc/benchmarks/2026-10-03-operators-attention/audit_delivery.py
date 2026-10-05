"""Verify the final package without constructing a model or rerunning a gate."""
import hashlib, json, re, subprocess, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
current=json.loads((HERE/'final-review/source.json').read_text())
bridges={}
for stage in ('decoder-xor','checkpoint-xor','operators-xor','attention-xor-02','xor-final',
              'controls-final','native-reading-review','native-global-review','native-qa-review',
              'native-grammar-reading-review','attention-profile-review','probes/nanochat-small-mechanism',
              'attention-default-final'):
 p=HERE/stage/'source.json'
 if not p.exists() and (HERE/stage/'source-manifest.json').exists():
  p=HERE/stage/'source-manifest.json'
 if not p.exists():
  options=list((HERE/stage).glob('*source*.json'))
  bridges[stage]={'source_manifest':None,'other_manifests':[str(x.relative_to(HERE)) for x in options]}
  continue
 old=json.loads(p.read_text())
 old=old.get('validated_source',old)
 changed=[path for path in sorted(set(old)|set(current)) if old.get(path)!=current.get(path)]
 bridges[stage]={'source_manifest':str(p.relative_to(HERE)), 'changed_files':changed,
  'production_changes':[p for p in changed if p.startswith('bin/')],
  'tests_changes':[p for p in changed if p.startswith('test/')],
  'data_changes':[p for p in changed if p.startswith('data/')],
  'no_training_repeated_for_bridge':True}
(HERE/'final-source-bridge.json').write_text(json.dumps(bridges,indent=2)+'\n')
ports=json.loads((HERE/'final-review/test-ports.json').read_text())
assert len(ports)==159
for port in ports:
 p=ROOT/port['path'];now=p.read_text() if p.exists() else None
 assert now==port['new'],port['path']
 if port['old'] is not None:
  before=subprocess.check_output(['git','show','HEAD:'+port['path']],cwd=ROOT,text=True)
  assert before==port['old'],port['path']
manifest=json.loads((HERE/'final-review/complete-source.json').read_text())
with zipfile.ZipFile(HERE/'final-review/source.zip') as archive:
 assert set(archive.namelist())==set(manifest)
 for p,h in manifest.items():
  assert hashlib.sha256(archive.read(p)).hexdigest()==h,p
  assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
assert 'test/fixtures/transitional_pos.grammar' in manifest
missing=[]
for path in (HERE/'README.md',ROOT/'doc/Testing.md',ROOT/'doc/Architecture.md',ROOT/'doc/Params.md',ROOT/'doc/NanoChatGrammarPilot.md'):
 for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
  if '://' in link or link.startswith(('#','mailto:')):continue
  target=link.split('#')[0].strip('<>')
  if target and not (path.parent/target).exists():missing.append({'file':str(path.relative_to(ROOT)),'target':target})
summary=dict(test_ports_verified=len(ports),archived_files_verified=len(manifest),
 non_python_fixture_included=True,local_links_missing=missing,
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
(HERE/'delivery-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
print('Final measurement changed production files:',bridges['xor-final']['production_changes'])
