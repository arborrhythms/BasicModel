"""Preserve delivered source and complete old/new tests against the §19 hold."""
from pathlib import Path
import ast,difflib,hashlib,json,subprocess,sys,zipfile
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'test'));import bounded_tests as bounded
folder=H/(sys.argv[1] if len(sys.argv)>1 else 'review20-source');folder.mkdir(exist_ok=False)
source=bounded.source_snapshot(ROOT)
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write(folder/'source.json',source)
names=set(source)|set(subprocess.check_output(['git','ls-files','test'],cwd=ROOT).decode().splitlines())
complete={p:sha(ROOT/p) for p in sorted(names)};write(folder/'complete-source.json',complete)
with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in complete:z.write(ROOT/p,p)
ports=[];seed_changes=[];diffs=[]
with zipfile.ZipFile(H/'review20-before/source.zip') as z:
 for p in sorted(source):
  before=z.read(p).decode() if p in z.namelist() else None;after=(ROOT/p).read_text()
  if before==after:continue
  diffs.extend(difflib.unified_diff((before or '').splitlines(True),after.splitlines(True),fromfile='review19/'+p,tofile='review20/'+p))
  if not p.startswith('test/') or not p.endswith('.py'):continue
  ports.append(dict(path=p,old=before,new=after))
  seeds=lambda s:[] if s is None else sorted(ast.get_source_segment(s,x) for x in ast.walk(ast.parse(s)) if isinstance(x,ast.Call) and 'seed' in ast.unparse(x.func).lower())
  if seeds(before)!=seeds(after):seed_changes.append(p)
write(folder/'test-ports.json',ports);write(folder/'seed-port-audit.json',dict(changed_seed_calls=seed_changes));assert not seed_changes
(folder/'changes-from-review19.patch').write_text(''.join(diffs))
helpers=json.loads((H/'review19-source/measurement-helpers.json').read_text())
helpers={p.replace('review19_campaign','review20_campaign'):sha(ROOT/p.replace('review19_campaign','review20_campaign')) for p in helpers}
write(folder/'measurement-helpers.json',helpers)
print(json.dumps(dict(source_files=len(source),test_ports=len(ports),seed_changes=seed_changes)))
