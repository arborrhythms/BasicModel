"""Append-only source, patch, full test-port and seed receipts for one stage."""
import ast,json,subprocess,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded
folder=HERE/sys.argv[1];folder.mkdir(exist_ok=False)
source=bounded.source_snapshot(ROOT)
bounded.write_json(folder/'source.json',source)
with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for name in source:archive.write(ROOT/name,name)
old=zipfile.ZipFile(HERE/'before.zip'); ports=[];seeds=[]
for name in sorted(set(old.namelist())|set(source)):
 if not name.startswith('test/'):continue
 before=old.read(name).decode() if name in old.namelist() else None
 after=(ROOT/name).read_text() if name in source else None
 if before==after:continue
 ports.append(dict(path=name,old=before,new=after))
 def calls(value):
  if value is None or not name.endswith('.py'):return []
  return sorted(ast.get_source_segment(value,n) for n in ast.walk(ast.parse(value)) if isinstance(n,ast.Call) and 'seed' in ast.unparse(n.func).lower())
 if calls(before)!=calls(after):seeds.append(dict(path=name,old=calls(before),new=calls(after)))
bounded.write_json(folder/'test-ports.json',ports)
bounded.write_json(folder/'seed-port-audit.json',dict(changed_seed_calls=seeds))
(folder/'changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
print(json.dumps(dict(files=len(source),ports=len(ports),seed_differences=len(seeds))))
