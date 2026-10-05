"""Freeze a review checkpoint and preserve whole old/new test files."""
import ast, hashlib, json, subprocess, sys, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded
source=bounded.source_snapshot(ROOT)
bounded.write_json(HERE/'review-source.json',source)
with zipfile.ZipFile(HERE/'review-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for name in source:archive.write(ROOT/name,name)
old=zipfile.ZipFile(HERE/'before.zip')
ports=[]
for name in sorted(set(old.namelist()) | set(source)):
 if not name.startswith('test/'):continue
 before=old.read(name).decode() if name in old.namelist() else None
 after=(ROOT/name).read_text() if name in source else None
 if before==after:continue
 ports.append({'path':name,'old':before,'new':after})
(HERE/'test-ports.json').write_text(json.dumps(ports,indent=2)+'\n')
patch=subprocess.check_output(['git','diff','--binary'],cwd=ROOT)
(HERE/'review.patch').write_bytes(patch)
# New files are complete in the source archive and ports, not lost from git diff.
seeds=[]
for port in ports:
 def calls(text):
  if not port['path'].endswith('.py') or text is None:return []
  tree=ast.parse(text)
  return sorted(ast.get_source_segment(text,n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ('seed' in ast.unparse(n.func).lower()))
 a,b=calls(port['old']),calls(port['new'])
 if a!=b:seeds.append({'path':port['path'],'old':a,'new':b})
(HERE/'seed-port-audit.json').write_text(json.dumps({'changed_seed_calls':seeds},indent=2)+'\n')
print(json.dumps({'files':len(source),'complete_test_ports':len(ports),'seed_call_differences':len(seeds)}))
