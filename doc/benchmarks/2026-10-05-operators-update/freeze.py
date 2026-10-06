"""Freeze the review candidate and audit unchanged seeds before the full sweep."""
import ast, hashlib, json, subprocess, sys, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests
OUT=HERE/'delivered-source'
OUT.mkdir(exist_ok=False)
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def seeds(source):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source))
            if isinstance(n,ast.Call) and 'seed' in ast.unparse(n.func).lower()]
source=bounded_tests.source_snapshot(ROOT)
save('source.json',source)
with zipfile.ZipFile(OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in source: archive.write(ROOT/name,name)
changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
ports={}
seed_changes=[]
for name in changed:
    if name.startswith('test/') and name.endswith('.py'):
        before=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT,text=True)
        after=(ROOT/name).read_text()
        saved=HERE/'before'/name
        if not saved.exists():
            saved.parent.mkdir(parents=True,exist_ok=True);saved.write_text(before)
        assert saved.read_text()==before
        ports[name]=dict(before=before,after=after)
        if seeds(before)!=seeds(after): seed_changes.append(name)
save('test-ports.json',ports)
save('seed-port-audit.json',dict(changed_seed_calls=seed_changes))
assert not seed_changes, 'existing seeds must be unchanged'
helpers=list(HERE.glob('*.py'))+list((HERE.parent/'2026-10-03-operators-attention').glob('*.py'))
helpers += [HERE/'review22-miss.json',HERE/'pole-bodies.json']
save('measurement-helpers.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(helpers)})
(OUT/'changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
print(json.dumps(dict(source_files=len(source),ported_tests=len(ports),changed_seed_calls=seed_changes,helpers=len(helpers))))
