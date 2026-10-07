"""Freeze source and measurement helpers relative to the exact round-3a Git landing."""
import ast, difflib, hashlib, json, sys, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests
OUT=HERE/'delivered-source'
OUT.mkdir(exist_ok=False)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def seeds(text):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text))
            if isinstance(n,ast.Call) and 'seed' in ast.unparse(n.func).lower()]
source=bounded_tests.source_snapshot(ROOT)
save('source.json',source)
with zipfile.ZipFile(OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in source:archive.write(ROOT/name,name)
old=zipfile.ZipFile(HERE/'before/source.zip')
baseline=json.loads((HERE/'before/source.json').read_text())
ports={};seed_changes=[];patch=[]
for name in sorted(set(source)|set(baseline)):
    if source.get(name)==baseline.get(name):continue
    before=old.read(name).decode() if name in baseline else ''
    after=(ROOT/name).read_text() if name in source else ''
    patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='round3a/'+name,tofile='round4a0/'+name))
    if name.startswith('test/') and name.endswith('.py'):
        ports[name]=dict(before=before,after=after)
        if seeds(before)!=seeds(after):seed_changes.append(name)
ports['observer_probe.py']=dict(before=(HERE.parent/'2026-10-07-operators-round3a/observer_probe.py').read_text(),
                                after=(HERE/'observer_probe.py').read_text())
save('test-ports.json',ports);save('seed-port-audit.json',dict(changed_seed_calls=seed_changes))
assert not seed_changes
(OUT/'changes.patch').write_text(''.join(patch))
helpers=[ROOT/'test/fixtures/when-readers-round4a0.json', HERE/'measurement-protocol.json',
         HERE.parent/'2026-10-01-item6-9-review/separator_campaign.py']+list(HERE.glob('*.py'))+list(HERE.glob('*.py.txt'))+list((HERE.parent/'2026-10-03-operators-attention').glob('*.py'))
save('measurement-helpers.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(helpers)})
print(json.dumps(dict(source_files=len(source),ported_tests=len(ports),changed_seed_calls=seed_changes,helpers=len(helpers))))
