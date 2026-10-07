"""Freeze the repair relative to the measured combined source, without seeds."""
import ast, difflib, hashlib, json, sys, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import source_snapshot
OUT=HERE/'measured-source'
OUT.mkdir(exist_ok=False)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
source=source_snapshot(ROOT)
save('source.json',source)
with zipfile.ZipFile(OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in source:archive.write(ROOT/name,name)
old=zipfile.ZipFile(HERE/'before/source.zip')
baseline=json.loads((HERE/'before/source.json').read_text())
patch=[]; changed=[]
for name in sorted(set(source)|set(baseline)):
    if source.get(name)==baseline.get(name):continue
    before=old.read(name).decode() if name in baseline else ''
    after=(ROOT/name).read_text() if name in source else ''
    patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='measured-combined/'+name,tofile='operators-final-b/'+name))
    changed.append(name)
(OUT/'changes.patch').write_text(''.join(patch))
helpers=[ROOT/'test/fixtures/when-readers-round4a0.json', HERE/'measurement-protocol.json', HERE.parent/'2026-10-01-item6-9-review/separator_campaign.py']+list(HERE.glob('*.py'))+list((HERE.parent/'2026-10-03-operators-attention').glob('*.py'))
save('measurement-helpers.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(helpers)})
save('delta.json',dict(changed_files=changed,baseline='measured combined source',seed=None))
print(json.dumps(dict(source_files=len(source),changed_files=changed,helpers=len(helpers))))
