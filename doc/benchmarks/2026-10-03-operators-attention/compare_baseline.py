"""Reproduce saved failures at published HEAD in the same tree; restore in finally."""
import hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
old=zipfile.ZipFile(HERE/'before.zip')
current=zipfile.ZipFile(HERE/'operators-source.zip')
new_names=set(current.namelist())-set(old.namelist())
failures=[]
for f in sorted((HERE/'default-01').glob('worker-[0-9][0-9][0-9].json')):
 for report in json.loads(f.read_text())['reports']:
  if report['outcome']=='failed': failures.append(report['nodeid'])
(HERE/'baseline-reproductions.json').write_text(json.dumps(failures,indent=2)+'\n')
# Refuse to replace an unsaved edit. Only the snapshotted source is exchanged.
for name in current.namelist():
 if (ROOT/name).read_bytes()!=current.read(name): raise RuntimeError(f'unsaved source: {name}')
try:
 for name in new_names: (ROOT/name).unlink()
 for name in old.namelist(): (ROOT/name).write_bytes(old.read(name))
 code=subprocess.call([sys.executable,'test/test_report.py','--run-dir',str(HERE/'baseline-reproductions'),*failures],cwd=ROOT)
finally:
 for name in current.namelist():
  path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(current.read(name))
 restored=all((ROOT/name).read_bytes()==current.read(name) for name in current.namelist())
 (HERE/'baseline-restored.json').write_text(json.dumps({'restored':restored,'source':'operators-source.json','files':len(current.namelist())},indent=2)+'\n')
 if not restored: raise RuntimeError('current source restore failed')
raise SystemExit(code)
