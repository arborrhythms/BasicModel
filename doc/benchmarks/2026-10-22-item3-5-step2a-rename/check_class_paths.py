"""Inspect saved bytes for the two class paths before renaming; never load a model."""
import hashlib
import json
from pathlib import Path
import pickletools
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
registry = json.loads((ROOT/'test/fixtures/phase-artifacts.json').read_text())
paths = set(subprocess.check_output(['rg', '--files', '--hidden', '--no-ignore',
    'output', 'data', 'test', '-g', '*.ckpt', '-g', '*.pth', '-g', '*.pkl'], cwd=ROOT, text=True).splitlines())
paths.update(subprocess.check_output(['rg', '--files', 'test', 'data', '-g', '*.pt'], cwd=ROOT, text=True).splitlines())
replays = json.loads((ROOT/'doc/benchmarks/2026-10-21-item3-5-reference-reconstruction/replay-inputs.json').read_text())
for item in replays['inputs']:
    paths.add(item['path'])
    if item.get('retained_copy'):
        paths.add(item['retained_copy'])
for artifact in registry['phases'].values():
    if artifact is not None:
        paths.add(artifact['path'])
retired = [p for p in paths if '/retired-artifacts/' in p]
paths.difference_update(retired)
records = []
for name in sorted(paths):
    path = ROOT/name
    digest, found, tail = hashlib.sha256(), set(), b''
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
            block = tail+chunk
            for word in (b'TernaryTruthStore', b'PartStore'):
                if word in block:
                    found.add(word.decode())
            tail = block[-32:]
    globals_, strings = [], []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith('.pkl'):
                    continue
                for op, arg, offset in pickletools.genops(archive.read(member)):
                    if op.name == 'GLOBAL':
                        globals_.append(arg)
                    if isinstance(arg, str) and any(word in arg for word in ('TernaryTruthStore','PartStore')):
                        strings.append(dict(member=member, opcode=op.name, offset=offset, text=arg))
    records.append(dict(path=name, bytes=path.stat().st_size, sha256=digest.hexdigest(),
        old_names_anywhere=sorted(found), pickle_globals=sorted(set(globals_)), old_name_opcodes=strings))
    print(name, sorted(found), flush=True)
result = dict(source_before_rename=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    registered_phases=registry['phases'], files_checked=len(records), files=records,
    retired_artifacts_read=False, excluded_retired_paths=retired,
    method='Raw-byte scan plus pickle opcode inspection; no deserialization or rewriting.',
    old_name_hits=[r['path'] for r in records if r['old_names_anywhere']])
(HERE/'checkpoint-class-paths.json').write_text(json.dumps(result,indent=2)+'\n')
print('checked',len(records),'files; old-name hits',result['old_name_hits'])
raise SystemExit(bool(result['old_name_hits']))
