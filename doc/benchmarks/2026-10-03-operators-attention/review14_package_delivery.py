"""Archive the saved §14 review evidence; no execution of model or tests."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'review14-delivery'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


source = read(HERE / 'review14-source/source.json')
assert source == source_snapshot(ROOT)
assert source == read(OUT / 'source.json')
assert source == read(HERE / 'review14-measurements/source.json')
complete = read(OUT / 'complete-source.json')
assert all(digest(ROOT / path) == value for path, value in complete.items())
assert not read(OUT / 'seed-port-audit.json')['changed_seed_calls']
assert len(read(OUT / 'test-ports.json')) == 169
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert head == '802abb1acc95e1bddc8cb237b13230a336681c49'
assert read(HERE / 'review14-measurements/complete.json')['source_matched']
assert read(HERE / 'review14-measurements/complete.json')['retries'] == 0
assert all(read(HERE / 'review14-contracts-final.json')['protected'].values())

links = re.findall(r'\]\(([^)]+)\)', (HERE / 'README.md').read_text())
missing = [link for link in links if not link.startswith(('https:', 'http:', '#'))
           and not (HERE / link.split('#')[0]).exists()]
assert not missing, missing

files = {ROOT / 'todo.md', HERE / 'README.md', HERE / 'README-before-review14.md'}
for path in HERE.iterdir():
    if path.name.startswith('review14') and path != OUT:
        files.update([path] if path.is_file() else (p for p in path.rglob('*') if p.is_file()))
for folder in (HERE / 'probes').glob('review14*'):
    files.update(p for p in folder.rglob('*') if p.is_file())
files = sorted(p for p in files if '__pycache__' not in p.parts and p.suffix != '.pyc')
manifest = {str(p.relative_to(ROOT)): digest(p) for p in files}
write_new(OUT / 'supplement-files.json', manifest)
with zipfile.ZipFile(OUT / 'review-supplement.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(ROOT))

bridge = dict(
    head=head, source_matched=True, whole_old_new_test_ports=169,
    changed_seed_calls=0, receipt_links_verified=len(links),
    source_files=len(source), delivery_files=len(complete), supplement_files=len(files),
    frozen_source_manifest_sha256=digest(HERE / 'review14-source/source.json'),
    measurement_source_manifest_sha256=digest(HERE / 'review14-measurements/source.json'),
    delivery_source_manifest_sha256=digest(OUT / 'source.json'),
    source_zip_sha256=digest(OUT / 'source.zip'),
    supplement_zip_sha256=digest(OUT / 'review-supplement.zip'),
    supplement_manifest_sha256=digest(OUT / 'supplement-files.json'),
    comparison=read(HERE / 'review14-measurements/summary.json')['counts'],
    single_sweep=read(HERE / 'review14-sweep-summary.json')['cases'],
    note='Saved measurements only. The supplement includes the receipt, todo, reporting helpers, all §14 probes, source snapshots, sweep and gate evidence. Nothing committed.')
write_new(OUT / 'bridge.json', bridge)
print(json.dumps(bridge))
