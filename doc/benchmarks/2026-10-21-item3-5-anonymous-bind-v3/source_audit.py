"""Compare every gate's actual source manifest with the frozen Git blobs."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = sys.argv[1]
assert subprocess.check_output(['git', 'rev-parse', source], cwd=ROOT, text=True).strip() == source
manifests = sorted(HERE.glob('*/source-manifest.json'))
assert manifests
records = [(path, json.loads(path.read_text())['validated_source']) for path in manifests]
helpers = sorted(path.relative_to(ROOT).as_posix() for path in HERE.glob('*.py'))
helpers.extend('doc/benchmarks/2026-10-20-item3-5-stm1/' + name for name in (
    'closing_probe.py', 'decomposition_certificate.py', 'forced_decomposition_grammar.py',
    'closing_episode_state.py'))
paths = sorted(set(helpers).union(*(set(record) for _, record in records)))
data = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
    input=''.join(f'{source}:{path}\n' for path in paths).encode())
stream, hashes = io.BytesIO(data), {}
for path in paths:
    header = stream.readline().decode().split()
    assert len(header) == 3 and header[1] == 'blob', (path, header)
    body = stream.read(int(header[2]))
    assert stream.read(1) == b'\n'
    hashes[path] = hashlib.sha256(body).hexdigest()
checks = []
for path, record in records:
    mismatches = [name for name, digest in record.items() if hashes[name] != digest]
    checks.append(dict(run=path.parent.name, files=len(record), mismatches=mismatches))
for path in helpers:
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == hashes[path], path
dirty = subprocess.check_output(['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'],
                                 cwd=ROOT, text=True).splitlines()
report = dict(source=source, checks=checks, helpers={path: hashes[path] for path in helpers},
              production_diff=dirty, matched=not dirty and all(not row['mismatches'] for row in checks))
(HERE/'source-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(source=source, runs=len(checks), files=sorted({row['files'] for row in checks}),
                      matched=report['matched'])))
assert report['matched']
