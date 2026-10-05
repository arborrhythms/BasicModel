"""Append-only complete review source, including non-Python test fixtures."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


folder = HERE / sys.argv[1]
folder.mkdir(exist_ok=False)
head = git('rev-parse', 'HEAD').decode().strip()
assert head == '802abb1acc95e1bddc8cb237b13230a336681c49'
tracked = set(git('ls-files', '-z').decode().split('\0')) - {''}
untracked = set(git('ls-files', '--others', '--exclude-standard', '-z',
                    'bin', 'test').decode().split('\0')) - {''}
changed = set(git('diff', '--name-only', '-z', 'HEAD').decode().split('\0')) - {''}
source = bounded.source_snapshot(ROOT)
names = set(source) | {p for p in tracked | untracked if p.startswith('test/')}
names |= {p for p in changed if p.startswith('doc/')}
names = sorted(p for p in names if (ROOT / p).is_file())
complete = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in names}
bounded.write_json(folder / 'source.json', source)
bounded.write_json(folder / 'complete-source.json', complete)
with zipfile.ZipFile(folder / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in names:
        archive.write(ROOT / path, path)
ports, seeds = [], []
for path in sorted((tracked | untracked) & {p for p in changed | untracked if p.startswith('test/')}):
    before = git('show', f'HEAD:{path}').decode() if path in tracked else None
    after = (ROOT / path).read_text() if (ROOT / path).exists() else None
    if before == after:
        continue
    ports.append(dict(path=path, old=before, new=after))
    def seed_calls(value):
        if value is None or not path.endswith('.py'):
            return []
        return sorted(ast.get_source_segment(value, node) for node in ast.walk(ast.parse(value))
                      if isinstance(node, ast.Call) and 'seed' in ast.unparse(node.func).lower())
    if seed_calls(before) != seed_calls(after):
        seeds.append(dict(path=path, old=seed_calls(before), new=seed_calls(after)))
bounded.write_json(folder / 'test-ports.json', ports)
bounded.write_json(folder / 'seed-port-audit.json', dict(changed_seed_calls=seeds))
(folder / 'changes.patch').write_bytes(git('diff', '--binary', 'HEAD'))
bounded.write_json(folder / 'manifest.json', dict(head=head, files=len(names),
    whole_old_new_test_ports=len(ports), changed_seed_calls=len(seeds),
    extra_source_files=sorted(set(names) - set(source)),
    note='Older snapshots omit non-Python test fixtures; this archive supplements them from the actual tree, and old test contents come from published HEAD.'))
print(json.dumps(dict(files=len(names), ports=len(ports), seed_differences=len(seeds))))
