"""Compare the review-start archive with the actual candidate, without running it."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def assertions(source):
    return [ast.unparse(node) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assert)]


def seeds(source):
    return sorted(ast.get_source_segment(source, node) for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.Call) and 'seed' in ast.unparse(node.func).lower())


archive = zipfile.ZipFile(HERE/'review11-before/source.zip')
protected = ['test/test_explicit_dimensions.py', 'test/test_mm_xor.py',
    'test/test_reconstruction_roundtrip.py', 'test/bounded_tests.py',
    'Makefile', 'pytest.ini', 'data/eval/nanochat_grammar_gate.json']
checks = {name: (ROOT/name).read_bytes() == git('show', 'HEAD:'+name) for name in protected}
changed, ports, same_assertions, same_seeds = [], [], {}, {}
for name in archive.namelist():
    if not name.startswith(('bin/', 'test/', 'data/')):
        continue
    old = archive.read(name)
    new = (ROOT/name).read_bytes()
    if old == new:
        continue
    changed.append(name)
    if name.endswith('.py'):
        same_seeds[name] = seeds(old.decode()) == seeds(new.decode())
    if name.startswith('test/'):
        ports.append(dict(path=name, old=old.decode(), new=new.decode()))
        same_assertions[name] = assertions(old.decode()) == assertions(new.decode())
review = 'doc/plans/2026-09-27-item-6-8-one-attention.md'
result = dict(head=git('rev-parse', 'HEAD').decode().strip(),
    protected_files_unchanged_from_head=checks, review11_changed_existing_source=changed,
    existing_assertions_unchanged=same_assertions, existing_seed_calls_unchanged=same_seeds,
    production_xml_unchanged=not any(name.startswith('data/') for name in changed),
    claude_review_unchanged=(ROOT/review).read_bytes() == archive.read(review),
    worktrees=git('worktree','list','--porcelain').decode(),
    complete_changed_test_ports=ports)
target = HERE/sys.argv[1]
with target.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
assert all(checks.values()) and all(same_assertions.values()) and all(same_seeds.values())
assert result['production_xml_unchanged'] and result['claude_review_unchanged']
print(json.dumps({key:value for key,value in result.items() if key not in ('complete_changed_test_ports','worktrees')}))
