"""Audit the grammar contract and freeze the complete gate source manifest."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = 'bb1678d170c8b7e98a8c6e2b1abe3250887d0087'
sys.path.insert(0, str(ROOT/'test'))
from bounded_tests import source_snapshot


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def functions(text):
    return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}


def assertions(node):
    return [ast.dump(item) for item in ast.walk(node) if isinstance(item, ast.Assert)]


assert not git('diff', 'HEAD', '--name-only', '--', 'bin', 'test', 'data').strip()
assert not git('ls-files', '--others', '--exclude-standard', '--', 'bin', 'test', 'data').strip()
protected = 'test/forced_math_grammar.py'
assert (ROOT/protected).read_text() == git('show', f'{BASE}:{protected}')
allowed = {
    'test/test_bind_thought.py': {
        'test_g2_generated_leaf_without_object_uses_same_resolver',
        'test_selected_open_formation_keeps_one_site_and_its_objectless_cue'},
    'test/test_reference_arrivals.py': {'test_unary_bind_keeps_the_objectless_word_site'},
}
ports = []
for path in git('diff', BASE, '--name-only', '--', 'test').splitlines():
    if not path.endswith('.py'):
        continue
    if subprocess.run(['git', 'cat-file', '-e', f'{BASE}:{path}'], cwd=ROOT,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        continue
    before, after = functions(git('show', f'{BASE}:{path}')), functions((ROOT/path).read_text())
    for name, node in before.items():
        assert name in after, (path, name, 'removed test')
        if assertions(node) != assertions(after[name]):
            assert name in allowed.get(path, ()), (path, name, 'unauthorized assertion change')
            ports.append(dict(path=path, test=name))
    if path.endswith('test_math_chain_section14.py'):
        name = 'test_live_pending_premise_closes_in_both_orders'
        assert ast.dump(before[name]) == ast.dump(after[name])

checks = {'bin/ClauseJournal.py': 'finish_clause', 'bin/ReferenceClose.py': 'sites',
          'bin/ReferenceContext.py': 'reference_requests', 'bin/PartRead.py': 'argument_reader',
          'bin/PartClosing.py': '_definition', 'bin/Language.py': 'resolve_lexical_references',
          'bin/Models.py': '_sentence_reference_bank'}
for path, name in checks.items():
    node = next(n for n in ast.walk(ast.parse((ROOT/path).read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'no_object' for n in ast.walk(node)), path
    if name in ('sites', 'argument_reader', 'reference_requests'):
        assert not any(isinstance(n, ast.Attribute) and n.attr == 'definitions' for n in ast.walk(node)), path
for path in ('bin/ClauseJournal.py', 'bin/ReferenceContext.py', 'bin/ReferenceClose.py'):
    tree = ast.parse((ROOT/path).read_text())
    assert not any(isinstance(n, ast.keyword) and n.arg == 'mode'
                   and isinstance(n.value, ast.Constant) and n.value.value == 'pronoun'
                   for n in ast.walk(tree)), path
tree = ast.parse((ROOT/'bin/ReferenceReconstruction.py').read_text())
assert not any(isinstance(n, ast.Attribute) and n.attr in ('detach', 'store', 'mtm')
               for n in ast.walk(tree))
source = git('rev-parse', 'HEAD').strip()
manifest = source_snapshot(ROOT)
result = dict(source=source, base=BASE, source_manifest=manifest,
    changed_assertion_tests=ports, forced_grammar_unchanged=True,
    live_learning_certificate_unchanged=True, pending_certificate_assertions_unchanged=True,
    classification_audit='passed', reference_term_reads_preview_records_only=True)
target = HERE/'frozen-source.json'
assert not target.exists()
target.write_text(json.dumps(result, indent=2)+'\n')
print(source, len(manifest), 'files; source audit passed')
