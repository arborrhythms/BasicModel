"""Freeze the production manifest and audit the authorized fixture changes."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = 'bb1678d170c8b7e98a8c6e2b1abe3250887d0087'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def functions(text):
    return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}


def assertions(node):
    return [ast.dump(item) for item in ast.walk(node) if isinstance(item, ast.Assert)]


assert not git('diff', 'HEAD', '--name-only', '--', 'bin', 'test', 'data').strip()
protected = 'test/forced_math_grammar.py'
assert (ROOT/protected).read_text() == git('show', f'{BASE}:{protected}')
ported = {
    'test/test_bind_thought.py': {
        'test_g2_generated_leaf_without_object_uses_same_resolver',
        'test_selected_open_formation_keeps_one_site_and_its_objectless_cue'},
    'test/test_reference_arrivals.py': {'test_unary_bind_keeps_the_objectless_word_site'},
    'test/test_math_chain_section14.py': set(),
    'test/test_reference_participants.py': set(),
}
changed_assertions = []
for path, allowed in ported.items():
    before, after = functions(git('show', f'{BASE}:{path}')), functions((ROOT/path).read_text())
    for name, node in before.items():
        assert name in after, (path, name, 'removed test')
        if assertions(node) != assertions(after[name]):
            assert name in allowed, (path, name, 'unauthorized assertion change')
            changed_assertions.append(dict(path=path, test=name))
    if path.endswith('test_math_chain_section14.py'):
        name = 'test_live_pending_premise_closes_in_both_orders'
        assert ast.dump(before[name]) == ast.dump(after[name])

checks = {'bin/ClauseJournal.py': 'finish_clause', 'bin/ReferenceClose.py': 'sites',
          'bin/ReferenceContext.py': 'reference_requests', 'bin/PartRead.py': 'argument_reader',
          'bin/PartClosing.py': '_definition', 'bin/Language.py': 'resolve_lexical_references'}
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

paths = git('ls-files', '--', 'bin', 'test', 'data').splitlines()
manifest = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in paths if (ROOT/path).is_file()}
result = dict(source=git('rev-parse', 'HEAD').strip(), base=BASE,
    files=manifest, changed_assertion_tests=changed_assertions,
    forced_grammar_unchanged=True, live_learning_certificate_unchanged=True,
    pending_certificate_assertions_unchanged=True, classification_audit='passed')
target = Path(__file__).with_name('frozen-source.json')
assert not target.exists()
target.write_text(json.dumps(result, indent=2)+'\n')
print(result['source'], len(manifest), 'files; source audit passed')
