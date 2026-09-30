"""Audit the round-5 delta, unchanged regression assertions, and commit state."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def read(path):
    return json.loads(path.read_text())


def body(source, class_name, method):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return ast.dump(function, include_attributes=False)


source = source_snapshot(ROOT)
assert source == read(HERE / 'final-source.json')
before = read(HERE / 'incoming-source.json')
changed = sorted(p for p in source.keys() | before.keys() if source.get(p) != before.get(p))
assert set(changed) == {'bin/ClauseJournal.py', 'bin/ClauseRow.py', 'bin/Queries.py',
    'test/test_grammatical_query_vps.py', 'test/test_thought_operation_catalog.py',
    'test/test_item7_predicate_identity.py'}, changed
for file in ('test/test_selected_relation_meaning.py', 'test/test_item9b_interpret.py',
             'test/test_item7_unindexed_relations.py'):
    assert (ROOT / file).read_bytes() == (HERE / 'before-source' / file).read_bytes()
assert not any(p.startswith('data/') for p in changed)
for path in (ROOT / 'test').glob('test_item7_*.py'):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else '')
            assert name not in ('manual_seed', 'seed', 'seed_everything'), path
baseline = subprocess.check_output(['git', 'show', 'HEAD:bin/Language.py'], cwd=ROOT, text=True)
current = (ROOT / 'bin/Language.py').read_text()
for cls in ('NonLayer', 'ConjunctionLayer'):
    assert body(baseline, cls, 'forward') == body(current, cls, 'forward')
git = {}
incoming_git = read(HERE / 'incoming-git-state.json')
for root in (ROOT, ROOT.parent):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    index = subprocess.check_output(['git', 'diff', '--cached', '--stat'], cwd=root, text=True)
    assert head == incoming_git[str(root)]['head']
    assert index == incoming_git[str(root)]['index'] == ''
    git[str(root)] = dict(head=head, index=index,
        status=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True))
report = dict(changed_files=changed, source_files=len(source),
    source_digest=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
    original_three_assertions_unchanged=True, configuration_changes=[],
    no_item7_seed_setting=True, nonlayer_and_conjunction_unchanged_from_head=True,
    commit_state_unchanged=True, git=git)
(HERE / 'final-source-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'git'}, indent=2))
