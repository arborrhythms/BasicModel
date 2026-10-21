"""Audit the authorized assertion ports and every frozen-source run."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = 'f56fdf8ff337021ef744070cb3238c9c4fc7e8fa'
sys.path.insert(0, str(ROOT/'test'))
from bounded_tests import source_snapshot

def old(path):
    return subprocess.check_output(['git', 'show', BASE+':'+path], cwd=ROOT, text=True)

def assertions(text):
    return Counter(ast.dump(node, include_attributes=False)
                   for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Assert))

ports = {}
path = 'test/test_reference_arrivals.py'
before = old(path)
expected = before.replace("assert sum(item['token'] == token for item in refined['arrivals']) == 2",
                          "assert sum(item['token'] == token for item in refined['arrivals']) == 1")
assert (ROOT/path).read_text() == expected
ports[path] = 'one count: 2 to 1; all other bytes unchanged'
path = 'test/test_reference_close.py'
assert assertions(old(path)) == assertions((ROOT/path).read_text())
ports[path] = 'all assertion ASTs unchanged, including the four carried partition cases'
path = 'test/test_bind_thought.py'
retired = {'test_bind_cost_trains_existing_form_and_content_only_after_kept_commit',
           'test_paired_bind_trials_use_r_plus_a_with_no_work_reward'}
before = {node.name: node for node in ast.parse(old(path)).body if isinstance(node, ast.FunctionDef)}
after = {node.name: node for node in ast.parse((ROOT/path).read_text()).body if isinstance(node, ast.FunctionDef)}
for name, node in before.items():
    if name not in retired:
        assert ast.dump(node, include_attributes=False) == ast.dump(after[name], include_attributes=False)
ports[path] = 'only the two superseded local-cost tests ported to episode-advantage credit'
path = 'test/test_math_chain_section14.py'
assert (ROOT/path).read_text() == old(path)
ports[path] = 'both pending-premise certificates byte-identical'
assert (ROOT/'test/forced_math_grammar.py').read_text() == old('test/forced_math_grammar.py')
frozen = HERE/'frozen-source.json'
source = None
runs = []
if frozen.exists():
    data = json.loads(frozen.read_text())
    source = data['source']
    assert source_snapshot(ROOT) == data['source_manifest']
    assert not subprocess.check_output(['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'],
                                       cwd=ROOT, text=True).strip()
    for manifest in sorted(HERE.glob('*/source-manifest.json')):
        name = manifest.parent.name
        if 'final' not in name:
            continue
        recorded = json.loads(manifest.read_text())
        assert recorded['validated_source'] == data['source_manifest'], name
        result = json.loads((manifest.parent/'result.json').read_text())
        runs.append(dict(receipt=name, exit=result['exit_code'], reason=result['reason']))
audit = dict(base=BASE, source=source, ports=ports, runs=runs,
             source_files=len(source_snapshot(ROOT)))
(HERE/('source-audit.json' if source else 'pre-freeze-audit.json')).write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2))
