"""Exact source and assertion ledger for the scoped sweep repair."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = 'fcaff2d19624a581206fed3a17dc1b5b3d752612'
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)
changed = git('diff', BASE, '--name-only', '--', 'bin', 'test', 'data').splitlines()
prior = json.loads((HERE.parents[1]/'final/source-05/source-audit.json').read_text())
paths = set(prior['files']) | {str(p.relative_to(ROOT)) for p in HERE.glob('*.py')}
files = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)}
(HERE/'source-audit.json').write_text(json.dumps(dict(base=BASE, changed=changed, files=files), indent=2)+'\n')
(HERE/'test-ports.patch').write_text(git('diff', BASE, '--', 'test'))
(HERE/'production.patch').write_text(git('diff', BASE, '--', 'bin', 'data'))
def checks(source):
    result = {}
    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix+node.name+'.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix+node.name] = [ast.unparse(n) for n in ast.walk(node)
                    if isinstance(n, ast.Assert) or isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr.startswith('assert')]
    visit(ast.parse(source).body)
    return result
ledger = []
for name in changed:
    if not name.startswith('test/') or not name.endswith('.py'):
        continue
    before, after = checks(git('show', BASE+':'+name)), checks((ROOT/name).read_text())
    for function in sorted(before.keys() | after.keys()):
        if before.get(function) != after.get(function):
            ledger.append(dict(file=name, function=function,
                before=before.get(function), after=after.get(function)))
(HERE/'assertion-audit.json').write_text(json.dumps(dict(base=BASE, changes=ledger,
    renamed={'TestBindGrammarRules.test_bind_has_no_chart_rule_ids':
             'TestBindGrammarRules.test_bind_has_one_forward_chart_rule_and_no_inverse'},
    retired=[]), indent=2)+'\n')
print(len(changed), 'changed test files;', len(ledger), 'assertion records;', len(files), 'audited files')
