"""Verify protected source, unselected item-7 initialization, and todo links."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def head(path):
    return subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT, text=True)


def named(source, name):
    return next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def dumped(node):
    return ast.dump(node, include_attributes=False)


def assertions(node):
    return [dumped(n) for n in ast.walk(node) if isinstance(n, ast.Assert) or
            (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr.startswith('assert'))]


def main():
    before = json.loads((HERE / 'starting-source.json').read_text())
    paths = list(json.loads((HERE / 'protected-preimages.json').read_text()))
    paths.remove('test/test_mm_xor.py')
    paths.remove('test/test_packed_reconstruction_parity.py')
    paths += ['data/MM_grammar.xml', 'test/test_explicit_dimensions.py',
              'test/test_thinking_kernel.py', 'test/test_stm_relative_sentence_end_state.py', 'pytest.ini']
    protected = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == before[p] for p in paths}
    for name in ('NonLayer', 'ConjunctionLayer'):
        protected[name + ' AST'] = dumped(named(head('bin/Language.py'), name)) == dumped(named((ROOT / 'bin/Language.py').read_text(), name))
    name = 'test_mm_grammar_learns_xor_signal'
    protected[name + ' AST'] = dumped(named(head('test/test_mm_xor.py'), name)) == dumped(named((ROOT / 'test/test_mm_xor.py').read_text(), name))
    name = 'test_forward_keeps_continuous_symbols'
    protected[name + ' assertions'] = assertions(named(head('test/test_mm_xor.py'), name)) == assertions(named((ROOT / 'test/test_mm_xor.py').read_text(), name))
    parity = (ROOT / 'test/test_packed_reconstruction_parity.py').read_text()
    old = (HERE / 'preimages/packed-parity.py').read_text()
    protected['parity assertions'] = assertions(ast.parse(old)) == assertions(ast.parse(parity))
    protected['parity test functions and tolerances'] = [dumped(n) for n in ast.parse(old).body if isinstance(n, (ast.Assign, ast.FunctionDef)) and not (isinstance(n, ast.FunctionDef) and n.name == 'measure_layout')] == [dumped(n) for n in ast.parse(parity).body if isinstance(n, (ast.Assign, ast.FunctionDef)) and not (isinstance(n, ast.FunctionDef) and n.name == 'measure_layout')]
    seed_calls = []
    for path in sorted((ROOT / 'test').glob('test_item7_*.py')):
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Call):
                name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ''
                if name in ('seed', 'manual_seed', 'manual_seed_all', 'set_seed'):
                    seed_calls.append(dict(path=str(path.relative_to(ROOT)), line=n.lineno, call=name))
    todo = (ROOT / 'todo.md').read_text()
    local = [unquote(v.split('#', 1)[0].strip('<>')) for v in re.findall(r'\]\(([^)]+)\)', todo)
             if not re.match(r'[a-z]+:', v) and not v.startswith('#')]
    missing = [v for v in local if not (ROOT / v).exists()]
    report = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        protected=protected, parity_observer_port='parity-capture.patch',
        item7_seed_calls=seed_calls, todo_links_checked=len(local), todo_missing=missing,
        todo_numbered_backreference_damage=bool(re.search(r'README\.md\d+', todo)))
    (HERE / 'protected-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    assert all(protected.values()) and not seed_calls and not missing and not report['todo_numbered_backreference_damage']
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
