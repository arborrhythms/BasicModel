"""Save source-bound failures and complete old/new bodies before review."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot, write_json


def freeze(label, files):
    output = HERE / label
    output.mkdir(exist_ok=False)
    for name in files:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(output / 'source.json', source_snapshot(ROOT))


def bodies(path):
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    result = {}
    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                result[name] = ''.join(lines[first - 1:node.end_lineno])
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + '.')
    visit(ast.parse(source).body)
    return result


def changes(label, files):
    output = HERE / label
    ledger, patch = [], []
    for name in files:
        oldpath, newpath = output / name, ROOT / name
        old, new = bodies(oldpath), bodies(newpath)
        for symbol in sorted(old.keys() | new.keys()):
            if old.get(symbol) != new.get(symbol):
                ledger.append(dict(file=name, symbol=symbol,
                    old_body=old.get(symbol), new_body=new.get(symbol)))
        patch.extend(difflib.unified_diff(oldpath.read_text().splitlines(True),
            newpath.read_text().splitlines(True), fromfile='a/' + name, tofile='b/' + name))
    write_json(output / 'bodies.json', ledger)
    (output / 'repair.patch').write_text(''.join(patch))
    write_json(output / 'after-source.json', source_snapshot(ROOT))


if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2], sys.argv[3:])
