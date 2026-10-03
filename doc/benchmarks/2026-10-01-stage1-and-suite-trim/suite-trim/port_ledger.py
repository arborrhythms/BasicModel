"""Receipt helpers: retain complete bodies when moving or retiring tests."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEDGER = HERE / 'ports-and-retirements.json'


def body(source, node):
    first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return ''.join(source.splitlines(keepends=True)[first - 1:node.end_lineno])


def definitions(source):
    result = {}
    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                result[name] = body(source, node)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + '::')
            elif isinstance(node, ast.If):
                visit(node.body, prefix)
                visit(node.orelse, prefix)
    visit(ast.parse(source).body)
    return result


def record(old_file, old_name, old_body, destinations, reason, evidence):
    """destinations is [(relative filename, qualified definition name), ...]."""
    rows = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    ident = old_file + '::' + old_name
    assert not any(r['old_id'] == ident for r in rows), ident
    new = []
    for filename, name in destinations:
        contents = (ROOT / filename).read_text()
        new.append(dict(id=filename + '::' + name,
                        body=definitions(contents)[name]))
    rows.append(dict(old_id=ident, old_body=old_body, new=new,
                     reason=reason, evidence=evidence))
    LEDGER.write_text(json.dumps(rows, indent=2) + '\n')


def remove_definitions(path, names):
    """Return the old complete definitions and remove their source spans."""
    target = ROOT / path
    source = target.read_text()
    spans = []
    old = {}
    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                if name in names:
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    spans.append((start - 1, node.end_lineno))
                    old[name] = body(source, node)
                elif isinstance(node, ast.ClassDef):
                    visit(node.body, name + '::')
    visit(ast.parse(source).body)
    assert set(old) == set(names), (path, set(names) - set(old))
    lines = source.splitlines(keepends=True)
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    target.write_text(''.join(lines))
    return old
