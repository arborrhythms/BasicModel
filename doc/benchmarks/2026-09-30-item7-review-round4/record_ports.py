"""Retain original and current bodies of every round-four test port."""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parents[2]
RENAMES = {'test_native_phrase_admission_is_atomic': 'test_referenced_phrase_admission_is_atomic'}


def bodies(source):
    result = {}
    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + '.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([node.lineno, *[item.lineno for item in node.decorator_list]]) - 1
                result[prefix + node.name] = '\n'.join(source.splitlines()[start:node.end_lineno])
    visit(ast.parse(source).body)
    return result


records = []
for path in sorted((HERE / 'pre-port-source' / 'test').glob('*.py')):
    relative = Path('test') / path.name
    before = path.read_text()
    after = (ROOT / relative).read_text()
    old, new = bodies(before), bodies(after)
    for name, body in old.items():
        new_name = RENAMES.get(name, name)
        current = new.get(new_name)
        if body != current:
            assert current is not None, f'Unrecorded retirement: {relative}::{name}'
            records.append(dict(file=str(relative), old_name=name, new_name=new_name,
                                before=body, after=current))
    if before != after:
        target = HERE / 'post-port-source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(after)
(HERE / 'test-ports.json').write_text(json.dumps(records, indent=2) + '\n')
print(f'{len(records)} ports retain both bodies.')
