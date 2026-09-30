"""Keep complete before/after bodies for the three predicate-ownership ports."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REASONS = {
    'test_compose_query_and_converse_faces_share_the_same_canonical_vp':
        'The shared part point is its identity code, not an inventory vector. Keep all shared-meaning checks and assert the predicate has no inventory row.',
    'test_missing_or_retired_vp_binding_fails_without_lazy_reinstallation':
        'Retain the missing-inventory-VP rejection and no-allocation assertions on equal, which still owns an inventory binding. Part has no such row to retire; the new full-inventory probe checks its availability.',
    'test_tiny_concept_inventory_keeps_thought_families_structural_not_partial':
        'Keep two missing inventory VPs as the all-or-nothing capacity probe (equal and lookup), beside row-free part. Part remains executable while neither missing inventory VP is minted.',
}


def bodies(source):
    lines = source.splitlines()
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            first = min([node.lineno, *[decorator.lineno for decorator in node.decorator_list]]) - 1
            result[node.name] = '\n'.join(lines[first:node.end_lineno])
    return result


records = []
for file in ('test/test_grammatical_query_vps.py', 'test/test_thought_operation_catalog.py'):
    before = (HERE / 'before-source' / file).read_text()
    after = (ROOT / file).read_text()
    old, new = bodies(before), bodies(after)
    for name, body in old.items():
        if body != new.get(name):
            assert name in REASONS and name in new, (file, name)
            records.append(dict(file=file, name=name, reason=REASONS[name], before=body, after=new[name]))
    target = HERE / 'after-ports' / file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(after)
assert len(records) == 3
(HERE / 'test-ports.json').write_text(json.dumps(records, indent=2) + '\n')
print('Three ports; all original and replacement bodies preserved.')
