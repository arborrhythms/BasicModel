"""Apply the observation-only parity fixture port after the frozen runs finish.

The six reconstruction runs and numerical parity import model code and
reading_fixtures, never this test module. Record the sole source difference
explicitly; the final gates and complete sweep still match exactly.
"""
import ast
import difflib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-28-item7-review')]
from bounded_tests import source_snapshot
from review_source import supporting_inputs


def assertions(source):
    tree = ast.parse(source)
    return [ast.dump(node, include_attributes=False) for node in ast.walk(tree)
            if isinstance(node, ast.Assert) or (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'assert_close')]


def main():
    for label in ('head', 'candidate'):
        processes = json.loads((HERE / f'{label}-reconstruction/processes.json').read_text())
        assert set(processes) == {'0', '1', '2'}
        assert (HERE / f'{label}-reconstruction/driver-hashes.json').exists()
    assert (HERE / 'parity/driver-hashes.json').exists()
    assert set(json.loads((HERE / 'parity/processes.json').read_text())) == {'packed', 'single'}
    before = source_snapshot(ROOT)
    for manifest in ('candidate-reconstruction/source-manifest.json', 'parity/source-manifest.json'):
        assert json.loads((HERE / manifest).read_text()) == before
    initial_gates = json.loads((HERE / 'explicit/source-manifest.json').read_text())
    assert initial_gates['validated_source'] == before
    assert initial_gates['supporting_inputs'] == supporting_inputs(ROOT)
    name = 'test/test_packed_reconstruction_parity.py'
    path = ROOT / name
    old = path.read_text()
    work = Path(json.loads((HERE / 'working-copy.json').read_text())['root'])
    new = (work / name).read_text()
    assert old != new
    assert assertions(old) == assertions(new)
    def protected_parts(source):
        return [ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.Assign)) and
                not (isinstance(node, ast.FunctionDef) and node.name == 'measure_layout')]
    assert protected_parts(old) == protected_parts(new), 'only the capture helper may change'
    preimage = HERE / 'preimages'
    preimage.mkdir(exist_ok=True)
    (preimage / 'packed-parity.py').write_text(old)
    (HERE / 'parity-capture.patch').write_text(''.join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile='before/' + name, tofile='after/' + name)))
    path.write_text(new)
    after = source_snapshot(ROOT)
    changed = [p for p in set(before) | set(after) if before.get(p) != after.get(p)]
    assert changed == [name]
    assert supporting_inputs(ROOT) == initial_gates['supporting_inputs']
    report = dict(reason=__doc__, changed_files={name: dict(before=before[name], after=after[name])},
        runtime_and_imported_fixtures_identical=True, all_assertions_and_tolerances_identical=True,
        before_manifest='candidate-reconstruction/source-manifest.json',
        after_source=after, supporting_inputs=initial_gates['supporting_inputs'],
        failing_probe='explicit/group-06/result.json', patch='parity-capture.patch',
        preimage='preimages/packed-parity.py')
    (HERE / 'measurement-source-bridge.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['changed_files'], indent=2))


if __name__ == '__main__':
    main()
