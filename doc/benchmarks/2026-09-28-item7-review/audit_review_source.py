"""Recheck protected source and connect measurements to the final fixture ports."""
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
from review_source import supporting_inputs


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    source = source_snapshot(ROOT)
    directory = sys.argv[1] if len(sys.argv) > 1 else 'closing-measurements'
    measured = json.loads((HERE / directory / 'source-manifest.json').read_text())
    changed = {name: dict(measured=measured.get(name), current=source.get(name))
               for name in sorted(source.keys() | measured.keys())
               if source.get(name) != measured.get(name)}
    assert not changed, 'source changed since the reissued measurements'
    write(HERE / directory / 'review-source-equivalence.json', dict(
        exact_source_match=True, differences=changed, runtime_executable_unchanged=True,
        runtime_evidence='Every validated source file is byte-identical to the reissued measurement source, including the unindexed-operand closing repair.',
        data_configuration_and_build_files_exact=True,
        fixture_disposition='../sweep-fixtures/integration.json',
        retired_cases='../sweep-fixtures/retired-cases.json',
        closing_repair='../mm-admission/repair.json'))

    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    protected = json.loads((HERE / 'protected-contracts.json').read_text())
    for name, record in protected.items():
        assert source[name] == record['sha256'], f'protected source changed: {name}'
        if record.get('exact_bytes'):
            original = subprocess.check_output(['git', 'show', f'{head}:{name}'], cwd=ROOT)
            assert (ROOT / name).read_bytes() == original
    original = ast.parse(subprocess.check_output(['git', 'show', f'{head}:bin/Language.py'], cwd=ROOT, text=True))
    current = ast.parse((ROOT / 'bin/Language.py').read_text())
    classes = {}
    for name in ('NonLayer', 'ConjunctionLayer'):
        before = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name == name)
        after = next(node for node in current.body if isinstance(node, ast.ClassDef) and node.name == name)
        classes[name] = ast.dump(before) == ast.dump(after)
    assert all(classes.values())
    write(HERE / 'review-source-audit.json', dict(
        head=head, source_files=len(source),
        source_digest=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        protected_source_matches_prior_audit=True,
        protected_contracts='protected-contracts.json',
        deferred_operator_classes_AST_identical_to_HEAD=classes,
        supporting_inputs=supporting_inputs(ROOT),
        measurement_comparison=directory + '/review-source-equivalence.json'))
    print(json.dumps(dict(source_files=len(source), changed_from_measurement=len(changed),
                         deferred_operators=classes, protected_contracts_unchanged=True)))


if __name__ == '__main__':
    main()
