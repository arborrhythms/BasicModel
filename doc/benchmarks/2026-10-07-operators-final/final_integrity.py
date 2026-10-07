"""Validate saved evidence and the exact review tree; no model or training."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot
from verification import validate


def read(path):
    return json.loads(path.read_text())


def counts(result):
    by_node = defaultdict(list)
    for worker in result['workers']:
        for row in worker['reports']:
            by_node[row['nodeid']].append(row)
    assert set(by_node) == set(result['selected']) == set(result['completed'])
    outcome = Counter()
    for reports in by_node.values():
        # The flags test emits three additional successful subtest reports.
        # Its final regular call report decides that selected case.
        outcome[reports[-1]['outcome']] += 1
    return dict(cases=len(by_node), outcomes=outcome,
        extra_subtest_reports=sum(len(rows)-1 for rows in by_node.values()))


def main():
    measured = read(HERE / 'delivered-source/source.json')
    current = source_snapshot(ROOT)
    review = read(HERE / 'review-source/source.json')
    assert current == review
    changed = sorted(name for name in set(current) | set(measured)
                     if current.get(name) != measured.get(name))
    formatting = read(HERE / 'postmeasurement-formatting.json')
    assert changed == [formatting['file']]
    with zipfile.ZipFile(HERE / 'delivered-source/source.zip') as archive:
        before = archive.read(changed[0]).decode()
    after = (ROOT / changed[0]).read_text()
    assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))
    assert measured == read(HERE / 'measurements/source.json')
    validate(measured)
    helpers = read(HERE / 'delivered-source/measurement-helpers.json')
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
               for name, sha in helpers.items())
    summary = read(HERE / 'measurements/summary.json')
    assert summary['complete']['completed'] and len(summary['complete']['jobs']) == 30
    storage = read(HERE / 'results-validation.json')
    assert storage['storage_checks_passed']
    bisected = read(HERE / 'bisection-report.json')
    assert len(bisected['rows']) == 6 and bisected['source_matched']
    assert all(row['effective_switch']['verified'] and row['entry_state_files_identical']
               for row in bisected['rows'])
    for filename in ('cost-and-priming-audit.json', 'bisection-report.json'):
        audit = read(HERE / filename)
        assert all(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == sha
                   for name, sha in audit['inputs'].items())
    links = re.findall(r'\]\(([^)]+)\)', (HERE / 'README.md').read_text())
    missing = [name for name in links if name.split('#')[0]
               and not (HERE / name.split('#')[0]).exists()]
    assert not missing, missing
    check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head == 'cce3a4f7b6e72fab4a9a27fd70d3fa7602553814'
    result = dict(baseline=head, review_source_matched=True, measured_executable_ast_unchanged=True,
        byte_delta_after_measurement=changed, frozen_helpers_matched=len(helpers),
        full_sweep=counts(read(HERE / 'full-sweep/result.json')),
        focused_repairs=counts(read(HERE / 'focused-repairs-01/result.json')),
        counts=summary['counts'], store_checks_passed=True, verified_bisections=6,
        invalid_diagnostic_full_runs_retained=6, invalid_diagnostic_prefixes_retained=3,
        standing_trainings=30, standing_retries=0, standing_replacements=0,
        review_findings=['Reconstruction 9/10', 'Sum at quarter 9/10',
                        'Absolute R and XOR delta R are nonzero',
                        'Raw MM numerical path changes through relative when, beyond RNG',
                        'Full sweep initially failed; focused repairs precede trainings',
                        'Initial diagnostic overrides were no-ops; retained and invalidated'],
        whitespace_check='passed', receipt_links='passed', committed=False,
        additional_reporting_scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob('*.py') if str(p.relative_to(ROOT)) not in helpers},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / 'final-integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
