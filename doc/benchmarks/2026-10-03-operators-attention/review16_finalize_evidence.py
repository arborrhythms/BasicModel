"""Summarize saved §16.3 evidence without running tests or model training."""
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    folder = HERE / 'review16-green-sweep'
    result = read(folder / 'result.json')
    outcomes, exceptional = {}, []
    for worker in result['workers']:
        for report in worker['reports']:
            node, outcome = report['nodeid'], report['outcome']
            if node not in outcomes or outcome == 'failed' or (
                    outcomes[node] == 'passed' and outcome != 'passed'):
                outcomes[node] = outcome
            if outcome in ('failed', 'xpassed'):
                exceptional.append(report)
    counts = dict(collections.Counter(outcomes.values()))
    assert counts == dict(passed=4900, skipped=286, xpassed=1), counts
    assert set(result['selected']) == set(result['completed']) == set(outcomes)
    assert all(worker['exit_code'] == 0 for worker in result['workers'])
    assert result['compile_cache_retries'] == []
    source = read(HERE / 'review16-source/source.json')
    assert source == read(folder / 'source-manifest.json')['validated_source']
    assert source == source_snapshot(ROOT)
    focus_files = (HERE / 'review16-focused-files.txt').read_text().splitlines()
    focused = {node: outcome for node, outcome in outcomes.items()
               if node.split('::')[0] in focus_files}
    assert set(focus_files) <= {node.split('::')[0] for node in focused}
    prior_focus = (HERE / 'review15-focused-files.txt').read_text().splitlines()
    assert set(prior_focus) <= set(focus_files)
    output_names = {'test_prepared_answer_boundary.py', 'test_trial_policy_ownership.py',
                    'test_generation_catalog.py', 'test_output_path_supervised.py',
                    'test_arithmetic_isolation.py'}
    fixed = {row['nodeid']: outcomes[row['nodeid']]
             for row in read(HERE / 'review14-sweep-summary.json')['failure_reports']
             if row['nodeid'].split('::')[0].split('/')[-1] in output_names}
    assert len(fixed) == 16 and set(fixed.values()) == {'passed'}
    summary = dict(
        counts=counts, selected=len(result['selected']), completed=len(result['completed']),
        elapsed_seconds=result['elapsed_seconds'], exit_code=result['exit_code'],
        reason=result['reason'], full_sweep_green=False, gates_started=False,
        all_pytest_worker_exit_codes_zero=True, worker_limit=result['limits']['workers'],
        limits=result['limits'], compile_cache_retries=[], warnings=result['warnings'],
        source_matched=True, exceptional_reports=exceptional,
        classification={exceptional[0]['nodeid']: {
            'kind': 'non-strict expected-failure unexpectedly passed',
            'test_and_bar_unchanged': True,
            'note': 'The .8 assertion passed. The bounded supervisor counts XPASS as failure. '
                    'No random rerun, guard edit, or test omission was used; decision pending.'}},
        original_output_regressions=fixed, focused_files=focus_files,
        focused_subset_counts=dict(collections.Counter(focused.values())),
        dropped_focused_files=[], added_since_section15=sorted(set(focus_files)-set(prior_focus)),
        focus_note='The final full sweep includes all 57 focused files; this subset is '
                   'extracted from that sweep, not another test execution.')
    write_new(folder / 'summary.json', summary)

    old_files = {}
    for name in ['review14-delivery', 'review14-addendum-delivery', 'review15-delivery']:
        for path, sha in read(HERE / name / 'supplement-files.json').items():
            if path.startswith(str(HERE.relative_to(ROOT)) + '/') and path != str(
                    (HERE / 'README.md').relative_to(ROOT)):
                if path in old_files:
                    assert old_files[path] == sha, path
                old_files[path] = sha
    changed = [path for path, sha in old_files.items()
               if not (ROOT / path).is_file() or hashlib.sha256(
                   (ROOT / path).read_bytes()).hexdigest() != sha]
    preservation = dict(checked=len(old_files), changed=changed,
                        manifests=['review14-delivery/supplement-files.json',
                                   'review14-addendum-delivery/supplement-files.json',
                                   'review15-delivery/supplement-files.json'],
                        excluded_mutable_files=['README.md', 'todo.md'], files=old_files)
    write_new(HERE / 'review16-historical-preservation.json', preservation)
    assert not changed, changed
    print(json.dumps(dict(counts=counts, focused_subset_counts=summary['focused_subset_counts'],
                          original_output_regressions=len(fixed),
                          historical_files_unchanged=len(old_files),
                          full_sweep_green=False, gates_started=False)))


if __name__ == '__main__':
    main()
