"""Classify a completed saved sweep; never launch or retry tests."""
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
folder = HERE / sys.argv[1]
result = json.loads((folder / 'result.json').read_text())
assert len(result['selected']) == len(result['completed'])
outcomes, failures = {}, []
for worker in result['workers']:
    for row in worker['reports']:
        node, outcome = row['nodeid'], row['outcome']
        if node not in outcomes or outcome == 'failed' or (
                outcomes[node] == 'passed' and outcome != 'passed'):
            outcomes[node] = outcome
        if outcome == 'failed':
            if any(name in node for name in ('test_echoic_decoder', 'test_readback_operand_unaries',
                                              'test_review11_contracts')) and 'decomposition_chooser' in row['message']:
                kind, reason = 'port', 'Minimal inverse fixture must supply the newly introduced independent scorer; numerical assertions unchanged.'
            elif 'test_alias_catalog_reordering' in node:
                kind, reason = 'port', 'Exact parameter/state inventory now includes the independent decomposition scorer.'
            elif 'test_review17_measurement_wiring' in node:
                kind, reason = 'regression', 'The new scorer was registered but not enlisted in SymbolSpace explicit optimizer parameters. Audit assertion retained.'
            elif 'test_free_trial_uses_no_reference_or_offsets_and_only_byte_cost' in node and 'reconstruction.decomposition' in row['message']:
                kind, reason = 'port', 'The loss registry now includes the authorized decomposition cross-entropy; witness-free numerical and ownership assertions remain unchanged.'
            else:
                kind, reason = 'unclassified', 'Inspect preserved failure before repair.'
            failures.append(dict(**row, classification=kind, reason=reason))
focus = (HERE / 'review17-focused-files.txt').read_text().splitlines()
assert set(focus) <= {node.split('::')[0] for node in outcomes}
old = json.loads((HERE / 'review14-sweep-summary.json').read_text())
names = ('test_prepared_answer_boundary.py', 'test_trial_policy_ownership.py',
         'test_generation_catalog.py', 'test_output_path_supervised.py', 'test_arithmetic_isolation.py')
original = {row['nodeid']: outcomes.get(row['nodeid']) for row in old['failure_reports']
            if row['nodeid'].split('::')[0].split('/')[-1] in names}
summary = dict(counts=dict(Counter(outcomes.values())), failures=failures,
    selected=len(result['selected']), completed=len(result['completed']),
    exit_code=result['exit_code'], reason=result['reason'], elapsed_seconds=result['elapsed_seconds'],
    original_output_regressions=original, focused_files=focus, dropped_files=[],
    focused_subset_counts=dict(Counter(outcome for node,outcome in outcomes.items()
                                      if node.split('::')[0] in focus)),
    compile_cache_retries=result['compile_cache_retries'],
    non_strict_xpass='Retained as xpassed in the receipt, accepted as pass by the supervisor; strict XPASS remains failed.')
with (folder / 'summary.json').open('x') as handle:
    json.dump(summary, handle, indent=2); handle.write('\n')
print(json.dumps({key:value for key,value in summary.items()
                  if key in ('counts', 'exit_code', 'elapsed_seconds', 'focused_subset_counts')}))
