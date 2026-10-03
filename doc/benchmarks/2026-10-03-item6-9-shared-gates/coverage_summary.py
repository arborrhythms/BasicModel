"""Join saved named, moved and sweep reports, without running any case."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from triage_results import all_outcomes, status

HERE = Path(__file__).resolve().parent
prior = json.loads((HERE.parent / '2026-10-03-item6-9-free-readback/coverage-summary.json').read_text())
by_node = defaultdict(list)
for row in all_outcomes():
    by_node[row['nodeid']].append(row)
groups = {}
for name, group in prior['groups'].items():
    cases = {node: status(by_node[node]) for node in group['cases']}
    groups[name] = dict(counts=dict(Counter(cases.values())), cases=cases)
extras = json.loads((HERE / 'extra-cases/complete.json').read_text())
moved = {node: status(by_node[node]) for node in extras['selected']}
result = dict(groups=groups, moved=dict(
    selected=len(moved), counts=dict(Counter(moved.values())),
    process_stops=extras['process_failures'], pending=extras['pending'],
    seconds=extras['seconds'], reused_named_cases=json.loads(
        (HERE / 'extra-cases/reused-table-cases.json').read_text())))
(HERE / 'coverage-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value['counts'] for key, value in groups.items()}))
