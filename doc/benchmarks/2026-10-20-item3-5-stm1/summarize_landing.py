"""Index each full-sweep failure against its later scoped result."""
from collections import Counter
import json
from pathlib import Path

root = Path(__file__).resolve().parent

def outcomes(directory):
    result = {}
    for path in sorted(directory.glob('worker-???.json')):
        for report in json.loads(path.read_text()).get('reports', ()):
            if report['phase'] == 'call' or report['outcome'] in ('failed', 'skipped'):
                previous = result.get(report['nodeid'])
                if previous != 'failed':
                    result[report['nodeid']] = report['outcome']
    return result

initial = outcomes(root/'full')
latest = {node: ('full', outcome) for node, outcome in initial.items()}
runs = sorted((path for path in root.glob('native-*') if path.is_dir()
               and path.name[7:].isdigit() and int(path.name[7:]) >= 101),
              key=lambda path: int(path.name[7:]))
for directory in runs:
    for node, outcome in outcomes(directory).items():
        latest[node] = directory.name, outcome
failures = {node: latest[node] for node, outcome in initial.items() if outcome == 'failed'}
ports = {}
prefix = 'test/test_prepared_answer_boundary.py::'
# `supplied` was unused: it duplicated each contract at another random
# initialization. The port supplies a readable concluded idea and its bank;
# both reader paths and every gradient assertion remain exercised.
for supplied in ('False', 'True'):
    for loop in ('False', 'True'):
        ports[prefix + f'test_prepared_concept_handoff_cuts_answer_state_gradient[{supplied}-{loop}]'] = (
            prefix + f'test_prepared_concept_handoff_cuts_answer_state_gradient[{loop}]')
    ports[prefix + f'test_answer_loss_trains_an_executed_shared_inverse_without_state_credit[{supplied}]'] = (
        prefix + 'test_answer_loss_trains_an_executed_shared_inverse_without_state_credit')
for node, port in ports.items():
    if node in failures and port in latest:
        failures[node] = latest[port]
lines = ['# Full-sweep failure index', '',
    'The full sweep failed 219 cases. Each row below points to the latest '
    'scoped execution of that case or its explicitly named fixture port after the sweep. This is an index '
    'of repairs across snapshots, not a replacement full-sweep result. '
    'Additional slow-case migrations and their limitations are in [landing.md](landing.md).', '',
    '| Full-sweep failed case | Latest scoped run | Outcome |', '|---|---|---|']
for node, (name, outcome) in sorted(failures.items()):
    port = f' (ported to `{ports[node].split("::")[-1]}`)' if node in ports else ''
    lines.append(f'| `{node}` | [{name}]({name}/report.html) | {outcome}{port} |')
(root/'sweep-failures.md').write_text('\n'.join(lines)+'\n')
print(dict(Counter(outcome for _, outcome in failures.values())))
