"""Summarize the twelve retained initializations without running a model."""
import hashlib
import json
from pathlib import Path

from arrival_tables import tables

HERE = Path(__file__).resolve().parent
summary, parameters = [], set()
text = ['# Twelve fresh arrival certificates', '',
    'Source `f9faae610f35d5ca660d0ac49249637cb23e7adf`.',
    'Six predeclared fresh initializations per order; all retained, no replacement',
    'draw or seed selection. Full `arrival_capture` state and both trial readings',
    'were collected on each first run. Forward assertion, D2 scope, graph count,',
    'departure assertion, forced scope, zero attention budget and keep law unchanged.', '']
for arm in ('forward', 'reverse'):
    cases = json.loads((HERE/f'fresh-final-{arm}.json').read_text())
    assert len(cases) == 6
    for case in cases:
        name = case['receipt']
        node = f'test_forced_ordinary_pending_premise_in_both_orders[{arm == "reverse"}]'
        state = HERE/(name+'-state')/node
        initial = json.loads((state/'initial-model.json').read_text())
        digest = hashlib.sha256(json.dumps(initial['parameters'], sort_keys=True).encode()).hexdigest()
        parameters.add(digest)
        outcome = json.loads((state/'outcome.json').read_text())['outcome']
        cert = json.loads((state/'certificate.json').read_text())
        resolved = cert['resolved']
        departures = sorted({row['departure'] for row in cert['rows'] if row['departure'] >= 0})
        assert case['exit'] == 0 and outcome == 'passed'
        assert (all(row['bound_correct'] and not row['open'] for row in resolved)
                if arm == 'forward' else sum(row['bound_correct'] for row in resolved) >= 2)
        assert len(departures) > 1
        for file in ('initial-rng.pt', 'after-model-rng.pt', 'generator-0.pt',
                     'initial-model.json', 'outcome.json', 'certificate.json'):
            assert (state/file).exists(), (name, file)
        rows = [json.loads(line) for line in (HERE/(name+'-keep.jsonl')).read_text().splitlines()]
        assert len(rows) == 2
        arrival = rows[1 if arm == 'forward' else 0]
        (HERE/(name+'-tables.md')).write_text(tables([arrival])+'\n')
        report = dict(**case, outcome=outcome, initial_parameters_sha256=digest,
            resolved=len(resolved), correct=sum(row['bound_correct'] for row in resolved),
            departures=departures, arrival_batch=arrival['batch'])
        summary.append(report)
        text += [f'## {name}', '', '| Check | Result |', '|---|---|',
            f'| Native result | {outcome} |',
            f"| {'Pending x rows correctly filled' if arm == 'forward' else 'Second-premise roots carrying three'} | {report['correct']}/{report['resolved']} |",
            f'| Distinct departure positions | {departures} |',
            f'| Parameter fingerprint | `{digest}` |',
            f'| Full capture | `{name}-state/{node}` |',
            f'| Both arrival trials, every document | [{name}-tables.md]({name}-tables.md), [{name}-keep.jsonl]({name}-keep.jsonl) |', '']
assert len(parameters) == 12, 'all twelve actual initializations must be distinct'
(HERE/'fresh-final-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
text += ['**12/12 pass.** The saved D1 pair also passes separately. No forward',
         'failure occurred in this predeclared set; no failed run was replaced.',
         'This gate does not waive the fresh MM capacity or reference-close failures.', '']
(HERE/'fresh-final.md').write_text('\n'.join(text))
print('12/12 passed; twelve distinct parameter fingerprints; full captures verified.')
