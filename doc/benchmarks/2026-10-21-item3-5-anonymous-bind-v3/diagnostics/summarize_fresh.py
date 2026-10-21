"""Account for every declared first draw, including failed certificates."""
import hashlib
import json
from pathlib import Path
import sys

here = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(here))
from arrival_tables import tables

summary, fingerprints = [], set()
for arm in ('forward', 'reverse'):
    cases = json.loads((here/f'fresh-final-{arm}.json').read_text())
    assert len(cases) == 6
    for case in cases:
        name = case['receipt']
        state = here/(name+'-state')/f'test_forced_ordinary_pending_premise_in_both_orders[{arm == "reverse"}]'
        initial = json.loads((state/'initial-model.json').read_text())
        fingerprint = hashlib.sha256(json.dumps(initial['parameters'], sort_keys=True).encode()).hexdigest()
        fingerprints.add(fingerprint)
        outcome = json.loads((state/'outcome.json').read_text())
        for file in ('initial-rng.pt', 'after-model-rng.pt', 'generator-0.pt',
                     'initial-model.json', 'outcome.json'):
            assert (state/file).exists(), (name, file)
        trials = [json.loads(line) for line in (here/(name+'-keep.jsonl')).read_text().splitlines()]
        assert len(trials) == 2 and all(len(batch['readings']) == 2 for batch in trials)
        (here/(name+'-tables.md')).write_text(tables(trials)+'\n')
        row = dict(**case, outcome=outcome['outcome'], initial_parameters_sha256=fingerprint)
        if row['outcome'] == 'passed':
            certificate = json.loads((state/'certificate.json').read_text())
            resolved = certificate['resolved']
            departures = sorted({item['departure'] for item in certificate['rows'] if item['departure'] >= 0})
            assert len(departures) > 1
            row.update(resolved=len(resolved), correct=sum(item['bound_correct'] for item in resolved),
                       departures=departures)
        else:
            for file in ('failure-store.json', 'failure-batch.json', 'failure-expression.json'):
                assert (state/file).exists(), (name, file)
            row['failure'] = outcome['failure']
        summary.append(row)
assert len(fingerprints) == 12
(here/'fresh-final-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
text = ['# Twelve first draws on the third frozen source', '',
    f"Source `{summary[0]['source']}`. Twelve distinct initial-parameter fingerprints.",
    'Six first draws per order; every initialization, generator state, both trials,',
    'R/E/A and keep decision captured from the outset. No failed draw was replaced.',
    'Assertions, forcing scope, zero budget, objective and keep law unchanged.', '',
    '| Draw | Result | Full keep tables |', '|---|---|---|']
for row in summary:
    name = row['receipt']
    text.append(f"| {name} | {row['outcome']} | [both batches]({name}-tables.md) |")
text += ['', '**10 passed, 2 failed.** All six forward draws pass. Reverse draws 2 and 3',
    'fail unchanged D2 and exactly replay; [draw 2](reverse-2-failure.md) and',
    '[draw 3](reverse-3-failure.md) retain the diagnoses. A kept first reading',
    'has no pending x site and its order-0 three complement is absent from',
    'the next frame’s bind menu. The failure is not relabeled or waived.', '',
    'The eight-file set’s additional captured reverse failure is separate;',
    'it is not one of these twelve predeclared initializations.', '']
(here/'fresh-final.md').write_text('\n'.join(text))
print('12 distinct captured first draws: 10 passed, 2 failed; all tables retained.')
