"""Summarize completed frozen-source draws, including failures and unrun slots.

Postprocessing only: no model construction, random draw or test execution.
The frozen fresh_tables.py is a pass-only renderer; this report retains the
stopped gate without claiming that its unrun slots passed.
"""
import hashlib
import json
from pathlib import Path

from arrival_tables import tables

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / 'frozen-source.json').read_text())['source']


def load(path):
    return json.loads(path.read_text())


def main():
    summary, fingerprints, manifests = [], set(), []
    output = ['# Frozen-source fresh arrival gate: stopped', '',
        f'Source `{SOURCE}`. Six fresh draws were declared per order.',
        'Every completed draw retains its full first-run initialization and',
        'both trial captures from the outset. No draw is replaced.', '',
        '| Order / draw | Result | Correct rows / roots | Full paired readings |',
        '|---|---|---|---|']
    for arm in ('forward', 'reverse'):
        cases = load(HERE / f'fresh-final-{arm}.json')
        assert [case['number'] for case in cases] == list(range(1, len(cases) + 1))
        assert len(cases) <= 6
        for case in cases:
            assert case['source'] == SOURCE
            name = case['receipt']
            node = f'test_forced_ordinary_pending_premise_in_both_orders[{arm == "reverse"}]'
            state = HERE / (name + '-state') / node
            initial = load(state / 'initial-model.json')
            digest = hashlib.sha256(json.dumps(initial['parameters'], sort_keys=True).encode()).hexdigest()
            fingerprints.add(digest)
            outcome = load(state / 'outcome.json')['outcome']
            result = load(HERE / name / 'result.json')
            assert case['exit'] == result['exit_code']
            assert (case['exit'] == 0) == (outcome == 'passed')
            for file in ('initial-rng.pt', 'after-model-rng.pt', 'generator-0.pt',
                         'initial-model.json', 'outcome.json', 'reads.jsonl'):
                assert (state / file).is_file(), (name, file)
            arrivals = [json.loads(line) for line in (HERE / (name + '-keep.jsonl')).read_text().splitlines()]
            assert len(arrivals) == 2
            assert all(row['source'] == SOURCE and row['replay'] is None and
                       len(row['readings']) == 2 for row in arrivals)
            table = tables(arrivals).replace('The forward\nassertion', f'The {arm}\nassertion')
            (HERE / (name + '-tables.md')).write_text(table.rstrip() + '\n')
            record = dict(**case, outcome=outcome, initial_parameters_sha256=digest,
                          full_first_draw_capture=True, captured_batches=2)
            correct = 'failure; no success certificate'
            if outcome == 'passed':
                cert = load(state / 'certificate.json')
                resolved = cert['resolved']
                departures = sorted({row['departure'] for row in cert['rows'] if row['departure'] >= 0})
                assert len(departures) > 1
                assert (all(row['bound_correct'] and not row['open'] for row in resolved)
                        if arm == 'forward' else sum(row['bound_correct'] for row in resolved) >= 2)
                record.update(resolved=len(resolved), correct=sum(row['bound_correct'] for row in resolved),
                              departures=departures)
                correct = f"{record['correct']}/{record['resolved']}"
            else:
                for file in ('failure-store.json', 'failure-batch.json', 'failure-expression.json'):
                    assert (state / file).is_file(), (name, file)
                record['failure'] = load(state / 'outcome.json')['failure']
            summary.append(record)
            manifests.append(load(HERE / name / 'source-manifest.json')['validated_source'])
            output.append(f"| {arm} / {case['number']} | {outcome} | {correct} | [{name}]({name}-tables.md) |")
        for number in range(len(cases) + 1, 7):
            summary.append(dict(number=number, source=SOURCE, arm=arm, outcome='not_run'))
            output.append(f'| {arm} / {number} | not run: stopped | — | — |')
    completed = [row for row in summary if row['outcome'] != 'not_run']
    assert len(fingerprints) == len(completed)
    assert all(manifest == manifests[0] for manifest in manifests)
    audit = dict(source=SOURCE, declared=12, completed=len(completed),
        passed=sum(row['outcome'] == 'passed' for row in summary),
        failed=sum(row['outcome'] == 'failed' for row in summary),
        not_run=sum(row['outcome'] == 'not_run' for row in summary),
        distinct_initial_parameter_fingerprints=len(fingerprints),
        identical_bounded_source_manifests=True, draws=summary)
    (HERE / 'fresh-final-summary.json').write_text(json.dumps(audit, indent=2) + '\n')
    output += ['', f"**{audit['completed']}/12 completed: {audit['passed']} passed, "
        f"{audit['failed']} failed; {audit['not_run']} not run.** All "
        f"{len(fingerprints)} completed initializations have distinct parameter fingerprints.", '',
        'Reverse draws [2](reverse-2-failure.md) and [5](reverse-5-failure.md)',
        'were exactly replayed before any production or assertion change.',
        'Their failing replays are diagnostics, not additional fresh draws.',
        'The gate stops for Alec’s contract/mechanism decision after draw 5.',
        'The eight-file set, both decomposition arms and both eight-seat MM',
        'gates are unrun on this source. No earlier-source result is promoted.', '']
    (HERE / 'fresh-final.md').write_text('\n'.join(output))
    print(json.dumps({key: value for key, value in audit.items() if key != 'draws'}, indent=2))


if __name__ == '__main__':
    main()
