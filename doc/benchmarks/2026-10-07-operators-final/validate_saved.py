"""Validate and tabulate saved evidence only; never instantiate a model."""
from collections import Counter
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


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    summary = read(HERE / 'measurements/summary.json')
    sweep = read(HERE / 'full-sweep/result.json')
    frozen = read(HERE / 'delivered-source/source.json')
    helpers = read(HERE / 'delivered-source/measurement-helpers.json')
    assert frozen == source_snapshot(ROOT)
    from verification import validate
    verification = validate(frozen)
    assert frozen == read(HERE / 'measurements/source.json')
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in helpers.items())
    assert len(sweep['completed']) == len(sweep['selected'])
    assert summary['complete']['completed'] and len(summary['complete']['jobs']) == 30
    assert summary['complete']['retries'] == 0
    assert not read(HERE / 'delivered-source/seed-port-audit.json')['changed_seed_calls']
    outcomes = Counter()
    for worker in sweep['workers']:
        for report in worker.get('reports', ()):
            outcomes[report['outcome']] += 1
    if not outcomes:
        for path in (HERE / 'full-sweep').glob('worker-*.json'):
            if '.process.' in path.name or '.request.' in path.name:
                continue
            for report in read(path).get('reports', ()):
                outcomes[report['outcome']] += 1
    storage = []
    expected_addresses = None
    for kind in ('sum', 'xor', 'mm'):
        for row in summary[kind]:
            final = row['storage']
            training = row.get('storage_after_training', final)
            check = dict(kind=kind, run=row['run'], after_training=training, final=final)
            check['within_capacity'] = final['rows_used'] <= final['capacity']
            if kind in ('sum', 'xor'):
                addresses = sorted(training['addresses'])
                if expected_addresses is None:
                    expected_addresses = addresses
                check.update(four_sentence_rows=training['sentence_rows'] == 4,
                    four_content_keys=len(set(training['content_keys'])) == 4,
                    four_hundred_witnesses=training['witness_counts'] == [400] * 4,
                    same_addresses_across_runs=addresses == expected_addresses,
                    final_preserves_addresses=sorted(final['addresses']) == addresses)
            else:
                check['raw_path_definitions_only'] = (final['sentence_rows'] == 0
                    and final['definition_rows'] == final['rows_used'] == 4)
            storage.append(check)
    checks = [value for row in storage for value in row.values() if isinstance(value, bool)]
    result = dict(source_matched=True, measurement_helpers_matched=True,
        gate_trainings=30, seed=None, retries=0, replacement_runs=0,
        sweep=dict(cases=len(sweep['selected']), outcomes=dict(outcomes), reason=sweep['reason'],
                   focused_repair_receipts=verification['focused_repairs']['receipts'],
                   seconds=sweep['elapsed_seconds']),
        counts=summary['counts'], below_comparison=summary['below_comparison'],
        storage_checks_passed=all(checks), storage=storage)
    save('results-validation.json', result)
    lines = ['# Operators final: all thirty trainings', '',
        'Each row is its one declared, unseeded run. No retry or replacement.', '',
        '| Run | MSE (MM: best) | Class | Reconstruction | Control | Rows/capacity | Sentence/DEF rows | Witnesses per sentence after training |',
        '|---|---:|---|---|---|---:|---:|---|']
    for kind in ('sum', 'xor', 'mm'):
        for row in summary[kind]:
            s = row['storage']
            train = row.get('storage_after_training', s)
            value = row['best'] if kind == 'mm' else row['mse']
            class_pass = row.get('class_pass', row.get('class_bar'))
            reconstruction = row.get('reconstruction_pass', row.get('reconstruction_bar'))
            control = row.get('sum_bar') if kind == 'sum' else row.get('passed') if kind == 'mm' else None
            label = lambda value: '—' if value is None else 'pass' if value else 'FAIL'
            witnesses = ', '.join(map(str, train['witness_counts'])) or '—'
            lines.append(f"| {kind}-{row['run']:02} | {value:.10g} | {label(class_pass)} | {label(reconstruction)} | {label(control)} | {s['rows_used']}/{s['capacity']} | {s['sentence_rows']}/{s['definition_rows']} | {witnesses} |")
    lines += ['', 'Evaluation also re-witnesses the existing addresses. Full final witness counts, timestamps, content keys and addresses are in [validation](results-validation.json).', '',
        'The unchanged raw MM control has no completed sentence closing and retains only its four DEF rows; no new closing was added for this receipt.', '']
    (HERE / 'measurement-results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(counts=result['counts'], sweep=result['sweep'],
        storage_checks_passed=result['storage_checks_passed'])))


if __name__ == '__main__':
    main()
