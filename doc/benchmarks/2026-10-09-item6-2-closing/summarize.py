"""Summarize the thirty original attempts without running or replacing one."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'test'))
from bounded_tests import source_snapshot
from verification import validate


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(output):
    source = source_snapshot(ROOT)
    validate(source)
    campaign = read(HERE/'measurements/complete.json')
    assert campaign['completed'] and campaign['source_matched'] and len(campaign['jobs']) == 30
    jobs = {job['name']: job for job in campaign['jobs']}
    assert set(jobs) == {f'{kind}-{n:02}' for kind in ('sum', 'xor', 'mm') for n in range(1, 11)}
    bisection_path = HERE/'mm-bisection/result.json'
    bisections = {} if not bisection_path.exists() else {
        row['name']: row for row in read(bisection_path)['cases']}
    outcomes = []
    for kind in ('sum', 'xor', 'mm'):
        for n in range(1, 11):
            name = f'{kind}-{n:02}'
            folder = HERE/'measurements'/name
            process = jobs[name]['process']
            thought = read(folder/'thinking.json')
            row = dict(name=name, kind=kind, process_exit=process['exit_code'],
                process_reason=process['reason'], seconds=process['elapsed_seconds'], thought=thought)
            if kind == 'sum':
                measured = read(folder/'measurement.json')
                row.update(sum_bar=measured['sum_bar'] and process['exit_code'] == 0,
                    floor_bar=measured['floor_bar'], mse=measured['mse'], contrast=measured['contrast'])
            else:
                observations = lines(folder/'observations.jsonl')
                reports = lines(folder/'reports.jsonl')
                calls = {item['nodeid'].rsplit('::', 1)[-1]: item['outcome']
                         for item in reports if item['phase'] == 'call'}
                if kind == 'xor':
                    measured, = [item for item in observations if item['kind'] == 'grammar']
                    consumers = [item['model_identity'] for item in observations
                                 if item['kind'] == 'shared_gate_consumer']
                    assert len(consumers) == 2 and len(set(consumers)) == 1
                    row.update(class_bar=calls.get('test_xor_class_accuracy') == 'passed',
                        reconstruction_bar=calls.get('test_piecewise_overall_at_least_50_pct') == 'passed',
                        mse=sum((p-y)**2 for p, y in zip(measured['predictions'], measured['targets']))
                            / len(measured['targets']), shared_training=True,
                        predictions=measured['predictions'], targets=measured['targets'],
                        inputs=measured['inputs'], reconstructions=measured['gate_reconstructions'])
                else:
                    measured, = [item for item in observations if item['kind'] == 'mm']
                    row.update(bar=calls.get('test_convergence') == 'passed', best=measured['best'],
                        steps=measured['calls'], diagnostic_landing_identical=
                            bisections.get(name, {}).get('landing_identical', False))
                    assert row['bar'] == (measured['best'] < .20)
            outcomes.append(row)
    counts = dict(sum=sum(row.get('sum_bar', False) for row in outcomes),
        sum_floor=sum(row.get('floor_bar', False) for row in outcomes),
        xor_class=sum(row.get('class_bar', False) for row in outcomes),
        xor_reconstruction=sum(row.get('reconstruction_bar', False) for row in outcomes),
        mm=sum(row.get('bar', False) for row in outcomes))
    mm = [row for row in outcomes if row['kind'] == 'mm']
    all_expected = all(row['process_exit'] == 0 for row in outcomes if row['kind'] != 'mm')
    all_expected &= all(row['process_exit'] == (0 if row['bar'] else 1) for row in mm)
    all_expected &= all(row['process_reason'] == 'exit' for row in outcomes)
    no_episodes = all(row['thought'] == dict(calls=0, open_episodes=0, bound_calls=0) for row in outcomes)
    condition = (all_expected and no_episodes and counts['sum'] == counts['sum_floor'] == 10
                 and counts['xor_class'] == counts['xor_reconstruction'] == 10
                 and all(row['bar'] or row['diagnostic_landing_identical'] for row in mm))
    result = dict(authority='Alec; thinking section 14.13', attempts=30, completed=30,
        source_files=len(source), source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        raw_bars=counts, outcomes=outcomes, seconds=campaign['seconds'],
        seed=None, retries=0, replacements=0, all_expected_process_outcomes=all_expected,
        all_thought_calls_and_episodes_zero=no_episodes, condition_satisfied=condition,
        bisection=None if not bisections else 'mm-bisection/result.json',
        diagnostic_cases=sorted(bisections), learning_claim=False,
        interpretation='Raw MM misses stay misses. Only exact diagnostic evidence can satisfy the operators section 20 exception.')
    destination = HERE/output
    with destination.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('attempts', 'raw_bars', 'condition_satisfied',
        'all_thought_calls_and_episodes_zero', 'diagnostic_cases', 'seconds')}))


if __name__ == '__main__':
    main(sys.argv[1])
