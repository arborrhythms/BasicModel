"""Read saved results only; never construct a model or start a training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def gate(name):
    folder = HERE / name
    value = read(folder / 'result.json')
    reports = [row for path in folder.glob('worker-*.json')
               if not path.name.endswith(('.request.json', '.process.json'))
               for row in read(path).get('reports', ())]
    outcomes = Counter((row['phase'], row['outcome']) for row in reports)
    cases = {}
    for row in reports:
        cases.setdefault(row['nodeid'], set()).add(row['outcome'])
    assert len(cases) == len(value['completed'])
    assert all(len(outcomes) == 1 for outcomes in cases.values())
    return dict(exit_code=value['exit_code'], reason=value['reason'],
        selected=len(value['selected']), completed=len(value['completed']),
        report_outcomes={f'{phase}:{outcome}': count for (phase, outcome), count in outcomes.items()},
        case_outcomes=dict(Counter(next(iter(outcomes)) for outcomes in cases.values())),
        compile_cache_retries=value['compile_cache_retries'])


def math_results():
    summary = read(HERE / 'summary.json')
    result = {}
    for condition, value in summary['conditions'].items():
        counts, operations, credits, errors = Counter(), Counter(), Counter(), Counter()
        questions = []
        norm_sum, norm_max = 0., 0.
        for run in value['runs']:
            folder = HERE / 'math-trainings' / f"{condition}-{run['run']:02}"
            observer = run['observer']
            counts.update(observer.get('counts', {}))
            operations.update(observer.get('operations', {}))
            credits.update(observer.get('credit', {}))
            errors.update([run['error']] if run['error'] else [])
            norm_sum += observer.get('raw_chooser_gradient_l2_sum', 0.)
            norm_max = max(norm_max, observer.get('raw_chooser_gradient_l2_max', 0.))
            if (folder / 'questions.jsonl').exists():
                questions.extend(lines(folder / 'questions.jsonl'))
        result[condition] = dict(attempted=len(value['runs']), completed=value['completed'],
            completed_epochs=sum(run['completed_epochs'] for run in value['runs']),
            errors=dict(errors), held_out_rows_tested=value['held_out_rows_tested'],
            held_out_accuracy=value['held_out_accuracy'], all_four_correct=value['all_four_correct'],
            held_out_chain_runs=value['held_out_chain_runs'],
            beyond_runs_evaluated=sum(run['beyond'] is not None for run in value['runs']),
            chooser_movement_runs_measured=sum(run['chooser_movement'] is not None for run in value['runs']),
            partial_observations=dict(counts=counts, operations=operations, credit=credits,
                raw_chooser_gradient_l2_sum=norm_sum, raw_chooser_gradient_l2_max=norm_max,
                questions=len(questions), question_episodes=sum(row['episode'] for row in questions),
                correct_committed_bindings=sum(row['bound_correct'] for row in questions),
                completed_chains=sum(row['chain_correct'] for row in questions),
                question_phases=dict(Counter(row['phase'] for row in questions)),
                questions_with_open_references=sum(bool(row['open']) for row in questions)))
    return result


def standing():
    folder = HERE / 'measurements'
    complete = read(folder / 'complete.json')
    assert complete['completed'] and complete['source_matched'] and len(complete['jobs']) == 30
    result = dict(sum=[], xor=[], mm=[], thinking=[])
    for run in range(1, 11):
        row = read(folder / f'sum-{run:02}/measurement.json')
        result['sum'].append(dict(run=run, mse=row['mse'], contrast=row['contrast'],
            sum_pass=row['sum_bar'], floor_pass=row['floor_bar']))
        where = folder / f'xor-{run:02}'
        events = lines(where / 'observations.jsonl')
        grammar, = [row for row in events if row['kind'] == 'grammar']
        consumers = [row for row in events if row['kind'] == 'shared_gate_consumer']
        assert len(consumers) == 2 and len({row['model_identity'] for row in consumers}) == 1
        calls = [row for row in lines(where / 'reports.jsonl') if row['phase'] == 'call']
        assert len(calls) == 2
        predicted, targets = grammar['predictions'], grammar['targets']
        mse = sum((a-b)**2 for a, b in zip(predicted, targets)) / len(targets)
        correct = sum((a > .5) == (b > .5) for a, b in zip(predicted, targets))
        recovered = sum(b is not None and Counter(a.split()) == Counter(b.replace(chr(0), ' ').split())
            for a, b in zip(grammar['inputs'], grammar['gate_reconstructions']))
        class_pass = next(row['outcome'] == 'passed' for row in calls if 'test_xor_class_accuracy' in row['nodeid'])
        reconstruction_pass = next(row['outcome'] == 'passed' for row in calls if 'test_piecewise_overall' in row['nodeid'])
        assert class_pass == (correct == 4 and mse < .05)
        assert reconstruction_pass == (recovered == 4 and not any(grammar['grammar_reconstruction_unavailable']))
        result['xor'].append(dict(run=run, mse=mse, correct=correct, reconstructed=recovered,
            class_pass=class_pass, reconstruction_pass=reconstruction_pass,
            joint=class_pass and reconstruction_pass, same_training=True))
        where = folder / f'mm-{run:02}'
        observation, = [row for row in lines(where / 'observations.jsonl') if row['kind'] == 'mm']
        call, = [row for row in lines(where / 'reports.jsonl') if row['phase'] == 'call']
        assert (call['outcome'] == 'passed') == (observation['best'] < .20)
        result['mm'].append(dict(run=run, passed=call['outcome'] == 'passed', best=observation['best']))
        for kind in ('sum', 'xor', 'mm'):
            name = f'{kind}-{run:02}'
            result['thinking'].append(dict(run=name, **read(folder / name / 'thinking.json')))
    result['counts'] = dict(sum=sum(row['sum_pass'] for row in result['sum']),
        sum_floor=sum(row['floor_pass'] for row in result['sum']),
        xor_class=sum(row['class_pass'] for row in result['xor']),
        xor_reconstruction=sum(row['reconstruction_pass'] for row in result['xor']),
        xor_joint=sum(row['joint'] for row in result['xor']),
        mm=sum(row['passed'] for row in result['mm']),
        thought_calls=sum(row['calls'] for row in result['thinking']),
        open_episodes=sum(row['open_episodes'] for row in result['thinking']))
    result['completed'] = len(complete['jobs'])
    result['processes_passed'] = sum(row['process']['exit_code'] == 0 for row in complete['jobs'])
    save('standing-summary.json', result)
    return result['counts']


def integrity():
    sys.path.insert(0, str(ROOT / 'test'))
    import bounded_tests
    from verification import validate
    source = bounded_tests.source_snapshot(ROOT)
    validate(source)
    frozen = read(HERE / 'measured-source/freeze.json')
    helpers = read(HERE / 'measured-source/measurement-helpers.json')
    supplemental = read(HERE / 'measurement-supplement.json')['files']
    protected = {name: (ROOT / name).read_bytes() == subprocess.check_output(
        ['git', 'show', frozen['base_commit'] + ':' + name], cwd=ROOT)
        for name in frozen['protected_matches']}
    value = dict(source_files=len(source), source_matches=True,
        original_helpers_match=all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in helpers.items()),
        supplemental_helpers_match=all(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
            for name, digest in supplemental.items()), protected_gate_files_unchanged=protected,
        original_receipts_unchanged=not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--',
            'doc/benchmarks/2026-10-07-item6-2', 'doc/benchmarks/2026-10-07-item6-2-repair'], cwd=ROOT).strip(),
        live_spec_sha256=hashlib.sha256((ROOT / 'doc/specs/2026-10-07-thinking.md').read_bytes()).hexdigest())
    assert value['original_helpers_match'] and value['supplemental_helpers_match']
    assert all(protected.values()) and value['original_receipts_unchanged']
    assert value['live_spec_sha256'] == read(HERE / 'protocol.json')['spec_sha256']
    save('integrity.json', value)
    return value


def paired_starts():
    import torch
    conditions = read(HERE / 'protocol.json')['conditions']
    rows = []
    for run in range(1, 11):
        folders = [HERE / 'math-trainings' / f'{condition}-{run:02}' for condition in conditions]
        states = [torch.load(folder / 'initial-chooser.pt', map_location='cpu', weights_only=True)
                  for folder in folders]
        matches = [states[0].keys() == state.keys() and
                   all(torch.equal(value, state[name]) for name, value in states[0].items())
                   for state in states[1:]]
        assert all(matches)
        rows.append(dict(run=run, initial_chooser_parameters_match_across_conditions=True,
            saved_entropy_sha256=hashlib.sha256((HERE / 'math-trainings' /
                f'paired-{run:02}' / 'initial-rng.pkl').read_bytes()).hexdigest()))
    save('paired-start-check.json', rows)
    return rows


def main():
    value = dict(status=read(HERE / 'summary.json')['status'],
        final_sweep=gate('final-sweep'), thinking_gate=gate('thinking-gate'),
        math=math_results(), standing=standing(), integrity=integrity(), paired_starts=paired_starts(),
        measurement_limit='Partial training observations are not held-out evaluation. Failed attempts have no final chooser movement measurement.',
        retry_policy='All declared trainings attempted once. Two standing audit-import startup failures preserved; first actual sum trainings restore the original saved entropy states.')
    save('review-summary.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
