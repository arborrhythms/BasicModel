"""Summarize verified switch diagnostics from saved records only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = {}


def read(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(HERE))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def lines(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(HERE))] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines()]


def observation(folder, kind):
    if kind == 'sum':
        row = read(folder / 'measurement.json')
        return dict(mse=row['mse'], passed=row['floor_bar'], additive=row['sum_bar'],
            reconstructed=row['reconstructed'], predictions=row['answers'],
            readbacks=row['read_backs'], storage=row['storage_after_training'])
    row, = [x for x in lines(folder / 'observations.jsonl') if x['kind'] == 'grammar']
    calls = [x for x in lines(folder / 'reports.jsonl') if x['phase'] == 'call']
    assert len(calls) == 2
    class_pass = next(x['outcome'] == 'passed' for x in calls if 'test_xor_class_accuracy' in x['nodeid'])
    recon_pass = next(x['outcome'] == 'passed' for x in calls if 'test_piecewise_overall' in x['nodeid'])
    predictions, targets = row['predictions'], row['targets']
    return dict(mse=sum((a-b)**2 for a,b in zip(predictions, targets))/len(targets),
        passed=class_pass and recon_pass, class_pass=class_pass, reconstruction_pass=recon_pass,
        predictions=predictions, readbacks=row['gate_reconstructions'],
        storage=row['storage_after_training'])


def audit(folder):
    raw = read(folder / 'run-audit.json')
    trials = [x for x in raw['sentence_trials'] if x['training']]
    steps = {(s['epoch'],s['batch_row']):s for s in raw['compose_score_function_steps']}
    costs, by_action = {}, defaultdict(Counter)
    for i, label in enumerate(('R','E','A')):
        values = [c[i] for t in trials for pair in t['components'] for c in pair]
        costs[label] = dict(maximum=max(values), nonzero_trial_rows=sum(v != 0 for v in values),
            nonzero_differences=sum(d[i] != 0 for t in trials for d in t['delta']),
            nonzero_epochs=sorted({t['epoch'] for t in trials if any(c[i] != 0 for pair in t['components'] for c in pair)}))
    for t in trials:
        for b, advantage in enumerate(t['advantage']):
            step = steps[t['epoch'],b]
            key = step['walk'] + '/' + step['action_name']
            c = by_action[key]
            c['departures'] += int(step['departed'])
            c['nonzero'] += int(advantage != 0)
            c['reward_explore'] += int(advantage < 0)
            c['reward_greedy'] += int(advantage > 0)
            c['advantage_sum'] += advantage
            for i,label in enumerate(('R','E','A')):
                c['delta_' + label + '_sum'] += t['delta'][b][i]
    result = dict(costs=costs, by_walk_action=by_action, priming=raw['priming'],
        centroid=raw['centroid'], final_operators=raw['final_committed_training_operators'],
        cost_records=str((folder / 'run-audit.json').relative_to(HERE)))
    return result, trials


def main():
    complete = read(HERE / 'verified-bisections/complete.json')
    assert complete['source_matched'] and len(complete['jobs']) == 6
    rows = []
    for job in complete['jobs']:
        original_folder = HERE / 'measurements' / f'{job["kind"]}-{job["run"]:02}'
        folder = HERE / 'verified-bisections' / job['name']
        effective = read(folder / 'effective-switch.json')
        assert effective['verified']
        original = observation(original_folder, job['kind'])
        variant = observation(folder, job['kind'])
        original_audit, before = audit(original_folder)
        variant_audit, after = audit(folder)
        assert len(before) == len(after) == 400
        first = {}
        for field in ('components','delta','actions','narrowing_actions','readers'):
            first[field] = next((a['epoch'] for a,b in zip(before,after) if a[field] != b[field]), None)
        if job['part'] == 'C':
            assert all(not c['enabled'] for c in variant_audit['centroid']['end'])
        if job['part'] == 'D':
            assert all(not row['received'] for row in variant_audit['priming'].values())
        states = [hashlib.sha256((p / 'unseeded-entry.pt').read_bytes()).hexdigest()
                  for p in (original_folder, folder)]
        rows.append(dict(name=job['name'], kind=job['kind'], part=job['part'], original=original,
            variant=variant, changed_pass=original['passed'] != variant['passed'],
            effective_switch=effective, first_difference_epoch=first,
            entry_state_sha256=states, entry_state_files_identical=states[0] == states[1],
            original_audit=original_audit, variant_audit=variant_audit, process=job['process']))
    rows.sort(key=lambda r:r['name'])
    result = dict(rows=rows, source_matched=True, inputs=INPUTS,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        standing_replacements=0, prior_invalid='diagnostic-override-error.json')
    (HERE / 'bisection-report.json').write_text(json.dumps(result,indent=2) + '\n')
    print(json.dumps([dict(name=r['name'], mse=r['variant']['mse'], passed=r['variant']['passed'],
        class_pass=r['variant'].get('class_pass'), reconstruction_pass=r['variant'].get('reconstruction_pass'),
        first_difference_epoch=r['first_difference_epoch'], rng_file_identical=r['entry_state_files_identical'],
        R_max=r['variant_audit']['costs']['R']['maximum'], R_delta=r['variant_audit']['costs']['R']['nonzero_differences']) for r in rows]))


if __name__ == '__main__':
    main()
