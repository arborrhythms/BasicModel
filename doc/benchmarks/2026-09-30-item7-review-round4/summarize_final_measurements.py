"""Report every declared trial; keep incomplete attempts beside completions."""
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def mm_grammar():
    receipts = {'HEAD': 'final-mm-grammar-head',
                'candidate': 'final3-mm-grammar-candidate'}
    data = {}
    for tree, receipt in receipts.items():
        directory = HERE / receipt
        processes = read(directory / 'processes.json')
        assert set(processes) == set(map(str, range(10)))
        data[tree] = [dict(trial=i + 1, process=processes[str(i)],
                          measurement=read(directory / f'run-{i:02}.json'))
                      for i in range(10)]
    lines = ['# MM_grammar: final ten full runs per tree', '',
             'All ten unseeded trials are reported. Each runs the unchanged configuration for 900 epochs under the 8 GiB guard. Rows show run order; the two trees do not share matched initializations.', '',
             '| Trial | HEAD ending MSE | Candidate ending MSE | HEAD after 900 updates | Candidate after 900 updates | HEAD GiB | Candidate GiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for head, candidate in zip(data['HEAD'], data['candidate']):
        h, c = head['measurement'], candidate['measurement']
        assert h['seed'] is None and c['seed'] is None
        for trial in (head, candidate):
            assert trial['process']['reason'] == 'exit' and trial['process']['exit_code'] == 0
            assert trial['measurement']['completed_epochs'] == 900
        values = [h['ending_training_mse'], c['ending_training_mse'],
                  h['after_900_updates_mse'], c['after_900_updates_mse']]
        peaks = [t['process']['peak_memory_bytes'] / 2**30 for t in (head, candidate)]
        lines.append(f"| {head['trial']} | " + ' | '.join(f'{v:.10g}' for v in values)
                     + ' | ' + ' | '.join(f'{v:.3f}' for v in peaks) + ' |')
    medians = {tree: statistics.median(t['measurement']['ending_training_mse'] for t in trials)
               for tree, trials in data.items()}
    lines += ['', f"Both trees complete 10/10. Median ending MSE: HEAD {medians['HEAD']:.10g}; candidate {medians['candidate']:.10g}.", '',
              'Ending MSE is the training error before the last update; the adjacent column measures after that update. No threshold or baseline is changed.', '',
              '[HEAD raw trials](final-mm-grammar-head/processes.json); [candidate raw trials](final3-mm-grammar-candidate/processes.json).', '']
    (HERE / 'final-mm-grammar-table.md').write_text('\n'.join(lines))
    (HERE / 'final-mm-grammar-summary.json').write_text(json.dumps(dict(trials=data, medians=medians), indent=2) + '\n')


def reconstruction():
    reuse = read(HERE / 'reconstruction-head-reuse.json')
    baseline = ROOT / reuse['baseline']
    head_processes = read(baseline / 'processes.json')
    reconciled = read(HERE / 'final3-reconstruction-reconciled.json')
    assert reconciled['all_guarded_completions']
    original = read(HERE / 'final3-reconstruction-candidate/processes.json')
    data = {}
    lines = ['# Reconstruction: eight predeclared seeds on each tree', '',
             'Seeds 0–7 were declared before measurement. The model configuration, driver, 8 GiB guard and 1,200-second deadline are unchanged. Every completed value is reported.', '',
             'The completed HEAD baseline is reused after exact source and driver verification. Original candidate deadline stops are retained below; only incomplete attempts receive one serial retry with the same seed and limits.', '',
             '| Seed | HEAD before | Candidate before | HEAD after training | Candidate after training | HEAD GiB | Candidate GiB | Candidate attempts |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for seed in range(8):
        key = str(seed)
        effective = reconciled['effective'][key]
        h, c = read(baseline / f'seed-{seed}.json'), read(HERE / effective['measurement'])
        hp, cp = head_processes[key], effective['process']
        assert c['source'] == reconciled['validated_source']
        assert h['source'] == reuse['validated_source']
        assert h['seed'] == c['seed'] == seed
        for measurement in (h, c):
            assert [phase['name'] for phase in measurement['phases']] == ['before_training', 'training', 'after_training']
        phases = [{p['name']: p for p in measurement['phases']} for measurement in (h, c)]
        values = [p['before_training']['reconstruction_mean'] for p in phases]
        values += [p['after_training']['reconstruction_mean'] for p in phases]
        peaks = [p['peak_memory_bytes'] / 2**30 for p in (hp, cp)]
        attempt = ('deadline stop; serial completion' if original[key]['reason'] == 'timeout' else 'completed once')
        lines.append(f'| {seed} | ' + ' | '.join(f'{v:.10g}' for v in values)
                     + ' | ' + ' | '.join(f'{v:.3f}' for v in peaks) + f' | {attempt} |')
        data[key] = dict(HEAD=dict(measurement=h, process=hp),
                         candidate=dict(measurement=c, process=cp, original_process=original[key]))
        if original[key]['reason'] == 'timeout':
            partial = read(HERE / f'final3-reconstruction-candidate/seed-{seed}.json')
            data[key]['candidate']['retry_prefix_comparison'] = dict(
                same_seed=partial['seed'] == c['seed'],
                same_configuration=partial['config_sha256'] == c['config_sha256'],
                same_initial_atom_prefix=partial['initial_atom_prefix_sha256'] == c['initial_atom_prefix_sha256'],
                same_before_training_reconstruction=[s['reconstruction'] for s in partial['phases'][0]['steps']]
                    == [s['reconstruction'] for s in c['phases'][0]['steps']])
    summaries = {}
    for tree in ('HEAD', 'candidate'):
        values = [t[tree]['measurement']['phases'][2]['reconstruction_mean'] for t in data.values()]
        summaries[tree] = dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))
    lines += ['', 'After-training summary (all eight):', '',
              '| Tree | Mean | Minimum | Maximum |', '|---|---:|---:|---:|']
    for tree, values in summaries.items():
        lines.append(f"| {tree} | {values['mean']:.10g} | {values['minimum']:.10g} | {values['maximum']:.10g} |")
    lines += ['', 'No re-baseline or learning threshold change is made.', '',
              '[HEAD reuse verification](reconstruction-head-reuse.json); [all original candidate attempts](final3-reconstruction-candidate/processes.json); [completion reconciliation](final3-reconstruction-reconciled.json).', '']
    (HERE / 'final-reconstruction-table.md').write_text('\n'.join(lines))
    (HERE / 'final-reconstruction-summary.json').write_text(json.dumps(dict(trials=data, summaries=summaries), indent=2) + '\n')


if __name__ == '__main__':
    import sys
    {'mm': mm_grammar, 'reconstruction': reconstruction}[sys.argv[1]]()
