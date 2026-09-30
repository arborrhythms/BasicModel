"""Report every declared run and the HEAD spread; never select a seed."""
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
rows = []
for label in ('head', 'candidate'):
    directory = HERE / (label + '-reconstruction')
    processes = json.loads((directory / 'processes.json').read_text())
    for seed in (0, 1, 2):
        path = directory / f'seed-{seed}.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        rows.append(dict(tree=label, seed=seed, process=processes[str(seed)],
            phases={phase['name']: phase['reconstruction_mean'] for phase in record.get('phases', ())},
            atom_prefix=record.get('initial_atom_prefix_sha256')))
complete = all(row['process']['exit_code'] == 0 and
               set(row['phases']) == {'before_training', 'training', 'after_training'} for row in rows)
spread = {}
if complete:
    for phase in ('before_training', 'training', 'after_training'):
        head = [r['phases'][phase] for r in rows if r['tree'] == 'head']
        candidate = [r['phases'][phase] for r in rows if r['tree'] == 'candidate']
        spread[phase] = dict(head_min=min(head), head_max=max(head), head_mean=statistics.mean(head),
            candidate_min=min(candidate), candidate_max=max(candidate), candidate_mean=statistics.mean(candidate),
            each_candidate_inside_head_spread=[min(head) <= v <= max(head) for v in candidate],
            candidate_mean_inside_head_spread=min(head) <= statistics.mean(candidate) <= max(head))
report = dict(protocol='4 validation, 7 training (2 warmup + 5 measured), 4 validation; batch 2; seeds 0,1,2 for both trees',
    all_six_complete=complete, runs=rows, spread=spread, baseline_changed=False,
    timing='Concurrent bounded diagnostic lanes; elapsed throughput is not a controlled performance comparison.')
(HERE / 'reconstruction-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(all_six_complete=complete, spread=spread), indent=2))
