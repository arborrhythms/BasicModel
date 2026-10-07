"""Read saved evidence, including absolute R (not only trial differences).

Added after source freeze as a reporting tool. It never builds or trains a
model. The input digests and this script's digest are retained in its output.
"""
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


def main():
    summary = read(HERE / 'measurements/summary.json')
    runs = []
    totals = defaultdict(Counter)
    for kind in ('sum', 'xor'):
        for row in summary[kind]:
            name = f'{kind}-{row["run"]:02}'
            audit = read(HERE / 'measurements' / name / 'run-audit.json')
            trials = [t for t in audit['sentence_trials'] if t['training']]
            steps = {(s['epoch'], s['batch_row']): s
                     for s in audit['compose_score_function_steps']}
            components = {}
            for index, label in enumerate(('R', 'E', 'A')):
                values = [c[index] for t in trials for pair in t['components'] for c in pair]
                components[label] = dict(minimum=min(values), maximum=max(values),
                    nonzero_trial_rows=sum(v != 0 for v in values),
                    nonzero_epochs=sorted({t['epoch'] for t in trials
                        if any(c[index] != 0 for pair in t['components'] for c in pair)}),
                    nonzero_differences=sum(d[index] != 0 for t in trials for d in t['delta']))
            by_action = defaultdict(Counter)
            for t in trials:
                for b, advantage in enumerate(t['advantage']):
                    s = steps[t['epoch'], b]
                    if s['walk'] != 'narrowing':
                        continue
                    c = by_action[s['action_name']]
                    c['selected'] += 1
                    c['departed'] += int(s['departed'])
                    c['nonzero_credit'] += int(advantage != 0)
                    c['reward_explore'] += int(advantage < 0)
                    c['reward_greedy'] += int(advantage > 0)
                    c['advantage_sum'] += advantage
                    for index, label in enumerate(('R', 'E', 'A')):
                        c['delta_' + label + '_sum'] += t['delta'][b][index]
                    if t['epoch'] > 380:
                        c['last20_selected'] += 1
                        c['last20_nonzero_credit'] += int(advantage != 0)
                        c['last20_reward_explore'] += int(advantage < 0)
                        c['last20_reward_greedy'] += int(advantage > 0)
            for action, counts in by_action.items():
                totals[kind + '/' + action].update(counts)
            runs.append(dict(name=name, absolute_components=components,
                narrowing_by_action=by_action, priming=audit['priming'],
                final_operators=audit['final_committed_training_operators'],
                rung0=audit['rung0_audit'], centroid=audit['centroid']))
    priming = runs[0]['priming']
    result = dict(scope='All twenty grammar trainings, original saved records only',
        priming_snapshot='Last recorded membership diffusion and resulting per-batch word weight; batch order is hello world, hello there, loving world, loving there.',
        priming_identical_across_grammar_runs=all(r['priming'] == priming for r in runs),
        shared_priming=priming, narrowing_by_action=totals, runs=runs,
        mm='The numerical MM path has no membership sentence nodes and no grammar narrowing trials.',
        inputs=INPUTS, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / 'cost-and-priming-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(priming_identical=result['priming_identical_across_grammar_runs'],
        absolute_components={r['name']:r['absolute_components'] for r in runs},
        narrowing_by_action=totals)))


if __name__ == '__main__':
    main()
