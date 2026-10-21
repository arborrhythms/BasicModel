"""Reconstruct the existing per-site arithmetic from captured probabilities."""
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parents[1]


def choice(item):
    selected, choices = item['selected'], item['candidates']
    if not choices or selected not in choices:
        return 1.0
    phi, probabilities = item['features'], item['probabilities']
    index = choices.index(selected)
    credits = [6*row[1] + row[2] + row[3] for row in phi]
    exps = [math.exp(value-max(credits)) for value in credits]
    total = sum(exps)
    entropy = -sum(value/total*math.log(probability) for value, probability in zip(exps, probabilities))
    return max(0, min(1, 1-phi[index][1])) - phi[index][2] + entropy


rows = []
for name, document in (('fresh-final-reverse-2', 2), ('fresh-final-reverse-3', 7)):
    batches = [json.loads(line) for line in (here/(name+'-keep.jsonl')).read_text().splitlines()]
    for batch in batches:
        assert all(target is None for target in batch['supplied_targets'])
        for trial_index, trial in enumerate(batch['readings']):
            plan = trial['reference_plans'][document]
            sites = [max(0, min(1, max((row[1] for row in item['features']), default=0)))
                     if item['mode'] == 'mint' and item['features'] is not None else
                     0 if item['mode'] == 'mint' else choice(item) for item in plan['items']]
            arrivals = [choice(item) for item in plan['arrivals']]
            assert math.isclose(sum(sites)+sum(arrivals), plan['cost'], abs_tol=1e-6)
            reference = trial['reference_credit'][document]
            assert math.isclose(reference, .1*plan['cost'], abs_tol=1e-7)
            answer = 0 if trial['answer_cost'] is None else trial['answer_cost'][document]
            components = batch['credit']['components'][document][trial_index]
            assert math.isclose(answer+reference, components[2], abs_tol=1e-7)
            rows.append(dict(case=name, document=document, batch=batch['batch'], trial=trial['trial'],
                R=components[0], answer=answer, reference=reference, A=components[2],
                sites=sites, arrivals=arrivals, plan_cost=plan['cost'], weight=.1,
                keep=batch['credit']['keep_decision'][document],
                deciding=batch['credit']['deciding'][document]))
(here/'reference-cost-breakdown.json').write_text(json.dumps(rows, indent=2)+'\n')
text = ['# Reverse failures: answer and reference terms of A', '',
    'Frozen source `3d453e6bb6927e5da1492c25106a4ef107b87bca`.',
    'All premise targets are None. Answer cost is absent (zero). A equals',
    'the captured reference credit, which is plan cost times .1. Per-site',
    'terms below are reconstructed from captured features and probabilities',
    'using the unchanged choice_cost, and checked against the captured sum.', '',
    '| Draw / document / batch | Trial | Site costs | Arrival costs | Plan | Answer | Reference = A | R | Keep |',
    '|---|---|---|---|---:|---:|---:|---:|---|']
for row in rows:
    text.append(f"| {row['case']} / {row['document']} / {row['batch']} | {row['trial']} | "
        f"{row['sites']} | {row['arrivals']} | {row['plan_cost']:.9f} | {row['answer']} | "
        f"{row['A']:.9f} | {row['R']:.9f} | {row['keep']} |")
text += ['', 'On the first sentence the named path costs 1 for its first bind plus',
    '3.079441542 for the arrival choice (1 + log(8)), yielding A=.407944173.',
    'The kept field/mint path has neither charge and A=0. R ties. Thus the',
    'earlier operation/keep decision removes the named site; the missing',
    'three candidate in the next frame then prevents recovery. These are',
    'two stages of the failure, not a lost answer target or an MTM eviction.', '',
    'No price, objective, keep, scope or assertion is changed by this analysis.', '']
(here/'reference-cost-breakdown.md').write_text('\n'.join(text))
print('All eight trial rows reconcile: A is entirely closing reference credit.')
