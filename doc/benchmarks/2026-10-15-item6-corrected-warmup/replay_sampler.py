"""Quantify the diversity assertion on frozen observed eligible menus.

Use the native reservoir and walk selector, with dummy snapshot payloads.
No model is trained or candidate score changed. This is conditional on the
recorded menus, not a new estimate of the full training test's failure rate.
"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import argparse
import json

import torch
from SentenceCredit import departure
from SentenceFork import SentenceFork
from identity_experiment import save


def run(report, trials=4096):
    torch.manual_seed(20261015)
    torch.set_num_threads(1)
    observations = []
    laws = []
    for draw, reservoir in zip(report['departures'], report['reservoirs']):
        batch = len(draw['compose_round'])
        menus = [reservoir['eligible'].get(str(row), []) for row in range(batch)]
        width = max(map(len, menus))
        active = torch.ones(trials * batch, dtype=torch.bool)
        fork = SentenceFork(active)
        snapshots = torch.zeros(trials * batch, 1)
        valid_steps = []
        for step in range(width):
            slots = torch.tensor([menu[step] if step < len(menu) else -1 for menu in menus]).repeat(trials)
            valid = slots >= 0
            valid_steps.append(valid)
            choice = SimpleNamespace(valid=valid, alternative_count=valid.long(), action=slots.clamp_min(0))
            fork.record(choice, slots, snapshots, word=torch.zeros_like(slots), latches=(snapshots,))
        # Every observed row in these two fixtures has all three walks.
        assert draw['walk_count'] == [3] * batch
        result = departure(SimpleNamespace(alternatives=active[:, None]),
            torch.stack(valid_steps, -1), active=active, compose_round=fork.slot,
            candidates=active[:, None])
        observations.append(result['compose_round'].reshape(trials, batch))
        for menu in menus:
            counts = Counter(menu)
            laws.append({None: 2/3, **{slot: count/(3*len(menu)) for slot, count in counts.items()}})
    values = torch.cat(observations, 1).tolist()
    distinct = [len(set(row)-{-1}) for row in values]
    p_none = 1.
    for law in laws: p_none *= law[None]
    p_failure = p_none
    for slot in set().union(*(set(law)-{None} for law in laws)):
        probability = 1.
        for law in laws: probability *= law[None] + law.get(slot, 0.)
        p_failure += probability - p_none
    expected_round = 6 if report['certificate'] == 'bound' else 24
    example = next((row for row in values if set(row)-{-1} == {expected_round}), None)
    return dict(certificate=report['certificate'], seed=20261015, trials=trials,
        rows_per_trial=len(laws), fail_distinct_le_one=sum(n <= 1 for n in distinct),
        theoretical_probability=p_failure, no_compose_probability=p_none,
        same_round_witness=example, witness_round=expected_round,
        note='Frozen menus; native independent reservoir/walk sampling. Keep costs are not inputs.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    results=[run(json.loads((args.root/(name+'-new-paired')/'result.json').read_text()))
             for name in ('bound','fork')]
    save(args.out, results)
    print(json.dumps(results,indent=2))
