"""Render retained arrival readings and compare saved initial states only."""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'2026-10-20-item3-5-stm1'/'review3'
NODE = 'test_forced_ordinary_pending_premise_in_both_orders[False]'


def load(path):
    return json.loads(path.read_text())


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def graph_text(graph):
    result = []
    for index, node in enumerate(graph['nodes']):
        forwards = node['meaning']['bindings'].get('_forward_references', ())
        result.append(f"n{index} refs={node['refs']}; forward={forwards}; "
            f"subject_word={node['subject_word_id']}; children={node['children']}")
    return '<br>'.join('`'+line.replace('|', '\\|')+'`' for line in result)


def tables(arrivals):
    result = ['# Arrival keep: one table per document', '',
        'Source `f9faae610f35d5ca660d0ac49249637cb23e7adf`. The forward',
        'assertion, budget, forced scope, objective and keep rule are unchanged.',
        '`exploit` is greedy; E is excluded. Readings are before admission.',
        'Complete refs/open fields, compositions, registries, credit costs,',
        'keep costs and departures are retained in the adjacent JSONL files.', '']
    for arrival in arrivals:
        credit = arrival['credit']
        for stream in range(len(credit['costs'])):
            result += [f"## {arrival['case']}, batch {arrival['batch']}, document {stream}", '',
                f"Kept **{credit['keep_decision'][stream]}**; deciding "
                f"`{credit['deciding'][stream]}`; compose departure "
                f"`{credit['departure']['compose_round'][stream]}`.", '',
                '| Trial | R | E (excluded) | A | R + A | Reading before admission |',
                '|---|---:|---:|---:|---:|---|']
            for index, trial in enumerate(arrival['readings']):
                r, e, a = credit['components'][stream][index]
                result.append(f"| {trial['trial']} | {r:.10f} | {e:.10f} | {a:.10f} | "
                    f"{credit['costs'][stream][index]:.10f} | {graph_text(trial['clauses'][stream])} |")
            result += ['']
    return '\n'.join(result)


def main():
    arrivals, validation = [], {}
    for label, original, stream in (
            ('forward-4', 'fresh-both-forward-4', 6), ('arrival-21', 'arrival-21', 7)):
        name = 'd1-'+label
        prior, captured = PRIOR/(original+'-state')/NODE, HERE/(name+'-state')/NODE
        rows = [json.loads(line) for line in (HERE/(name+'-keep.jsonl')).read_text().splitlines()]
        arrival = rows[-1]
        assert arrival['batch'] == 2 and arrival['sentences'] == ['x is three.']*8
        arrivals.append(arrival)
        a, b = load(prior/'initial-model.json'), load(captured/'initial-model.json')
        initial = {key: a[key] == b[key] for key in (
            'parameters', 'buffers', 'after_rng', 'source_manifest', 'torch_threads')}
        states = {name: same(torch.load(prior/name, weights_only=False),
                            torch.load(captured/name, weights_only=False))
                  for name in ('initial-rng.pt', 'after-model-rng.pt', 'generator-0.pt')}
        assert all(initial.values()) and all(states.values())
        credit = arrival['credit']
        validation[original] = dict(source=arrival['source'], initial=initial, rng_states=states,
            outcome=load(captured/'outcome.json')['outcome'], original_failing_stream=stream,
            costs=credit['costs'][stream], components=credit['components'][stream],
            keep=credit['keep_decision'][stream], deciding=credit['deciding'][stream])
    (HERE/'arrival-keep.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in arrivals))
    (HERE/'arrival-keep-tables.md').write_text(tables(arrivals)+'\n')
    (HERE/'arrival-keep-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
