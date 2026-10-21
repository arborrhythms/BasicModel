"""Build D1 tables and exact-replay evidence from the two retained probes.

This reads receipt files only. It does not construct or query a model.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CASES = (('forward-4', 'fresh-both-forward-4', 6), ('arrival-21', 'arrival-21', 7))
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
                      f"children={node['children']}")
    return '<br>'.join('`'+line.replace('|', '\\|')+'`' for line in result)


tables = ['# D1 arrival keep: one table per document', '',
    'Source: `266189edab25985ac1d000b80553ee398f873b69`; production and',
    'certificate unchanged. Each arrival batch contains eight document streams.',
    'These are premise-arrival checks, not supplied-answer or learning results.',
    'The original test fails in both exact replays; no failure is regraded.', '',
    '`exploit` is the greedy trial. E is reported but excluded from keep and',
    'credit. `clause` references index the local node’s children. A equals',
    'supplied-answer cost plus weighted reference-closing credit. All supplied',
    'targets are unavailable (`None`) here. Full readings, open slots, programs,',
    'component costs and departures are in [arrival-keep.jsonl](arrival-keep.jsonl).', '']
arrivals, validation = [], {}
for label, original, failing_stream in CASES:
    stem = 'd1-'+label+'-before'
    prior, captured = HERE/(original+'-state')/NODE, HERE/(stem+'-state')/NODE
    rows = [json.loads(line) for line in (HERE/(stem+'-keep.jsonl')).read_text().splitlines()]
    assert len(rows) == 2 and rows[1]['batch'] == 2
    arrival = rows[1]
    assert arrival['sentences'] == ['x is three.']*8
    assert arrival['supplied_targets'] == [None]*16
    arrivals.append(arrival)
    a, b = load(prior/'initial-model.json'), load(captured/'initial-model.json')
    initial = {key: a[key] == b[key] for key in (
        'parameters', 'buffers', 'after_rng', 'source_manifest', 'torch_threads')}
    a, b = load(prior/'failure-store.json'), load(captured/'failure-store.json')
    final = {key: (a[key] == b[key] if key != 'types' else
        {k: sorted(v) for k, v in a[key].items()} ==
        {k: sorted(v) for k, v in b[key].items()}) for key in a}
    states = {name: same(torch.load(prior/name, weights_only=False),
                        torch.load(captured/name, weights_only=False))
              for name in ('initial-rng.pt', 'after-model-rng.pt', 'generator-0.pt')}
    outcome = load(captured/'outcome.json')
    assert outcome['outcome'] == 'failed'
    assert all(initial.values()) and all(final.values()) and all(states.values())
    credit = arrival['credit']
    validation[label] = dict(original=original, replay=stem, source=arrival['source'],
        initial=initial, final=final, rng_states=states, outcome=outcome['outcome'],
        failing_stream=failing_stream, costs=credit['costs'][failing_stream],
        components=credit['components'][failing_stream],
        keep=credit['keep_decision'][failing_stream], deciding=credit['deciding'][failing_stream])
    for stream in range(8):
        tables += [f'## {original}, document stream {stream}', '',
            f"Departure: compose round `{credit['departure']['compose_round'][stream]}`. "
            f"Kept: **{credit['keep_decision'][stream]}**. "
            f"Deciding: `{credit['deciding'][stream]}`.", '',
            '| Trial | R | E (excluded) | A | R + A | Reading before admission |',
            '|---|---:|---:|---:|---:|---|']
        for index, trial in enumerate(arrival['readings']):
            r, e, a = credit['components'][stream][index]
            tables.append(f"| {trial['trial']} | {r:.10f} | {e:.10f} | {a:.10f} | "
                f"{credit['costs'][stream][index]:.10f} | {graph_text(trial['clauses'][stream])} |")
        tables += ['']

(HERE/'arrival-keep.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in arrivals))
(HERE/'arrival-keep-tables.md').write_text('\n'.join(tables))
paths = subprocess.check_output(['git', 'ls-files', 'bin', 'test', 'data'], cwd=ROOT, text=True).splitlines()
changed = subprocess.check_output(['git', 'diff', '--name-only', '266189eda', '--',
                                   'bin', 'test', 'data'], cwd=ROOT, text=True).splitlines()
assert not changed
validation['source'] = dict(base='266189edab25985ac1d000b80553ee398f873b69',
    changed_production_test_data=changed,
    sha256={path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in paths if (ROOT/path).is_file()})
validation['probe_sha256'] = hashlib.sha256((HERE/'arrival_keep_probe.py').read_bytes()).hexdigest()
(HERE/'arrival-keep-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
print('Two exact failing replays; 16 document tables; source unchanged from 266189eda.')
