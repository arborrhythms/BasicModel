"""Summarize the retained run; no model execution or additional lessons."""
from collections import Counter
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
output = Path(__file__).resolve().parent
result = json.loads((run/'results.json').read_text())
audits = json.loads((run/'lesson-audits.json').read_text())
records = json.loads((run/'kept-programs.json').read_text())
progress = json.loads((run/'progress.json').read_text())
extra_calls, previous = [], 0
for batch in progress:
    if batch['training'] and batch['optimizer_steps']-previous != 2:
        extra_calls.append(dict(batch=batch['batch'], ids=batch['ids'],
            calls=batch['optimizer_steps']-previous))
    previous = batch['optimizer_steps']
epochs = {}
for epoch in range(2):
    selected = [row for row in audits if row['batch']//18 == epoch]
    epochs[epoch] = {}
    for name in ('compose.wait', 'compose.choice'):
        terms = [row['terms'][name] for row in selected if name in row['terms']]
        count = sum(term['count'] for term in terms)
        epochs[epoch][name] = dict(count=count,
            raw_ce=sum(term['raw']*term['count'] for term in terms)/count)
held = [row for row in records if not row['training']]
summary = dict(gates=result['gates'], groups=result['groups'],
    optimizer_steps=result['optimizer_steps'], elapsed_seconds=result['elapsed_seconds'],
    paired_updates=2*sum(row['training'] for row in progress),
    batches_with_extra_optimizer_calls=extra_calls,
    training_presentations=sum(row['training'] for row in records), heldout_presentations=len(held),
    lesson_calls=len(audits), unavailable=sum('compose.unavailable' in row['terms'] for row in audits),
    epoch_ce=epochs,
    programs=dict(Counter(','.join(row['observed']['program']) for row in held)),
    closed_meaning_norms=[row['observed']['closed_meaning_norm'] for row in held],
    description_orders=dict(Counter(str(order) for row in result['results']
        for np in row['nps'] for order in np['description_orders'])),
    source_order_by_group={family: sum(row['observed']['leaf_positions'] ==
        list(range(len(row['observed']['leaf_positions']))) for row in result['results']
        if row['family'] == family) for family in result['groups']})
(output/'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
