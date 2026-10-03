"""Readable comparisons without replacing any raw attempt or historical receipt."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

stage = Path(sys.argv[1]).resolve()
trees = {}
for label in ('head', 'candidate'):
    path = stage / label
    summary = json.loads((path / 'summary.json').read_text())
    rows = []
    for group in summary['groups']:
        for node, outcome in group['outcomes'].items():
            row = dict(nodeid=node, gate=group['gate'], outcome=outcome,
                       peak_gib=group['peak_memory_bytes'] / 2**30,
                       observations=[o for o in group['observations'] if o.get('nodeid') == node])
            if node.endswith('::test_mm20m_xor_exact_roundtrip'):
                row['trial'] = group['gate'] - 13
            for observed in row['observations']:
                if observed['kind'] == 'grammar':
                    observed['mse'] = sum((a-b)**2 for a,b in zip(observed['predictions'], observed['targets'])) / 4
                    observed['correct'] = sum((a > .5) == (b > .5) for a,b in zip(observed['predictions'], observed['targets']))
                    # The report prints an empty string for each absent
                    # decoded row. Preserve that fact, not a made-up decode.
                    observed['reported_reconstructions'] = observed['reconstructions'] + [''] * (4-len(observed['reconstructions']))
            rows.append(row)
    assert len(rows) == 49, (label, len(rows))
    trees[label] = rows
assert [(r['nodeid'], r.get('trial')) for r in trees['head']] == [(r['nodeid'], r.get('trial')) for r in trees['candidate']]
lines = ['# Named XOR measurements', '',
         'Fresh unseeded attempts on the frozen HEAD and candidate source manifests. '
         'Each table includes all fifteen exact round trips and the explicit slow proofs.', '',
         '| Proof | HEAD | Candidate | Candidate peak GiB |', '|---|---|---|---|']
for head, candidate in zip(trees['head'], trees['candidate']):
    name = head['nodeid'].removeprefix('test/')
    if 'trial' in head:
        name += f" — trial {head['trial']}/15"
    lines.append(f"| {name} | {head['outcome']} | {candidate['outcome']} | {candidate['peak_gib']:.2f} |")
lines += ['', '[HEAD details](head/table.md); [candidate details](candidate/table.md).', '']
for label, rows in trees.items():
    exact = Counter(r['outcome'] for r in rows if 'trial' in r)
    lines.append(f"{label}: {dict(Counter(r['outcome'] for r in rows))}; exact round trips {dict(exact)}.")
    for row in rows:
        for observed in row['observations']:
            if observed['kind'] == 'grammar':
                lines += ['', f"{label}, `{row['nodeid'].split('::')[-2]}`: "
                    f"MSE {observed['mse']:.10f}, {observed['correct']}/4 answers correct. "
                    f"Answers {observed['predictions']}; reported reconstructions {observed['reported_reconstructions']}."]
(stage / 'comparison.json').write_text(json.dumps(trees, indent=2) + '\n')
(stage / 'comparison.md').write_text('\n'.join(lines) + '\n')
if (stage / 'head/mm').exists():
    mm = {label: [json.loads((stage / label / f'mm/run-{i:02}.json').read_text()) for i in range(10)] for label in trees}
    assert all(r['completed_epochs'] == 900 for rows in mm.values() for r in rows)
    lines = ['# Ten full MM_grammar runs', '', 'Fresh unseeded 900-epoch measurements. The configuration, learning rate and 8 GiB guard are unchanged.', '',
             '| Run | HEAD ending MSE | Candidate ending MSE |', '|---|---|---|']
    for i in range(10):
        lines.append(f"| {i+1} | {mm['head'][i]['after_900_updates_mse']:.10f} | {mm['candidate'][i]['after_900_updates_mse']:.10f} |")
    for label in mm:
        lines += ['', f"{label} median: {statistics.median(r['after_900_updates_mse'] for r in mm[label]):.10f}."]
    (stage / 'mm-table.json').write_text(json.dumps(mm, indent=2) + '\n')
    (stage / 'mm-table.md').write_text('\n'.join(lines) + '\n')
