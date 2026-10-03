"""Render every answer and gate reconstruction from the fixed ten-run campaign."""
from collections import Counter
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
lines = ['# Ten fresh runs of each XOR_grammar gate', '',
    summary['decision'] + '.', '',
    f"Class bar met in {summary['class_successes']}/10 runs: four correct answers and MSE < .05.", '',
    'Every run uses the unchanged 400-epoch fixture, no chosen seed, and the 8 GiB guard. '
    'Answers are the saved final evaluation. Reconstructions are the strings read by the current gate; '
    'their producer is unchanged until plan step 6.', '',
    '| Gate | Run | Result | MSE | Correct | Four answers | Four reconstructions |',
    '|---|---|---|---|---|---|---|']
for row in summary['rows']:
    label = 'class' if row['gate'] == 5 else 'reconstruction'
    result = 'passed' if row['process']['exit_code'] == 0 else 'failed'
    relative = f"gate-{row['gate']:02}-trial-{row['trial']-1:02}/run/result.json"
    if len(row['observations']) != 1:
        lines.append(f"| {label} | {row['trial']} | [{result}]({relative}) | unavailable | unavailable | unavailable | unavailable |")
        continue
    observed = row['observations'][0]
    answers = ', '.join(f'{value:.9f}' for value in observed['predictions'])
    strings = json.dumps(observed['gate_reconstructions'], ensure_ascii=True).replace('|', '&#124;')
    lines.append(f"| {label} | {row['trial']} | [{result}]({relative}) | {row['mse']:.10f} | {row['correct']}/4 | {answers} | `{strings}` |")
lines += ['', '[Full-precision observations, process receipts and stop decision](summary.json).', '']
(root / 'table.md').write_text('\n'.join(lines))
