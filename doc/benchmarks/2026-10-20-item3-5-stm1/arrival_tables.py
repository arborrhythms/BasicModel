"""Render the saved arrival diagnostics without rerunning a model."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
lines = ['# Saved arrival replays after the sweep', '',
    'The unchanged certificates own the verdicts. These tables show each '
    'retained forward-reference occurrence and its working read after the '
    'arrival; retained content is not overwritten. Claude’s unsaved state '
    'is unavailable and is not claimed as replayed.', '']


def cell(value):
    return '`' + json.dumps(value, ensure_ascii=False).replace('|', '\\|') + '`'


for name, saved, run in (('pending-122', '32', 'native-122'),
                         ('pending-123', '75', 'native-123'),
                         ('full-pending-replay-125', 'full-sweep failure', 'native-125')):
    for path in sorted((root / name).glob('*/store.json')):
        data = json.loads(path.read_text())
        reverse = '[True]' in path.parent.name
        lines += [f'## Initialization {saved}, reverse={reverse}', '',
                  f'[Certificate]({run}/report.html); [native store]({path.relative_to(root)}).', '']
        for row in data['rows']:
            raw, working = row['raw'], row['working']
            if raw is None or not raw['bindings'].get('_forward_references'):
                continue
            lines += [f'### Occurrence row {row["index"]}', '',
                      '| Measurement | Result |', '|---|---|']
            values = {'stream': row['stream'],
                      'retained forward names': raw['bindings']['_forward_references'],
                      'retained references': raw['refs'],
                      'working references': working['refs'],
                      'working open slots': working['open'],
                      'working forward names': working['bindings'].get('_forward_references', [])}
            lines += [f'| {key} | {cell(value)} |' for key, value in values.items()]
            lines.append('')

(root / 'arrival-replays.md').write_text('\n'.join(lines))
