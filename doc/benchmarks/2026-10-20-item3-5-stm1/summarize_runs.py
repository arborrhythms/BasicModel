"""Render completed native attempts without merging different source snapshots."""
from collections import Counter
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for result in root.glob('*/result.json'):
    data = json.loads(result.read_text())
    outcomes = {}
    for report in result.parent.glob('worker-???.json'):
        for item in json.loads(report.read_text()).get('reports', ()):
            if item['phase'] == 'call' or item['outcome'] in ('failed', 'skipped'):
                previous = outcomes.get(item['nodeid'])
                if previous != 'failed':
                    outcomes[item['nodeid']] = item['outcome']
    count = Counter(outcomes.values())
    rows.append((result.stat().st_mtime, result.parent.name, data, count))
text = [
    '# Scoped test runs', '',
    'Each row is its own source snapshot. Passes from different snapshots are '
    'not combined into a landing result. These are mechanism certificates, '
    'not learning claims.', '',
    '| Run | Completed/selected | Passed | Failed | Skipped | Result |',
    '|---|---:|---:|---:|---:|---|',
]
for _, name, data, count in sorted(rows):
    text.append(f'| [{name}]({name}/report.html) | '
                f'{len(data["completed"])}/{len(data["selected"])} | '
                f'{count["passed"]} | {count["failed"]} | {count["skipped"]} | '
                f'{data["reason"]} |')
text += ['', '`math-12` changed source during execution and is diagnostic only. '
    '`decomposition-diagnostic-11-run` and `decomposition-diagnostic-12-run` '
    'failed in their diagnostic plugin because `CLOSING_TRACE` was absent; '
    'neither is a decomposition verdict. Attempt `stm1-13` used the complete '
    'runner and records both actual certificate failures.',
    '', 'No full landing sweep or publication has occurred in this round.', '']
(root/'runs.md').write_text('\n'.join(text))
