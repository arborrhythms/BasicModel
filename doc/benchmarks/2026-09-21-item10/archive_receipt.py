"""Preserve a bounded receipt and its source map; distinguish unique cases."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def archive(run, prefix, log=None):
    dest = Path(__file__).parent
    result = json.loads((run / 'result.json').read_text())
    manifest = json.loads((run / 'source-manifest.json').read_text())
    reports = [r for w in result['workers'] for r in w['reports']]
    outcomes = {}
    for report in reports:
        node, outcome = report['nodeid'], report['outcome']
        if outcomes.get(node) not in ('failed', 'error'):
            outcomes[node] = outcome
    source = manifest['validated_source']
    summary = {key: result[key] for key in ('selected','completed','exit_code','reason','elapsed_seconds',
                                          'peak_aggregate_memory_bytes','compile_cache_retries','limits')}
    summary['selected'] = len(result['selected'])
    summary['completed'] = len(result['completed'])
    summary.update(run=run.name, outcomes=dict(Counter(outcomes.values())),
                   source_files=len(source),
                   source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                   phase_reports=dict(Counter(r['outcome'] for r in reports)),
                   failures=[r for r in reports if r['outcome'] in ('failed', 'error')])
    with gzip.open(dest / f'{prefix}-result.json.gz', 'wb') as out:
        out.write((run / 'result.json').read_bytes())
    shutil.copy2(run / 'source-manifest.json', dest / f'{prefix}-source-manifest.json')
    (dest / f'{prefix}-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if log:
        shutil.copy2(log, dest / f'{prefix}-run.txt')
    print(prefix, summary['outcomes'], summary['selected'], summary['completed'], summary['exit_code'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('prefix')
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    archive(args.run, args.prefix, args.log)
