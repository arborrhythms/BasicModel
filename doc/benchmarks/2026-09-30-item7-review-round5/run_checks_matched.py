"""Fresh bounded files, with one diagnostic repeat for each memory stop."""
import hashlib
import json
import os
from pathlib import Path
import sys

from run_xor_matched import ROOT, HERE, diagnostic
from bounded_tests import GIB, run_suite, source_snapshot


def main():
    output = HERE / sys.argv[1]
    output.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    requested = sys.argv[3:]
    probes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for selector in requested if (path := Path(selector.split('::')[0])).is_absolute()}
    (output / 'source-manifest.json').write_text(json.dumps(dict(
        validated_source=source, external_probes=probes), indent=2) + '\n')
    os.environ.pop('BASIC_SEED', None)
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager',
                      BASIC_AUTOLOAD='false', RUN_SLOW=os.environ.get('ITEM7_RUN_SLOW', '1'),
                      PYTHONPATH=os.pathsep.join((str(ROOT / 'bin'), str(ROOT / 'test'))))
    result = dict(groups=[], diagnostic_only=[], reason='running')
    for index, selector in enumerate(requested):
        group = run_suite(root=ROOT, selectors=[selector], run_dir=output / f'group-{index:02}',
                          memory_bytes=8 * GIB, workers=1, worker_memory_bytes=8 * GIB,
                          timeout=1800, suite_timeout=2100, batch_size=32, max_files=1)
        assert source_snapshot(ROOT) == source, 'tested source changed'
        result['groups'].append(dict(selector=selector, reason=group['reason'],
                                     exit_code=group['exit_code'], receipt=f'group-{index:02}/result.json'))
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        for worker in group['workers']:
            if worker['reason'] in ('memory', 'aggregate_memory'):
                progress = Path(worker['log']).with_suffix('.json')
                active = json.loads(progress.read_text()).get('active') if progress.exists() else None
                if active:
                    result['diagnostic_only'].append(diagnostic(active, output / f'unguarded-{index:02}'))
        assert source_snapshot(ROOT) == source, 'tested source changed during diagnostic'
    result['reason'] = 'failed' if any(g['exit_code'] for g in result['groups']) else 'passed'
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return int(result['reason'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
