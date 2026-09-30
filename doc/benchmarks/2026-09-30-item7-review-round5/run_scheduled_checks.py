"""One collection of the required affected files, with bounded concurrent workers."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-28-item7-review')]
import bounded_tests as bounded
from resource_schedule import History, scheduled
from review_source import supporting_inputs
from matching_diagnostic import diagnostic


def historical_results():
    paths = [HERE.parent / '2026-09-30-item7-review-round4/full-sweep/run/result.json']
    for label in ('thought-reasoning', 'final-item7'):
        paths.extend(sorted((HERE / label).glob('group-*/result.json')))
    return paths


def main():
    label = sys.argv[1]
    plan = json.loads((HERE / 'final-validation-plan.json').read_text())
    selectors = plan[label] if label in plan else sys.argv[2:]
    assert selectors
    source = json.loads((HERE / 'final-source.json').read_text())
    inputs = json.loads((HERE / 'final-inputs.json').read_text())
    assert bounded.source_snapshot(ROOT) == source
    assert supporting_inputs(ROOT) == inputs
    output = HERE / label
    output.mkdir(exist_ok=False)
    (output / 'source-manifest.json').write_text(json.dumps(dict(
        validated_source=source, supporting_inputs=inputs), indent=2) + '\n')
    aggregate = dict(groups=[], diagnostic_only=[], reason='running')
    bounded.write_json(output / 'result.json', aggregate)
    os.environ.pop('BASIC_SEED', None)
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                      BASIC_AUTOLOAD='false', PYTHONPATH=os.pathsep.join((str(ROOT / 'bin'), str(ROOT / 'test'))))
    # The selected-file pool retains its original 8 GiB aggregate limit.
    # With the two existing 8 GiB XOR/MM workers, the campaign stays within 24 GiB.
    with scheduled(bounded, history=History(historical_results()), budget=8 * bounded.GIB,
                   schedule_path=output / 'schedule.json'):
        run = bounded.run_suite(root=ROOT, selectors=selectors, run_dir=output / 'run',
            memory_bytes=8 * bounded.GIB, workers=10, worker_memory_bytes=8 * bounded.GIB,
            timeout=1800, suite_timeout=10800, batch_size=256, max_files=16)
    assert bounded.source_snapshot(ROOT) == source
    assert supporting_inputs(ROOT) == inputs
    aggregate['groups'].append(dict(selectors=selectors, reason=run['reason'],
        exit_code=run['exit_code'], receipt='run/result.json'))
    for index, worker in enumerate(run['workers']):
        if worker['reason'] not in ('memory', 'aggregate_memory'):
            continue
        progress = Path(worker['log']).with_suffix('.json')
        active = json.loads(progress.read_text()).get('active') if progress.exists() else None
        if active:
            aggregate['diagnostic_only'].append(diagnostic(active, output / f'unguarded-{index:03}'))
    aggregate['reason'] = 'passed' if run['exit_code'] == 0 else 'failed'
    bounded.write_json(output / 'result.json', aggregate)
    assert bounded.source_snapshot(ROOT) == source
    assert supporting_inputs(ROOT) == inputs
    print(json.dumps(dict(reason=run['reason'], selected=len(run['selected']),
        completed=len(run['completed']), elapsed_seconds=run['elapsed_seconds'])), flush=True)
    return run['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
