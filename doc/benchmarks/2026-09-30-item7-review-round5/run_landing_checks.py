"""Run only the fixture-port checks authorized by item 7, section 25."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LANDING = HERE / 'landing'
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-28-item7-review')]
import bounded_tests as bounded
from resource_schedule import History, scheduled
from review_source import supporting_inputs
from matching_diagnostic import diagnostic


def main():
    plan = json.loads((LANDING / 'validation-plan.json').read_text())
    source = json.loads((LANDING / 'source-manifest.json').read_text())
    inputs = json.loads((HERE / 'final-inputs.json').read_text())
    assert bounded.source_snapshot(ROOT) == source
    assert supporting_inputs(ROOT) == inputs
    os.environ.pop('BASIC_SEED', None)
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                      BASIC_AUTOLOAD='false',
                      PYTHONPATH=os.pathsep.join((str(ROOT / 'bin'), str(ROOT / 'test'))))
    for label in sys.argv[1:]:
        output = LANDING / label
        output.mkdir(exist_ok=False)
        with scheduled(bounded, history=History([HERE / 'full-sweep/run/result.json']),
                       budget=8 * bounded.GIB, schedule_path=output / 'schedule.json'):
            result = bounded.run_suite(root=ROOT, selectors=plan[label],
                run_dir=output / 'run', memory_bytes=8 * bounded.GIB,
                workers=10, worker_memory_bytes=8 * bounded.GIB,
                timeout=1800, suite_timeout=10800, batch_size=256, max_files=16)
        diagnostics = []
        for index, worker in enumerate(result['workers']):
            if worker['reason'] not in ('memory', 'aggregate_memory'):
                continue
            progress = Path(worker['log']).with_suffix('.json')
            active = json.loads(progress.read_text()).get('active') if progress.exists() else None
            if active:
                diagnostics.append(diagnostic(active, output / f'unguarded-{index:03}'))
        bounded.write_json(output / 'diagnostics.json', diagnostics)
        assert bounded.source_snapshot(ROOT) == source
        assert supporting_inputs(ROOT) == inputs
        print(json.dumps(dict(selection=label, reason=result['reason'],
            selected=len(result['selected']), completed=len(result['completed']),
            elapsed_seconds=result['elapsed_seconds'])), flush=True)
        if result['exit_code']:
            return result['exit_code']
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
