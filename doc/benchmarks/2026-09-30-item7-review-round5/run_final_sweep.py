"""One source-matched complete sweep, after all declared measurements/gates.

Prior timings choose dispatch and batch boundaries only. Every collected
selector runs once; no assertions, seeds, skips or thresholds are changed.
"""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / '2026-09-28-item7-review'
sys.path[:0] = [str(ROOT / 'test'), str(PRIOR)]
import bounded_tests as bounded
from review_source import supporting_inputs
from resource_schedule import History, scheduled
from run_scheduled_checks import historical_results


def main():
    output = HERE / 'full-sweep'
    source = bounded.source_snapshot(ROOT)
    inputs = supporting_inputs(ROOT)
    assert source == json.loads((HERE / 'final-source.json').read_text())
    assert inputs == json.loads((HERE / 'final-inputs.json').read_text())
    for label in ('final-item7', 'final-thought-reasoning', 'final-graph-release', 'final-xor-candidate'):
        result = json.loads((HERE / label / 'result.json').read_text())
        assert result['reason'] != 'running', label
        assert json.loads((HERE / label / 'source-manifest.json').read_text())['validated_source'] == source
        if label != 'final-xor-candidate':
            assert result['reason'] == 'passed', (label, 'requires inspection before full sweep')
    trials = json.loads((HERE / 'final-mm-grammar-candidate/processes.json').read_text())
    assert set(trials) == set(map(str, range(10)))
    assert all(value['reason'] == 'exit' and value['exit_code'] == 0 for value in trials.values())
    assert json.loads((HERE / 'final-mm-grammar-candidate/source-manifest.json').read_text())['validated_source'] == source
    output.mkdir(exist_ok=False)
    (output / 'source-manifest.json').write_text(json.dumps(dict(
        validated_source=source, supporting_inputs=inputs), indent=2) + '\n')
    os.environ.pop('BASIC_SEED', None)
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                      BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
    with scheduled(bounded, history=History(historical_results()), budget=24 * bounded.GIB,
                   schedule_path=output / 'schedule.json'):
        result = bounded.run_suite(root=ROOT, selectors=[], run_dir=output / 'run',
            memory_bytes=24 * bounded.GIB, workers=10, worker_memory_bytes=8 * bounded.GIB,
            timeout=1800, suite_timeout=10800, batch_size=256, max_files=16)
    assert source == bounded.source_snapshot(ROOT)
    assert inputs == supporting_inputs(ROOT)
    # Each worker stopped by the unchanged guard gets one diagnostic only.
    from matching_diagnostic import diagnostic
    repeats = []
    for index, worker in enumerate(result['workers']):
        if worker['reason'] not in ('memory', 'aggregate_memory'):
            continue
        path = Path(worker['log']).with_suffix('.json')
        active = json.loads(path.read_text()).get('active') if path.exists() else None
        if active:
            repeats.append(diagnostic(active, output / f'unguarded-{index:03}'))
    (output / 'diagnostics.json').write_text(json.dumps(repeats, indent=2) + '\n')
    assert source == bounded.source_snapshot(ROOT)
    assert inputs == supporting_inputs(ROOT)
    print(json.dumps(dict(reason=result['reason'], exit_code=result['exit_code'],
        selected=len(result['selected']), completed=len(result['completed']))), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
