"""One source-matched complete sweep, after all declared measurements/gates.

Prior timings choose dispatch and batch boundaries only. Every collected
selector runs once; no assertions, seeds, skips or thresholds are changed.
"""
from collections import Counter, defaultdict
import hashlib
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
    timings, history = defaultdict(float), {}
    for path in (HERE.parent / '2026-09-30-item7-review-round4/full-sweep/run/result.json',):
        raw = path.read_bytes()
        history[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        for worker in json.loads(raw)['workers']:
            for report in worker.get('reports', ()):
                timings[report['nodeid']] = max(timings[report['nodeid']], report.get('duration', 0.))
    isolated = {'test/test_stm_relative_sentence_end_state.py',
                'test/test_generation_catalog.py', 'test/test_output_walk.py',
                'test/test_output_path_supervised.py'}
    ordinary = bounded.make_batches
    def schedule(nodes, batch_size, max_files, devices):
        by_file = defaultdict(list)
        for node in nodes:
            by_file[node.split('::', 1)[0]].append(node)
        files = sorted(by_file, key=lambda path: (-sum(timings[node] for node in by_file[path]), path))
        batches = []
        for path in files:
            group = []
            duration = 0.
            for node in by_file[path]:
                # Expensive native tests get a fresh worker with the SAME
                # 1800-second cap. Small cases retain file-local fixtures.
                solo = path in isolated or timings[node] > 120
                if group and (solo or len(group) == batch_size or duration + timings[node] > 600):
                    batches.extend(ordinary(group, batch_size, 1, devices))
                    group, duration = [], 0.
                if solo:
                    batches.append([node])
                else:
                    group.append(node)
                    duration += timings[node]
            if group:
                batches.extend(ordinary(group, batch_size, 1, devices))
        assert Counter(node for batch in batches for node in batch) == Counter(nodes)
        (output / 'schedule.json').write_text(json.dumps(dict(
            rationale=__doc__, historical_inputs=history, batches=batches,
            expected_cases=len(nodes)), indent=2) + '\n')
        return batches
    os.environ.pop('BASIC_SEED', None)
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                      BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
    bounded.make_batches = schedule
    try:
        result = bounded.run_suite(root=ROOT, selectors=[], run_dir=output / 'run',
            memory_bytes=24 * bounded.GIB, workers=3, worker_memory_bytes=8 * bounded.GIB,
            timeout=1800, suite_timeout=10800, batch_size=32, max_files=1)
    finally:
        bounded.make_batches = ordinary
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
