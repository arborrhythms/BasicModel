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
    gates = json.loads((HERE / 'explicit-after-owner/result.json').read_text())
    assert gates['reason'] != 'running'
    gate_manifest = json.loads((HERE / 'explicit-after-owner/source-manifest.json').read_text())
    assert gate_manifest['validated_source'] == source
    assert gate_manifest['supporting_inputs'] == inputs
    for label in ('head', 'candidate'):
        processes = json.loads((HERE / (label + '-reconstruction' + ('-final' if label == 'candidate' else '')) / 'processes.json').read_text())
        assert set(processes) == set(map(str, range(8)))
        trials = json.loads((HERE / (label + '-mm-grammar' + ('-final' if label == 'candidate' else '')) / 'processes.json').read_text())
        assert set(trials) == set(map(str, range(10)))
    assert json.loads((HERE / 'candidate-reconstruction-final/source-manifest.json').read_text()) == source
    assert json.loads((HERE / 'candidate-mm-grammar-final/source-manifest.json').read_text())['validated_source'] == source
    assert json.loads((HERE / 'final-xor-after-owner/source-manifest.json').read_text())['validated_source'] == source
    assert json.loads((HERE / 'graph-release-head/result.json').read_text())['reason'] != 'running'
    output.mkdir(exist_ok=False)
    (output / 'source-manifest.json').write_text(json.dumps(dict(
        validated_source=source, supporting_inputs=inputs), indent=2) + '\n')
    timings, history = defaultdict(float), {}
    for path in (HERE.parent / '2026-09-28-item7-review-round2/full-sweep/combined-result.json',):
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
    print(json.dumps(dict(reason=result['reason'], exit_code=result['exit_code'],
        selected=len(result['selected']), completed=len(result['completed']))), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
