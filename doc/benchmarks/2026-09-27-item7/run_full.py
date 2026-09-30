"""One accepted-source sweep, using the preceding sweep's file costs.

Keep the reviewed resource caps and isolate the same expensive native output
and relative-STM cases. Ordering changes dispatch only, never case selection,
assertions, skips or seeds.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded


def main():
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                      BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
    costs, inputs = defaultdict(float), {}
    for path in (HERE.parent / '2026-09-27-item7-5-pressure/full/result.json.gz',
                 HERE.parent / '2026-09-27-item7-5-landing/fixtures/result.json'):
        if not path.exists():
            path = path.with_suffix(path.suffix + '.gz')
        raw = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        result = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
        measured = defaultdict(float)
        for worker in result['workers']:
            for report in worker.get('reports', []):
                measured[report['nodeid'].split('::', 1)[0]] += report.get('duration', 0.)
        for name, seconds in measured.items():
            costs[name] = max(costs[name], seconds)
    isolated = {'test/test_stm_relative_sentence_end_state.py',
                'test/test_generation_catalog.py', 'test/test_output_walk.py',
                'test/test_output_path_supervised.py'}
    ordinary = bounded.make_batches

    def schedule(nodes, batch_size, max_files, devices):
        by_file = defaultdict(list)
        for node in nodes:
            by_file[node.split('::', 1)[0]].append(node)
        files = sorted(by_file, key=lambda p: (-costs[p], p))
        batches = []
        for path in files:
            size = 1 if path in isolated else batch_size
            batches.extend(ordinary(by_file[path], size, 1, devices))
        assert Counter(n for batch in batches for n in batch) == Counter(nodes)
        (HERE / 'full-schedule.json').write_text(json.dumps(dict(
            rationale=__doc__, historical_inputs=inputs, expected_cases=len(nodes),
            files=[dict(path=p, prior_seconds=costs[p]) for p in files],
            batches=batches), indent=2) + '\n')
        return batches

    bounded.make_batches = schedule
    try:
        result = bounded.run_suite(root=ROOT, selectors=[], run_dir=HERE / 'full',
            memory_bytes=24 * bounded.GIB, workers=3, worker_memory_bytes=8 * bounded.GIB,
            timeout=1800, suite_timeout=10800, batch_size=32, max_files=1)
        print(json.dumps(dict(exit_code=result['exit_code'], reason=result['reason'],
            selected=len(result['selected']), completed=len(result['completed']))), flush=True)
        return result['exit_code']
    finally:
        bounded.make_batches = ordinary


if __name__ == '__main__':
    raise SystemExit(main())
