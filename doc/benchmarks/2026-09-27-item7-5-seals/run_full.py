"""One complete bounded sweep; same limits, expensive files first.

The prior receipt hit the 8 GiB cap when relative-STM cases shared a worker.
Isolate those cases and native output cases with fresh reconstruction graphs.
No selection, assertion, skip, seed or cap changes.
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
    previous = HERE.parent / '2026-09-26-item7-5'
    costs = defaultdict(float)
    inputs = {}
    for segment in ('full', 'full-memory-remainder', 'full-continuation'):
        path = previous / segment / 'result.json.gz'
        raw = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        result = json.loads(gzip.decompress(raw))
        for worker in result['workers']:
            for report in worker.get('reports', []):
                costs[report['nodeid'].split('::', 1)[0]] += report.get('duration', 0.)
    # The new sentence objectives add two native backwards. Use completed
    # affected-case timings as well as the old full sweep for dispatch order.
    path = HERE / 'affected-verified/result.json'
    raw = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    current_costs = defaultdict(float)
    for worker in json.loads(raw)['workers']:
        for report in worker.get('reports', []):
            current_costs[report['nodeid'].split('::', 1)[0]] += report.get('duration', 0.)
    for path, seconds in current_costs.items():
        costs[path] = max(costs[path], seconds)
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
        code, _ = bounded.run_suite(root=ROOT, selectors=[], run_dir=HERE / 'full',
            memory_bytes=24 * bounded.GIB, workers=3, worker_memory_bytes=8 * bounded.GIB,
            timeout=1800, suite_timeout=10800, batch_size=32, max_files=1)
        return code
    finally:
        bounded.make_batches = ordinary


if __name__ == '__main__':
    raise SystemExit(main())
