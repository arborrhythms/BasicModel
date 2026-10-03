"""Run declared probes/affected files with the standing resource limits."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-30-item7-review-round5')]
import bounded_tests as bounded
from resource_schedule import History, scheduled

out = HERE / sys.argv[1]
selectors = sys.argv[2:]
assert selectors
source = bounded.source_snapshot(ROOT)
out.mkdir(exist_ok=False)
bounded.write_json(out / 'source-manifest.json', dict(validated_source=source))
os.environ.pop('BASIC_SEED', None)
os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='1',
                  BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
history = History([HERE.parent / '2026-09-30-item7-review-round5/full-sweep/run/result.json'])
with scheduled(bounded, history=history, budget=24 * bounded.GIB,
               schedule_path=out / 'schedule.json'):
    result = bounded.run_suite(root=ROOT, selectors=selectors, run_dir=out / 'run',
        memory_bytes=24 * bounded.GIB, workers=10, worker_memory_bytes=8 * bounded.GIB,
        timeout=1800, suite_timeout=10800, batch_size=256, max_files=16)
assert source == bounded.source_snapshot(ROOT)
print(json.dumps(dict(reason=result['reason'], selected=len(result['selected']),
                     completed=len(result['completed']))))
raise SystemExit(result['exit_code'])
