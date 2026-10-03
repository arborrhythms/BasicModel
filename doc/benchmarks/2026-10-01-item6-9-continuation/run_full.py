"""One final source-matched full sweep; no statistical retries or guard waiver."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / '2026-09-30-item7-review-round5'
sys.path[:0] = [str(ROOT / 'test'), str(PRIOR),
               str(HERE.parent / '2026-09-28-item7-review')]
import bounded_tests as bounded
from resource_schedule import History, scheduled
from review_source import supporting_inputs

out = HERE / 'full-sweep'
out.mkdir(exist_ok=False)
source = bounded.source_snapshot(ROOT)
inputs = supporting_inputs(ROOT)
assert source == json.loads((HERE / 'final-source.json').read_text())
assert inputs == json.loads((HERE / 'final-inputs.json').read_text())
bounded.write_json(out / 'source-manifest.json', dict(validated_source=source,
    supporting_inputs=inputs))
os.environ.pop('BASIC_SEED', None)
os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                  BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
with scheduled(bounded, history=History([PRIOR / 'full-sweep/run/result.json']),
               budget=24 * bounded.GIB, schedule_path=out / 'schedule.json'):
    result = bounded.run_suite(root=ROOT, selectors=[], run_dir=out / 'run',
        memory_bytes=24 * bounded.GIB, workers=10, worker_memory_bytes=8 * bounded.GIB,
        timeout=1800, suite_timeout=10800, batch_size=256, max_files=16)
assert source == bounded.source_snapshot(ROOT)
assert inputs == supporting_inputs(ROOT)
summary = dict(reason=result['reason'], exit_code=result['exit_code'],
    selected=len(result['selected']), completed=len(result['completed']),
    missing=sorted(set(result['selected']) - set(result['completed'])),
    duplicate_completed={n:c for n,c in Counter(result['completed']).items() if c != 1},
    resource_stops=[dict(reason=w['reason'], log=w['log']) for w in result['workers']
                    if w['reason'] not in ('exit', 'recycle')])
bounded.write_json(out / 'coverage.json', summary)
print(json.dumps(summary))
raise SystemExit(result['exit_code'])
