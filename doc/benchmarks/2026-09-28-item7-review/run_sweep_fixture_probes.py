"""Bounded fixture probes in an isolated copy while the main sweep is frozen."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('ITEM7_FIXTURE_ROOT') or
            json.loads((HERE/'sweep-fixtures/base.json').read_text())['root'])
sys.path.insert(0, str(ROOT/'test'))
import bounded_tests as bounded

os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                  BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT/'bin'))
result = bounded.run_suite(root=ROOT, selectors=sys.argv[2:] or [
    'test/test_type_run_spans.py',
    'test/test_meronomy_ladder.py::test_word_and_grammatical_cuts_equal_the_meronomy_cut',
    'test/test_meronomy_ladder.py::test_tiling_ladder_nests_units_in_space_bounded_wholes',
], run_dir=HERE/'sweep-fixtures'/sys.argv[1], memory_bytes=8*bounded.GIB,
    workers=1, worker_memory_bytes=8*bounded.GIB, timeout=1800,
    suite_timeout=5400, batch_size=32, max_files=1)
print(json.dumps(dict(reason=result['reason'], exit_code=result['exit_code'],
    selected=len(result['selected']), completed=len(result['completed']))))
raise SystemExit(result['exit_code'])
