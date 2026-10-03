"""Run a receipt command once under the unchanged per-worker guard."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import run_guarded

label, *command = sys.argv[1:]
folder = Path(__file__).parent / 'repairs'
record = folder / (label + '.json')
if record.exists():
    raise RuntimeError('a receipt run label cannot overwrite an earlier attempt')
result = run_guarded(command, cwd=str(ROOT), env=os.environ.copy(),
    log_path=folder / (label + '.log'), memory_bytes=8 * 1024**3, timeout=1800)
record.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
raise SystemExit(result['exit_code'])
