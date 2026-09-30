"""Serialize large-memory XOR receipts behind named frozen-source receipts."""
import json
import os
from pathlib import Path
import sys
import time

here = Path(__file__).resolve().parent
label, root, *prerequisites = sys.argv[1:]
sys.path.insert(0, str(Path(root) / 'test'))
from bounded_tests import source_snapshot
source = source_snapshot(Path(root))
print(f'{label}: waiting for {prerequisites}', flush=True)
while True:
    ready = True
    for name in prerequisites:
        path = here / name / 'result.json'
        if not path.exists() or json.loads(path.read_text()).get('reason') == 'running':
            ready = False
            break
    if ready:
        break
    time.sleep(2)
assert source_snapshot(Path(root)) == source, 'queued measurement source changed'
os.execv(sys.executable, [sys.executable, str(here / 'run_xor.py'), label, root])
