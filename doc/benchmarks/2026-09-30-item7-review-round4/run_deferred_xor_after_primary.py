"""Release historical XOR after the two-worker reconstruction pool finishes.

Only one serial reconstruction retry can then overlap one historical table.
The full sweep still waits for every required campaign to finish.
"""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
label, root, *dependencies = sys.argv[1:]
sys.path.insert(0, str(Path(root) / 'test'))
from bounded_tests import source_snapshot
source = source_snapshot(Path(root))
print(label + ': waiting for the primary reconstruction pool and ' + repr(dependencies), flush=True)
while not (HERE / 'final3-reconstruction-candidate/driver-hashes.json').exists():
    time.sleep(2)
for name in dependencies:
    path = HERE / name / 'result.json'
    while not path.exists() or json.loads(path.read_text()).get('reason') == 'running':
        time.sleep(2)
assert source_snapshot(Path(root)) == source
os.execv(sys.executable, [sys.executable, str(HERE / 'run_xor_matched.py'), label, root])
