"""Dispatch the one full sweep after required measurements finish.

Historical tables run first so their unguarded diagnostics do not compete
with the sweep's bounded workers. The tested runtime stays frozen.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot

source = json.loads((HERE / 'ai-native-id-source.json').read_text())
assert source_snapshot(ROOT) == source
print('Waiting for reconstruction completion and final/historical XOR receipts', flush=True)
while True:
    ready = (HERE / 'final-reconstruction-complete.json').exists()
    for name in ('ah-candidate', 'ai-candidate', 'final3-xor-candidate', 'explicit-final3'):
        path = HERE / name / 'result.json'
        try:
            ready = ready and path.exists() and json.loads(path.read_text())['reason'] != 'running'
        except json.JSONDecodeError:
            ready = False
    if ready:
        break
    time.sleep(2)
assert source_snapshot(ROOT) == source
for label in ('ah-candidate', 'ai-candidate', 'final3-xor-candidate'):
    subprocess.run([sys.executable, str(HERE / 'summarize_xor.py'), str(HERE / label)], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(HERE / 'summarize_xor_comparison.py')], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(HERE / 'summarize_final_measurements.py'), 'reconstruction'], cwd=ROOT, check=True)
print('All prerequisites complete; starting the one full sweep', flush=True)
result = subprocess.run([sys.executable, str(HERE / 'run_full_sweep_final3.py')], cwd=ROOT)
print('Full sweep dispatch returned', result.returncode, flush=True)
raise SystemExit(result.returncode)
