"""Expose both already-running measurement families to the grammar guard."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded

parser = argparse.ArgumentParser()
parser.add_argument('--native-supervisor-pid', required=True, type=int)
args = parser.parse_args()
out = HERE / 'combined-companions.json'
while True:
    campaign = json.loads((HERE / 'step5/progress.json').read_text())
    active = list(campaign['active'])
    native_done = (HERE / 'native-after/driver.process.json').exists()
    if not native_done:
        active.append(dict(label='native-after', kind='native',
                           pid=args.native_supervisor_pid))
    bounded.write_json(out, dict(active=active, time=time.time(),
        note='Grammar guard samples all these process trees plus its own workers.'))
    if not active and native_done and (HERE / 'step5/complete.json').exists():
        break
    time.sleep(.25)
