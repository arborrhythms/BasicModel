"""Repeat only incomplete time-limited HEAD measurements, with the same limits."""
from pathlib import Path
import json,os,sys,time
here=Path(__file__).resolve().parent
original=here/'final-reconstruction-head'
while not (original/'driver-hashes.json').exists():time.sleep(2)
trials=json.loads((original/'processes.json').read_text())
seeds=sorted(int(seed) for seed,trial in trials.items() if trial['reason']=='timeout')
(here/'head-reconstruction-retry-decision.json').write_text(json.dumps(dict(
    seeds=seeds,reason='Complete only the declared seeds whose measurement timed out; no completed learning measurement is repeated, and the 8 GiB / 1200 second limits remain unchanged.'),indent=2)+'\n')
if seeds:
 os.execv(sys.executable,[sys.executable,str(here/'run_reconstruction_retries.py'),
    'reconstruction-head-timeout-retry',sys.argv[1],','.join(map(str,seeds))])
