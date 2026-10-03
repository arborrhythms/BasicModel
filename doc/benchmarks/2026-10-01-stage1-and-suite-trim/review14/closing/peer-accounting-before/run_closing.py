"""The final candidate receipt, in order; no retries of measured cases."""
from pathlib import Path
import json,subprocess,sys,time
H=Path(__file__).resolve().parent;ROOT=H.parents[4]
started=time.monotonic()
steps=[('measurements',['measure.py','campaign']),('measurement_summary',['summarize.py']),('full_sweep',['full_sweep.py']),('sweep_summary',['summarize_sweep.py'])]
for stage,argv in steps:
 (H/'status.json').write_text(json.dumps(dict(stage=stage,started_elapsed_seconds=time.monotonic()-started))+'\n')
 result=subprocess.run([sys.executable,str(H/argv[0]),*argv[1:]],cwd=ROOT)
 if result.returncode:
  (H/'status.json').write_text(json.dumps(dict(stage=stage,exit_code=result.returncode,elapsed_seconds=time.monotonic()-started))+'\n')
  raise SystemExit(result.returncode)
(H/'status.json').write_text(json.dumps(dict(stage='complete',elapsed_seconds=time.monotonic()-started))+'\n')
