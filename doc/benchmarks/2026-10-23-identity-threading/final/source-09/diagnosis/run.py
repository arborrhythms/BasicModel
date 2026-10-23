"""Re-observe a retained first draw before changing the frozen source."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
ROOT = SOURCE.parents[4]
name, selector = sys.argv[1:]
target = HERE/name
target.mkdir()
capture = SOURCE/'modules-a-state'/selector.replace('/', '_').replace('::', '--')
bench = ROOT/'doc/benchmarks'
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    IDENTITY_REPLAY_RNG=str(capture/'initial-rng.pt'), CONTRACT_CAPTURE=str(target/'state'),
    ARRIVAL_CAPTURE=str(target/'arrival'), ARRIVAL_KEEP_DIR=str(target/'keep'),
    ARRIVAL_KEEP_SOURCE=json.loads((SOURCE/'frozen-source.json').read_text())['source'],
    PYTEST_PLUGINS='identity_capture,arrival_capture,identity_keep_probe,boundary_probe',
    PYTHONPATH=os.pathsep.join(map(str, (HERE, SOURCE.parents[1]/'development',
        bench/'2026-10-21-item3-5-reference-reconstruction',
        bench/'2026-10-22-item3-5-step2b/final/decoder-domain', ROOT/'bin', ROOT/'test'))))
command=[str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', selector, '--tb=short']
record=dict(source=env['ARRIVAL_KEEP_SOURCE'], command=command,
    initial_rng=env['IDENTITY_REPLAY_RNG'], started=datetime.now(timezone.utc).isoformat(),
    scope='diagnostic replay only, not gate evidence; corpus private generator is not restored')
with (target/'run.log').open('x') as log:
    result=subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
record.update(exit=result.returncode, finished=datetime.now(timezone.utc).isoformat())
(target/'run.json').write_text(json.dumps(record,indent=2)+'\n')
print(name, result.returncode)
raise SystemExit(result.returncode)
