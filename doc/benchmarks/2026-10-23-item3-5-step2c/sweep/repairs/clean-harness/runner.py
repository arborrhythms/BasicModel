"""Retained scoped diagnosis/repair runs after the one step-2 sweep."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RECEIPT = HERE.parents[1]
BASE = ROOT/'doc/benchmarks'
name, *selectors = sys.argv[1:]
target = HERE/name
target.mkdir()
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
audit = json.loads((RECEIPT/'final/source-05/source-audit.json').read_text())
manifest = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in audit['files']}
(target/'source.json').write_text(json.dumps(dict(base=source, files=manifest), indent=2)+'\n')
(target/'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--', 'bin', 'test', 'data'], cwd=ROOT))
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='0',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    CONTRACT_CAPTURE=str(target/'state'), ARRIVAL_CAPTURE=str(target/'arrival-state'),
    ARRIVAL_KEEP_DIR=str(target/'arrival-keep'), ARRIVAL_KEEP_SOURCE=source,
    PYTEST_PLUGINS='sweep_capture,arrival_capture,identity_keep_probe',
    PYTHONPATH=os.pathsep.join(map(str, (HERE, HERE.parent/'source-01', RECEIPT/'development',
        BASE/'2026-10-21-item3-5-reference-reconstruction',
        BASE/'2026-10-22-item3-5-step2b/final/decoder-domain', ROOT/'bin', ROOT/'test'))))
for key in ('ARRIVAL_REPLAY', 'ARRIVAL_CLOSING_BASELINE', 'CLOSING_REPLAY',
            'IDENTITY_REPLAY_RNG', 'IDENTITY_REPLAY_GENERATORS', 'ARRIVAL_KEEP_FILE'):
    env.pop(key, None)
replay = os.environ.get('SWEEP_REPLAY')
if replay:
    captured = HERE.parent/'source-01/state'/replay.replace('/', '_').replace('::', '--')
    env['IDENTITY_REPLAY_RNG'] = str(captured/'initial-rng.pt')
    env['IDENTITY_REPLAY_GENERATORS'] = str(captured)
if os.environ.get('SWEEP_PROBE'):
    env['PYTEST_PLUGINS'] += ','+os.environ['SWEEP_PROBE']
if os.environ.get('SWEEP_CLEAN'):
    env.pop('PYTEST_PLUGINS', None)
command = [str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *selectors, '--tb=short']
record = dict(source=source, command=command, replay=replay,
    plugins=env.get('PYTEST_PLUGINS'), started=datetime.now(timezone.utc).isoformat())
(target/'start.json').write_text(json.dumps(record, indent=2)+'\n')
with (target/'run.log').open('x') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
record.update(exit=result.returncode, finished=datetime.now(timezone.utc).isoformat(),
    source_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in manifest.items()))
(target/'run.json').write_text(json.dumps(record, indent=2)+'\n')
print((target/'run.log').read_text()[-12000:])
raise SystemExit(result.returncode)
