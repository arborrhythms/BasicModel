"""Line 3's one default sweep, with isolated observational capture."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'output/item3-5-stage4e-np-20261023/source-05/sweep'
BASE = ROOT / 'doc/benchmarks'
SOURCE = '3b34b3f388a7d4d5f455fe12bec7ea1a6b5007e2'
AUDIT = BASE / '2026-10-23-item3-5-stage4e-np/source-audit.json'


def verify():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
    assert not subprocess.check_output(['git', 'diff', SOURCE, '--name-only', '--', 'bin', 'test', 'data']).strip()
    for name, digest in json.loads(AUDIT.read_text())['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


verify()
OUT.mkdir(parents=True)
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='0',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    CONTRACT_CAPTURE=str(OUT/'state'), ARRIVAL_CAPTURE=str(OUT/'arrival-state'),
    ARRIVAL_KEEP_DIR=str(OUT/'arrival-keep'), ARRIVAL_KEEP_SOURCE=SOURCE,
    PYTEST_PLUGINS='capture_environment,sweep_capture,arrival_capture,identity_keep_probe',
    PYTHONPATH=os.pathsep.join(map(str, (HERE,
        BASE/'2026-10-23-item3-5-step2c/sweep/source-01',
        BASE/'2026-10-23-item3-5-step2c/development',
        BASE/'2026-10-21-item3-5-reference-reconstruction',
        BASE/'2026-10-22-item3-5-step2b/final/decoder-domain', ROOT/'bin', ROOT/'test'))))
for key in ('ARRIVAL_REPLAY', 'ARRIVAL_CLOSING_BASELINE', 'CLOSING_REPLAY',
            'IDENTITY_REPLAY_RNG', 'IDENTITY_REPLAY_GENERATORS',
            'BASICMODEL_PHASE_ARTIFACTS', 'ARRIVAL_KEEP_FILE'):
    env.pop(key, None)
command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py',
    '--workers', '4', '--memory-gib', '28', '--timeout', '1800',
    '--suite-timeout', '10800', '--batch-size', '128', '--max-files', '8',
    '--run-dir', str(OUT/'full')]
record = dict(source=SOURCE, audited_files=json.loads(AUDIT.read_text())['files'], command=command,
    measurement_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (HERE/'full_sweep.py', HERE/'capture_environment.py')},
    default_suite=True, run_slow=False, seeded_retry=False,
    capture_environment='Observers already loaded in each worker; remove their environment inheritance for nested harnesses.',
    started=datetime.now(timezone.utc).isoformat())
(OUT/'start.json').write_text(json.dumps(record, indent=2)+'\n')
with (OUT/'run.log').open('x') as log:
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
verify()
record.update(exit=result.returncode, source_unchanged=True, finished=datetime.now(timezone.utc).isoformat())
(OUT/'run.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('source','exit','source_unchanged','finished')}))
