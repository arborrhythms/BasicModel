"""Step 2's single default sweep on the reviewed production/test source."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RECEIPT = HERE.parents[1]
BASE = ROOT / 'doc/benchmarks'
SOURCE = 'fcaff2d19624a581206fed3a17dc1b5b3d752612'
AUDIT = RECEIPT / 'final/source-05/source-audit.json'


def verify():
    assert not subprocess.check_output(['git', 'diff', SOURCE, '--name-only', '--',
                                       'bin', 'test', 'data'], cwd=ROOT).strip()
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard',
                                       '--', 'bin', 'test', 'data'], cwd=ROOT).strip()
    for name, digest in json.loads(AUDIT.read_text())['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


verify()
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='0',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    CONTRACT_CAPTURE=str(HERE/'state'), ARRIVAL_CAPTURE=str(HERE/'arrival-state'),
    ARRIVAL_KEEP_DIR=str(HERE/'arrival-keep'), ARRIVAL_KEEP_SOURCE=SOURCE,
    PYTEST_PLUGINS='sweep_capture,arrival_capture,identity_keep_probe',
    PYTHONPATH=os.pathsep.join(map(str, (HERE, RECEIPT/'development',
        BASE/'2026-10-21-item3-5-reference-reconstruction',
        BASE/'2026-10-22-item3-5-step2b/final/decoder-domain',
        ROOT/'bin', ROOT/'test'))))
for key in ('ARRIVAL_REPLAY', 'ARRIVAL_CLOSING_BASELINE', 'CLOSING_REPLAY',
            'IDENTITY_REPLAY_RNG', 'IDENTITY_REPLAY_GENERATORS',
            'BASICMODEL_PHASE_ARTIFACTS', 'ARRIVAL_KEEP_FILE'):
    env.pop(key, None)
command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py',
    '--workers', '4', '--memory-gib', '28', '--timeout', '1800',
    '--suite-timeout', '10800', '--batch-size', '128', '--max-files', '8',
    '--run-dir', str(HERE/'full')]
record = dict(source=SOURCE,
    execution_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    audited_files=json.loads(AUDIT.read_text())['files'], command=command,
    measurement_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (HERE/'run.py', HERE/'sweep_capture.py')},
    default_suite=True, run_slow=False, seeded_retry=False,
    started=datetime.now(timezone.utc).isoformat())
(HERE/'start.json').write_text(json.dumps(record, indent=2)+'\n')
with (HERE/'run.log').open('x') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
verify()
record.update(exit=result.returncode, source_unchanged=True,
              finished=datetime.now(timezone.utc).isoformat())
(HERE/'run.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('source','exit','source_unchanged','finished')}))
