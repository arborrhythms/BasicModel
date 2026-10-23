"""One scoped pass by frozen source; completed groups are never rerun."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = ROOT/'doc/benchmarks'
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
selections = json.loads((HERE/'selections.json').read_text())

def unchanged():
    assert not subprocess.check_output(['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'],
                                      cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', 'bin', 'test', 'data'],
                                      cwd=ROOT, text=True).strip()
    for name, digest in json.loads((HERE/'source-audit.json').read_text())['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name

unchanged()
manifest = HERE/'frozen-source.json'
if manifest.exists():
    assert json.loads(manifest.read_text())['source'] == source
else:
    manifest.write_text(json.dumps(dict(source=source, selectors=selections,
        started=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2)+'\n')
for name in sys.argv[1:] or selections:
    unchanged()
    assert name in selections
    outcome = HERE/(name+'-run.json')
    assert not outcome.exists() and not (HERE/name).exists()
    env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
        DEVELOPER_DIR='/Library/Developer/CommandLineTools', IDENTITY_GATE_DIR=str(HERE),
        PYTHONPATH=os.pathsep.join(map(str, (HERE, HERE.parents[1]/'development',
            BASE/'2026-10-21-item3-5-reference-reconstruction',
            BASE/'2026-10-22-item3-5-step2b/final/decoder-domain',
            BASE/'2026-10-20-item3-5-stm1', BASE/'2026-10-20-item3-5-stm1/review3',
            BASE/'2026-10-19-item4-step0-round2', ROOT/'bin', ROOT/'test'))))
    for key in ('ARRIVAL_REPLAY', 'ARRIVAL_CLOSING_BASELINE', 'CLOSING_REPLAY'):
        env.pop(key, None)
    env.update(CONTRACT_CAPTURE=str(HERE/(name+'-state')),
        PYTEST_PLUGINS='identity_capture,arrival_capture,identity_keep_probe',
        ARRIVAL_CAPTURE=str(HERE/(name+'-arrival-state')),
        ARRIVAL_KEEP_DIR=str(HERE/(name+'-keep')), ARRIVAL_KEEP_SOURCE=source,
        ITEM35_BIND_RECEIPT=str(HERE/(name+'-generated')))
    if name.startswith('decomposition-'):
        env.update(PYTEST_PLUGINS='identity_capture,closing_probe,identity_decomposition,decomposition_keep_probe',
            BIND_GAIN_ARM=name.split('-')[1], DECOMPOSITION_KEEP_FILE=str(HERE/(name+'-keep.jsonl')),
            CLOSING_TRACE=str(HERE/(name+'-trace.jsonl')), SCOPED_REPORTS=str(HERE/(name+'-reports')))
    command = [str(ROOT/'.venv/bin/python'), 'test/test_report.py', *selections[name],
        '--workers', '1', '--memory-gib', '8', '--timeout', '900', '--suite-timeout', '7200',
        '--batch-size', '16', '--max-files', '1', '--run-dir', str(HERE/name)]
    record = dict(source=source, command=command, started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (HERE/(name+'.log')).open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit=result.returncode, finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    outcome.write_text(json.dumps(record, indent=2)+'\n')
    unchanged()
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
