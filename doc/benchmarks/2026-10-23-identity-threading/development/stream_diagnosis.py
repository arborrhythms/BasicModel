"""Retain every development run, its source diff, initial state and failure."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT/'doc/benchmarks'
name, *selectors = sys.argv[1:]
target = HERE/name
target.mkdir()
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
fixtures = [BASE/'2026-10-20-item3-5-stm1'/name for name in
            ('forced_decomposition_grammar.py', 'decomposition_certificate.py')]
(target/'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--', 'bin', 'test', 'data',
    *(str(p.relative_to(ROOT)) for p in fixtures)], cwd=ROOT))
files = [p for directory in ('bin', 'test', 'data') for p in (ROOT/directory).rglob('*')
         if p.is_file() and p.suffix in ('.py', '.grammar', '.xml', '.xsd', '.json')] + fixtures
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(target/'source.json').write_text(json.dumps(dict(base=source, files=manifest), indent=2)+'\n')
import shutil
untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', 'bin', 'test', 'data'],
                                    cwd=ROOT, text=True).splitlines()
for name in untracked:
    copied = target/'untracked'/name
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/name, copied)
env = dict(os.environ, BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    CONTRACT_CAPTURE=str(target/'state'), PYTEST_PLUGINS='identity_capture,arrival_capture,identity_keep_probe,stream_boundary_probe',
    ARRIVAL_CAPTURE=str(target/'arrival-state'), ARRIVAL_KEEP_DIR=str(target/'arrival-keep'),
    ARRIVAL_KEEP_SOURCE=source,
    PYTHONPATH=os.pathsep.join(map(str, (HERE, HERE.parent/'final/source-09/diagnosis', BASE/'2026-10-21-item3-5-reference-reconstruction',
        BASE/'2026-10-22-item3-5-step2b/final/decoder-domain', ROOT/'bin', ROOT/'test'))))
arm = os.environ.get('DECOMPOSITION_ARM')
if arm:
    assert arm in ('zero', 'enabled')
    frozen = HERE.parent/'final/source-06'
    for filename in ('identity_decomposition.py', 'original-complete.grammar'):
        shutil.copy2(frozen/filename, target/filename)
    env.update(IDENTITY_GATE_DIR=str(target), BIND_GAIN_ARM=arm,
        PYTEST_PLUGINS='identity_capture,closing_probe,identity_decomposition,decomposition_keep_probe',
        DECOMPOSITION_KEEP_FILE=str(target/'keep.jsonl'), CLOSING_TRACE=str(target/'trace.jsonl'),
        SCOPED_REPORTS=str(target/'reports'))
    env['PYTHONPATH'] = os.pathsep.join((str(target), env['PYTHONPATH'],
        str(BASE/'2026-10-20-item3-5-stm1'), str(BASE/'2026-10-20-item3-5-stm1/review3'),
        str(BASE/'2026-10-19-item4-step0-round2')))
command = [str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *selectors, '--tb=short']
record = dict(command=command, started=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (target/'run.log').open('x') as output:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
record.update(exit=result.returncode, finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
(target/'run.json').write_text(json.dumps(record, indent=2)+'\n')
print((target/'run.log').read_text()[-16000:])
raise SystemExit(result.returncode)
