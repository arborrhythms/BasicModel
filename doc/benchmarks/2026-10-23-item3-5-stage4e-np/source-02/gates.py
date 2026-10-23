"""Scoped checks for the regenerated lesson drivers on the frozen source."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import source_snapshot
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True)
files = ['test/test_modification_lessons.py','test/test_parts_of_speech.py',
    'test/test_constituent_lesson.py','test/test_grammar_lesson_context.py',
    'test/test_phase_artifacts.py','test/test_identity_experiment.py',
    'test/test_identity_threading.py','test/test_reading_curriculum.py']
base = ROOT/'doc/benchmarks'
capture = base/'2026-10-23-item3-5-step2c'
env = dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    CONTRACT_CAPTURE=str(out/'state'),ARRIVAL_CAPTURE=str(out/'arrival-state'),
    ARRIVAL_KEEP_DIR=str(out/'arrival-keep'),
    PYTEST_PLUGINS='sweep_capture,arrival_capture,identity_keep_probe',
    PYTHONPATH=os.pathsep.join(map(str,(capture/'sweep/source-01',capture/'development',
        base/'2026-10-21-item3-5-reference-reconstruction',
        base/'2026-10-22-item3-5-step2b/final/decoder-domain',ROOT/'bin',ROOT/'test'))))
source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
env['ARRIVAL_KEEP_SOURCE'] = source
command = [str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*files,'--tb=short']
audit = source_snapshot(ROOT)
record = dict(source=source,files=audit,command=command,started=datetime.now(timezone.utc).isoformat())
(out/'start.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'run.log').open('x') as log:
    result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
record.update(exit=result.returncode,source_unchanged=source_snapshot(ROOT)==audit,
              finished=datetime.now(timezone.utc).isoformat())
(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
print((out/'run.log').read_text()[-8000:])
raise SystemExit(result.returncode)
