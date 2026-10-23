"""Retained development checks; these are not frozen-source gate evidence."""
import datetime, hashlib, json, os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
run=HERE/'development-04'
run.mkdir(exist_ok=False)
patch=subprocess.check_output(['git','diff','--binary','--','bin','test','data'],cwd=ROOT)
(run/'working-source.patch').write_bytes(patch)
new=ROOT/'test/test_step2b_storage.py'
(run/new.name).write_bytes(new.read_bytes())
selectors=[f'test/test_{name}.py' for name in (
 'sentence_end_state','forgetting','learning_readiness','reasoning_cde_model',
 'review14_contracts','identity_corpus','query_executors','query_vp_boundaries')]
base=HERE.parents[1]
env=dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',
 DEVELOPER_DIR='/Library/Developer/CommandLineTools',
 PYTHONPATH=os.pathsep.join(map(str,(HERE,base/'2026-10-21-item3-5-reference-reconstruction',
 base/'2026-10-20-item3-5-stm1',base/'2026-10-20-item3-5-stm1/review3',
 base/'2026-10-19-item4-step0-round2',ROOT/'bin',ROOT/'test'))),
 PYTEST_PLUGINS='storage_capture',CONTRACT_CAPTURE=str(run/'captures'))
command=[str(ROOT/'.venv/bin/python'),'test/test_report.py',*selectors,
 '--workers','1','--memory-gib','8','--timeout','900','--suite-timeout','3600',
 '--batch-size','16','--max-files','1','--run-dir',str(run/'tests')]
record=dict(base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 patch_sha256=hashlib.sha256(patch).hexdigest(),new_test_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
 started=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command)
(run/'run.json').write_text(json.dumps(record,indent=2)+'\n')
with (run/'console.log').open('x') as log:
 result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
record.update(exit=result.returncode,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run/'run.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
raise SystemExit(result.returncode)
