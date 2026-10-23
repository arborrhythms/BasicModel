"""Retained development checks; these are not frozen-source gate evidence."""
import datetime, hashlib, json, os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
run=HERE/'development-03'
run.mkdir(exist_ok=False)
patch=subprocess.check_output(['git','diff','--binary','--','bin','test','data'],cwd=ROOT)
(run/'working-source.patch').write_bytes(patch)
new=ROOT/'test/test_step2b_storage.py'
(run/new.name).write_bytes(new.read_bytes())
selectors=[f'test/test_{name}.py' for name in (
 'clause_acceptance','sentence_end_state','sentence_identity','sentence_retention',
 'truth_text','forgetting','part_read','operators_round4a0','operators_round3a',
 'operators_final','operators_final_b','review14_contracts',
 'learning_readiness','structural_checkpoint','ternary_truth_store','two_truth_stores',
 'assertion_provenance','existence_provenance','nested_retention_owners','expectation_retention',
 'occurrence_coordinates','live_query_occurrences','truth_grounded_reasoning','reasoning',
 'reasoning_cde_model','mereological','reference_participants','reference_arrival_review',
 'reference_choices','reference_review10','reference_reconstruction')]
base=HERE.parents[1]
env=dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',
 DEVELOPER_DIR='/Library/Developer/CommandLineTools',
 PYTHONPATH=os.pathsep.join(map(str,(base/'2026-10-21-item3-5-reference-reconstruction',
 base/'2026-10-20-item3-5-stm1',base/'2026-10-20-item3-5-stm1/review3',
 base/'2026-10-19-item4-step0-round2',ROOT/'bin',ROOT/'test'))),
 PYTEST_PLUGINS='contract_capture',CONTRACT_CAPTURE=str(run/'captures'))
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
