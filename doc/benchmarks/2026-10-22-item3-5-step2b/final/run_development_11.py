"""Retained development checks; these are not frozen-source gate evidence."""
import datetime, hashlib, json, os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
run=HERE/'development-11'
run.mkdir(exist_ok=False)
patch=subprocess.check_output(['git','diff','--binary','--','bin','test','data'],cwd=ROOT)
(run/'working-source.patch').write_bytes(patch)
new=ROOT/'test/test_step2b_storage.py'
(run/new.name).write_bytes(new.read_bytes())
selectors=['test/test_joint_objectives.py::test_real_packed_runbatch_trains_prediction_and_representation_with_teacher', 'test/test_lies_can_be_learned.py::test_conflicting_claim_preserves_support_and_counterevidence', 'test/test_negative_expectation.py::test_declared_not_is_a_thought_act_not_an_observation', 'test/test_predicate_identity.py::test_first_three_slot_predicate_occurrence[False-part]', 'test/test_predicate_identity.py::test_first_three_slot_predicate_occurrence[False-implies]', 'test/test_predicate_identity.py::test_first_three_slot_predicate_occurrence[True-part]', 'test/test_predicate_identity.py::test_first_three_slot_predicate_occurrence[True-implies]', 'test/test_relative_sentence_codebook_insertion.py::test_closing_reuses_relation_and_preserves_independent_evidence', 'test/test_replacement_kernel_contracts.py::test_direct_and_depth_one_two_chains_keep_native_premises[1]', 'test/test_replacement_kernel_contracts.py::test_direct_and_depth_one_two_chains_keep_native_premises[2]', 'test/test_replacement_kernel_contracts.py::test_direct_and_depth_one_two_chains_keep_native_premises[3]', 'test/test_taxonomy_policy.py::test_nested_taxonomy_result_keeps_each_record_source_and_premise_ablation']
base=HERE.parents[1]
env=dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',
 DEVELOPER_DIR='/Library/Developer/CommandLineTools',
 PYTHONPATH=os.pathsep.join(map(str,(HERE/'review-repair',base/'2026-10-21-item3-5-reference-reconstruction',
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
