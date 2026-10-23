import datetime,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];BASE=HERE.parents[2]
name='test_test_joint_objectives.py--test_real_packed_runbatch_trains_prediction_and_representation_with_teacher'
run=HERE/'rung0-repaired-replay';run.mkdir(exist_ok=False)
env=dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',PYTEST_PLUGINS='rung0_lexical_probe',
 RUNG0_REPLAY=str(HERE/'modules-state'/name/'initial-rng.pt'),RUNG0_CAPTURE=str(run/'capture'),
 DEVELOPER_DIR='/Library/Developer/CommandLineTools',
 PYTHONPATH=os.pathsep.join(map(str,(HERE,ROOT/'bin',ROOT/'test'))))
command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-q','test/test_joint_objectives.py::test_real_packed_runbatch_trains_prediction_and_representation_with_teacher','--tb=short']
record=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),command=command,
 started=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (run/'console.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
record.update(exit=result.returncode,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run/'run.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));raise SystemExit(result.returncode)
