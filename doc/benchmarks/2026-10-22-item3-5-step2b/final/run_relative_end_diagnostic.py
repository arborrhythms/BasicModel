"""Replay the failed state; this diagnosis is not gate evidence."""
import json
import os
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
run=HERE/'relative-end-diagnostic'
run.mkdir(exist_ok=False)
env=dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',
    DEVELOPER_DIR='/Library/Developer/CommandLineTools',
    PYTHONPATH=os.pathsep.join(map(str,(HERE,ROOT/'bin',ROOT/'test'))),
    PYTEST_PLUGINS='relative_end_probe',
    END_STATE_REPLAY=str(HERE/'modules-state/test_test_sentence_end_state.py--test_one_native_sentence_owns_only_its_end_state_after_the_boundary/initial-rng.pt'),
    END_STATE_REPORT=str(run/'row.json'))
command=[str(ROOT/'.venv/bin/python'),'test/test_report.py',
    'test/test_sentence_end_state.py::test_one_native_sentence_owns_only_its_end_state_after_the_boundary',
    '--workers','1','--memory-gib','8','--timeout','900','--suite-timeout','1200',
    '--run-dir',str(run/'tests')]
with (run/'console.log').open('x') as output:
    result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
(run/'run.json').write_text(json.dumps(dict(source=subprocess.check_output(
    ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),command=command,
    replay=env['END_STATE_REPLAY'],exit=result.returncode),indent=2)+'\n')
raise SystemExit(result.returncode)
