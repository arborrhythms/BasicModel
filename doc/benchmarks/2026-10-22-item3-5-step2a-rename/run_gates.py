"""Rename-only gates: the retained eight-file set, then regression files."""
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent/'2026-10-21-item3-5-reference-reconstruction'
STM = HERE.parent/'2026-10-20-item3-5-stm1'
source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff',source,'--name-only','--','bin','test','data'],cwd=ROOT,text=True).strip()
selections = dict(native=[f'test/test_{name}.py' for name in (
    'math_chain_section14','reference_close','reference_mentions','bind_thought',
    'identity_rows','identity_cube','part_store')]+[
    'test/test_operators_final_b.py::test_both_training_trials_share_bootstrap_snapshot'],
    regressions=[f'test/test_{name}.py' for name in (
    'medium_term_memory','reference_arrivals','reference_arrival_review','reference_choices',
    'part_closing','reference_occurrence_views','reference_review10','reference_participants')])
assert not (HERE/'frozen-source.json').exists()
(HERE/'frozen-source.json').write_text(json.dumps(dict(source=source,selectors=selections,
    production_change='class/module identifier rename only; no safe culls yet'),indent=2)+'\n')
results=[]
for name,selectors in selections.items():
    env=dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',
        DEVELOPER_DIR='/Library/Developer/CommandLineTools',
        PYTHONPATH=os.pathsep.join(map(str,(PRIOR,STM,STM/'review3',
            ROOT/'doc/benchmarks/2026-10-19-item4-step0-round2',ROOT/'bin',ROOT/'test'))))
    for key in ('ARRIVAL_REPLAY','ARRIVAL_CLOSING_BASELINE','CLOSING_REPLAY'):
        env.pop(key,None)
    if name=='native':
        env.update(PYTEST_PLUGINS='closing_probe,arrival_capture,arrival_keep_probe,early_failure_probe,reference_probe,bootstrap_probe',
            CLOSING_TRACE=str(HERE/(name+'-trace.jsonl')),SCOPED_REPORTS=str(HERE/(name+'-reports')),
            ITEM35_BIND_RECEIPT=str(HERE/(name+'-generated')),
            BOOTSTRAP_CAPTURE=str(HERE/(name+'-bootstrap-state')),
            ARRIVAL_KEEP_DIR=str(HERE/(name+'-keep')),ARRIVAL_KEEP_SOURCE=source,
            REFERENCE_EVIDENCE=str(HERE/(name+'-reference-evidence')),
            ARRIVAL_CAPTURE=str(HERE/(name+'-state')))
    else:
        env.update(PYTEST_PLUGINS='contract_capture',CONTRACT_CAPTURE=str(HERE/(name+'-state')))
    command=[str(ROOT/'.venv/bin/python'),'test/test_report.py',*selectors,
        '--workers','1','--memory-gib','8','--timeout','900','--suite-timeout','1800',
        '--batch-size','16','--max-files','1','--run-dir',str(HERE/name)]
    assert not (HERE/name).exists()
    with (HERE/(name+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    results.append(dict(source=source,name=name,command=command,exit=result.returncode))
    (HERE/'gate-runs.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,result.returncode,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
