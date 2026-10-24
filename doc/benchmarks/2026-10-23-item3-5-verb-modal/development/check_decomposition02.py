import os,subprocess,shutil
from pathlib import Path
root=Path.cwd();base=root/'doc/benchmarks';out=root/'output/item3-5-verb-modal-20261023/swap-verb/development/decomposition-check02';out.mkdir(parents=True)
original=out.parents[1]/'final/source-01'
for name in ('identity_decomposition.py','original-complete.grammar'):shutil.copy2(original/name,out/name)
for arm in ('zero','enabled'):
 env=dict(os.environ,RUN_SLOW='1',OMP_NUM_THREADS='1',BASICMODEL_DEVICE='cpu',IDENTITY_GATE_DIR=str(out),BIND_GAIN_ARM=arm,CONTRACT_CAPTURE=str(out/(arm+'-state')),
  PYTHONPATH=os.pathsep.join(map(str,(out,base/'2026-10-23-item3-5-step2c/development',base/'2026-10-21-item3-5-reference-reconstruction',base/'2026-10-19-item4-step0-round2',root/'bin',root/'test'))),
  PYTEST_PLUGINS='identity_capture,closing_probe,identity_decomposition,decomposition_keep_probe',
  DECOMPOSITION_KEEP_FILE=str(out/(arm+'-keep.jsonl')),CLOSING_TRACE=str(out/(arm+'-trace.jsonl')),SCOPED_REPORTS=str(out/(arm+'-reports')))
 with (out/(arm+'.log')).open('x') as log:
  r=subprocess.run([str(root/'.venv/bin/python'),'test/test_report.py','test/test_math_chain_decomposition.py','--workers','1','--memory-gib','8','--timeout','900','--suite-timeout','1800','--batch-size','16','--max-files','1','--run-dir',str(out/arm)],env=env,stdout=log,stderr=subprocess.STDOUT)
 print(arm,r.returncode,flush=True)
