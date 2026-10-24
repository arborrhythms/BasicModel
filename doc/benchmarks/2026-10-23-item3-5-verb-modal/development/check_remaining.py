import json,os,subprocess
from pathlib import Path
root=Path.cwd(); out=root/'output/item3-5-verb-modal-20261023/development'
groups=json.loads((out.parent/'swap-verb/final/source-01/selections.json').read_text())
env=dict(os.environ,RUN_SLOW='1',OMP_NUM_THREADS='1',BASICMODEL_DEVICE='cpu')
for name in ('identity-and-driver','xor-standing'):
 with (out/('28-'+name+'.log')).open('x') as log:
  r=subprocess.run([str(root/'.venv/bin/python'),'test/test_report.py',*groups[name],'--workers','1','--memory-gib','8','--timeout','900','--suite-timeout','7200','--batch-size','16','--max-files','1','--run-dir',str(out/('28-'+name))],env=env,stdout=log,stderr=subprocess.STDOUT)
 print(name,r.returncode,flush=True)
