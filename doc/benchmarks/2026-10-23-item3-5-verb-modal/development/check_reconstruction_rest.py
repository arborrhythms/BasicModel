import os,subprocess
from pathlib import Path
root=Path.cwd();out=root/'output/item3-5-verb-modal-20261023/development/30-reconstruction-rest.log'
files=('reconstruction_roundtrip','mixing_reconstruction_bank','generation_catalog','grammar_lesson_context','mind_generativity','stored_idea_generation','reverse_traversal','active_decode_candidates','understanding_reconstruct','reading_curriculum')
with out.open('x') as log:
 for name in files:
  env=dict(os.environ,OMP_NUM_THREADS='1',BASICMODEL_DEVICE='cpu',RUN_SLOW='0' if name=='reconstruction_roundtrip' else '1')
  log.write('\nFILE '+name+' RUN_SLOW='+env['RUN_SLOW']+'\n');log.flush()
  r=subprocess.run([str(root/'.venv/bin/python'),'-m','pytest','-q','test/test_'+name+'.py','--tb=short'],env=env,stdout=log,stderr=subprocess.STDOUT)
  log.write('\nEXIT '+str(r.returncode)+'\n');log.flush();print(name,r.returncode,flush=True)
