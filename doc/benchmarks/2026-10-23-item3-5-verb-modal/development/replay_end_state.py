import os,subprocess
from pathlib import Path
root=Path.cwd();out=root/'output/item3-5-verb-modal-20261023/development'
base=root/'doc/benchmarks'
paths=(out,base/'2026-10-23-item3-5-step2c/development',base/'2026-10-23-item3-5-step2c/sweep/source-01',base/'2026-10-23-item3-5-stage4e-np/source-05',base/'2026-10-21-item3-5-reference-reconstruction',base/'2026-10-22-item3-5-step2b/final/decoder-domain',base/'2026-10-20-item3-5-stm1',base/'2026-10-20-item3-5-stm1/review3',base/'2026-10-19-item4-step0-round2',root/'bin',root/'test')
state=root/'output/item3-5-verb-modal-20261023/swap-verb/final/source-03/identity-and-driver-state/test_test_sentence_end_state.py--test_one_native_sentence_owns_only_its_end_state_after_the_boundary/initial-rng.pt'
env=dict(os.environ,OMP_NUM_THREADS='1',BASICMODEL_DEVICE='cpu',RUN_SLOW='1',PYTHONPATH=os.pathsep.join(map(str,paths)),PYTEST_PLUGINS='sweep_capture,arrival_capture,identity_keep_probe,end_state_probe',CONTRACT_CAPTURE=str(out/'35-end-state-capture'),IDENTITY_REPLAY_RNG=str(state),END_STATE_PROBE=str(out/'35-end-state-probe.json'))
with (out/'35-end-state-replay.log').open('x') as log:
 r=subprocess.run([str(root/'.venv/bin/python'),'-m','pytest','-q','test/test_sentence_end_state.py::test_one_native_sentence_owns_only_its_end_state_after_the_boundary','--tb=short'],env=env,stdout=log,stderr=subprocess.STDOUT)
print(r.returncode)
