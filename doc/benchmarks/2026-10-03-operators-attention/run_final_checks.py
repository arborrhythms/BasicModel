"""Named source-matched review measurements, with the unchanged resource guard."""
import json, os, sys, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded
kind,name=sys.argv[1:3]
folder=HERE/name;folder.mkdir(exist_ok=False)
source=bounded.source_snapshot(ROOT)
bounded.write_json(folder/'source.json',source)
with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for path in source: archive.write(ROOT/path,path)
env=bounded.worker_environment(ROOT)
env.pop('BASIC_SEED',None)
env.update(MODEL_COMPILE='none',BASICMODEL_DEVICE='cpu',RUN_SLOW='1',BASIC_AUTOLOAD='0',BASIC_AUTOSAVE='0')
env['PYTHONPATH']=os.pathsep.join((str(HERE),env['PYTHONPATH']))
pytest=[sys.executable,'-m','pytest','-q','-s']
if kind=='native':
 config=sys.argv[3]
 env.update(ATTENTION_REFERENCE_CONFIG=config,ATTENTION_REFERENCE_OUTPUT=str(folder/'measurement.json'))
 command=pytest+[str(HERE/'native_config_probe.py')]
elif kind=='controls':
 env.update(PYTEST_PLUGINS='xor_observer',ITEM7_XOR_GATE='6',ITEM7_XOR_MEASUREMENTS=str(folder/'observations.jsonl'))
 command=pytest+['test/test_mm_xor.py::TestMMXorConvergence::test_convergence',
  'test/test_grounded_xor.py','test/test_word_admission.py::test_33_grounded_xor_with_word_boundary',
  'test/test_explicit_dimensions.py::TestXorExactCliReconstruction']
elif kind=='nanochat':
 command=[sys.executable,'bin/eval_nanochat_grammar.py','score','--fresh','--device','cpu','--output',str(folder/'measurement.json')]
elif kind=='profile':
 env['ATTENTION_PROFILE_OUTPUT']=str(folder/'measurement.json')
 command=pytest+[str(HERE/'attention_profile.py')]
else:raise ValueError(kind)
bounded.write_json(folder/'plan.json',dict(command=command,seed=None if kind!='nanochat' else 'existing XML evaluator seed',
 memory_bytes=8*bounded.GIB,timeout=1800,additional_training_runs=0 if kind in ('native','nanochat','profile') else 'unchanged selected controls'))
result=bounded.run_guarded(command,cwd=ROOT,env=env,log_path=folder/'run.log',memory_bytes=8*bounded.GIB,timeout=1800)
bounded.write_json(folder/'process.json',result)
assert source==bounded.source_snapshot(ROOT)
bounded.write_json(folder/'complete.json',dict(source_matched=True,completed=True,retries=0))
print(json.dumps(result))
print((folder/'run.log').read_text()[-10000:])
