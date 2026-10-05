"""One named §25 measurement, bounded and source-matched; no campaigns."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'test'))
import bounded_tests as bounded

kind = sys.argv[1]
selectors = {
    'xor': ['test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy',
            'test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct'],
}[kind]
folder = HERE/(sys.argv[2] if len(sys.argv) > 2 else kind)
folder.mkdir(exist_ok=False)
source = bounded.source_snapshot(ROOT)
bounded.write_json(folder/'source.json', source)
assert source == bounded.source_snapshot(ROOT)
env = bounded.worker_environment(ROOT)
env.pop('BASIC_SEED', None)
env.update(MODEL_COMPILE='none', BASICMODEL_DEVICE='cpu', RUN_SLOW='1',
           BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false',
           PYTHONPATH=os.pathsep.join((str(HERE), str(ROOT/'bin'), str(ROOT/'test'))))
if kind == 'xor':
    env.update(PYTEST_PLUGINS='xor_observer', ITEM7_XOR_GATE='5',
               ITEM7_XOR_MEASUREMENTS=str(folder/'observations.jsonl'),
               OWNERSHIP_OBSERVER_OUTPUT=str(folder/'ownership'))
bounded.write_json(folder/'plan.json', dict(selectors=selectors, runs=1,
    memory_bytes=8*bounded.GIB, timeout=1800, seed=None, compile='none', device='cpu'))
process = bounded.run_guarded([sys.executable, '-m', 'pytest', '-q', *selectors],
    cwd=ROOT, env=env, log_path=folder/'run.log', memory_bytes=8*bounded.GIB, timeout=1800)
bounded.write_json(folder/'process.json', process)
assert source == bounded.source_snapshot(ROOT)
bounded.write_json(folder/'complete.json', dict(source_matched=True, completed=True, retries=0))
print(json.dumps(process))
