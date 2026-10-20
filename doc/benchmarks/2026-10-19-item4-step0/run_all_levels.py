"""One paired comparison of unchanged standing certificates, no rescue runs."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
receipt = Path(__file__).resolve().parent
selectors = [
    'test/test_grounded_xor.py',
    'test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse',
    'test/test_explicit_dimensions.py::TestXorExactCliReconstruction',
    'test/test_mm_xor.py::TestMMXorConvergence::test_convergence',
    'test/test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal',
    'test/test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal',
    'test/test_explicit_dimensions.py::TestXorGrammarLearnsXor',
    'test/test_explicit_dimensions.py::TestXorGrammarReconstruction',
    'test/test_math_chain.py', 'test/test_math_chain_repair.py',
    'test/test_math_chain_repair2.py', 'test/test_math_chain_decomposition.py',
    'test/test_math_chain_section14.py',
]
protocol = receipt/'all-levels-protocol.json'
if protocol.exists():
    raise SystemExit('Protocol already exists; do not repeat a completed learning comparison.')
nonce = secrets.token_hex(16)
protocol.write_text(json.dumps(dict(nonce=nonce, selectors=selectors,
    initialization='fresh entropy, one reproducible seed per fixture/shared-training group; identical across arms',
    gains={'zero':[0,0,0,0], 'enabled':[.5,.5,.5,.5]}, prime=True, phi=.5,
    assertions='unchanged', training_budget='unchanged',
    parameter_comparison='common per-stage constructor; exact tensor equality'), indent=2)+'\n')
for arm in ('zero', 'enabled'):
    environment = os.environ.copy()
    environment.update(ITEM4_GATE_ARM=arm, ITEM4_GATE_NONCE=nonce, RUN_SLOW='1',
        PYTEST_PLUGINS='item4_gate',
        PYTHONPATH=os.pathsep.join([str(root/'bin'), str(root/'test'), str(receipt)]),
        DEVELOPER_DIR='/Library/Developer/CommandLineTools')
    command = [sys.executable, 'test/test_report.py', *selectors,
        '--workers','3','--memory-gib','24','--batch-size','8','--max-files','1',
        '--timeout','1800','--suite-timeout','7200', '--run-dir',
        str(receipt/('all-levels-'+arm))]
    with (receipt/('all-levels-'+arm+'-runner.log')).open('w') as output:
        result = subprocess.run(command, cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT)
    print(arm, result.returncode, flush=True)
