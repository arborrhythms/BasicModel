"""Check the empty-menu repair, then replay both original decomposition arms."""
import os
from pathlib import Path
import subprocess
import sys

receipt = Path(__file__).resolve().parent
root = receipt.parents[2]
previous = receipt.parent / '2026-10-19-item4-step0'
base = os.environ.copy()
base.update(
    RUN_SLOW='1', BASICMODEL_DEVICE='cpu', PYTEST_PLUGINS='closing_probe',
    PYTHONPATH=os.pathsep.join(map(str, (receipt, previous, root / 'bin', root / 'test'))))
selections = [('post-sweep', [
    'test/test_item4_step0.py', 'test/test_math_chain_repair2.py',
    'test/test_math_chain_section14.py', 'test/test_item6_2_thinking.py',
    'test/test_unified_thought_controller.py', 'test/test_walk_trials.py',
    'test/test_reference_mentions.py', 'test/test_reference_close.py',
    'test/test_operators_final_b.py::test_both_training_trials_share_bootstrap_snapshot'])]
selections += [('post-sweep-decomposition-' + arm, ['test/test_math_chain_decomposition.py'])
               for arm in ('zero', 'enabled')]
for name, selectors in selections:
    env = dict(base, CLOSING_TRACE=str(receipt / (name + '-trace.jsonl')),
               SCOPED_REPORTS=str(receipt / (name + '-reports')))
    if 'decomposition' in name:
        env.update(
            ITEM4_GATE_ARM=name.split('-')[-1],
            ITEM4_GATE_NONCE='d15d7bc18ad8170f2bf77e628184fd4a',
            ITEM4_GATE_METRICS='../2026-10-19-item4-step0-round2/post-sweep-decomposition-metrics',
            PYTEST_PLUGINS='closing_probe,item4_gate')
    command = [sys.executable, 'test/test_report.py', *selectors,
               '--workers', '4', '--memory-gib', '28', '--timeout', '900',
               '--suite-timeout', '1800', '--batch-size', '16', '--max-files', '1',
               '--run-dir', str(receipt / name)]
    with (receipt / (name + '.log')).open('w') as output:
        result = subprocess.run(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(name, result.returncode, flush=True)
    if result.returncode:
        sys.exit(result.returncode)
