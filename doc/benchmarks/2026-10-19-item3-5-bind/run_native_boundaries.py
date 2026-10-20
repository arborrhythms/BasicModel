"""Check bind boundaries and preserve both original decomposition arms."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

receipt = Path(__file__).resolve().parent
root = receipt.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--decomposition-only', action='store_true')
parser.add_argument('--suffix', required=True)
args = parser.parse_args()
step0 = receipt.parent / '2026-10-19-item4-step0'
round2 = receipt.parent / '2026-10-19-item4-step0-round2'
base = dict(os.environ, RUN_SLOW='1', BASICMODEL_DEVICE='cpu',
    PYTEST_PLUGINS='closing_probe',
    ITEM35_BIND_RECEIPT=str(receipt / ('generated-'+args.suffix)),
    PYTHONPATH=os.pathsep.join(map(str, (round2, step0, root/'bin', root/'test'))))
selections = [] if args.decomposition_only else [('native', [
    'test/test_bind_thought.py', 'test/test_math_chain_section14.py',
    'test/test_reference_close.py', 'test/test_reference_mentions.py',
    'test/test_operators_final_b.py::test_both_training_trials_share_bootstrap_snapshot'])]
selections += [('bind-decomposition-' + arm, ['test/test_math_chain_decomposition.py'])
               for arm in ('zero', 'enabled')]
for name, selectors in selections:
    arm = name.split('-')[-1] if name.startswith('bind-decomposition-') else None
    name += '-'+args.suffix
    if (receipt/name).exists():
        raise FileExistsError(receipt/name)
    env = dict(base, CLOSING_TRACE=str(receipt / (name+'-trace.jsonl')),
               SCOPED_REPORTS=str(receipt / (name+'-reports')))
    if arm is not None:
        env.update(BIND_GAIN_ARM=arm,
            PYTHONPATH=os.pathsep.join(map(str, (receipt, round2, root/'bin', root/'test'))),
            PYTEST_PLUGINS='closing_probe,bind_gate')
    command = [sys.executable, 'test/test_report.py', *selectors,
        '--workers', '3', '--memory-gib', '24', '--timeout', '900',
        '--suite-timeout', '1800', '--batch-size', '16', '--max-files', '1',
        '--run-dir', str(receipt/name)]
    with (receipt/(name+'.log')).open('w') as output:
        result = subprocess.run(command, cwd=root, env=env, stdout=output,
                                stderr=subprocess.STDOUT)
    print(name, result.returncode, flush=True)
    if result.returncode:
        sys.exit(result.returncode)
