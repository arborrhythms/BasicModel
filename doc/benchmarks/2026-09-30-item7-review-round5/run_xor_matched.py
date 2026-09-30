"""All XOR gates, source matched, with separately recorded unguarded diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, ProcessTree, run_suite, source_snapshot, documentation_snapshot, worker_environment

SELECTORS = [
    'test/test_grounded_xor.py',
    'test/test_concept_output.py',
    'test/test_mm_xor.py',
    'test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp',
    'test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct',
    'test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy',
    'test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct',
    'test/test_basicmodel.py::TestSPNN::test_xor_training',
    'test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor]',
    'test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise]',
    'test/test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live',
    'test/test_reconstruction_roundtrip.py::test_xor_recon_grads_flow',
    'test/test_reconstruction_roundtrip.py::test_xor_percepts_tile_words',
    'test/test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget',
] + ['test/test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip'] * 15



def diagnostic(selector, directory):
    """One unguarded repeat, with a time deadline and measured memory, never a gate."""
    directory.mkdir()
    started = time.monotonic()
    with (directory / 'pytest.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'pytest', '-q', '--tb=short',
                                 '-p', 'no:cacheprovider', selector],
                                cwd=ROOT, env=worker_environment(ROOT), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        tree = ProcessTree(proc.pid)
        peak = 0
        reason = 'completed'
        while proc.poll() is None:
            peak = max(peak, tree.sample())
            if time.monotonic() - started > 1800:
                tree.terminate(proc, .5)
                reason = 'timeout'
                break
            time.sleep(.1)
    result = dict(selector=selector, diagnostic_only=True, memory_guard=None,
                  returncode=proc.returncode, reason=reason, peak_memory_bytes=peak,
                  elapsed_seconds=time.monotonic() - started)
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    output = HERE / sys.argv[1]
    output.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    (output / 'source-manifest.json').write_text(json.dumps(dict(
        validated_source=source, recorded_documentation=documentation_snapshot(ROOT)), indent=2) + '\n')
    os.environ.pop('BASIC_SEED', None)
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager',
                      BASIC_AUTOLOAD='false', RUN_SLOW='1',
                      PYTHONPATH=os.pathsep.join((str(HERE), str(ROOT / 'bin'), str(ROOT / 'test'))),
                      PYTEST_PLUGINS='xor_observer_round5',
                      ITEM7_XOR_MEASUREMENTS=str(output / 'measurements.jsonl'))
    result = dict(root=str(ROOT), groups=[], diagnostic_only=[], reason='running')
    for index, selector in enumerate(SELECTORS):
        os.environ['ITEM7_XOR_GATE'] = str(index)
        print(f'GATE {index}: {selector}', flush=True)
        group = run_suite(root=ROOT, selectors=[selector], run_dir=output / f'gate-{index:02}',
                          memory_bytes=8 * GIB, workers=1, worker_memory_bytes=8 * GIB,
                          timeout=1800, suite_timeout=2100, batch_size=32, max_files=1)
        assert source_snapshot(ROOT) == source, 'tested source changed'
        result['groups'].append(dict(gate=index, selector=selector, reason=group['reason'],
                                     exit_code=group['exit_code'], receipt=f'gate-{index:02}/result.json'))
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        for worker in group['workers']:
            if worker['reason'] in ('memory', 'aggregate_memory'):
                progress = Path(worker['log']).with_suffix('.json')
                active = json.loads(progress.read_text()).get('active') if progress.exists() else None
                if active:
                    result['diagnostic_only'].append(diagnostic(active, output / f'unguarded-{index:02}'))
        assert source_snapshot(ROOT) == source, 'tested source changed during diagnostic'
    result['reason'] = 'failed' if any(g['exit_code'] for g in result['groups']) else 'passed'
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return int(result['reason'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
