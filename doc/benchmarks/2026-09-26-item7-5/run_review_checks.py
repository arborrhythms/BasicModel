"""One bounded worker for final compiler, ownership, gate and parity receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def main():
    source = source_snapshot(ROOT)
    env = os.environ.copy()
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='1',
               BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT/'bin'))
    jobs = {
        'explicit-final': [
            'test/test_compiled_word_chunk.py::test_tensor_peer_complete_forward_is_one_graph_across_runtime_lengths',
            'test/test_word_store.py::test_forward_records_kind_tagged_trace',
            'test/test_reverse_traversal.py::test_packed_trace_records_pre_fold_operand_rows_at_every_binary',
            'test/test_reverse_traversal.py::test_packed_rows_reconstruct_each_sentence_separately'],
        'trie-verified': [
            'test/test_compose_records.py',
            'test/test_word_store.py::test_two_epoch_training_severs_cross_batch_graph'],
        'xor-mm-final': [
            'test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy',
            'test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct',
            'test/test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal'],
    }
    results = {}
    for name, selectors in jobs.items():
        command = [sys.executable, 'test/test_report.py', *selectors,
                   '--workers', '1', '--memory-gib', '8', '--max-files', '1',
                   '--batch-size', '1', '--run-dir', str(HERE/name)]
        results[name] = subprocess.run(command, cwd=ROOT, env=env).returncode
        assert source_snapshot(ROOT) == source, 'source changed during review checks'
        (HERE/'review-checks-processes.json').write_text(json.dumps(results, indent=2)+'\n')
    results['measurements'] = subprocess.run(
        [sys.executable, str(HERE/'run_measurements.py')], cwd=ROOT, env=env).returncode
    assert source_snapshot(ROOT) == source, 'source changed during measurements'
    (HERE/'review-checks-processes.json').write_text(json.dumps(results, indent=2)+'\n')
    return int(any(results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
