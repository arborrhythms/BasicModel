"""Finish the repaired source's measurements and its one complete sweep."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HEAD = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot

source = source_snapshot(ROOT)
status = dict(stage='waiting_for_item7_acceptance', steps=[], finished=False,
              retained_controls=['head-reconstruction', 'head-mm-grammar'],
              prior_candidate_receipts_retained=True)
status_path = HERE / 'finish-after-owner-status.json'


def save():
    status_path.write_text(json.dumps(status, indent=2) + '\n')


def run(stage, arguments, allowed=(0, 1, 137)):
    assert source_snapshot(ROOT) == source
    status['stage'] = stage
    save()
    print('START', stage, flush=True)
    started = time.monotonic()
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT)
    status['steps'].append(dict(stage=stage, exit_code=result.returncode,
                                seconds=time.monotonic() - started))
    save()
    assert source_snapshot(ROOT) == source
    if result.returncode not in allowed:
        raise RuntimeError(f'{stage}: infrastructure exit {result.returncode}')
    print('END', stage, result.returncode, flush=True)


save()
try:
    while True:
        result = json.loads((HERE / 'item7-after-owner/result.json').read_text())
        if result['reason'] != 'running':
            assert result['reason'] == 'passed', 'acceptance files failed before measurements'
            break
        time.sleep(5)
    assert json.loads((HERE / 'item7-after-owner/source-manifest.json').read_text())['validated_source'] == source
    assert json.loads((HERE / 'head-reconstruction/source-manifest.json').read_text()) == source_snapshot(HEAD)
    assert json.loads((HERE / 'head-mm-grammar/source-manifest.json').read_text())['validated_source'] == source_snapshot(HEAD)
    os.environ['ITEM7_RUN_SLOW'] = '1'
    run('graph_release_final_source', [HERE / 'run_checks.py', 'graph-release-candidate-after-owner', ROOT,
        'test/test_word_store.py::test_two_epoch_training_severs_cross_batch_graph'])
    run('candidate_reconstruction_final_source', [HERE / 'run_candidate_remeasure.py', 'reconstruction'], allowed=(0,))
    os.environ['ITEM7_RUN_SLOW'] = '0'
    run('native_definition_context_final_source', [HERE / 'run_checks.py', 'native-definition-context-after-owner',
        ROOT, str(HERE / 'probe_definition_context_after_owner.py') + '::test_definition_share_of_native_predictor_reads'])
    run('final_source_xor', [HERE / 'run_xor.py', 'final-xor-after-owner', ROOT])
    run('summarize_final_source_xor', [HERE / 'summarize_xor.py', HERE / 'final-xor-after-owner'], allowed=(0,))
    run('candidate_mm_grammar_final_source', [HERE / 'run_candidate_remeasure.py', 'mm-grammar'], allowed=(0,))
    run('summarize_final_source_measurements', [HERE / 'summarize_measurements_final.py'], allowed=(0,))
    run('explicit_final_source', [HERE / 'run_explicit_after_owner.py'])
    run('summarize_explicit_final_source', [HERE / 'summarize_explicit_after_owner.py'], allowed=(0,))
    run('one_full_sweep', [HERE / 'run_full_sweep_after_owner.py'])
    result = json.loads((HERE / 'full-sweep/run/result.json').read_text())
    if set(result['selected']) != set(result['completed']):
        run('unfinished_sweep_coverage', [HERE / 'continue_sweep.py'], allowed=(137,))
    run('summarize_sweep', [HERE / 'summarize_sweep.py'], allowed=(0,))
    status.update(stage='ready_for_receipt_review', finished=True)
except BaseException as error:
    status.update(stage='infrastructure_error', error=repr(error))
    raise
finally:
    save()
