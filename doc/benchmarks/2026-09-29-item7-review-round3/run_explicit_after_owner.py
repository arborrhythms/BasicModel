"""Final explicit gates, with completed measurements reused only at identical source.

The complete XOR table and the two-epoch guarded gate/unguarded diagnostic are
part of this receipt. They are not repeated after an already recorded outcome.
"""
import json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'test'),str(HERE.parent/'2026-09-28-item7-review')]
from bounded_tests import GIB,run_suite,source_snapshot,documentation_snapshot,ProcessTree
from review_source import supporting_inputs
import subprocess


def diagnostic(selector, directory):
    directory.mkdir()
    started=time.monotonic()
    with (directory/'pytest.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',selector],
            cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=log,start_new_session=True)
        tree,peak,reason=ProcessTree(proc.pid),0,'completed'
        while proc.poll() is None:
            peak=max(peak,tree.sample())
            if time.monotonic()-started>1800:
                tree.terminate(proc,.5);reason='timeout';break
            time.sleep(.1)
    return dict(selector=selector,diagnostic_only=True,memory_guard=None,exit_code=proc.returncode,
        reason=reason,peak_memory_bytes=peak,elapsed_seconds=time.monotonic()-started,log=str(directory/'pytest.log'))


def main():
    out=HERE/'explicit-after-owner';out.mkdir(exist_ok=False)
    source,inputs=source_snapshot(ROOT),supporting_inputs(ROOT)
    (out/'source-manifest.json').write_text(json.dumps(dict(validated_source=source,
        supporting_inputs=inputs,recorded_documentation=documentation_snapshot(ROOT)),indent=2)+'\n')
    result=dict(reason='running',groups=[],selected=[],completed=[],workers=[],diagnostic_only=[])
    def add(group,receipt,reused=False):
        result['groups'].append(dict(reason=group['reason'],exit_code=group['exit_code'],receipt=receipt,reused=reused))
        for key in ('selected','completed','workers'):result[key].extend(group[key])
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    xor=HERE/'final-xor-after-owner'
    assert json.loads((xor/'source-manifest.json').read_text())['validated_source']==source
    for group in json.loads((xor/'result.json').read_text())['groups']:
        path=xor/group['receipt'];add(json.loads(path.read_text()),str(path.relative_to(HERE)),True)
    memory=HERE/'graph-release-candidate-after-owner'
    assert json.loads((memory/'source-manifest.json').read_text())['validated_source']==source
    add(json.loads((memory/'group-00/result.json').read_text()),'graph-release-candidate-after-owner/group-00/result.json',True)
    result['diagnostic_only'].extend(json.loads((memory/'result.json').read_text())['diagnostic_only'])
    selectors=[
        'test/test_compiled_word_chunk.py::test_tensor_peer_complete_forward_is_one_graph_across_runtime_lengths',
        'test/test_reverse_traversal.py::test_packed_trace_records_pre_fold_operand_rows_at_every_binary',
        'test/test_reverse_traversal.py::test_packed_rows_reconstruct_each_sentence_separately',
        'test/test_thinking_kernel.py',
        'test/test_packed_reconstruction_parity.py',
        'test/test_category_em_smoke.py',
        'test/test_nanochat_grammar_eval.py::test_64_word_trace_reduces_online_in_stm8_without_part_truncation',
    ]
    # The canonical selector is checked by collection; all §7 / §17 item tests
    # accompany the explicit numerical gates on this same snapshot.
    selectors += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'test').glob('test_item7_*.py'))]
    os.environ.pop('BASIC_SEED',None)
    os.environ.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false',RUN_SLOW='1',
                      PYTHONPATH=os.pathsep.join((str(ROOT/'bin'),str(ROOT/'test'))))
    for index,selector in enumerate(selectors):
        group=run_suite(root=ROOT,selectors=[selector],run_dir=out/f'group-{index:02}',memory_bytes=8*GIB,
            workers=1,worker_memory_bytes=8*GIB,timeout=1800,suite_timeout=2100,batch_size=32,max_files=1)
        assert source_snapshot(ROOT)==source and supporting_inputs(ROOT)==inputs
        add(group,f'explicit-after-owner/group-{index:02}/result.json')
        for worker in group['workers']:
            if worker['reason'] not in ('memory','aggregate_memory'):continue
            progress=Path(worker['log']).with_suffix('.json')
            active=json.loads(progress.read_text()).get('active') if progress.exists() else None
            if active:result['diagnostic_only'].append(diagnostic(active,out/f'unguarded-{index:02}'))
        assert source_snapshot(ROOT)==source
    result['reason']='gate_failures' if any(g['exit_code'] for g in result['groups']) else 'passed'
    result['exit_code']=int(result['reason']!='passed')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result['exit_code']
if __name__=='__main__':raise SystemExit(main())
