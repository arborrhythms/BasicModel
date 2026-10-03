"""Ten shared XOR trainings (both bars per run), ten sum controls, one table, ten MM.

Both named XOR_grammar rows reuse the predeclared first shared training.
No HEAD run, selected initialization, or additional audit training.
"""
import argparse, difflib, hashlib, json, os, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'test'),str(HERE)]
import bounded_tests as bounded
SELECTORS={
0:'test/test_grounded_xor.py',1:'test/test_concept_output.py',2:'test/test_mm_xor.py',
3:'test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp',
4:'test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct',
5:'test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy',
6:'test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct',
8:'test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor]',
9:'test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise]',
10:'test/test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live',
11:'test/test_reconstruction_roundtrip.py::test_xor_recon_grads_flow',
12:'test/test_reconstruction_roundtrip.py::test_xor_percepts_tile_words',
13:'test/test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget',
14:'test/test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip',
}

def environment():
    env=bounded.worker_environment(ROOT)
    env.pop('BASIC_SEED',None)
    env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',RUN_SLOW='1',BASIC_AUTOLOAD='false',
        PYTHONPATH=os.pathsep.join((str(HERE),str(ROOT/'bin'),str(ROOT/'test'))))
    return env

def child(kind, output, gate=None):
    os.environ.update(environment());output.mkdir(parents=True,exist_ok=True)
    if kind in ('gate', 'xor'):
        os.environ.update(PYTEST_PLUGINS='xor_observer', ITEM7_XOR_GATE=str(5 if kind=='xor' else gate),
                          ITEM7_XOR_MEASUREMENTS=str(output/'observations.jsonl'))
        if kind == 'xor' and output.name == 'xor-01':
            os.environ['OWNERSHIP_OBSERVER_OUTPUT'] = str(HERE/'xor-ownership')
        selectors=[SELECTORS[5],SELECTORS[6]] if kind=='xor' else [SELECTORS[gate]]
        result=bounded.run_suite(root=ROOT,selectors=selectors,run_dir=output/'run',
            memory_bytes=8*bounded.GIB,worker_memory_bytes=8*bounded.GIB,workers=1,
            timeout=1800,suite_timeout=2100,batch_size=32,max_files=1)
        return result['exit_code']
    if kind=='mm':
        import runpy
        sys.path.insert(0,str(ROOT/'bin'))
        sys.argv=[str(HERE/'measure_mm_grammar.py'),str(output/'measurement.json')]
        runpy.run_path(sys.argv[0],run_name='__main__')
        return 0
    if kind=='sum':
        import importlib.util
        sys.path.insert(0,str(ROOT/'bin'))
        source=ROOT/'doc/benchmarks/2026-10-01-item6-9-review/separator_campaign.py'
        spec=importlib.util.spec_from_file_location('prior_sum_control',source)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.child('sum',output,str(HERE/'XOR_grammar_sum_control.xml'))
        path=output/'measurement.json'
        observation=json.loads(path.read_text())
        observation['sum_bar']=abs(observation['contrast']) <= 1e-4 and not observation['class_bar']
        observation['criterion']='abs(checkerboard contrast) <= 1e-4 and class bar not met (plan section 14)'
        bounded.write_json(path,observation)
        return 0
    raise ValueError(kind)

def campaign():
    out=HERE/'measurements';out.mkdir(exist_ok=False)
    old=(ROOT/'data/XOR_grammar.xml').read_text()
    rule='<rule>S = not.forward(S)</rule>\n            <rule>S = conjunction.forward(S, S)</rule>\n            <rule>S = disjunction.forward(S, S)</rule>'
    assert rule in old
    new=old.replace(rule,'<rule>S = sum.forward(S, S)</rule>')
    (HERE/'XOR_grammar_sum_control.xml').write_text(new)
    (HERE/'sum-control.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='data/XOR_grammar.xml',tofile='receipt/sum_control.xml')))
    source=bounded.source_snapshot(ROOT)
    bounded.write_json(out/'manifest.json',dict(source=source,seed=None,selectors=SELECTORS,
        xor_trainings=10,bars_per_training=['class','reconstruction','both'],table_rows_from='xor-01',sum_runs=10,mm_runs=10,exact_roundtrips=1,
        worker_gib=8,max_workers=3,own_maximum_gib=24,combined_with_weekly_gib=24,
        while_native='native measurement runs separately under its 24 GiB guard',
        removed=dict(selector='test/test_basicmodel.py::TestSPNN::test_xor_training',reason='suite-trim item 3: inline smoke test, not a proof'),
        baseline='accepted item 7: 44/49, exact 12/15, MM median ending MSE .1066; no HEAD run'))
    jobs=[]
    jobs.extend(dict(kind='xor',gate=None,trial=i,name=f'xor-{i:02}') for i in range(1,11))
    jobs.extend(dict(kind='sum',gate=None,trial=i,name=f'sum-{i:02}') for i in range(1,11))
    jobs.extend(dict(kind='gate',gate=g,trial=1,name=f'gate-{g:02}') for g in SELECTORS if g not in (5,6))
    jobs.extend(dict(kind='mm',gate=None,trial=i,name=f'mm-{i:02}') for i in range(1,11))
    active=[];done=[];start=time.monotonic()
    try:
        while jobs or active:
            assert source==bounded.source_snapshot(ROOT), 'candidate changed during closing measurements'
            for job in list(active):
                result=job['process'].poll()
                if result is not None:
                    bounded.write_json(job['output']/'process.json',result)
                    done.append({k:v for k,v in job.items() if k not in ('process','output')}|dict(process=result))
                    active.remove(job)
                    print(json.dumps(dict(name=job['name'],exit_code=result['exit_code'],reason=result['reason'])),flush=True)
            used=sum(j['process'].current_memory_bytes for j in active)
            if used>24*bounded.GIB and active:
                max(active,key=lambda j:j['process'].current_memory_bytes)['process'].stop(
                    exit_code=137,reason='combined_memory')
            while jobs and len(active)<3:
                job=jobs.pop(0);dest=out/job['name'];dest.mkdir()
                command=[sys.executable,str(Path(__file__).resolve()),'child','--kind',job['kind'],'--output',str(dest)]
                if job['gate'] is not None:command+=['--gate',str(job['gate'])]
                process=bounded.GuardedProcess(command,cwd=ROOT,env=environment(),log_path=dest/'driver.log',
                    memory_bytes=8*bounded.GIB,timeout=2150).start()
                active.append(job|dict(process=process,output=dest))
            bounded.write_json(out/'progress.json',dict(done=done,pending=len(jobs),seconds=time.monotonic()-start,
                active=[dict(name=j['name'],pid=j['process'].proc.pid,memory_bytes=j['process'].current_memory_bytes) for j in active]))
            time.sleep(.5)
    finally:
        for j in active:j['process'].stop(exit_code=130,reason='campaign_stopped')
    bounded.write_json(out/'complete.json',dict(jobs=done,seconds=time.monotonic()-start,source_matched=source==bounded.source_snapshot(ROOT)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('campaign','child'));p.add_argument('--kind');p.add_argument('--output',type=Path);p.add_argument('--gate',type=int);a=p.parse_args()
    if a.mode=='campaign':campaign()
    else:raise SystemExit(child(a.kind,a.output.resolve(),a.gate))
