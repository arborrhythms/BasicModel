"""Item 6.2: exactly ten shared XOR gates, ten unchanged MM gates, ten sum controls."""
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE/'measurements'
PRIOR = HERE.parent/'2026-10-03-operators-attention'
sys.path[:0] = [str(ROOT/'test'), str(HERE), str(PRIOR)]
import bounded_tests as bounded

XOR = ['test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy',
       'test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct']
MM = ['test/test_mm_xor.py::TestMMXorConvergence::test_convergence']
WORKERS = min(2, max(1, (os.cpu_count() or 1)-4))


def environment():
    env = bounded.worker_environment(ROOT)
    env.pop('BASIC_SEED', None)
    env.pop('OPERATORS_REPLAY_RNG', None)
    env.pop('OPERATORS_DISABLE', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.pop('OWNERSHIP_OBSERVER_OUTPUT', None)
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='none', RUN_SLOW='1',
               BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false',
               PYTHONPATH=os.pathsep.join((str(HERE), str(PRIOR), str(ROOT/'bin'), str(ROOT/'test'))))
    return env


def sum_child(folder):
    source = ROOT/'doc/benchmarks/2026-10-01-item6-9-review/separator_campaign.py'
    spec = importlib.util.spec_from_file_location('prior_sum_control', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from operators_gate_observer import capture_final_derivations
    from operators_run_audit import observe_run, json_value
    from rng_replay import entry, switches
    from thinking_observer import observe
    entry(folder)
    with observe(folder/'thinking.json'), switches(), capture_final_derivations() as derivations, observe_run() as run_audit:
        module.child('sum', folder, str(OUT/'XOR_grammar_sum_control.xml'))
    (folder/'run-audit.json').write_text(json.dumps(run_audit, indent=2, default=json_value)+'\n')
    path = folder/'measurement.json'
    observation = json.loads(path.read_text())
    observation['sum_bar'] = abs(observation['contrast']) <= 1e-4 and not observation['class_bar']
    observation['floor_bar'] = observation['sum_bar'] and abs(observation['mse']-.25) <= .02
    observation['criterion'] = 'abs(checkerboard contrast) <= 1e-4 and class bar not met (unchanged operators final-b control)'
    observation['final_greedy_compose'] = [derivations[key] for key in sorted(derivations)]
    observation['storage'] = run_audit['storage']
    observation['storage_after_training'] = run_audit['storage_after_training']
    bounded.write_json(path, observation)


def campaign():
    source = bounded.source_snapshot(ROOT)
    assert source == json.loads((HERE/'measured-source/source.json').read_text())
    helpers = json.loads((HERE/'measured-source/measurement-helpers.json').read_text())
    def helpers_match():
        return all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
                   for name,sha in helpers.items())
    assert helpers_match()
    from verification import validate
    validate(source)
    OUT.mkdir(exist_ok=False)
    bounded.write_json(OUT/'source.json', source)
    original = (ROOT/'data/XOR_grammar.xml').read_text()
    rules = '<rule>S = not.forward(S)</rule>\n            <rule>S = conjunction.forward(S, S)</rule>\n            <rule>S = disjunction.forward(S, S)</rule>'
    assert original.count(rules) == 1
    control = original.replace(rules, '<rule>S = sum.forward(S, S)</rule>')
    (OUT/'XOR_grammar_sum_control.xml').write_text(control)
    (OUT/'sum-control.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), control.splitlines(True),
        fromfile='data/XOR_grammar.xml', tofile='receipt/XOR_grammar_sum_control.xml')))
    jobs = [dict(kind=kind, run=run, name=f'{kind}-{run:02}')
            for kind in ('sum', 'xor', 'mm') for run in range(1, 11)]
    bounded.write_json(OUT/'plan.json', dict(jobs=jobs, selectors=dict(xor=XOR, mm=MM),
        epochs=dict(xor=400, sum=400, mm_maximum=200), seed=None,
        ownership='xor-10; same single training consumed by both unchanged bars',
        bands=dict(at_zero='MSE < .05', at_quarter='abs(MSE - .25) <= .02',
                   between='remaining MSE < .25', above_quarter='remaining MSE > .25'),
        comparison=dict(round3a_landing=dict(xor_class=10,xor_reconstruction=10,joint=10,sum=10,mm=10),round2_landing=dict(xor_class=9,xor_reconstruction=10,joint=9,sum=10,mm=10),section22=dict(xor_class=7,xor_reconstruction=9,joint=6,sum=10,mm=10),section12=dict(xor_class=0,xor_reconstruction=7,joint=0,sum=10,mm=10),
            section13=dict(xor_class=0,xor_reconstruction=0,joint=0,sum=10,mm=10),
            section14=dict(xor_class=1,xor_reconstruction=8,joint=1,sum=10,mm=10),
            section17=dict(xor_class=1,xor_reconstruction=10,joint=1,sum=10,mm=10),
            section18=dict(xor_class=0,xor_reconstruction=10,joint=0,sum=10,mm=10),
            section20=dict(xor_class=2,xor_reconstruction=8,joint=2,sum=10,mm=10),
            closing69=dict(class_mse=.1147481948,reconstructed_sentences=0,total_sentences=4,
                           ownership_conflicts=0,prior_class=9,prior_reconstruction=5)),
        order='Sum ten and read the 10/10 control; XOR ten; then unchanged raw-forward MM ten. Complete every declared run even if a gate misses.',
        worker_bytes=8*bounded.GIB, worker_timeout=1800,
        maximum_workers=WORKERS, aggregate_reservation_bytes=16*bounded.GIB,
        retries=0, commits=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    active, done, sum_read = [], [], False
    start = time.monotonic()
    try:
        while jobs or active:
            assert source == bounded.source_snapshot(ROOT), 'source changed during item 6.2 measurements'
            assert helpers_match(), 'measurement helpers changed after freeze'
            for job in list(active):
                result = job['process'].poll()
                if result is not None:
                    bounded.write_json(job['folder']/'process.json', result)
                    done.append(dict(kind=job['kind'], run=job['run'], name=job['name'], process=result))
                    active.remove(job)
                    print(json.dumps(dict(name=job['name'], exit_code=result['exit_code'], reason=result['reason'])), flush=True)
            if sum(job['process'].current_memory_bytes for job in active) > 16*bounded.GIB:
                raise RuntimeError('aggregate memory guard')
            while jobs and len(active) < WORKERS:
                if jobs[0]['kind'] != 'sum' and not sum_read:
                    if any(job['kind'] == 'sum' for job in active):
                        break
                    observations = [json.loads((OUT/f'sum-{run:02}'/'measurement.json').read_text())
                                    for run in range(1, 11)]
                    passed = sum(row['sum_bar'] for row in observations)
                    bounded.write_json(OUT/'sum-read-first.json', dict(passed=passed,
                        total=10, before_xor_or_mm=True,
                        completed_processes=[row for row in done if row['kind']=='sum'],
                        observations=observations))
                    print(json.dumps(dict(sum_read_first=passed, total=10)), flush=True)
                    assert len(observations)==10, 'complete the unchanged sum control first'
                    sum_read = True
                if jobs[0]['kind'] == 'mm' and any(job['kind'] == 'xor' for job in active):
                    break
                job = jobs.pop(0)
                folder = OUT/job['name']
                folder.mkdir()
                env = environment()
                if job['kind'] == 'sum':
                    command = [sys.executable, str(Path(__file__).resolve()), 'sum', str(folder)]
                else:
                    env.update(PYTEST_PLUGINS='operators_gate_observer,thinking_observer',
                        THINKING_OBSERVER_OUTPUT=str(folder/'thinking.json'),
                        ITEM7_XOR_GATE='5' if job['kind']=='xor' else '2',
                        ITEM7_XOR_MEASUREMENTS=str(folder/'observations.jsonl'),
                        REVIEW17_REPORTS=str(folder/'reports.jsonl'))
                    if job['kind']=='xor' and job['run']==10:
                        env['OWNERSHIP_OBSERVER_OUTPUT'] = str(folder/'ownership')
                    command = [sys.executable, '-m', 'pytest', '-q', *(XOR if job['kind']=='xor' else MM)]
                process = bounded.GuardedProcess(command, cwd=ROOT, env=env,
                    log_path=folder/'run.log', memory_bytes=8*bounded.GIB, timeout=1800).start()
                active.append(dict(job, folder=folder, process=process))
            bounded.write_json(OUT/'progress.json', dict(done=done, pending=len(jobs),
                active=[dict(name=job['name'], pid=job['process'].proc.pid,
                             memory_bytes=job['process'].current_memory_bytes) for job in active],
                seconds=time.monotonic()-start))
            time.sleep(.5)
    finally:
        for job in active:
            job['process'].stop(exit_code=130, reason='campaign_stopped')
    bounded.write_json(OUT/'complete.json', dict(jobs=done, source_matched=source==bounded.source_snapshot(ROOT),
        completed=len(done)==30, seconds=time.monotonic()-start, retries=0))


if __name__ == '__main__':
    if len(sys.argv)>1:
        sum_child(Path(sys.argv[2]))
    else:
        campaign()
