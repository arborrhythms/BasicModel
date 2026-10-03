"""Source-matched item 6.9 measurements; unchanged bounded pytest workers.

Every XOR selector is the accepted item-7 selector, with one fresh
exact round trip per the October 1 receipt rule. Concurrency changes dispatch only, using that receipt's
measured memory plus headroom, a 24 GiB aggregate cap, and 8 GiB workers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
PRIOR = HERE.parent / '2026-09-30-item7-review-round5'
sys.path.insert(0, str(MAIN / 'test'))
import bounded_tests as bounded

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
] + ['test/test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip']


def write(path, value):
    bounded.write_json(path, value)


def environment(root):
    env = bounded.worker_environment(root)
    env.pop('BASIC_SEED', None)
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager',
               BASIC_AUTOLOAD='false', RUN_SLOW='1',
               PYTHONPATH=os.pathsep.join((str(HERE), str(root / 'bin'), str(root / 'test'))))
    return env


def group(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    os.environ.update(environment(root))
    os.environ.update(PYTEST_PLUGINS='xor_observer',
                      ITEM7_XOR_GATE=str(args.gate),
                      ITEM7_XOR_MEASUREMENTS=str(output / 'observations.jsonl'))
    result = bounded.run_suite(root=root, selectors=[SELECTORS[args.gate]],
        run_dir=output / 'run', memory_bytes=8 * bounded.GIB,
        workers=1, worker_memory_bytes=8 * bounded.GIB,
        timeout=1800, suite_timeout=2100, batch_size=32, max_files=1)
    return result['exit_code']


def campaign(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = {label: Path(path).resolve() for label, path in
             (entry.split('=', 1) for entry in args.tree)}
    assert set(roots) == {'candidate'}, 'October 1: measure candidate only; accepted item 7 is the baseline'
    sources = {label: bounded.source_snapshot(root) for label, root in roots.items()}
    harness = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in HERE.glob('*.py')}
    write(output / 'plan.json', dict(roots={k: str(v) for k, v in roots.items()},
        selectors=SELECTORS, mm_trials=10 if args.mm else 0,
        no_seed=True, per_worker_bytes=8 * bounded.GIB,
        aggregate_bytes=24 * bounded.GIB, max_workers=10,
        harness=harness, started_at=time.time()))
    pending, active, completed = [], [], []
    for label, root in roots.items():
        treeout = output / label
        treeout.mkdir()
        write(treeout / 'source-manifest.json', dict(validated_source=sources[label]))
        for index, selector in enumerate(SELECTORS):
            previous = json.loads((PRIOR / f'final-xor-candidate/gate-{index:02}/result.json').read_text())
            peak = max(w['peak_memory_bytes'] for w in previous['workers'])
            reserve = min(8 * bounded.GIB, max(1.5 * bounded.GIB, 1.3 * peak + .35 * bounded.GIB))
            jobout = treeout / f'gate-{index:02}'
            pending.append(dict(label=label, kind='xor', gate=index, selector=selector,
                root=root, output=jobout, reserved_bytes=reserve,
                command=[sys.executable, str(__file__), 'group', '--root', str(root),
                         '--output', str(jobout), '--gate', str(index)]))
        if args.mm:
            mmout = treeout / 'mm'
            mmout.mkdir()
            for trial in range(10):
                pending.append(dict(label=label, kind='mm', trial=trial, root=root,
                    output=mmout / f'trial-{trial:02}', reserved_bytes=1.5 * bounded.GIB,
                    command=[sys.executable, str(HERE / 'measure_mm_grammar.py'),
                             str(mmout / f'run-{trial:02}.json')]))
    started = time.monotonic()
    peak_aggregate = 0

    def verify():
        for label, root in roots.items():
            assert bounded.source_snapshot(root) == sources[label], (label, 'source changed')
        assert all(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
                   for name, digest in harness.items()), 'measurement harness changed'

    def persist():
        write(output / 'progress.json', dict(completed=completed,
            active=[dict(label=j['label'], kind=j['kind'], gate=j.get('gate'),
                         trial=j.get('trial'), pid=j['process'].pid) for j in active],
            pending=len(pending), elapsed_seconds=time.monotonic() - started,
            peak_aggregate_memory_bytes=peak_aggregate))

    try:
        while pending or active:
            current = 0
            for job in list(active):
                proc, tree = job['process'], job['tree']
                used = tree.sample()
                job['current'] = used
                job['peak'] = max(used, job['peak'])
                current += used
                if time.monotonic() - job['started'] > 2150:
                    tree.terminate(proc, .5)
                    job['stop_reason'] = 'timeout'
                code = proc.poll()
                if code is None:
                    continue
                job['log'].close()
                info = {key: job[key] for key in ('label', 'kind', 'reserved_bytes')}
                info.update(gate=job.get('gate'), trial=job.get('trial'),
                            exit_code=code, peak_memory_bytes=job['peak'],
                            reason=job.get('stop_reason', 'exit'),
                            elapsed_seconds=time.monotonic() - job['started'])
                write(job['output'] / 'process.json', info)
                completed.append(info)
                active.remove(job)
                verify()
                print(json.dumps(info), flush=True)
            peak_aggregate = max(current, peak_aggregate)
            if current > 24 * bounded.GIB:
                largest = max(active, key=lambda j: j['current'])
                largest['stop_reason'] = 'aggregate_memory'
                largest['tree'].terminate(largest['process'], .5)
                raise RuntimeError('24 GiB aggregate guard stopped the campaign')
            occupied = sum(max(j['reserved_bytes'], j['current']) for j in active)
            while len(active) < 10:
                eligible = next((j for j in pending if j['reserved_bytes'] + occupied <= 24 * bounded.GIB), None)
                if eligible is None:
                    break
                job = eligible
                verify()
                pending.remove(job)
                job['output'].mkdir()
                log = (job['output'] / 'driver.log').open('x')
                command = (bounded.bounded_command(job['command'], 8 * bounded.GIB)
                           if job['kind'] == 'mm' else job['command'])
                proc = subprocess.Popen(command, cwd=job['root'], env=environment(job['root']),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(process=proc, tree=bounded.ProcessTree(proc.pid), log=log,
                           started=time.monotonic(), peak=0, current=0)
                active.append(job)
                occupied += job['reserved_bytes']
                print(f"START {job['label']} {job['kind']} {job.get('gate', job.get('trial'))} pid={proc.pid}", flush=True)
            persist()
            time.sleep(.25)
        verify()
        for label, root in roots.items():
            treeout = output / label
            groups, observations = [], []
            for index, selector in enumerate(SELECTORS):
                receipt = json.loads((treeout / f'gate-{index:02}/run/result.json').read_text())
                groups.append(dict(gate=index, selector=selector, reason=receipt['reason'],
                    exit_code=receipt['exit_code'], receipt=f'gate-{index:02}/run/result.json'))
                path = treeout / f'gate-{index:02}/observations.jsonl'
                if path.exists():
                    observations.extend(path.read_text().splitlines())
            (treeout / 'measurements.jsonl').write_text('\n'.join(observations) + '\n')
            write(treeout / 'result.json', dict(root=str(root), groups=groups,
                diagnostic_only=[], reason='failed' if any(g['exit_code'] for g in groups) else 'passed'))
            subprocess.run([sys.executable, str(HERE / 'summarize_xor.py'), str(treeout)], check=True)
        write(output / 'complete.json', dict(elapsed_seconds=time.monotonic() - started,
            peak_aggregate_memory_bytes=peak_aggregate, completed=completed))
    finally:
        for job in active:
            job['tree'].terminate(job['process'], .5)
            job['log'].close()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    child = commands.add_parser('group')
    child.add_argument('--root', required=True)
    child.add_argument('--output', required=True)
    child.add_argument('--gate', type=int, required=True)
    parent = commands.add_parser('campaign')
    parent.add_argument('--output', required=True)
    parent.add_argument('--tree', action='append', required=True)
    parent.add_argument('--mm', action='store_true')
    arguments = parser.parse_args()
    raise SystemExit(group(arguments) if arguments.command == 'group' else campaign(arguments))
