"""Run the bank-only XOR stage-1 arms, once each. Native production arms are already complete."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT / 'doc/benchmarks/2026-09-28-item7-review')]
import bounded_tests as bounded
from review_source import supporting_inputs


def environment():
    env = bounded.worker_environment(ROOT)
    env.pop('BASIC_SEED', None)
    env.update(RUN_SLOW='1', BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager',
               BASIC_AUTOLOAD='false', OBJECTIVE_CONFLICTS_OUTPUT=str(HERE))
    return env


def main(mode):
    os.environ.update(environment())
    source, inputs = bounded.source_snapshot(ROOT), supporting_inputs(ROOT)
    fixture = HERE.parent / 'stage1/XOR_grammar_reconstruction.xml'
    fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
    observer = ROOT / 'test/objective_conflicts_probe.py'
    manifest = dict(source=source, supporting_inputs=inputs,
        observer_sha256=hashlib.sha256(observer.read_bytes()).hexdigest(),
        fixture_sha256=fixture_hash, seed=None, retries=0,
        native_batch=28, native_worker_gib=24, xor_worker_gib=8,
        aggregate_gib=24, worker_seconds=1800, mode=mode)
    bounded.write_json(HERE / (mode + '-manifest.json'), manifest)

    def verify():
        assert source == bounded.source_snapshot(ROOT)
        assert inputs == supporting_inputs(ROOT)
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == fixture_hash

    results = []
    for arm in ('step5a', 'cut'):
        verify()
        if mode == 'native':
            output = HERE / ('BasicModel_answers_tied_benchmark-' + arm)
            run_dir = HERE / ('native-' + arm + '-pytest')
            assert not output.exists() and not run_dir.exists(), 'Never overwrite or retry an arm silently.'
            result = bounded.run_suite(root=ROOT,
                selectors=['test/test_objective_conflicts_slow.py::test_native_production_objective_measurements[' + arm + ']'],
                run_dir=run_dir, memory_bytes=24*bounded.GIB, workers=1,
                worker_memory_bytes=24*bounded.GIB, timeout=1800, suite_timeout=2100,
                batch_size=1, max_files=1)
            if output.exists() and result['workers']:
                bounded.write_json(output/'process.json', result['workers'][-1])
        else:
            output = HERE / ('XOR_grammar-' + arm)
            output.mkdir(exist_ok=False)
            shutil.copyfile(observer, output/'observer-source.py.txt')
            worker = bounded.GuardedProcess(
                [sys.executable, str(observer), '--config', 'XOR_grammar',
                 '--config-xml', str(fixture), '--arm', arm, '--output', str(output)],
                cwd=ROOT, env=environment(), log_path=output/'run.log',
                memory_bytes=8*bounded.GIB, timeout=1800).start()
            started = time.monotonic()
            try:
                while (result := worker.poll()) is None:
                    bounded.write_json(output/'progress.json', dict(pid=worker.proc.pid,
                        seconds=time.monotonic()-started, memory_bytes=worker.current_memory_bytes))
                    time.sleep(.25)
            finally:
                if not worker.finished:
                    worker.stop(exit_code=130, reason='measurement_stopped')
            bounded.write_json(output/'process.json', result)
        verify()
        bounded.write_json(output/'verification.json', dict(source_matched=True,
            supporting_inputs_matched=True, fixture_matched=True))
        results.append(dict(arm=arm, reason=result['reason'], exit_code=result['exit_code']))
        print(json.dumps(results[-1]), flush=True)
    bounded.write_json(HERE/(mode+'-complete.json'), dict(results=results, source_matched=True))
    return int(any(r['exit_code'] for r in results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('xor',))
    raise SystemExit(main(parser.parse_args().mode))
