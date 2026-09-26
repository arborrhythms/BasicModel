"""Repeat the unchanged item-10 measurement driver under an 8 GiB guard."""
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_guarded, source_snapshot


def main():
    output = ROOT / 'output/item9b-occurrence-measurements'
    output.mkdir(parents=True, exist_ok=True)
    source = source_snapshot(ROOT)
    (output / 'source-manifest.json').write_text(json.dumps(source, indent=2) + '\n')
    env = os.environ.copy()
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join(str(ROOT / path) for path in
                   ('bin', 'test', 'doc/benchmarks/2026-09-21-item10')))
    driver = ROOT / 'doc/benchmarks/2026-09-21-item10/probe.py'
    config = ROOT / 'doc/benchmarks/2026-09-26-item9b-corrections/measurements/parity.xml'
    results = {}
    for name, args in (
            ('serial-baseline', []),
            ('packed', ['--config', str(config), '--parity', 'packed']),
            ('single', ['--config', str(config), '--parity', 'single'])):
        results[name] = run_guarded(
            [str(ROOT / '.venv/bin/python'), str(driver), '--out',
             str(output / (name + '.json')), *args],
            cwd=ROOT, env=env, log_path=output / (name + '.log'),
            memory_bytes=8 * GIB, timeout=900)
        assert source_snapshot(ROOT) == source, 'source changed during measurement'
        (output / 'processes.json').write_text(json.dumps(results, indent=2) + '\n')
        print(name, results[name]['exit_code'], flush=True)
        if results[name]['exit_code']:
            raise SystemExit(results[name]['exit_code'])
    packed = json.loads((output / 'packed.json').read_text())['parity']
    single = json.loads((output / 'single.json').read_text())['parity']
    comparison = {
        'source_sha256': hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        'initial_parameters_equal': packed['initial_parameters_sha256'] == single['initial_parameters_sha256'],
        'initial_dictionary_equal': packed['initial_dictionary_sha256'] == single['initial_dictionary_sha256'],
        'sentences_equal': packed['sentences'] == single['sentences'],
        'packed_byte_cost': packed['mean_sentence_byte_cost'],
        'single_byte_cost': single['mean_sentence_byte_cost'],
    }
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps(comparison), flush=True)


if __name__ == '__main__':
    main()
