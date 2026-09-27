"""Item 7.5: repeat the reviewed controls; report movement without tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_guarded, source_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parity-only', action='store_true')
    args = parser.parse_args()
    output = ROOT / ('doc/benchmarks/2026-09-26-item7-5/parity-measurements' if args.parity_only
                     else 'doc/benchmarks/2026-09-26-item7-5/measurements')
    output.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    (output / 'source-manifest.json').write_text(json.dumps(source, indent=2) + '\n')
    env = os.environ.copy()
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
            ('bin', 'test', 'doc/benchmarks/2026-09-21-item10')))
    baseline = ROOT / 'doc/benchmarks/2026-09-26-item8/measure.py'
    legacy = ROOT / 'doc/benchmarks/2026-09-21-item10/probe.py'
    config = ROOT / 'doc/benchmarks/2026-09-26-item7-5/parity.xml'
    drivers = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (Path(__file__), baseline, legacy, config)}
    (output / 'drivers.json').write_text(json.dumps(drivers, indent=2) + '\n')
    jobs = [] if args.parity_only else [('serial-baseline', [baseline, 'baseline'])]
    jobs += [(layout, [legacy, '--config', config, '--parity', layout])
             for layout in ('packed', 'single')]
    results = {}
    for name, args in jobs:
        results[name] = run_guarded([str(ROOT / '.venv/bin/python'),
            *map(str, args), '--out', str(output / (name + '.json'))], cwd=ROOT, env=env,
            log_path=output / (name + '.log'), memory_bytes=8 * GIB, timeout=1200)
        assert source_snapshot(ROOT) == source, 'source changed during measurement'
        (output / 'processes.json').write_text(json.dumps(results, indent=2) + '\n')
        print(name, results[name]['exit_code'], flush=True)
    return int(any(r['exit_code'] for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
