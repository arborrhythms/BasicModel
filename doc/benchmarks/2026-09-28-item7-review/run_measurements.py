"""Reissue the reviewed fixed-seed reconstruction protocol, with CPU timing."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_guarded, source_snapshot


def main():
    output = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'measurements')
    output.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    (output / 'source-manifest.json').write_text(json.dumps(source, indent=2) + '\n')
    env = os.environ.copy()
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
            ('bin', 'test', 'doc/benchmarks/2026-09-28-item7-review')))
    baseline = HERE / 'profile_baseline.py'
    legacy = HERE / 'probe.py'
    parity = HERE / 'parity.py'
    config = HERE / 'parity.xml'
    drivers = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), baseline, legacy, parity, config,
                  HERE / 'measure.py')}
    (output / 'drivers.json').write_text(json.dumps(drivers, indent=2) + '\n')
    jobs = [('serial-baseline', [baseline])]
    jobs += [(layout, [legacy, '--config', config, '--parity', layout])
             for layout in ('packed', 'single')]
    results = {}
    for name, arguments in jobs:
        results[name] = run_guarded([str(ROOT / '.venv/bin/python'),
            *map(str, arguments), '--out', str(output / (name + '.json'))], cwd=ROOT, env=env,
            log_path=output / (name + '.log'), memory_bytes=8 * GIB, timeout=1200)
        assert source_snapshot(ROOT) == source, 'source changed during measurement'
        (output / 'processes.json').write_text(json.dumps(results, indent=2) + '\n')
        print(name, results[name]['exit_code'], flush=True)
    return int(any(r['exit_code'] for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
