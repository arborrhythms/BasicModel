"""Capped, source-frozen supervisor for the predeclared native comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import run_guarded, source_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, choices=(1,), default=1)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--controls', nargs='+', default=['ordered', 'shuffled', 'context_free', 'reconstruction_only'])
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ('bin', 'test')),
               BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('BASIC_SEED', None)
    manifest = dict(source=source, workers=args.workers, memory_gib_per_worker=8, completed=[],
                    harness={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                             for name in ('native.py', 'run_native.py', 'PROTOCOL.md')})
    def execute(seed, control):
        name = f'{control}-{seed}'
        result = run_guarded([sys.executable, str(HERE/'native.py'), '--seed', str(seed),
            '--control', control, '--out', str(out/name)], cwd=ROOT, env=env,
            log_path=out/(name+'.log'), memory_bytes=8*2**30, timeout=7200)
        return dict(seed=seed, control=control, **result)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for seed in args.seeds:
        for control in args.controls:
            result = execute(seed, control)
            manifest['completed'].append(result)
            manifest['source_unchanged'] = source_snapshot(ROOT) == source
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(result['control'], result['seed'], 'exit', result['exit_code'], flush=True)
    assert manifest['source_unchanged']
    raise SystemExit(int(any(r['exit_code'] for r in manifest['completed'])))


if __name__ == '__main__':
    main()
