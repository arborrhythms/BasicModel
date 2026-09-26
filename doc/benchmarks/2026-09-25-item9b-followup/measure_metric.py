"""Measure only the fixed-probe distance reducer, with no model or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def worker(out):
    sys.path.insert(0, str(ROOT / 'bin'))
    import torch
    from CategoricalDiscrimination import FIXED_PROBES, fixed_probe_discrimination
    torch.set_num_threads(1)
    # Full physical production inventory, both poles. Synthetic captured
    # readings exercise the cost; these are not model-quality measurements.
    width = 2 * 65536
    basis = torch.linspace(0., 1., width)
    readings = {name: torch.linspace(0., 1., len(probes['texts']))[:, None] * basis
                for name, probes in FIXED_PROBES.items()}
    fixed_probe_discrimination(readings)
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        result = fixed_probe_discrimination(readings)
        durations.append(time.perf_counter() - start)
    out.write_text(json.dumps(dict(
        scope='Reducer only; no probe inference, model mutation, training or device transfer measured',
        device='cpu', threads=1, physical_concept_rows=65536, poles=2,
        probes={name: len(p['texts']) for name, p in FIXED_PROBES.items()},
        durations_seconds=durations, median_seconds=statistics.median(durations),
        input_bytes=sum(x.numel()*x.element_size() for x in readings.values()),
        example=result), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.worker:
        worker(args.out)
        return
    sys.path.insert(0, str(ROOT / 'test'))
    from bounded_tests import run_guarded, source_snapshot
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    result = run_guarded([sys.executable, __file__, '--worker', '--out', str(out/'metric.json')],
        cwd=ROOT, env=dict(os.environ, BASICMODEL_DEVICE='cpu'),
        log_path=out/'worker.log', memory_bytes=8*2**30, timeout=120)
    (out/'manifest.json').write_text(json.dumps(dict(source=source,
        source_unchanged=source_snapshot(ROOT)==source, result=result), indent=2)+'\n')
    if result['exit_code']:
        raise SystemExit(result['exit_code'])
    assert source_snapshot(ROOT) == source
    print((out/'metric.json').read_text())


if __name__ == '__main__':
    main()
