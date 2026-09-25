"""Run current/selective folds on the preserved serial workload, in order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import run_guarded, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--selected-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    probe = Path(__file__).with_name('measure_native_selected_only.py' if args.selected_only
                                     else 'measure_native.py')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ('bin', 'test')),
               BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('BASIC_SEED', None)
    report = dict(source=source, probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(), runs=[])
    for enabled in (False, True):
        name = 'normalized' if enabled else 'current'
        tree = ET.parse(ROOT / 'data/MM_ladder.xml')
        arch = tree.getroot().find('architecture')
        for key, value in (('normalizeSigma', enabled), ('normalizePi', False)):
            element = arch.find(key)
            if element is None:
                element = ET.SubElement(arch, key)
            element.text = str(value).lower()
        config = out / f'{name}.xml'
        tree.write(config)
        result = run_guarded([sys.executable, str(probe), '--config', str(config),
                              '--out', str(out / f'{name}.json')], cwd=ROOT, env=env,
                             log_path=out / f'{name}.log', memory_bytes=6 * 2**30, timeout=900)
        (out / f'{name}-process.json').write_text(json.dumps(result, indent=2) + '\n')
        report['runs'].append(dict(name=name, exit_code=result['exit_code']))
        report['source_unchanged'] = source_snapshot(ROOT) == source
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, result['exit_code'], flush=True)
        if result['exit_code'] or not report['source_unchanged']:
            raise SystemExit(result['exit_code'] or 1)


if __name__ == '__main__':
    main()
