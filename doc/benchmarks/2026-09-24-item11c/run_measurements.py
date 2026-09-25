"""Preserved serial reconstruction and primitive-prior comparison for 11c."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '2026-09-21-item10'
PRIOR = HERE.parent / '2026-09-23-item11a'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import run_guarded, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ('bin', 'test')),
               BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('BASIC_SEED', None)
    source = source_snapshot(ROOT)
    scripts = (Path(__file__), BASE / 'probe.py', PRIOR / 'priors.py',
               PRIOR / 'reconstruction_ablation.py')
    manifest = dict(source=source, measurement_seed=42, completed=[],
                    probes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in scripts})
    workloads = []
    for enabled in (False, True):
        key = 'on' if enabled else 'off'
        config = ET.parse(ROOT / 'data/MM_ladder.xml')
        architecture = config.getroot().find('architecture')
        entry = architecture.find('conceptualPi')
        if entry is None:
            entry = ET.SubElement(architecture, 'conceptualPi')
        entry.text = str(enabled).lower()
        path = out / f'{key}.xml'
        config.write(path, encoding='utf-8', xml_declaration=True)
        workloads.append((key, [sys.executable, str(BASE / 'probe.py'), '--config', str(path),
                                '--out', str(out / f'{key}.json')]))
    workloads.extend([
        ('frozen', [sys.executable, str(PRIOR / 'reconstruction_ablation.py'),
                    '--config', str(out / 'off.xml'), '--out', str(out / 'frozen.json')]),
        ('priors', [sys.executable, str(PRIOR / 'priors.py'), '--out', str(out / 'priors.json')]),
    ])
    for key, command in workloads:
        receipt = run_guarded(command, cwd=ROOT, env=env, log_path=out / f'{key}.log',
                              memory_bytes=4 * 2**30, timeout=600)
        (out / f'{key}-process.json').write_text(json.dumps(receipt, indent=2) + '\n')
        manifest['completed'].append(dict(name=key, exit_code=receipt['exit_code']))
        manifest['source_unchanged'] = source_snapshot(ROOT) == source
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{key}: exit {receipt["exit_code"]}', flush=True)
        if receipt['exit_code'] or not manifest['source_unchanged']:
            raise SystemExit(receipt['exit_code'] or 1)
    baseline = json.loads((BASE / 'final-source/baseline.json').read_text())
    comparison = {'reference': 'd4dc385', 'native': {}}
    for key in ('off', 'on', 'frozen'):
        current = json.loads((out / f'{key}.json').read_text())
        comparison['native'][key] = [dict(phase=b['name'], baseline=a['reconstruction_mean'],
            current=b['reconstruction_mean'], delta=b['reconstruction_mean'] - a['reconstruction_mean'])
            for a, b in zip(baseline['phases'], current['phases'])]
    (out / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')


if __name__ == '__main__':
    main()
