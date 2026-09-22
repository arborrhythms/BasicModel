"""Repeat d4dc385's native/packed/single reconstruction with pi off and on."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'doc/benchmarks/2026-09-21-item10'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import run_guarded, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / d) for d in ('bin', 'test')),
               BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('BASIC_SEED', None)
    source = source_snapshot(ROOT)
    manifest = dict(validated_source=source, reconstruction_measurement_seed=42, completed=[])
    workloads = [('rungs', [sys.executable, str(Path(__file__).with_name('rungs.py')), str(out / 'rungs.json')])]
    for enabled in (False, True):
        name = 'on' if enabled else 'off'
        for workload in ('baseline', 'packed', 'single'):
            config = ET.parse(ROOT / 'data/MM_ladder.xml' if workload == 'baseline' else BASE / 'parity.xml')
            architecture = config.getroot().find('architecture')
            ET.SubElement(architecture, 'conceptualPi').text = str(enabled).lower()
            path = out / f'{name}-{workload}.xml'
            config.write(path, encoding='utf-8', xml_declaration=True)
            key = f'{name}-{workload}'
            command = [sys.executable, str(BASE / 'probe.py'), '--config', str(path), '--out', str(out / f'{key}.json')]
            if workload != 'baseline':
                command += ['--parity', workload]
            workloads.append((key, command))
    for key, command in workloads:
        receipt = run_guarded(command, cwd=ROOT, env=env, log_path=out / f'{key}.log',
                              memory_bytes=8 * 2**30, timeout=600)
        (out / f'{key}-process.json').write_text(json.dumps(receipt, indent=2) + '\n')
        manifest['completed'].append(dict(name=key, exit_code=receipt['exit_code']))
        manifest['source_unchanged'] = source_snapshot(ROOT) == source
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{key}: exit {receipt["exit_code"]}', flush=True)
        if receipt['exit_code'] or not manifest['source_unchanged']:
            raise SystemExit(receipt['exit_code'] or 1)
    old = json.loads((BASE / 'final-source/baseline.json').read_text())
    compare = {'reference': 'd4dc385', 'native': {}, 'parity_payload_identical': {}}
    for setting in ('off', 'on'):
        new = json.loads((out / f'{setting}-baseline.json').read_text())
        compare['native'][setting] = [dict(phase=b['name'], baseline=a['reconstruction_mean'],
                                         current=b['reconstruction_mean'], delta=b['reconstruction_mean']-a['reconstruction_mean'])
                                      for a, b in zip(old['phases'], new['phases'])]
        for mode in ('packed', 'single'):
            previous = json.loads((BASE / f'final-source/{mode}.json').read_text())['parity']
            current = json.loads((out / f'{setting}-{mode}.json').read_text())['parity']
            compare['parity_payload_identical'][f'{setting}-{mode}'] = current == previous
    (out / 'comparison.json').write_text(json.dumps(compare, indent=2) + '\n')


if __name__ == '__main__':
    main()
