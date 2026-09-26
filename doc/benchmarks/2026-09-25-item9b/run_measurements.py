"""Re-run the published serial and packed receipts under an 8 GiB cap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import run_guarded, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    previous = HERE.parent / '2026-09-25-item9-parity'
    config = ET.parse(previous / 'parity.xml')
    parts = config.getroot().find('PartSpace')
    parts.remove(parts.find('maxVectors'))
    config.write(out / 'parity.xml', encoding='unicode')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ('bin', 'test')),
               BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('BASIC_SEED', None)
    source = source_snapshot(ROOT)
    manifest = dict(source=source, seed=42, completed=[],
                    configuration_migration='remove retired maxVectors; retain nVectors=4096',
                    config_sha256=hashlib.sha256((out / 'parity.xml').read_bytes()).hexdigest())
    for mode in ('baseline', 'packed', 'single', 'comparison'):
        if mode == 'baseline':
            command = [sys.executable, str(HERE.parent / '2026-09-21-item10/probe.py'),
                       '--out', str(out / 'baseline.json')]
        elif mode == 'comparison':
            command = [sys.executable, str(previous / 'compare.py'), str(out)]
        else:
            command = [sys.executable, str(previous / 'probe.py'), '--config', str(out / 'parity.xml'),
                       '--parity', mode, '--out', str(out / (mode + '.json'))]
        result = run_guarded(command, cwd=ROOT, env=env, log_path=out / (mode + '.log'),
                             memory_bytes=8 * 2**30, timeout=600)
        manifest['completed'].append(result)
        manifest['source_unchanged'] = source_snapshot(ROOT) == source
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if result['exit_code'] or not manifest['source_unchanged']:
            raise SystemExit(result['exit_code'] or 1)
        print(mode + ': completed', flush=True)


if __name__ == '__main__':
    main()
