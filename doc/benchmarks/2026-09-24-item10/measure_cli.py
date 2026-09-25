"""Diagnose the inherited XOR_exact null; a fixed seed pairs measurements only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import run_guarded, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline_dir.resolve()
    source = source_snapshot(ROOT)
    manifest = dict(source=source, baseline_source=source_snapshot(baseline),
                    baseline_commit='606683a8e32aac66569669a5e607f64eeec3ae32',
                    measurement_seed=42, completed=[],
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config = ET.parse(ROOT / 'data/XOR_exact.xml')
    arch = config.getroot().find('architecture')
    ET.SubElement(arch, 'normalizeSigma').text = 'true'
    ET.SubElement(arch, 'normalizePi').text = 'false'
    configured = out / 'XOR_exact-normalized.xml'
    config.write(configured, encoding='utf-8', xml_declaration=True)
    env = dict(os.environ, BASIC_SEED='42', MODEL_COMPILE='eager',
               BASICMODEL_DEVICE='cpu', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    rows = []
    for name, root, path in (
            ('baseline', baseline, 'data/XOR_exact.xml'),
            ('current', ROOT, 'data/XOR_exact.xml'),
            ('normalized', ROOT, str(configured))):
        receipt = run_guarded([sys.executable, str(root / 'bin/Models.py'), path],
                              cwd=root, env=env, log_path=out / f'{name}.log',
                              memory_bytes=4 * 2**30, timeout=600)
        (out / f'{name}-process.json').write_text(json.dumps(receipt, indent=2) + '\n')
        text = (out / f'{name}.log').read_text()
        matches = re.findall(r'label=(-?\d+\.\d+)\s+predicted=(-?\d+\.\d+)\s+(OK|MISMATCH)', text)
        row = dict(name=name, exit_code=receipt['exit_code'], predictions=matches)
        if matches:
            row.update(output_mse=sum((float(y)-float(p))**2 for y, p, _ in matches)/len(matches),
                       reconstruction_matches=sum(ok == 'OK' for _, _, ok in matches))
        rows.append(row)
        manifest['completed'].append(row)
        manifest['source_unchanged'] = source_snapshot(ROOT) == source
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(row, flush=True)
        if receipt['exit_code'] or not manifest['source_unchanged']:
            raise SystemExit(receipt['exit_code'] or 1)
    (out / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
