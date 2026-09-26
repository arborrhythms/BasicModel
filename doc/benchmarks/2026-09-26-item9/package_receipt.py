"""Preserve every declared learning arm and the harness diagnostics."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def compressed(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))


def main():
    source = source_snapshot(ROOT)
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    assert source == json.loads((HERE.parent / '2026-09-26-item9b-corrections' /
                                'source-manifest.json').read_text())
    final_runs, artifacts, processes = set(), {}, []
    for seed in (0, 1, 2):
        run = ROOT / 'output' / f'item9-native-release-{seed}'
        final_runs.add(run)
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['source'] == source and manifest['source_unchanged']
        assert len(manifest['completed']) == 4
        assert all(r['exit_code'] == 0 for r in manifest['completed'])
        for name, digest in manifest['harness'].items():
            assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
        destination = HERE / 'native' / f'seed-{seed}'
        destination.mkdir(parents=True, exist_ok=True)
        for path in run.rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(run)
            target = destination / relative
            if path.suffix == '.ckpt':
                # These temporary restore checkpoints are not repository assets.
                artifacts[str(target.relative_to(HERE))] = dict(
                    archived=False, bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == '.log':
                target = target.with_suffix('.log.gz')
                compressed(path, target)
            else:
                shutil.copyfile(path, target)
            artifacts[str(target.relative_to(HERE))] = dict(
                archived=True, bytes=target.stat().st_size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        processes.extend(manifest['completed'])
    text_run = ROOT / 'output' / 'item9-semantic-diagnostic-complete'
    final_runs.add(text_run)
    text_data = json.loads((HERE / 'semantic' / 'text.json').read_text())
    assert text_data['source'] == source and text_data['source_unchanged']
    process = json.loads((text_run / 'process.json').read_text())
    assert process['exit_code'] == 1 and not text_data['curriculum_passed']
    for path in text_run.iterdir():
        if path.suffix == '.log':
            compressed(path, HERE / 'semantic' / (path.name + '.gz'))
        elif path.is_file():
            shutil.copyfile(path, HERE / 'semantic' / path.name)
    diagnostics = HERE / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    index = []
    candidates = sorted(set((ROOT / 'output').glob('item9-native-*')) |
                        set((ROOT / 'output').glob('item9-semantic-*')))
    for run in candidates:
        if run in final_runs or not run.is_dir():
            continue
        entry = dict(run=run.name, files=[])
        for path in run.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.log', '.xml'):
                relative = path.relative_to(run)
                target = diagnostics / run.name / (str(relative) + '.gz')
                compressed(path, target)
                entry['files'].append(str(target.relative_to(diagnostics)))
        if entry['files']:
            index.append(entry)
    (diagnostics / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    summary = dict(source_sha256=source_hash, native_processes=processes,
        text_process=process, artifacts=artifacts,
        harness={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in HERE.glob('*.py')},
        protocol_sha256=hashlib.sha256((HERE / 'PROTOCOL.md').read_bytes()).hexdigest())
    (HERE / 'provenance.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(source_sha256=source_hash, native_arms=len(processes),
        text_exit_code=process['exit_code'], diagnostic_runs=len(index)), indent=2))


if __name__ == '__main__':
    main()
