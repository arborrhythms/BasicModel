"""Verify every committed executable/configuration blob against the sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main(revision, output):
    commit = subprocess.check_output(['git', 'rev-parse', revision + '^{commit}'],
                                     cwd=ROOT, text=True).strip()
    source = json.loads((HERE / 'source-manifest.json').read_text())
    names = subprocess.check_output(['git', 'ls-tree', '-r', '-z', '--name-only', commit],
                                    cwd=ROOT).decode().split('\0')
    suffixes = {'bin': {'.py'}, 'test': {'.py'},
                'data': {'.xml', '.xsd', '.grammar', '.json'}}
    committed = {name for name in names if name and (
        name in {'pytest.ini', 'Makefile', 'requirements.txt', 'README.md'} or
        Path(name).suffix in suffixes.get(Path(name).parts[0], set()))}
    assert committed == set(source), dict(missing=sorted(set(source) - committed),
                                          extra=sorted(committed - set(source)))
    requests = ''.join(f'{commit}:{path}\n' for path in sorted(source)).encode()
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT,
                            input=requests, capture_output=True, check=True)
    cursor = 0
    for path in sorted(source):
        end = result.stdout.index(b'\n', cursor)
        _object, kind, size = result.stdout[cursor:end].split()
        assert kind == b'blob', path
        cursor = end + 1
        blob = result.stdout[cursor:cursor + int(size)]
        assert hashlib.sha256(blob).hexdigest() == source[path], path
        cursor += int(size)
        assert result.stdout[cursor:cursor + 1] == b'\n'
        cursor += 1
    assert cursor == len(result.stdout)
    record = dict(commit=commit, source_files=len(source),
        source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        all_committed_source_blobs_match=True, source_paths_match=True,
        full_receipt='full/result.json.gz', depth3_campaign='red; unchanged assertion retained')
    if output:
        Path(output).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('revision', nargs='?', default='HEAD')
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.revision, args.output)
