"""Check staged source bytes and LFS evidence before the authorized landing."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def git(*args, input=None):
    return subprocess.run(['git', *args], cwd=ROOT, input=input, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def blobs(names):
    data = git('cat-file', '--batch', input=('\n'.join(names) + '\n').encode())
    cursor = 0
    result = []
    for name in names:
        stop = data.index(b'\n', cursor)
        header = data[cursor:stop].split()
        assert len(header) == 3 and header[1] == b'blob', (name, header)
        size = int(header[2])
        cursor = stop + 1
        result.append(data[cursor:cursor + size])
        cursor += size
        assert data[cursor:cursor+1] == b'\n'
        cursor += 1
    assert cursor == len(data)
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    source = json.loads((HERE / 'measured-source/source.json').read_text())
    names = sorted(source)
    staged = {name: hashlib.sha256(data).hexdigest()
              for name, data in zip(names, blobs([':' + name for name in names]))}
    assert staged == source
    changed = git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0')
    metadata = git('cat-file', '--batch-check',
                   input=('\n'.join(':' + name for name in changed) + '\n').encode()).decode().splitlines()
    entries = {}
    for name, line in zip(changed, metadata):
        fields = line.split()
        assert len(fields) == 3 and fields[1] == 'blob', (name, fields)
        entries[name] = dict(oid=fields[0], size=int(fields[2]))
    assert len(entries) == len(changed)
    oversized = {name: item['size'] for name, item in entries.items() if item['size'] > 100*1024*1024}
    assert not oversized, oversized
    small = [name for name, item in entries.items() if item['size'] < 1024]
    pointers = {}
    for name, content in zip(small, blobs([':' + name for name in small])):
        if content.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            match = re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n', content)
            assert match, name
            sha, size = match[1].decode(), int(match[2])
            assert (ROOT/name).stat().st_size == size and digest(ROOT/name) == sha, name
            pointers[name] = dict(sha256=sha, bytes=size, working_bytes_match=True)
    result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        source_files=len(source),source_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        all_staged_source_blobs_match=True, staged_changed_files=len(changed),
        by_top_level=dict(Counter(name.split('/')[0] for name in changed)),
        staged_blobs_over_100_mib=oversized, lfs_evidence=pointers,
        tracked_source_changes_after_measurement=False)
    with (HERE/'staged-source-verification.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
