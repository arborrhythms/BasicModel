"""Losslessly retain large tensor captures with their original byte hashes.

Restore a named capture with gzip -dk path.pt.gz. No tensor is reserialized.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
manifest = HERE/'packed-captures.json'
records = json.loads(manifest.read_text()) if manifest.exists() else []
for path in sorted(HERE.rglob('*.pt')):
    if path.stat().st_size < 32 * 1024**2:
        continue
    archive = path.with_suffix(path.suffix+'.gz')
    assert not archive.exists(), archive
    original = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    with path.open('rb') as source, gzip.open(archive, 'wb', compresslevel=6) as output:
        shutil.copyfileobj(source, output)
    with gzip.open(archive, 'rb') as source:
        restored = hashlib.file_digest(source, 'sha256').hexdigest()
    assert original == restored
    records.append(dict(original=str(path.relative_to(HERE)),
        compressed=str(archive.relative_to(HERE)), original_bytes=path.stat().st_size,
        original_sha256=original, compressed_bytes=archive.stat().st_size,
        compressed_sha256=hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()))
    manifest.write_text(json.dumps(records, indent=2)+'\n')
    path.unlink()
    print(records[-1]['compressed'], records[-1]['compressed_bytes'], flush=True)
