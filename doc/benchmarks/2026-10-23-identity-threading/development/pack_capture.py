"""Lossless capture packaging with original and compressed byte hashes."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def pack(path, ledger):
    path, ledger = Path(path), Path(ledger)
    if path.stat().st_size <= 16 * 1024 * 1024:
        return
    output = path.with_name(path.name+'.gz')
    assert not output.exists(), output
    original = digest(path)
    with path.open('rb') as source, output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as target:
            shutil.copyfileobj(source, target, length=1024*1024)
    with gzip.open(output, 'rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == original
    entries = json.loads(ledger.read_text()) if ledger.exists() else []
    entries.append(dict(original=str(path.relative_to(ledger.parent)),
        compressed=str(output.relative_to(ledger.parent)), bytes=path.stat().st_size,
        gzip_bytes=output.stat().st_size, sha256=original, gzip_sha256=digest(output)))
    ledger.write_text(json.dumps(entries, indent=2)+'\n')
    path.unlink()


def save(value, path):
    import torch
    torch.save(value, path)
    pack(path, Path(path).parent/'packed-states.json')
