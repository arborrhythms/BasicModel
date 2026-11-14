"""Download and verify the external files declared in data/external-files.json."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREPARE_COMMAND = '.venv/bin/python bin/prepare_data.py'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, entry):
    if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
        raise ValueError(f'content mismatch: {path}')


def require_local(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path} is absent; run {PREPARE_COMMAND}')
    return path


def prepare(root=ROOT, manifest=None):
    root = Path(root)
    manifest = manifest or json.loads((ROOT / 'data/external-files.json').read_text())
    with tempfile.TemporaryDirectory(prefix='basicmodel-data-') as temporary:
        cached = {}
        for entry in manifest['files']:
            destination = root / entry['path']
            if destination.is_file():
                verify(destination, entry)
                print(f'verified {entry["path"]}')
                continue
            source = entry.get('archive', entry)
            url = entry['url']
            if url not in cached and 'archive' not in entry:
                # A plain file needs no archive cache or second copy. Download
                # beside its destination, authenticate the staged bytes, then
                # install atomically. A repeated URL can reuse this exact file.
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                    staged = Path(output.name)
                    try:
                        request = urllib.request.Request(url, headers={'User-Agent': 'BasicModel data preparation'})
                        with urllib.request.urlopen(request, timeout=120) as response:
                            shutil.copyfileobj(response, output)
                        output.flush()
                        verify(staged, entry)
                        staged.replace(destination)
                    finally:
                        staged.unlink(missing_ok=True)
                cached[url] = destination
                print(f'prepared {entry["path"]}')
                continue
            if url not in cached:
                downloaded = Path(temporary) / str(len(cached))
                request = urllib.request.Request(url, headers={'User-Agent': 'BasicModel data preparation'})
                with urllib.request.urlopen(request, timeout=120) as response, downloaded.open('wb') as output:
                    shutil.copyfileobj(response, output)
                verify(downloaded, source)
                cached[url] = downloaded
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Stage beside the destination so replace is atomic on its filesystem.
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                staged = Path(output.name)
                try:
                    if 'archive' in entry:
                        with zipfile.ZipFile(cached[url]) as archive:
                            payload = archive.read(entry['archive']['member'])
                        member = entry['archive']['payload']
                        if len(payload) != member['bytes'] or hashlib.sha256(payload).hexdigest() != member['sha256']:
                            raise ValueError(f'archive member mismatch: {entry["path"]}')
                        if 'header' in entry:
                            ending = entry['line_ending'].encode()
                            payload = entry['header'].encode() + ending + ending.join(payload.splitlines()) + ending
                        output.write(payload)
                    else:
                        with cached[url].open('rb') as stream:
                            shutil.copyfileobj(stream, output)
                    output.flush()
                    verify(staged, entry)
                    staged.replace(destination)
                finally:
                    staged.unlink(missing_ok=True)
            print(f'prepared {entry["path"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    prepare(args.root)
