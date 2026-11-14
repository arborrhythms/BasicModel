"""Preparation authenticates downloads before installing any payload; no network."""
import hashlib
import io
from pathlib import Path
import zipfile
import pytest
import prepare_data


def identity(payload):
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def test_archive_roundtrip_is_exact_and_download_is_shared(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('a.csv', b'1,2\n3,4\n')
        archive.writestr('b.csv', b'5,6\n')
    content = buffer.getvalue()
    calls = []
    def fetch(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(content)
    monkeypatch.setattr(prepare_data.urllib.request, 'urlopen', fetch)
    entries = []
    for name, raw in (('a.csv', b'1,2\n3,4\n'), ('b.csv', b'5,6\n')):
        expected = b'x,y\r\n' + raw.replace(b'\n', b'\r\n')
        entries.append(dict(path=name, url='https://example.invalid/archive', **identity(expected),
            archive=dict(member=name, payload=identity(raw), **identity(content)),
            header='x,y', line_ending='\r\n'))
    prepare_data.prepare(tmp_path, dict(files=entries))
    assert calls == ['https://example.invalid/archive']
    assert (tmp_path/'a.csv').read_bytes() == b'x,y\r\n1,2\r\n3,4\r\n'
    prepare_data.prepare(tmp_path, dict(files=entries))
    assert len(calls) == 1


def test_digest_failure_installs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'bad'))
    with pytest.raises(ValueError, match='content mismatch'):
        prepare_data.prepare(tmp_path, dict(files=[dict(path='a', url='https://example.invalid/a', **identity(b'good'))]))
    assert not list(tmp_path.iterdir())


def test_missing_data_names_the_preparation_command(tmp_path):
    with pytest.raises(FileNotFoundError, match='bin/prepare_data.py'):
        prepare_data.require_local(tmp_path/'absent')


def test_plain_payload_is_verified_before_install_and_shared(tmp_path, monkeypatch):
    payload = b'fixed source bytes\x00\xff'
    calls = []
    def fetch(request, timeout):
        calls.append(request.full_url)
        assert not (tmp_path/'first').exists()
        return io.BytesIO(payload)
    monkeypatch.setattr(prepare_data.urllib.request, 'urlopen', fetch)
    prepare_data.prepare(tmp_path, dict(files=[dict(
        path=name, url='https://example.invalid/raw', **identity(payload))
        for name in ('first', 'second')]))
    assert calls == ['https://example.invalid/raw']
    assert (tmp_path/'first').read_bytes() == (tmp_path/'second').read_bytes() == payload
    assert sorted(p.name for p in tmp_path.iterdir()) == ['first', 'second']


def test_interrupted_plain_download_leaves_no_staged_or_final_file(tmp_path, monkeypatch):
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            raise OSError('interrupted transfer')
    monkeypatch.setattr(prepare_data.urllib.request, 'urlopen', lambda *a, **k: Interrupted(b''))
    with pytest.raises(OSError, match='interrupted transfer'):
        prepare_data.prepare(tmp_path, dict(files=[dict(
            path='raw', url='https://example.invalid/raw', **identity(b'complete'))]))
    assert not list(tmp_path.iterdir())
