"""Observe conclusion references and both trial costs, without changing them."""
import gzip
import json
import os
from pathlib import Path
import pytest
from arrival_capture import plain, meaning
from boundary_probe import boundary_probe

@pytest.fixture(autouse=True)
def thought_boundary_probe(monkeypatch):
    import ThoughtStream
    write = ThoughtStream.write
    records = []
    def observed(model, value, **kwargs):
        try:
            return write(model, value, **kwargs)
        except Exception as error:
            frames = []
            trace = error.__traceback__
            while trace:
                local = trace.tb_frame.f_locals
                if trace.tb_frame.f_code.co_name == 'clause':
                    frames.append(dict(name=local.get('name'), reference=local.get('reference'),
                        role=local.get('role'), meaning=plain(meaning(local.get('value')))))
                trace = trace.tb_next
            records.append(dict(error=repr(error), meaning=plain(meaning(value)), frames=frames))
            raise
    monkeypatch.setattr(ThoughtStream, 'write', observed)
    yield
    p = Path(os.environ['CONTRACT_CAPTURE']).parent/'thought-boundary-probe.json.gz'
    p.write_bytes(gzip.compress((json.dumps(plain(records), indent=2)+'\n').encode(), mtime=0))
