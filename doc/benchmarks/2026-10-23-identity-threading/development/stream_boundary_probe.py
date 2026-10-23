"""Read-only failed-draw probe of pending-name stream filtering."""
import gzip
import json
import os
from pathlib import Path
import pytest
from arrival_capture import plain
from boundary_probe import boundary_probe

@pytest.fixture(autouse=True)
def stream_boundary_probe(monkeypatch):
    from ReferenceClose import ReferenceClose
    pending = ReferenceClose.pending_names
    records = []
    def observed(resolver, clause, individuals, stream):
        try:
            return pending(resolver, clause, individuals, stream)
        except Exception as error:
            frames=[]
            trace=error.__traceback__
            while trace:
                if trace.tb_frame.f_code.co_name=='pending_names':
                    local=trace.tb_frame.f_locals
                    index=local.get('index')
                    frames.append(dict(index=index, names=local.get('names'),
                        address=local.get('address'), stream=stream,
                        stream_shape=tuple(resolver.store.stream.shape),
                        row_stream=None if index is None else resolver.store.stream[index]))
                trace=trace.tb_next
            records.append(dict(error=repr(error),frames=plain(frames)))
            raise
    monkeypatch.setattr(ReferenceClose, 'pending_names', observed)
    yield
    p=Path(os.environ['CONTRACT_CAPTURE']).parent/'stream-boundary-probe.json.gz'
    p.write_bytes(gzip.compress((json.dumps(plain(records),indent=2)+'\n').encode(),mtime=0))
