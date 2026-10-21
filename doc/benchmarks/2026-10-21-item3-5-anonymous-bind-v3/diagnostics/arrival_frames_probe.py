"""Read-only frame inventory for the captured reverse-2 failure."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def observe_frames(monkeypatch):
    from ReferenceClose import ReferenceClose
    original = ReferenceClose.accessible
    target = Path(os.environ['ARRIVAL_FRAMES_FILE'])
    assert not target.exists()

    def accessible(resolver, frames):
        result = original(resolver, frames)
        store = resolver.store
        rows = []
        for frame in frames:
            rows.append(dict(row=frame.row_id, order=frame.order,
                symbols=frame.symbol_ids, activations=frame.activations))
        records = []
        for address, (location, context) in result.items():
            row = store.row(location)
            records.append(dict(address=address, location=location,
                order=int(row['order']), identity=context['identity'],
                refs=row['meaning'].role_refs))
        with target.open('a') as stream:
            stream.write(json.dumps(dict(frames=rows, records=records))+'\n')
        return result

    monkeypatch.setattr(ReferenceClose, 'accessible', accessible)
