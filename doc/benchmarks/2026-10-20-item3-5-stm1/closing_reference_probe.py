"""Read-only diagnosis of the saved native-61 closing initialization."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def record_closing_reads(monkeypatch):
    from ClauseRow import ClauseRows
    from ThoughtReferences import bindings, open_slots

    read = ClauseRows.read_reference_bindings
    path = Path(os.environ['CLOSING_REFERENCE_TRACE'])

    def described(value):
        return dict(refs=value.role_refs, open=open_slots(value), bindings=bindings(value))

    def measured(store, index, meaning, **kwargs):
        result = read(store, index, meaning, **kwargs)
        if open_slots(result) != open_slots(meaning):
            names = bindings(meaning).get('_forward_references', ())
            witnesses = []
            for address in store._forward_addresses(name for _, name in names):
                location = store._index_occurrences.get(('ltm', store.address_domain, address))
                if location is not None:
                    witnesses.append(dict(index=location, value=described(store.raw_meaning_of(location))))
            record = dict(index=index, exclusions=kwargs.get('exclude', ()),
                          before=described(meaning), after=described(result),
                          candidate_occurrences=witnesses)
            with path.open('a') as output:
                output.write(json.dumps(record, default=str) + '\n')
        return result
    monkeypatch.setattr(ClauseRows, 'read_reference_bindings', measured)
