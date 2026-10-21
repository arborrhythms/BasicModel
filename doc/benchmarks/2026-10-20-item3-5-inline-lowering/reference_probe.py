"""Retain actual candidate/partition reads from the scoped reference cases."""
import json
import os
from pathlib import Path

import pytest

from arrival_capture import plain


@pytest.fixture(autouse=True)
def observe_reference_reads(request, monkeypatch):
    if request.node.path.name != 'test_reference_close.py':
        return
    from ReferenceClose import ReferenceClose
    records = request.node._reference_evidence = []
    accessible = ReferenceClose.accessible
    individuals = ReferenceClose.individuals
    commit = ReferenceClose.commit

    def read_accessible(resolver, frames):
        result = accessible(resolver, frames)
        records.append(plain(dict(call='accessible', frames=[str(frame) for frame in frames],
                                  records=result)))
        return result

    def read_individuals(resolver, inputs):
        result = individuals(resolver, inputs)
        records.append(plain(dict(call='individuals', inputs=inputs, output=result)))
        return result

    def close(resolver, *args, **kwargs):
        result = commit(resolver, *args, **kwargs)
        records.append(plain(dict(call='commit', audit=resolver.last_audit,
            rows=[dict(index=i, occurrence=resolver.store.occurrence_of(i),
                       refs=resolver.store.raw_meaning_of(i).role_refs,
                       point=resolver.store.slots[i, 0]) for i in range(len(resolver.store))])))
        return result

    monkeypatch.setattr(ReferenceClose, 'accessible', read_accessible)
    monkeypatch.setattr(ReferenceClose, 'individuals', read_individuals)
    monkeypatch.setattr(ReferenceClose, 'commit', close)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    records = getattr(item, '_reference_evidence', None)
    if records is None or report.when != 'call':
        return
    folder = Path(os.environ['REFERENCE_EVIDENCE'])
    folder.mkdir(parents=True, exist_ok=True)
    (folder/(item.name+'.json')).write_text(json.dumps(dict(node=item.nodeid,
        outcome=report.outcome, failure=str(report.longrepr), reads=records), indent=2)+'\n')
