"""Read the actual close footprint without changing the admission."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def inspect_storage(monkeypatch, request):
    from ClauseRow import ClauseRows
    write = ClauseRows.write_clause

    def closing(store, clause, **kwargs):
        def describe(node):
            return dict(relation=node.relation, order=node.order,
                minted=node.minted, request=None if node.reference_request is None
                else node.reference_request['mode'], refs=str(node.refs),
                children=[describe(c) for c in node.children],
                companions=[describe(c) for c in node.companions])
        record = dict(test=request.node.nodeid, before=len(store),
            capacity=store.capacity, clause=describe(clause))
        try:
            result = write(store, clause, **kwargs)
            record['result'] = result
            return result
        finally:
            record['after'] = len(store)
            record['kinds'] = [store.KINDS[int(k)] for k in store.record_kind[:len(store)]]
            if 'bootstrap' in request.node.nodeid:
                owner = store._concept_owner()
                record['postings'] = {str(k): v for k, v in store._leaf_postings.items()}
                record['parts'] = store.parts.refs[store.parts.indices()].tolist()
                derived = owner.similarity_codebook.mereology
                record['context'] = {str(k): dict(count=n, sum=float(v.sum()))
                                     for k, (n, v) in derived.occurrence_terms().items()}
                record['codes'] = {str(r): dict(form=owner.word_surface_for_row(r),
                     norm=float(owner.similarity_codebook.W[r].norm()))
                     for r in range(len(owner.similarity_codebook.W))}
            path = Path(os.environ['STORAGE_DIAGNOSTIC'])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as output:
                output.write(json.dumps(record, default=str)+'\n')
    monkeypatch.setattr(ClauseRows, 'write_clause', closing)
