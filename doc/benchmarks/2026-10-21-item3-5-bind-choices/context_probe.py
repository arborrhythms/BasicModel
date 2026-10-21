"""Read the retained occurrence contexts after the bootstrap test's batches."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def contexts(request, monkeypatch):
    if request.node.name != 'test_both_training_trials_share_bootstrap_snapshot':
        return
    from Models import BasicModel
    from PartRead import occurrence_roots, word_rows
    from arrival_capture import plain, snapshot
    original = BasicModel.runBatch

    def batch(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        owner = model._concept_owner()
        store = owner._closed_clause_store()
        roots = occurrence_roots(owner, store)
        report = dict(snapshot=snapshot(model), contexts=[dict(record,
            words=tuple(word_rows(owner, store.parts, record['root']))) for record in roots],
            rows=[dict(index=i, order=int(store.order[i]), kind=store.KINDS[int(store.record_kind[i])],
                ref=store.raw_meaning_of(i).role_refs, when=store.when[i]) for i in range(len(store))])
        with (Path(os.environ['BOOTSTRAP_CAPTURE'])/'context.jsonl').open('a') as output:
            output.write(json.dumps(plain(report))+'\n')
        return result
    monkeypatch.setattr(BasicModel, 'runBatch', batch)
