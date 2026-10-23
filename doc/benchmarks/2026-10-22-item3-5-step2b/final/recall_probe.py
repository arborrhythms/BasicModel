"""Observe batch/context identity at the retained reasoning failure."""
import json, os, traceback
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def observe_recall(monkeypatch):
    from Models import BasicModel
    original=BasicModel._check_recall_batch
    def check(model,batch=None):
        state=dict(requested=batch,old=model.__dict__.get('_what_recall_batch_size'),
            inferred=model._priming_batch_size(),
            rows={k:len(v) for k,v in model.__dict__.get('_what_recall_sentence_history',{}).items()},
            stack=traceback.format_stack(limit=7))
        with Path(os.environ['RECALL_PROBE']).open('a') as stream:stream.write(json.dumps(state)+'\n')
        return original(model,batch)
    monkeypatch.setattr(BasicModel,'_check_recall_batch',check)
