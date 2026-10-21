"""Retain both actual thought readings and their unchanged R+A keep."""
import json
import os
from pathlib import Path
import pytest
from arrival_capture import plain, meaning


@pytest.fixture(autouse=True)
def capture_decomposition_keep(request, monkeypatch):
    if 'test_forced_native_decomposition_through_training_driver' not in request.node.name:
        return
    from Models import BasicModel
    import WalkTrials
    pending = []
    once = BasicModel._run_selected_thought_once
    def execute(model, *args, **kwargs):
        result = once(model, *args, **kwargs)
        pending.append(dict(row=kwargs['row'], fork=kwargs.get('fork') is not None,
            work=result.work.spent, meaning=meaning(result.meaning),
            records=[dict(kind=r.kind, operation=r.operation, level=r.level,
                meaning=None if r.meaning is None else meaning(r.meaning)) for r in result.records]))
        return result
    monkeypatch.setattr(BasicModel, '_run_selected_thought_once', execute)
    observe = WalkTrials.observe_comparison
    def compared(model, kind, audit, **kwargs):
        if kind == 'think':
            with Path(os.environ['DECOMPOSITION_KEEP_FILE']).open('a') as output:
                output.write(json.dumps(plain(dict(case=request.node.nodeid, row=kwargs.get('row'),
                    readings=list(pending), comparison=audit)))+'\n')
            pending.clear()
        return observe(model, kind, audit, **kwargs)
    monkeypatch.setattr(WalkTrials, 'observe_comparison', compared)
