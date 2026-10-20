"""Read-only call trace of the unchanged decomposition's thought writes."""
import json, os, traceback
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def writes(monkeypatch, request):
    from Models import BasicModel
    from PartClosing import PartClosing
    from ThoughtReferences import bindings
    import ThoughtStream
    running = [False]
    outer = BasicModel.run_selected_thought
    def run(*args, **kwargs):
        running[0] = True
        try:
            return outer(*args, **kwargs)
        finally:
            running[0] = False
    def log(record):
        record['test'] = request.node.nodeid
        with Path(os.environ['THOUGHT_WRITES']).open('a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    write = ThoughtStream.write
    def record(model, meaning, **kwargs):
        if running[0]:
            log(dict(kind='write', refs=meaning.role_refs, children=len(meaning.constituents),
                bindings={k:v for k,v in bindings(meaning).items() if k not in ('_reference_sites','_observed_symbol_activations')},
                stack=traceback.format_stack(limit=7)))
        return write(model, meaning, **kwargs)
    commit = PartClosing._commit
    def plan(closing):
        if running[0]:
            log(dict(kind='Part', records=[dict(symbol=v['symbol'],pair=v['pair'],seat=v['seat'],
                operation=v['value'].operation,formal_name=v['value'].formal_name) for v in closing._plan],
                occupations=closing._occupations,stack=traceback.format_stack(limit=9)))
        return commit(closing)
    monkeypatch.setattr(BasicModel,'run_selected_thought',run)
    monkeypatch.setattr(ThoughtStream,'write',record)
    monkeypatch.setattr(PartClosing,'_commit',plan)
