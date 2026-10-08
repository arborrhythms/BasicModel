"""Observe episode entry without changing a standing training's choices."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

@contextmanager
def observe(path):
    from Models import BasicModel
    from ThoughtReferences import open_slots
    original=BasicModel.run_selected_thought
    counts=dict(calls=0,open_episodes=0,bound_calls=0)
    def wrapped(model,meaning,*args,**kwargs):
        counts['calls']+=1
        counts['open_episodes' if open_slots(meaning) else 'bound_calls']+=1
        return original(model,meaning,*args,**kwargs)
    BasicModel.run_selected_thought=wrapped
    try:
        yield counts
    finally:
        BasicModel.run_selected_thought=original
        Path(path).write_text(json.dumps(counts,indent=2)+'\n')

_active=None
def pytest_sessionstart(session):
    global _active
    path=os.environ.get('THINKING_OBSERVER_OUTPUT')
    if path:
        _active=observe(path)
        _active.__enter__()

def pytest_sessionfinish(session,exitstatus):
    if _active is not None:
        _active.__exit__(None,None,None)
