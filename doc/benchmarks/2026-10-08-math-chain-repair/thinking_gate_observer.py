"""Observe credit and episode entry; reject explicit seed calls in the repaired unseeded fixtures."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch

_stack=None
_data=None


def pytest_sessionstart(session):
    global _stack,_data
    import torch
    from Models import BasicModel
    import ThoughtCredit
    from ThoughtReferences import open_slots
    folder=Path(os.environ['THINKING_GATE_OUTPUT']);folder.mkdir(exist_ok=True)
    _data=dict(pid=os.getpid(),nodes=[],seed_calls_rejected=[],episodes=0,credit=[])
    _stack=ExitStack()
    original=BasicModel.run_selected_thought
    register=ThoughtCredit.register
    def unseeded(seed):
        _data['seed_calls_rejected'].append(int(seed))
        raise AssertionError('thinking gate must be unseeded')
    def episode(model,meaning,*args,**kwargs):
        _data['episodes']+=int(bool(open_slots(meaning)))
        return original(model,meaning,*args,**kwargs)
    def credited(model,value,*,costs,source):
        _data['credit'].append(dict(source=source,costs=list(map(float,costs)),
            requires_grad=value is not None and value.requires_grad,
            surrogate=None if value is None else float(value.detach())))
        return register(model,value,costs=costs,source=source)
    _stack.enter_context(patch.object(torch,'manual_seed',unseeded))
    _stack.enter_context(patch.object(BasicModel,'run_selected_thought',episode))
    _stack.enter_context(patch.object(ThoughtCredit,'register',credited))


def pytest_runtest_logreport(report):
    _data['nodes'].append(dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,
        reason=None if report.passed else str(report.longrepr)))


def pytest_sessionfinish(session,exitstatus):
    _stack.close()
    _data['exit_code']=int(exitstatus)
    folder=Path(os.environ['THINKING_GATE_OUTPUT'])
    (folder/f'{os.getpid()}.json').write_text(json.dumps(_data,indent=2)+'\n')
