"""Record the old uniform-only observer's failed scale comparison."""
import json
import os
from pathlib import Path

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    yield
    if call.excinfo is None:
        return
    for frame in call.excinfo.traceback:
        values = frame.frame.f_locals
        if frame.name != 'stepped' or 'alternatives' not in values:
            continue
        entry, row = values['entry'], values['row']
        result = dict(test=item.nodeid, scale=values['scale'],
            walks=values['walks'], rounds=values['rounds'],
            alternatives=values['alternatives'], walk=entry['walk'],
            fraction=entry['module'].exploration_fraction,
            temperature=entry['module'].exploration_temperature,
            logits=entry['original'][row].tolist(),
            eligible=entry['eligible'][row].tolist(),
            action=values['action'])
        with Path(os.environ['PROPOSAL_TRACE']).open('a') as stream:
            stream.write(json.dumps(result)+'\n')
