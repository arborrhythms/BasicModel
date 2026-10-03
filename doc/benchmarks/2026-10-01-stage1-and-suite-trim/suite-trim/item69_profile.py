"""Receipt-local per-case call profile; does not change fixture execution."""
import cProfile
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pstats
import time
import pytest

@contextmanager
def record_profile(item, phase):
    profile=cProfile.Profile()
    started=time.monotonic()
    profile.enable()
    yield
    profile.disable()
    elapsed=time.monotonic()-started
    root=Path(os.environ['ITEM69_PROFILE_DIR']);root.mkdir(parents=True,exist_ok=True)
    name=hashlib.sha256(item.nodeid.encode()).hexdigest()[:16] + ('.setup' if phase == 'setup' else '')
    path=root/(name+'.prof');profile.dump_stats(path)
    stats=pstats.Stats(profile)
    rows=[]
    for (file,line,function),(primitive,calls,own,cumulative,callers) in stats.stats.items():
        rows.append(dict(file=file,line=line,function=function,primitive_calls=primitive,
                         calls=calls,self_seconds=own,cumulative_seconds=cumulative))
    result=dict(nodeid=item.nodeid,phase=phase,elapsed_seconds=elapsed,profile_file=str(path),
                functions=sorted(rows,key=lambda row:row['cumulative_seconds'],reverse=True)[:100])
    (root/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    with record_profile(item, 'setup'):
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    with record_profile(item, 'call'):
        yield
