"""Central RUN_SLOW gate, including historical decorators and MPS opt-ins."""
import ast
from functools import lru_cache
import inspect
import os
import textwrap
import pytest


@lru_cache(maxsize=None)
def _switch_gated(obj, decorators_only=False):
    """Recognize the existing gate itself, never a prose mention of slow tests."""
    if obj is None:
        return False
    try:
        tree=ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError,TypeError,IndentationError,SyntaxError):
        return False
    definition=tree.body[0]
    nodes=list(getattr(definition,'decorator_list',()))
    if not decorators_only:
        nodes.extend(n.test for n in ast.walk(definition) if isinstance(n,ast.If))
    for outer in nodes:
        for node in ast.walk(outer):
            if isinstance(node,ast.Name) and node.id in ('RUN_SLOW','_RUN_SLOW','RUN_MPS_SLOW'):
                return True
            if isinstance(node,ast.Call) and node.args and isinstance(node.args[0],ast.Constant):
                function=node.func
                if (isinstance(function,ast.Attribute) and function.attr in ('getenv','get')
                        and node.args[0].value in ('RUN_SLOW','RUN_MPS_SLOW')):
                    return True
    return False


def is_slow(item):
    return (item.get_closest_marker('slow') is not None
            or any('RUN_SLOW' in str(marker.kwargs.get('reason',''))
                   for marker in item.iter_markers('skipif'))
            or _switch_gated(getattr(item,'obj',None))
            or _switch_gated(getattr(item,'cls',None), True))


def pytest_configure(config):
    config.addinivalue_line(
        'markers','slow: long training/quality/compilation check; opt in with RUN_SLOW=1')


def pytest_collection_modifyitems(items):
    enabled=os.environ.get('RUN_SLOW')=='1'
    skipped=pytest.mark.skip(reason='slow -- set RUN_SLOW=1')
    for item in items:
        if is_slow(item):
            marked=item.get_closest_marker('slow') is not None
            if not marked:
                item.add_marker(pytest.mark.slow)
            # Historical gates keep their own exact opt-in conditions (for
            # example RUN_MPS_SLOW alone). Add metadata, not a second gate.
            if not enabled and marked:
                item.add_marker(skipped)
