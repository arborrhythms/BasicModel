"""pytest configuration -- add bin/etc to sys.path for optional submodules."""
import os
import sys

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_ETC = os.path.join(_BIN, "etc")
if _ETC not in sys.path:
    sys.path.insert(0, _ETC)

import pytest


@pytest.fixture(autouse=True)
def _reset_global_singletons():
    """Reset cross-test global singletons before every test.

    ``Language.TheGrammar`` and ``util.TheXMLConfig`` are module-level
    singletons. A prior test that configured a different model leaves
    ``TheGrammar._configured=True`` (so a later test's ``from_config``
    skips grammar reconfiguration and builds with the wrong level
    shapes -> ``n_vectors=0`` / stale-dim dynamic embedding) and stale
    ``validate_config`` requirement closures. Many helpers already do
    ``Language.TheGrammar._configured = False`` ad hoc; doing it
    uniformly here makes the suite order-independent (tests that pass
    in isolation no longer fail by file-ordering).
    """
    import Language
    from util import TheXMLConfig
    Language.TheGrammar._configured = False
    TheXMLConfig._requirements.clear()
    yield
