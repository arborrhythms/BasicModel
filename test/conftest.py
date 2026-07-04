"""pytest configuration -- add bin/etc to sys.path for optional submodules."""
import os
import sys

# The suite's canonical device is CPU, declared HERE because conftest
# executes before any test module imports util (whose import freezes the
# process default device). 120 of the test files already declare this
# intent with their own ``os.environ.setdefault("BASICMODEL_DEVICE",
# "cpu")`` -- but a per-module setdefault only wins for SOLO runs (in a
# full run, whichever module imports util first decides for everyone).
# Historically the suite ran ~all-CPU anyway by ACCIDENT: an early
# module's un-restored ``init_device("cpu")`` leaked into every later
# module, and dozens of tests silently depended on that leak (the
# leak's victims were the full-suite-only test_heat_reverse_wiring
# failures: modules caught between import-time MPS globals and the
# leaked CPU default died on cross-device ops, which the heat path's
# never-break-generation ``except`` converted into a silent ON==OFF
# collapse). Declaring CPU centrally makes the de-facto contract real
# and order-independent. An explicit BASICMODEL_DEVICE in the
# environment still overrides (setdefault); device-detection unit tests
# (test_util_device) pop/restore the variable themselves.
os.environ.setdefault("_BASICMODEL_DEVICE_EXPLICIT", "1" if "BASICMODEL_DEVICE" in os.environ else "0")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
# The project root too: a handful of tests import via the package form
# (``from bin import Spaces``), which resolves only when basicmodel/ is
# on sys.path. ``python -m pytest`` gets that via the CWD entry, but the
# ``make test`` runner (``python test/test_report.py``) and any
# from-elsewhere invocation do not.
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

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


@pytest.fixture(autouse=True, scope="module")
def _restore_process_device():
    """Unwind cross-module leaks of the process-wide device.

    Several test modules flip the device at runtime
    (``util.init_device("cpu")`` in build helpers or ``setUpClass``)
    and never restore it. The flip leaks into every LATER module: a
    victim then mixes import-time globals built under the original
    default device with fresh tensors on the flipped one, and dies with
    "Expected all tensors to be on the same device" -- or worse,
    silently: ``LanguageLayer.unreduce``'s heat path catches the device
    mix in its broad never-break-generation ``except`` and degrades to
    the plain reverse, collapsing ON==OFF
    (test_heat_reverse_wiring's full-suite-only failures; minimal
    repro: test_active_payload_audit.py + test_compile_static_loop.py
    + test_heat_reverse_wiring.py).

    Module scope is deliberate: capture happens when a module's first
    test starts (AFTER collection-time imports settled the baseline)
    and restore when its last test ends -- so ``setUpClass``-era flips
    survive across the module's own tests but never reach the next
    module. The env var is restored alongside so ``auto_device()``
    re-resolution agrees.
    """
    prev_env = os.environ.get("BASICMODEL_DEVICE")
    from util import TheDevice, init_device
    prev_dev = str(TheDevice.get())
    yield
    if prev_env is None:
        os.environ.pop("BASICMODEL_DEVICE", None)
    else:
        os.environ["BASICMODEL_DEVICE"] = prev_env
    init_device(prev_dev)
