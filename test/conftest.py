"""pytest configuration — add bin/etc to sys.path for optional submodules."""
import os
import sys

_ETC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "etc")
if _ETC not in sys.path:
    sys.path.insert(0, _ETC)
