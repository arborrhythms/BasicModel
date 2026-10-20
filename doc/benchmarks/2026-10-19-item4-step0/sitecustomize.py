"""Measurement-only overlay, inherited by the standing CLI subprocesses."""
import os
if os.environ.get('ITEM4_GATE_ARM'):
    import gate_runtime
    gate_runtime.install()
