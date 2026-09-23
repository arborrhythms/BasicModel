"""Measurement-only ablation: isolate training of primitive memberships."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
from Models import BaseModel

build = BaseModel.from_config


def without_definition_updates(*args, **kwargs):
    result = build(*args, **kwargs)
    for ws in result[0].wholeSpaces:
        primitive = getattr(ws.subspace.what, 'primitive_properties', None)
        if primitive is not None:
            primitive.members.requires_grad_(False)
    return result


BaseModel.from_config = staticmethod(without_definition_updates)
runpy.run_path(str(ROOT / 'doc/benchmarks/2026-09-21-item10/probe.py'), run_name='__main__')
