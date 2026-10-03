"""Verify the owner resets the registry before entering a captured forward."""
from pathlib import Path
import sys,pytest
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
from Spaces import ConceptualSpace
from Layers import Error
old=ConceptualSpace.Start
def start(self):
    old(self)
    self._intra_errors=Error()
ConceptualSpace.Start=start
raise SystemExit(pytest.main(['-q','--tb=short','test/test_query_phase_fullgraph.py']))
