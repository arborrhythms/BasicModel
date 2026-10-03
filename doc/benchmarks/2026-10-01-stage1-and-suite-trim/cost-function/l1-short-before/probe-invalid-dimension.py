"""Run the unchanged L1 assertions with a short, uncompiled native fixture."""
import sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import pytest, util
import test_compiled_word_chunk as fixtures
from test_concept_readout_l1 import test_real_runbatch_stages_l1_once_and_reports_it_separately
original=fixtures._tiny_canonical_model
with pytest.MonkeyPatch.context() as monkeypatch:
 def small(tmp_path, monkeypatch, **kwargs):
  model=original(tmp_path,monkeypatch,input_width=32,word_buckets='8',dimension=8,stm_capacity=4,**kwargs)
  monkeypatch.setattr(util,'TheCompileBackend','none')
  return model
 monkeypatch.setattr(fixtures,'_tiny_canonical_model',small)
 with tempfile.TemporaryDirectory() as directory:
  test_real_runbatch_stages_l1_once_and_reports_it_separately(Path(directory),monkeypatch,bool(int(sys.argv[1])))
