"""Executed against the unchanged HEAD archive before Stage B edits."""
import os, sys
from pathlib import Path
archive=Path(sys.argv[1]); destination=Path(sys.argv[2])
sys.path[:0]=[str(archive/'bin'),str(archive/'test')]
os.environ['BASICMODEL_DEVICE']='cpu'
import torch
from test_cs_sparse_weights import _cs
from test_structural_checkpoint import _model_with
from types import SimpleNamespace
cs=_cs(nS=64,order=3)
word,obj,meta=cs.interpret_word([7],[1],key='legacy')
model=_model_with(cs,SimpleNamespace())
torch.save(dict(source='1678ee1fb79c474ffaee5725fd035716de5f1913',word=word,obj=obj,meta=meta,
                structural=model._collect_structural_extras()),destination)
print(word,obj,meta)
