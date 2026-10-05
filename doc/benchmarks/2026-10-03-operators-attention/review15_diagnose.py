"""Read-only native answer-path diagnosis; no training or seed override.

Fixture files: test_output_path_supervised.py, test_output_walk.py,
test_meronomy_ladder.py. Uses their unchanged model construction.
"""
import json
import sys
from pathlib import Path
import tempfile
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'test'), str(ROOT/'bin')]
from test_output_path_supervised import _native_answer_model
from test_output_walk import _capture_program_probe
from Spaces import _concept_alloc_of
from What import What

def describe(value):
    if not torch.is_tensor(value):
        return str(value)
    v = value.detach().cpu()
    return dict(shape=list(v.shape), nonzero=int(v.count_nonzero()),
                norm=float(v.float().norm()), values=v.tolist() if v.numel()<100 else None)

model = _native_answer_model(Path(tempfile.mkdtemp()), False)
model.eval()
owner = model._concept_owner()
book = owner.similarity_codebook
result = dict(initial_basis=describe(book.getW()), initial_rows=list(_concept_alloc_of(owner).layer()._tensor_row_keys))
with torch.no_grad():
    u = _capture_program_probe(model, ['1 plus 2', '3 plus 4'])
    held = model.resolveAnswer(u, (What.supervised(0), What.supervised(1)))
    c = model.reverseOutput(u, held)
result.update(basis=describe(book.getW()), native_ps=describe(model.perceptualSpace.subspace.what.getW()),
              evidence=book.mereology._definitions(range(len(book.W))),
              conceptual_answer=describe(held.conceptual_answer), concepts=describe(c.concepts),
              percepts=describe(c.percepts), actual=describe(c.actual),
              words={name:describe(getattr(model.inputSpace,name,None)) for name in
                     ('_ar_word_concept_rows','_ar_word_object_rows','_ar_word_atoms','_ar_word_concept_activations','_ar_grammar_object_atoms')},
              pushed=describe(getattr(model,'_tensor_pushed_ideas',None)),
              root=describe(getattr(model,'_stm_single_S',None)))
print(json.dumps(result,indent=2))
model.End(); model.symbolSpace.soft_reset()
