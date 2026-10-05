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
events = []
choose = model.languageSpace.choose_operation
def observed(state, *args, **kwargs):
    proposal = choose(state, *args, **kwargs)
    choice = proposal[0] if isinstance(proposal, tuple) and not hasattr(proposal, 'kind') else proposal
    events.append(dict(depth=state[1].tolist(), before=describe(state[0]),
                       kind=choice.kind.tolist(), op=choice.local_op.tolist(), candidate=describe(choice.candidate), operands=describe(proposal[-1]) if not hasattr(proposal, 'kind') else None))
    return proposal
model.languageSpace.choose_operation = observed
with torch.no_grad():
    u = _capture_program_probe(model, ['1 plus 2', '3 plus 4'])
    held = model.resolveAnswer(u, (What.supervised(0), What.supervised(1)))
    c = model.reverseOutput(u, held)
result.update(compose_rules=[[(r.method_name, r.canonical, str(r.reference_orders)) for r in rules] for rules in (model.languageSpace._compose_binary_rules, model.languageSpace._compose_unary_rules)], events=events, binary_names=model.languageSpace._generate_binary_names, unary_names=model.languageSpace._generate_unary_names, basis=describe(book.getW()), native_ps=describe(model.perceptualSpace.subspace.what.getW()),
              evidence=book.mereology._definitions(range(len(book.W))),
              conceptual_answer=describe(held.conceptual_answer), concepts=describe(c.concepts),
              percepts=describe(c.percepts), actual=describe(c.actual),
              words={name:describe(getattr(model.inputSpace,name,None)) for name in
                     ('_ar_word_concept_rows','_ar_word_object_rows','_ar_word_concept_atoms','_ar_word_object_atoms','_ar_readout_coefficients','_ar_grammar_object_atoms')},
              pushed=describe(getattr(model,'_tensor_pushed_ideas',None)),
              root=describe(getattr(model,'_stm_single_S',None)))
print(json.dumps(result,indent=2))
model.End(); model.symbolSpace.soft_reset()
