"""Inference-only ablations; original native-output regression fixture and seed."""
import ast, json, tempfile, sys
from pathlib import Path
from dataclasses import replace
import pytest, torch
from reading_fixtures import use_eager_reading
from test_output_path_supervised import _native_answer_model
from test_output_walk import _capture_program_probe
from What import What
import Language, Attention, Models
HERE=Path(__file__).resolve().parent
mode=sys.argv[1] if len(sys.argv)>1 else 'current'
def old_function(module, filename, owner, name):
    tree=ast.parse((HERE/'before/bin'/filename).read_text())
    container=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner) if owner else tree
    function=next(n for n in container.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    scope=dict(vars(module))
    exec(compile(ast.Module(body=[function],type_ignores=[]),filename,'exec'),scope)
    setattr(getattr(module,owner) if owner else module,name,scope[name])
if mode=='attention': old_function(Attention,'Attention.py',None,'narrow_words')
if mode=='not': old_function(Language,'Language.py','NotLayer','forward')
if mode=='non': old_function(Language,'Language.py','NonLayer','forward')
if mode=='intersection': old_function(Language,'Language.py','IntersectionLayer','_butterfly_pair_op')
if mode=='decomp': old_function(Language,'Language.py','LanguageSpace','_bounded_binary_reconstruction')
if mode=='walk': old_function(Models,'Models.py','BasicModel','_output_generate_walk')
def stats(t):
    return {'shape':list(t.shape),'norm':t.flatten(1).norm(dim=-1).tolist(),
            'first':t[0].flatten()[:24].tolist(), 'row_diff':float((t[0]-t[1]).abs().max()),
            'slots':t.ne(0).any(-1).tolist()}
with tempfile.TemporaryDirectory() as td, pytest.MonkeyPatch.context() as mp:
    use_eager_reading(mp)
    m=_native_answer_model(Path(td),False,concept_width=264);m.eval()
    with torch.no_grad():
        u=_capture_program_probe(m,['1 plus 2','3 plus 4'])
        owned=replace(u,symbolic_state=None,conceptual_state=None,answer_seed=None)
        c=m.reverseOutput(owned,m.resolveAnswer(owned,(What.supervised(0),What.supervised(1))))
    result={'mode':mode,'concepts':stats(c.concepts),'percepts':stats(c.percepts),'actual':stats(c.actual),
            'walk':m._last_output_walk_trace[0].tolist(),'truncated':m._output_truncated.tolist()}
    (HERE/f'diagnose-output-{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    m.End();m.symbolSpace.soft_reset()
