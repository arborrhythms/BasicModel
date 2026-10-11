"""Teach only the pair chooser on the retained forward artifact, then replay.
This is a reachability diagnostic, not a learned language/order certificate.
"""
import ast, hashlib, json, sys, warnings
from pathlib import Path
sys.path[:0]=['bin','test']
import torch
import Language
import test_stm_recon_from_cleared_cache as fixture
from DecompositionChooser import DecompositionChooser
warnings.filterwarnings('ignore');torch.set_num_threads(1);torch.manual_seed(20261010)
folder=Path(__file__).resolve().parent
saved=torch.load(folder/'probe.pt',weights_only=False)
model=fixture._make_serial_model()
args,kwargs,_=saved['captures'][0]
original=Language.LanguageSpace._bounded_binary_reconstruction
calls=[]
def capture(*a,**kw):
    if torch.equal(a[1],args[0]) and type(a[0]).__name__=='IntersectionLayer':calls.append((a,kw))
    return original(*a,**kw)
Language.LanguageSpace._bounded_binary_reconstruction=staticmethod(capture)
with torch.no_grad():model._decode_conceptual_sentence(*args,**kwargs)
Language.LanguageSpace._bounded_binary_reconstruction=staticmethod(original)
a,kw=calls[0];kw=dict(kw);kw['return_details']=True
chooser=model.languageSpace.decomposition_chooser
rows=torch.arange(a[5].shape[1])[None];target=torch.tensor([[0,1]])
optimizer=torch.optim.SGD(chooser.parameters(),lr=.1)
results=[]
for step in range(65):
    result=original(*a,**kw);detail=result[3]
    loss,present,correct=DecompositionChooser.teacher_loss(detail,rows,rows,target)
    if step in (0,64):
        results.append(dict(step=step,loss=float(loss.detach()),present=present.tolist(),
            correct=correct.tolist(),weight=chooser.weight.detach().tolist(),selected=detail['selected'].tolist(),
            left=fixture._topk_decode(saved['codebook'],result[0][:,None]).tolist(),
            right=fixture._topk_decode(saved['codebook'],result[1][:,None]).tolist()))
    if step<64:
        optimizer.zero_grad();loss.mean().backward();optimizer.step()
with torch.no_grad():
    r=model._decode_conceptual_sentence(*args,**kwargs)
    tk=fixture._topk_decode(saved['codebook'],r[0][:,:2])
    overlap=fixture._topk_overlap(saved['target'],tk)
# Replay the reviewed old raw-residual override with those same learned weights.
path=Path('output/item6-followup/h-source/bin/Language.py')
tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LanguageSpace')
node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_bounded_binary_reconstruction');node.decorator_list=[]
namespace=vars(Language).copy();exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
Language.LanguageSpace._bounded_binary_reconstruction=staticmethod(namespace[node.name])
with torch.no_grad():
    old=model._decode_conceptual_sentence(*args,**kwargs)
    oldtk=fixture._topk_decode(saved['codebook'],old[0][:,:2]);oldoverlap=fixture._topk_overlap(saved['target'],oldtk)
Language.LanguageSpace._bounded_binary_reconstruction=staticmethod(original)
report=dict(protocol=__doc__,artifact_sha256=hashlib.sha256((folder/'probe.pt').read_bytes()).hexdigest(),
    presentations=64,optimizer='SGD .1; pair chooser only',results=results,
    repaired_overlap=overlap,old_override_overlap=oldoverlap,repaired_topk=tk.tolist(),old_override_topk=oldtk.tolist())
(folder/'learned-probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));model.End()
