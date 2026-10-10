import sys, os, json, hashlib, warnings, types, ast
from pathlib import Path
source=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]).resolve()
sys.path[:0]=[str(source/'bin'),str(source/'test')]
os.environ['MODEL_COMPILE']='none'; os.environ['BASICMODEL_DEVICE']='cpu'
import torch
import test_stm_recon_from_cleared_cache as t
import Models, Language
warnings.filterwarnings('ignore')
torch.set_num_threads(1); torch.manual_seed(20261010)
model=t._make_serial_model()
def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
initial={k:digest(v) for k,v in model.state_dict().items() if torch.is_tensor(v)}
captures=[]
original=model._decode_conceptual_sentence
def freeze(x):
    if torch.is_tensor(x): return x.detach().clone()
    if isinstance(x,dict): return {k:freeze(v) for k,v in x.items()}
    if isinstance(x,tuple):
        return type(x)(*(freeze(v) for v in x)) if hasattr(x,'_fields') else tuple(freeze(v) for v in x)
    return x
def decode(*args,**kwargs):
    result=original(*args,**kwargs)
    captures.append((freeze(args),freeze(kwargs),freeze(result)))
    return result
model._decode_conceptual_sentence=decode
S, target=t._run_forward(model)
t._clear_word_cache(model.symbolSpace)
with torch.no_grad():
    surface=model.reverseReconstruct(model._test_understanding)
    ideas=model._test_understanding.input_reconstruction.ideas
    cols=model._test_word_columns
    recon=ideas.gather(1,cols[...,None].expand(*cols.shape,ideas.shape[-1]))
owner=model._concept_owner(); W=owner.interpret.binding_atoms(owner.similarity_codebook.getW())
topk=t._topk_decode(W,recon)
result={'seed':20261010,'source':str(source),'initial':initial,'target':target.tolist(),'columns':cols.tolist(),'recon':recon.tolist(),'topk':topk.tolist(),'overlap':t._topk_overlap(target,topk),'surface_type':str(type(surface)),'captures':[]}
for a,k,d in captures:
    result['captures'].append({'root':a[0].tolist(),'end_depth':a[2].tolist(),'basis_valid':a[4].tolist(),'options':list(k),'count':d[1].tolist(),'truncated':d[2].tolist(),'actions':d[4].tolist(),'leaves':d[0].tolist()})
# Replay the same frozen forward artifacts using progressively restored paths.
def method(path, cls, name):
    tree=ast.parse(Path(path).read_text()); c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    n=next(n for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    n.decorator_list=[]; module=ast.Module(body=[n],type_ignores=[])
    namespace=vars(Models if cls=='BasicModel' else Language).copy()
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[name]
base=Path(sys.argv[3]).resolve() if len(sys.argv)>3 else source
old_walk=method(base/'bin/Models.py','BasicModel','_output_generate_walk')
old_elig=method(base/'bin/Language.py','LanguageSpace','decoder_eligibility')
current_walk=model._output_generate_walk; current_elig=Language.LanguageSpace.decoder_eligibility
result['replays']={}
search_calls=[]
bounded=Language.LanguageSpace._bounded_binary_reconstruction
def inspect_pair(*a,**k):
    detail=k.pop('return_details',False)
    r=bounded(*a,**k,return_details=True)
    d=r[3]; fits=d['relative_residual'].masked_fill(~d['allowed'],torch.inf)
    idx=d['selected']; width=fits.shape[-1]
    best=fits.flatten(1).amin(-1)
    search_calls.append(dict(operator=type(a[0]).__name__,left_indices=d['left_indices'].tolist(),
        right_indices=d['right_indices'].tolist(),selected=idx.tolist(),best=best.tolist(),
        exact=d['exact'].tolist(),ties=(fits==best[:,None,None]).nonzero().tolist(),
        selected_residual=fits.flatten(1).gather(1,idx[:,None]).tolist(),
        left_topk=t._topk_decode(W,r[0][:,None]).tolist(),right_topk=t._topk_decode(W,r[1][:,None]).tolist(),
        root_is_first=bool(torch.equal(a[1],captures[0][0][0]))))
    return r if detail else r[:3]
Language.LanguageSpace._bounded_binary_reconstruction=staticmethod(inspect_pair)
for variant in ('original','old_eligibility','old_walk_and_eligibility','no_live_constituents'):
    Language.LanguageSpace.decoder_eligibility=staticmethod((lambda *a,**k:old_elig(*a,**{key:value for key,value in k.items() if key=='case_bank'})) if variant in ('old_eligibility','old_walk_and_eligibility') else current_elig)
    if variant=='old_walk_and_eligibility':
        def walk(self,*a,**k):
            for key in ('constituents','constituent_valid','constituent_families','terminal_valid'): k.pop(key,None)
            return old_walk(self,*a,**k)
        model._output_generate_walk=types.MethodType(walk,model)
    else: model._output_generate_walk=current_walk
    rows=[]; search_calls.clear()
    with torch.no_grad():
        for a,k,d in captures:
            k=dict(k)
            if variant=='no_live_constituents':
                for key in ('constituents','constituent_valid','constituent_families'):k.pop(key,None)
            r=original(*a,**k)
            tk=t._topk_decode(W,r[0][:,:target.shape[1]])
            rows.append({'count':r[1].tolist(),'truncated':r[2].tolist(),'topk':tk.tolist(),'overlap':t._topk_overlap(target,tk),'actions':r[4].tolist()})
    result['replays'][variant]=rows
    result.setdefault('search',{})[variant]=list(search_calls)
Language.LanguageSpace.decoder_eligibility=staticmethod(current_elig)
model._output_generate_walk=current_walk
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)+'\n')
torch.save({'initial':initial,'captures':captures,'codebook':W.detach(),'target':target,'columns':cols},out.with_suffix('.pt'))
print(json.dumps({k:v for k,v in result.items() if k not in ('initial','captures','recon','search')},indent=2))
