"""Follow the paired MM discrepancy at the first forward; no optimizer steps."""
import hashlib,json,os,random,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'mm-when-readers'

def child(variant):
    sys.path[:0]=[str(Path.cwd()/'bin'),str(Path.cwd()/'test')]
    import torch,numpy as np
    from test_mm_xor import _fresh_model
    from util import init_device
    import Spaces
    from unittest.mock import patch
    from contextlib import ExitStack
    init_device('cpu');torch.set_num_threads(1)
    state=torch.load(HERE/'mm-bisection/unseeded-entry.pt',weights_only=False,map_location='cpu')
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    def rng():return dict(torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest())
    model,_,data=_fresh_model(str(Path.cwd()/'data/MM_xor.xml'))
    result=dict(variant=variant,construction_rng=rng(),training_steps=0)
    raw,target=next(iter(data.data_loader(split='train',num_streams=4)))
    inp=model.inputSpace.prepInput(raw)
    result.update(raw=raw.tolist() if torch.is_tensor(raw) else raw,input=inp.detach().tolist(),before_rng=rng())
    with ExitStack() as stack:
        import traceback
        original=Spaces.WhenStartDurationEncoding.encode
        relative=Spaces.DocumentWhenEncoding.encode
        sites={'input': {'Spaces.py:_lex_and_embed'},
               'percept': {'Spaces.py:_embed_radix'},
               'attention': {'PerceptField.py:read_percepts'},
               'definitions': {'Layers.py:append_meaning'},
               'input_percept': {'Spaces.py:_lex_and_embed','Spaces.py:_embed_radix'}}[variant]
        def altered(self,start,end=None):
            caller=next((f'{Path(f.filename).name}:{f.name}' for f in reversed(traceback.extract_stack()) if '/bin/' in f.filename),None)
            return original(self,torch.zeros_like(torch.as_tensor(start))) if caller in sites else relative(self,start,end)
        stack.enter_context(patch.object(Spaces.DocumentWhenEncoding,'encode',altered))
        encoded=[]
        encoder=Spaces.WhenEncoding.encode
        import traceback
        def encode(self,*args,**kwargs):
            value=encoder(self,*args,**kwargs)
            encoded.append(dict(caller=[f'{Path(f.filename).name}:{f.name}' for f in traceback.extract_stack() if '/bin/' in f.filename][-5:],values=value.detach().tolist()))
            return value
        stack.enter_context(patch.object(Spaces.WhenEncoding,'encode',encode))
        _,_,value,_=model.forward(inp)
        result.update(predictions=value.detach().tolist(),after_rng=rng(),ir_mask=model._ir_mask_positions.detach().tolist(),when_calls=encoded)
    (OUT/(variant+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    model.End();model.symbolSpace.soft_reset()

def main():
    OUT.mkdir(exist_ok=False)
    original=json.loads((HERE/'mm-when-bisection/round3a.json').read_text())
    unchanged=json.loads((HERE/'mm-when-bisection/relative.json').read_text())
    results=[]
    for variant in ('input','percept','attention','definitions','input_percept'):
        env=dict(os.environ,BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env.pop('BASIC_SEED',None)
        with (OUT/(variant+'.log')).open('w') as log:
            process=subprocess.run([sys.executable,str(Path(__file__).resolve()),variant],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        actual=json.loads((OUT/(variant+'.json')).read_text())
        results.append(dict(variant=variant,exit_code=process.returncode,restores_landing=actual['predictions']==original['predictions'],
            changes_candidate=actual['predictions']!=unchanged['predictions'],rng_equal=actual['after_rng']==original['after_rng'],mask_equal=actual['ir_mask']==original['ir_mask']))
    (OUT/'result.json').write_text(json.dumps(dict(seed=None,training_steps=0,variants=results,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(results),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[1])
    else:main()
