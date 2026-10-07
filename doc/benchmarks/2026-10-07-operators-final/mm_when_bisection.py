"""Follow the paired MM discrepancy at the first forward; no optimizer steps."""
import hashlib,json,os,random,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'mm-when-bisection'

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
        if variant=='relative_with_old_band':
            original=Spaces.WhenStartDurationEncoding.encode
            stack.enter_context(patch.object(Spaces.DocumentWhenEncoding,'encode',lambda self,start,end=None:original(self,torch.zeros_like(torch.as_tensor(start)))))
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
    processes=[]
    for variant,cwd in [('round3a',HERE/'mm-bisection/round3a-source'),('relative',ROOT),('relative_with_old_band',ROOT)]:
        env=dict(os.environ,BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env.pop('BASIC_SEED',None)
        with (OUT/(variant+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),variant],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        processes.append(dict(variant=variant,exit_code=p.returncode))
    (OUT/'processes.json').write_text(json.dumps(processes,indent=2)+'\n')
    a,b,c=[json.loads((OUT/(v+'.json')).read_text()) for v in ('round3a','relative','relative_with_old_band')]
    result=dict(seed=None,training_steps=0,construction_rng_equal=a['construction_rng']==b['construction_rng'],
        prepared_input_equal=a['input']==b['input'],before_rng_equal=a['before_rng']==b['before_rng'],
        after_rng_equal=a['after_rng']==b['after_rng'],ir_mask_equal=a['ir_mask']==b['ir_mask'],
        relative_changes_first_forward=a['predictions']!=b['predictions'],old_band_restores_prediction=a['predictions']==c['predictions'],
        old_band_restores_rng=a['after_rng']==c['after_rng'],old_band_restores_mask=a['ir_mask']==c['ir_mask'],
        when_readers=sorted({name for call in b['when_calls'] for name in call['caller']}),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[1])
    else:main()
