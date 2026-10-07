"""One unseeded paired MM trajectory, separate from the thirty standing runs."""
import hashlib,json,os,random,subprocess,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'mm-bisection'

def child(version):
    import numpy as np
    import torch
    sys.path[:0]=[str(Path.cwd()/'bin'),str(Path.cwd()/'test')]
    from test_mm_xor import _fresh_model
    from util import init_device
    init_device('cpu');torch.set_num_threads(1)
    state=torch.load(OUT/'unseeded-entry.pt',weights_only=False,map_location='cpu')
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    def digest(model):
        h=hashlib.sha256()
        for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return dict(parameters=h.hexdigest(),torch_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    model,_,data=_fresh_model(str(Path.cwd()/'data/MM_xor.xml'))
    result=dict(version=version,construction=digest(model),trajectory=[])
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    loader=data.data_loader(split='train',num_streams=4)
    for epoch in range(200):
        raw,answer=next(iter(loader))
        inp=model.inputSpace.prepInput(raw);target=model.outputSpace.prepOutput(answer)
        optimizer.zero_grad();_,_,value,_=model.forward(inp)
        target=target.to(value.device)
        while target.dim()<value.dim():target=target.unsqueeze(-1)
        loss=torch.nn.functional.mse_loss(value,target.expand_as(value))
        loss.backward();optimizer.step()
        result['trajectory'].append(dict(epoch=epoch+1,mse=float(loss.detach()),predictions=value.detach().cpu().flatten().tolist(),**digest(model)))
    (OUT/(version+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    model.End();model.symbolSpace.soft_reset()


def main():
    sys.path.insert(0,str(ROOT/'test'))
    import bounded_tests as bounded
    source=bounded.source_snapshot(ROOT)
    assert source==json.loads((HERE/'delivered-source/source.json').read_text())
    OUT.mkdir(exist_ok=False)
    import numpy as np
    import torch
    torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state()),OUT/'unseeded-entry.pt')
    baseline=OUT/'round3a-source'
    with zipfile.ZipFile(HERE/'before/source.zip') as archive:archive.extractall(baseline)
    plan=dict(seed=None,epochs=200,retries=0,replacements=0,baseline='cce3a4f7b',
        scope='paired raw MM path; captured OS-initialized RNG state, all 200 epochs, no threshold early stop')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    processes=[]
    for version,cwd in [('round3a',baseline),('final',ROOT)]:
        env=bounded.worker_environment(cwd);env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false')
        env.pop('BASIC_SEED',None)
        result=bounded.run_guarded([sys.executable,str(Path(__file__).resolve()),version],cwd=cwd,env=env,log_path=OUT/(version+'.log'),memory_bytes=8*bounded.GIB,timeout=1800)
        processes.append(dict(version=version,**result));print(json.dumps(processes[-1]),flush=True)
    (OUT/'processes.json').write_text(json.dumps(processes,indent=2)+'\n')
    a,b=[json.loads((OUT/(v+'.json')).read_text()) for v in ('round3a','final')]
    equal=[x==y for x,y in zip(a['trajectory'],b['trajectory'],strict=True)]
    result=dict(plan=plan,construction_equal=a['construction']==b['construction'],
        all_trajectories_identical=all(equal),identical_epochs=sum(equal),
        first_different_epoch=next((i+1 for i,v in enumerate(equal) if not v),None),
        first_forward_equal=a['trajectory'][0]['predictions']==b['trajectory'][0]['predictions'],
        source_matched=source==bounded.source_snapshot(ROOT))
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[1])
    else:main()
