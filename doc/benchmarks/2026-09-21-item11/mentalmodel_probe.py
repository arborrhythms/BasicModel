import sys, os, json, gc, random, warnings, traceback
from pathlib import Path
root = Path(sys.argv[1])
os.environ['BASICMODEL_DEVICE'] = 'cpu'
os.environ['MODEL_COMPILE'] = 'eager'
sys.path[:0] = [str(root/'bin'), str(root/'test')]
import torch
import numpy as np
import Models, Language
from test_hierarchical import _make_model

original = Language.binary_tiling_soft_dp
seed = None
def stats(value):
    finite = value[torch.isfinite(value)]
    return dict(shape=list(value.shape), finite=int(finite.numel()), total=value.numel(),
                nan=int(value.isnan().sum()), posinf=int(value.isposinf().sum()),
                neginf=int(value.isneginf().sum()),
                max_abs=float(finite.abs().max()) if finite.numel() else None)
def inspect_dp(c, r, *args, **kwargs):
    out = original(c, r, *args, **kwargs)
    if not torch.isfinite(out['reduce_marginal_op']).all():
        caller = sys._getframe(1).f_locals
        report = dict(seed=seed, root=str(root), copy=stats(c), reduce=stats(r),
                      output={k:stats(v) for k,v in out.items()})
        for name in ('x','stacked_reduced','copy_score','reduce_score'):
            value=caller.get(name)
            if torch.is_tensor(value): report[name]=stats(value)
        print('NONFINITE '+json.dumps(report),flush=True)
        path=Path('/tmp')/('item11-mental-'+root.name+'.pt')
        torch.save(dict(c=c.detach(),r=r.detach(),output={k:v.detach() for k,v in out.items()},
                        x=caller.get('x'), candidates=caller.get('stacked_reduced')),path)
    return out
Language.binary_tiling_soft_dp = inspect_dp
for seed in range(32):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = _make_model('MentalModel.xml')
    try:
        with Models.TheData.runtime_batch(['hello world'],[torch.tensor([0.0])]), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train_input,_=model.inputSpace.getTrainData()
            x=model.inputSpace.prepInput(train_input[:1])
            with torch.no_grad(): result=model.forward(x)
        print(json.dumps(dict(seed=seed,status='pass')),flush=True)
    except Exception as exc:
        print(json.dumps(dict(seed=seed,status='fail',error=str(exc))),flush=True)
        break
    finally:
        model.End(); model.symbolSpace.soft_reset(); del model
        torch._dynamo.reset(); gc.collect()
