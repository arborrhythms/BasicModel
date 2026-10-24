import copy,sys,time,json,resource
sys.path[:0]=['bin','test']
import torch
from MeaningKernels import MeaningAction
# Compare the identical shared basis expression, including parameter adjoints.
torch.set_num_threads(1)
old=MeaningAction(128,modifier=True).double();new=copy.deepcopy(old)
with torch.no_grad():
 old.readout.fill_(.007);old.groups.vectors.add_(torch.linspace(0,.01,old.groups.vectors.numel()).reshape_as(old.groups.vectors))
new.load_state_dict(old.state_dict())
x=torch.linspace(.1,.8,64*128,dtype=torch.float64).reshape(64,128).requires_grad_();v=x.flip(-1).detach().requires_grad_()
start=time.perf_counter();y=old(x,v);g=torch.autograd.grad(y.square().sum(),(x,v,*old.parameters()),allow_unused=True);elapsed_old=time.perf_counter()-start
original=MeaningAction.basis
def basis(self,value,**kw):
 if self.modifier:
  q=self.groups(torch.zeros((),dtype=torch.long,device=value.device))
  return q,torch.ones_like(value[...,:self.readout.shape[0]],dtype=torch.bool)
 return original(self,value,**kw)
new.basis=basis.__get__(new)
start=time.perf_counter();z=new(x,v);h=torch.autograd.grad(z.square().sum(),(x,v,*new.parameters()),allow_unused=True);elapsed_new=time.perf_counter()-start
torch.testing.assert_close(y,z,rtol=1e-12,atol=1e-12)
for a,b in zip(g,h):
 if a is None:assert b is None
 else:torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)
print(json.dumps(dict(old_seconds=elapsed_old,shared_seconds=elapsed_new,max_output_error=float((y-z).abs().max()),gradients='all equal within 1e-9')))
